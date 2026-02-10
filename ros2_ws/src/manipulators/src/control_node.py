"""
Main control node for Kinova Gen3 torque control.

Startup:  connect → clear faults → home → low-level mode → torque mode → run loop
Shutdown: stop loop → position mode → high-level mode → home → disconnect
"""

import os
import time
import logging
import threading
import numpy as np

import rclpy

# Enable debug logging for hardware module to see action diagnostics
logging.getLogger('manipulators.hardware').setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(levelname)s: %(message)s')
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor

from .hardware import KinovaHardware
from .robot_model import RobotModel
from .diff_ik_controller import DiffIKController
from .osc_controller import OSCController
from .utility import kinova_degrees_to_radians, matrix_to_quat


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # -- Shared parameters --
        self.declare_parameter('robot_ip', '192.168.1.10')
        self.declare_parameter('username', 'admin')
        self.declare_parameter('password', 'admin')
        self.declare_parameter('urdf_file', 'assets/robots/kinova/urdf/gen3_2f85.urdf')
        self.declare_parameter('ee_frame', 'gen3_end_effector_link')
        self.declare_parameter('home_position_deg', [0.0, 344.0, 180.0, 214.0, 0.0, 315.0, 90.0])
        self.declare_parameter('initial_pose', 'home')
        self.declare_parameter('control_rate_hz', 400.0)
        self.declare_parameter('max_torque', [30.0, 30.0, 30.0, 30.0, 7.0, 7.0, 7.0])
        self.declare_parameter('controller_type', 'diff_ik')

        # -- Controller gains (shared names, values set per controller config) --
        self.declare_parameter('kp_task', [150.0, 150.0, 150.0, 80.0, 80.0, 80.0])
        self.declare_parameter('kp_joint', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('kd_joint', [12.0, 12.0, 12.0, 12.0, 4.0, 4.0, 4.0])

        # -- Diff-IK only --
        self.declare_parameter('damping', 0.01)
        self.declare_parameter('max_joint_velocity', 1.5)

        # -- OSC only --
        self.declare_parameter('kd_task', [50.0, 50.0, 50.0, 30.0, 30.0, 30.0])
        self.declare_parameter('max_wrench', [100.0, 100.0, 100.0, 30.0, 30.0, 30.0])

        # Read shared params
        self.robot_ip = self.get_parameter('robot_ip').value
        self.username = self.get_parameter('username').value
        self.password = self.get_parameter('password').value
        self.ee_frame = self.get_parameter('ee_frame').value
        self.home_deg = np.array(self.get_parameter('home_position_deg').value)
        self.initial_pose_deg = self._resolve_pose(
            self.get_parameter('initial_pose').value
        )
        self.rate_hz = self.get_parameter('control_rate_hz').value
        self.max_torque = np.array(self.get_parameter('max_torque').value)
        self.controller_type = self.get_parameter('controller_type').value

        # Resolve URDF path
        urdf_file = self.get_parameter('urdf_file').value
        if not os.path.isabs(urdf_file):
            pkg_share = get_package_share_directory('manipulators')
            urdf_file = os.path.join(pkg_share, urdf_file)
        self.urdf_path = urdf_file

        # -- Publishers --
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.ee_pose_pub = self.create_publisher(PoseStamped, 'ee_pose', 10)

        # -- Subscribers --
        self.create_subscription(PoseStamped, 'target_pose', self._on_target_pose, 1)
        self.create_subscription(Float64, 'gripper_command', self._on_gripper_command, 1)

        # -- Services --
        self.create_service(Trigger, 'e_stop', self._on_e_stop)

        # -- Shared state (written by callbacks, read by control thread) --
        self._lock = threading.Lock()
        self._target_pos = None       # (3,)  set after startup FK
        self._target_quat = None      # (4,)  xyzw
        self._gripper_target = 0.0    # 0.0=open, 1.0=closed

        # -- Control thread --
        self._running = False
        self._thread = None
        self.hw = None
        self.model = None
        self.controller = None

        self._joint_names = [f"gen3_joint_{i}" for i in range(1, 8)]

    # ------------------------------------------------------------------
    # Pose lookup
    # ------------------------------------------------------------------

    _DEFAULT_POSE_DEG = [0.0, 344.0, 180.0, 214.0, 0.0, 315.0, 90.0]

    def _resolve_pose(self, name: str) -> np.ndarray:
        """Look up a named pose parameter, fall back to home, then default."""
        if name == 'home':
            return self.home_deg

        param_name = f'{name}_deg'
        try:
            self.declare_parameter(param_name, descriptor=ParameterDescriptor(dynamic_typing=True))
            val = self.get_parameter(param_name).value
            if val is not None and len(val) == 7:
                self.get_logger().info(f"Using pose '{name}': {val}")
                return np.array(val, dtype=float)
        except Exception:
            pass

        self.get_logger().warn(
            f"Pose '{name}' (param '{param_name}') not found, falling back to home"
        )
        return self.home_deg

    # ------------------------------------------------------------------
    # Startup / shutdown sequences
    # ------------------------------------------------------------------

    def startup(self):
        """Full startup: connect → home → torque mode → start loop."""
        self.get_logger().info(f"Connecting to robot at {self.robot_ip} ...")
        self.hw = KinovaHardware(self.robot_ip, self.username, self.password)
        self.hw.connect()
        self.get_logger().info("Connected.")

        # Clear faults and wait
        self.hw.clear_faults()
        if not self.hw.wait_until_ready():
            raise RuntimeError("Arm not ready after clearing faults")

        # Log current state before homing
        try:
            arm_state = self.hw.base.GetArmState()
            self.get_logger().info(f"Arm state: {arm_state}")
            current_angles = self.hw.base.GetMeasuredJointAngles()
            angles_list = [j.value for j in current_angles.joint_angles]
            self.get_logger().info(f"Current joint angles (deg): {angles_list}")
            self.get_logger().info(f"Target initial position (deg): {list(self.initial_pose_deg)}")
        except Exception as e:
            self.get_logger().warning(f"Could not read pre-homing state: {e}")

        # Move to initial pose (high-level mode)
        self.get_logger().info("Moving to initial position ...")
        if not self.hw.go_to_joints(self.initial_pose_deg):
            raise RuntimeError("Moving to initial pose failed")
        self.get_logger().info("Initial position reached.")

        # Build robot model
        self.get_logger().info("Loading robot model ...")
        self.model = RobotModel(self.urdf_path, self.ee_frame)

        # Read initial state and set initial target = current EE pose
        state = self.hw.read_state()
        q = kinova_degrees_to_radians(state.positions_deg)
        ee_pos, ee_rot = self.model.fk(q)
        ee_quat = matrix_to_quat(ee_rot)
        quat_norm = np.linalg.norm(ee_quat)
        self.get_logger().info(f"Home EE pose: pos={ee_pos.round(4)}, quat={ee_quat.round(4)}, quat_norm={quat_norm:.4f}")
        with self._lock:
            self._target_pos = ee_pos.copy()
            self._target_quat = ee_quat

        # Create controller
        if self.controller_type == 'osc':
            q_ref = kinova_degrees_to_radians(np.array(self.home_deg))
            self.controller = OSCController(
                model=self.model,
                kp_task=np.array(self.get_parameter('kp_task').value),
                kd_task=np.array(self.get_parameter('kd_task').value),
                kp_null=np.array(self.get_parameter('kp_joint').value),
                kd_null=np.array(self.get_parameter('kd_joint').value),
                q_ref=q_ref,
                max_joint_torque=self.max_torque,
                max_wrench=np.array(self.get_parameter('max_wrench').value),
            )
        else:
            self.controller = DiffIKController(
                model=self.model,
                kp_task=np.array(self.get_parameter('kp_task').value),
                kp_joint=np.array(self.get_parameter('kp_joint').value),
                kd_joint=np.array(self.get_parameter('kd_joint').value),
                dt=1.0 / self.rate_hz,
                damping=self.get_parameter('damping').value,
                max_joint_velocity=self.get_parameter('max_joint_velocity').value,
                max_torque=self.max_torque,
            )
        self.get_logger().info(f"Controller: {self.controller_type}")

        # Switch to low-level torque mode
        self.get_logger().info("Entering torque mode ...")
        self.hw.set_servoing_mode(low_level=True)
        time.sleep(0.5)
        self.hw.set_torque_mode(True)

        # Start control loop
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"Control loop running at {self.rate_hz} Hz. Ready for targets."
        )

    def shutdown(self):
        """Full shutdown: stop loop → position mode → home → disconnect."""
        self.get_logger().info("Shutting down ...")

        # Stop control loop
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self.hw is None:
            return

        try:
            # Back to position mode
            if self.hw.in_torque_mode:
                self.hw.set_torque_mode(False)
                time.sleep(0.5)
            self.hw.set_servoing_mode(low_level=False)
            time.sleep(1.0)

            # Home
            self.hw.clear_faults()
            if self.hw.wait_until_ready(timeout=5.0):
                self.get_logger().info("Moving to home before disconnect ...")
                self.hw.go_to_joints(self.home_deg)
        except Exception as e:
            self.get_logger().warn(f"Shutdown sequence error: {e}")

        self.hw.disconnect()
        self.get_logger().info("Disconnected. Shutdown complete.")

    # ------------------------------------------------------------------
    # Control loop (runs in dedicated thread)
    # ------------------------------------------------------------------

    def _control_loop(self):
        dt = 1.0 / self.rate_hz

        # Seed initial state
        state = self.hw.read_state()

        while self._running:
            t_start = time.perf_counter()

            try:
                q = kinova_degrees_to_radians(state.positions_deg)
                dq = np.deg2rad(state.velocities_deg)

                # Get current target
                with self._lock:
                    target_pos = self._target_pos.copy()
                    target_quat = self._target_quat.copy()
                    gripper = self._gripper_target

                # Compute torques
                torques = self.controller.compute(target_pos, target_quat, q, dq)

                # Send to robot and get fresh state for next cycle
                state = self.hw.send_torques(torques, state.positions_deg, gripper)

                # Publish state
                self._publish_state(state, q)

            except Exception as e:
                self.get_logger().error(f"Control loop error: {e}")
                self._running = False
                break

            # Sleep for remaining time
            elapsed = time.perf_counter() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _publish_state(self, state, q_rad: np.ndarray):
        """Publish joint states and EE pose."""
        now = self.get_clock().now().to_msg()

        # JointState
        js = JointState()
        js.header.stamp = now
        js.name = self._joint_names
        js.position = q_rad.tolist()
        js.velocity = np.deg2rad(state.velocities_deg).tolist()
        js.effort = state.torques.tolist()
        self.joint_state_pub.publish(js)

        # EE pose
        ee_pos, ee_rot = self.model.fk(q_rad)
        ee_quat = matrix_to_quat(ee_rot)

        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = "world"
        ps.pose.position.x = float(ee_pos[0])
        ps.pose.position.y = float(ee_pos[1])
        ps.pose.position.z = float(ee_pos[2])
        ps.pose.orientation.x = float(ee_quat[0])
        ps.pose.orientation.y = float(ee_quat[1])
        ps.pose.orientation.z = float(ee_quat[2])
        ps.pose.orientation.w = float(ee_quat[3])
        self.ee_pose_pub.publish(ps)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_target_pose(self, msg: PoseStamped):
        with self._lock:
            self._target_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ])
            self._target_quat = np.array([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ])

    def _on_gripper_command(self, msg: Float64):
        with self._lock:
            self._gripper_target = max(0.0, min(1.0, msg.data))

    def _on_e_stop(self, request, response):
        self.get_logger().warn("E-STOP triggered!")
        self._running = False
        try:
            if self.hw and self.hw.in_torque_mode:
                self.hw.set_torque_mode(False)
            if self.hw:
                self.hw.stop()
        except Exception:
            pass
        response.success = True
        response.message = "E-stop executed"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()

    try:
        node.startup()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().fatal(f"Fatal: {e}")
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
