"""
Keyboard teleop node.

Reads keyboard input and publishes incremental EE pose targets.

Keys:
  W/S  - X forward/back
  A/D  - Y left/right
  Q/E  - Z up/down
  I/K  - pitch +/-
  J/L  - yaw +/-
  U/O  - roll +/-
  G    - toggle gripper open/close
  M    - toggle mode (accumulated / relative-to-EE)
  +/=  - increase delta scale (x2)
  -    - decrease delta scale (x0.5)
  ESC  - quit
"""

import sys
import tty
import termios
import select
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64

from .utility import quat_to_matrix, matrix_to_quat


USAGE = """
Keyboard Teleop — Kinova Gen3
──────────────────────────────
  W/S : X forward/back
  A/D : Y left/right
  Q/E : Z up/down
  I/K : pitch +/-
  J/L : yaw +/-
  U/O : roll +/-
  G   : toggle gripper
  M   : toggle mode (accumulated / relative)
  +/= : increase delta scale (x2)
  -   : decrease delta scale (x0.5)
  ESC : quit
──────────────────────────────
"""

# Map key -> (axis_index, sign)
# Position: indices 0-2 (x,y,z)  Rotation: indices 3-5 (roll,pitch,yaw)
KEY_MAP = {
    'w': (0, +1), 's': (0, -1),   # X
    'a': (1, +1), 'd': (1, -1),   # Y
    'q': (2, +1), 'e': (2, -1),   # Z
    'u': (3, +1), 'o': (3, -1),   # roll
    'i': (4, +1), 'k': (4, -1),   # pitch
    'j': (5, +1), 'l': (5, -1),   # yaw
}


def _get_key(timeout=0.1):
    """Read a single keypress (non-blocking with timeout)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return None


def _rotation_matrix(axis: int, angle: float) -> np.ndarray:
    """Small rotation matrix around axis 0=X, 1=Y, 2=Z."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(3)
    if axis == 0:  # roll (X)
        R[1, 1] = c; R[1, 2] = -s
        R[2, 1] = s; R[2, 2] = c
    elif axis == 1:  # pitch (Y)
        R[0, 0] = c; R[0, 2] = s
        R[2, 0] = -s; R[2, 2] = c
    elif axis == 2:  # yaw (Z)
        R[0, 0] = c; R[0, 1] = -s
        R[1, 0] = s; R[1, 1] = c
    return R


class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')

        self.declare_parameter('linear_step', 0.005)
        self.declare_parameter('angular_step', 0.05)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('linear_threshold', 0.001)
        self.declare_parameter('angular_threshold', 0.01)

        self.linear_step = self.get_parameter('linear_step').value
        self.angular_step = self.get_parameter('angular_step').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.linear_threshold = self.get_parameter('linear_threshold').value
        self.angular_threshold = self.get_parameter('angular_threshold').value

        self.pose_pub = self.create_publisher(PoseStamped, 'target_pose', 1)
        self.gripper_pub = self.create_publisher(Float64, 'gripper_command', 1)

        # Wait for initial EE pose
        self._target_pos = None
        self._target_rot = None
        self._gripper_open = True
        self._relative_mode = False  # False = accumulated, True = relative-to-EE
        self._scale = 1.0

        # Live EE pose (continuously updated)
        self._ee_pos = None
        self._ee_rot = None

        self.create_subscription(PoseStamped, 'ee_pose', self._on_ee_pose, 1)

    def _on_ee_pose(self, msg: PoseStamped):
        """Capture EE pose. Initializes target on first call, then keeps tracking live EE."""
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        rot = quat_to_matrix(np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]))
        self._ee_pos = pos
        self._ee_rot = rot
        if self._target_pos is None:
            self._target_pos = pos.copy()
            self._target_rot = rot.copy()
            self.get_logger().info("Got initial EE pose. Keyboard active.")

    def run(self):
        """Main loop: read keys, update target, publish."""
        print(USAGE)
        self.get_logger().info("Waiting for initial EE pose from control node ...")

        rate_dt = 1.0 / self.publish_rate

        while rclpy.ok():
            # Spin once to process ee_pose subscription
            rclpy.spin_once(self, timeout_sec=0.01)

            if self._target_pos is None:
                continue

            key = _get_key(timeout=rate_dt)

            if key == '\x1b':  # ESC
                self.get_logger().info("ESC pressed, exiting.")
                break

            if key and key.lower() == 'g':
                self._gripper_open = not self._gripper_open
                msg = Float64()
                msg.data = 0.0 if self._gripper_open else 1.0
                self.gripper_pub.publish(msg)
                status = "OPEN" if self._gripper_open else "CLOSED"
                self.get_logger().info(f"Gripper: {status}")
                continue

            if key and key.lower() == 'm':
                self._relative_mode = not self._relative_mode
                mode = "RELATIVE (EE + delta)" if self._relative_mode else "ACCUMULATED"
                if not self._relative_mode:
                    # Switching back to accumulated: re-seed target from current EE
                    self._target_pos = self._ee_pos.copy()
                    self._target_rot = self._ee_rot.copy()
                self.get_logger().info(f"Mode: {mode}")
                continue

            if key and key in ('+', '='):
                self._scale *= 2.0
                self.get_logger().info(f"Scale: {self._scale:.4f}")
                continue

            if key and key == '-':
                self._scale *= 0.5
                self.get_logger().info(f"Scale: {self._scale:.4f}")
                continue

            moved = False
            if key and key.lower() in KEY_MAP:
                axis, sign = KEY_MAP[key.lower()]
                eff_linear = self.linear_step * self._scale
                eff_angular = self.angular_step * self._scale

                if self._relative_mode:
                    self._target_pos = self._ee_pos.copy()
                    self._target_rot = self._ee_rot.copy()

                if axis < 3:
                    if not self._relative_mode or eff_linear >= self.linear_threshold:
                        delta = np.zeros(3)
                        delta[axis] = sign * eff_linear
                        self._target_pos += delta
                        moved = True
                else:
                    if not self._relative_mode or eff_angular >= self.angular_threshold:
                        rot_axis = axis - 3
                        dR = _rotation_matrix(rot_axis, sign * eff_angular)
                        self._target_rot = dR @ self._target_rot
                        moved = True

            # Publish target
            if self._relative_mode:
                if moved:
                    self._publish_target()
                # No key / below threshold: don't publish, let controller hold last target
            else:
                self._publish_target()

    def _publish_target(self):
        self._publish_target_from(self._target_pos, self._target_rot)

    def _publish_target_from(self, pos, rot):
        quat = matrix_to_quat(rot)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
