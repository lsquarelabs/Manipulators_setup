"""
Keyboard teleop node.

Reads keyboard input and publishes incremental EE pose targets.
Uses pynput for proper key press/release detection, enabling true
simultaneous multi-key input.

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

import threading
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from pynput import keyboard

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

# Map key char -> (axis_index, sign)
# Position: indices 0-2 (x,y,z)  Rotation: indices 3-5 (roll,pitch,yaw)
KEY_MAP = {
    'w': (0, +1), 's': (0, -1),   # X
    'a': (1, +1), 'd': (1, -1),   # Y
    'q': (2, +1), 'e': (2, -1),   # Z
    'u': (3, +1), 'o': (3, -1),   # roll
    'i': (4, +1), 'k': (4, -1),   # pitch
    'j': (5, +1), 'l': (5, -1),   # yaw
}


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

        self.pose_pub = self.create_publisher(PoseStamped, 'ee_target_pose', 1)
        self.gripper_pub = self.create_publisher(Float64, 'gripper_command', 1)

        # Wait for initial EE pose
        self._target_pos = None
        self._target_rot = None
        self._gripper_open = True
        self._relative_mode = False  # False = accumulated, True = relative-to-EE
        self._scale = 0.25

        # Live EE pose (continuously updated)
        self._ee_pos = None
        self._ee_rot = None

        self.create_subscription(PoseStamped, 'ee_state', self._on_ee_state, 1)

        # --- Key state (thread-safe) ---
        self._held_keys = set()       # currently held key chars
        self._lock = threading.Lock()
        self._quit = False
        # Pending one-shot actions from press callbacks
        self._pending_actions = []

    def _on_ee_state(self, msg: PoseStamped):
        """Capture EE state. Initializes target on first call, then keeps tracking live EE."""
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

    # ── pynput callbacks (run in listener thread) ──

    def _on_press(self, key):
        char = self._key_to_char(key)
        if char is None:
            return
        with self._lock:
            if char in KEY_MAP:
                self._held_keys.add(char)
            elif char == 'g':
                self._pending_actions.append('gripper')
            elif char == 'm':
                self._pending_actions.append('mode')
            elif char in ('+', '='):
                self._pending_actions.append('scale_up')
            elif char == '-':
                self._pending_actions.append('scale_down')
            elif char == '\x1b':
                self._quit = True

    def _on_release(self, key):
        char = self._key_to_char(key)
        if char is None:
            return
        with self._lock:
            self._held_keys.discard(char)

    @staticmethod
    def _key_to_char(key):
        """Convert a pynput key to a lowercase char, or None."""
        if key == keyboard.Key.esc:
            return '\x1b'
        try:
            return key.char.lower() if key.char else None
        except AttributeError:
            return None

    # ── Main loop ──

    def run(self):
        """Main loop: read held keys, update target, publish."""
        print(USAGE)
        self.get_logger().info("Waiting for initial EE pose from control node ...")

        rate_dt = 1.0 / self.publish_rate

        listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        listener.start()

        try:
            while rclpy.ok() and not self._quit:
                rclpy.spin_once(self, timeout_sec=rate_dt)

                if self._target_pos is None:
                    continue

                # Snapshot key state
                with self._lock:
                    held = set(self._held_keys)
                    actions = list(self._pending_actions)
                    self._pending_actions.clear()

                # --- Handle one-shot actions ---
                if self._quit:
                    self.get_logger().info("ESC pressed, exiting.")
                    break

                for action in actions:
                    if action == 'gripper':
                        self._gripper_open = not self._gripper_open
                        msg = Float64()
                        msg.data = 0.0 if self._gripper_open else 1.0
                        self.gripper_pub.publish(msg)
                        status = "OPEN" if self._gripper_open else "CLOSED"
                        self.get_logger().info(f"Gripper: {status}")
                    elif action == 'mode':
                        self._relative_mode = not self._relative_mode
                        mode = "RELATIVE (EE + delta)" if self._relative_mode else "ACCUMULATED"
                        if not self._relative_mode:
                            self._target_pos = self._ee_pos.copy()
                            self._target_rot = self._ee_rot.copy()
                        self.get_logger().info(f"Mode: {mode}")
                    elif action == 'scale_up':
                        self._scale *= 2.0
                        self.get_logger().info(f"Scale: {self._scale:.4f}")
                    elif action == 'scale_down':
                        self._scale *= 0.5
                        self.get_logger().info(f"Scale: {self._scale:.4f}")

                # --- Accumulate per-axis motion from held keys ---
                axis_sign = {}
                for k in held:
                    if k in KEY_MAP:
                        axis, sign = KEY_MAP[k]
                        axis_sign[axis] = axis_sign.get(axis, 0) + sign

                eff_linear = self.linear_step * self._scale
                eff_angular = self.angular_step * self._scale

                moved = False
                if axis_sign:
                    if self._relative_mode:
                        self._target_pos = self._ee_pos.copy()
                        self._target_rot = self._ee_rot.copy()

                    for axis, net in axis_sign.items():
                        if net == 0:  # opposite keys cancel
                            continue
                        sign = 1 if net > 0 else -1
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
                    else:
                        self._publish_target_from(self._ee_pos, self._ee_rot)
                else:
                    self._publish_target()

        finally:
            listener.stop()

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
