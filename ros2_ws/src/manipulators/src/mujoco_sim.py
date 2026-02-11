"""
MuJoCo simulation server for Kinova Gen3.

Standalone process that runs physics and communicates with the control
node over UDP, mimicking the real Kortex real-time interface.

Usage:
    ros2 run manipulators mujoco_sim
    ros2 run manipulators mujoco_sim -- --port 10001 --headless
"""

import os
import argparse
import select
import socket
import time
import numpy as np

import mujoco
import mujoco.viewer

from .sim_protocol import (
    NUM_JOINTS,
    HEADER_SIZE,
    CMD_TORQUE, CMD_READ_STATE, CMD_GO_TO_JOINTS, CMD_CONNECT, CMD_DISCONNECT,
    unpack_header, unpack_torque_payload, unpack_go_to_joints_payload,
    pack_state_resp, pack_ack_resp,
)

DEFAULT_PORT = 10001


class MujocoSim:
    """MuJoCo physics server for Kinova Gen3 + Robotiq 2F-85."""

    def __init__(self, mjcf_path: str, port: int = DEFAULT_PORT, control_dt: float = 0.0025):
        self.port = port

        # Load model
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # Map arm joint names → qpos/qvel indices
        self.arm_qpos_idx = []
        self.arm_qvel_idx = []
        for i in range(1, NUM_JOINTS + 1):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint_{i}')
            if jid == -1:
                raise ValueError(f"Joint 'joint_{i}' not found in MJCF")
            self.arm_qpos_idx.append(self.model.jnt_qposadr[jid])
            self.arm_qvel_idx.append(self.model.jnt_dofadr[jid])

        # Gripper actuator (position-controlled via tendon)
        self.gripper_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'fingers_actuator'
        )

        # Disable the built-in position actuators on the arm joints
        # so we can apply torques directly via qfrc_applied.
        for i in range(NUM_JOINTS):
            self.model.actuator_gainprm[i, :] = 0
            self.model.actuator_biasprm[i, :] = 0

        # Physics substeps per control command.
        # Set model timestep to evenly divide the control period.
        self.n_substeps = max(1, round(control_dt / self.model.opt.timestep))
        self.model.opt.timestep = control_dt / self.n_substeps

        # Reset to home keyframe
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

        self.sock = None

    # ── Degree conversion (matches Kinova 0-360 convention) ──

    @staticmethod
    def _rad_to_kinova_deg(rad: np.ndarray) -> np.ndarray:
        return np.rad2deg(rad) % 360.0

    @staticmethod
    def _kinova_deg_to_rad(deg: np.ndarray) -> np.ndarray:
        signed = deg.copy()
        signed[signed > 180.0] -= 360.0
        return np.deg2rad(signed)

    # ── Physics helpers ──

    def _read_state(self):
        """Read current joint state from simulation."""
        pos_rad = np.array([self.data.qpos[i] for i in self.arm_qpos_idx])
        vel_rad = np.array([self.data.qvel[i] for i in self.arm_qvel_idx])
        torques = np.array([self.data.qfrc_applied[i] for i in self.arm_qvel_idx])

        pos_deg = self._rad_to_kinova_deg(pos_rad)
        vel_deg = np.rad2deg(vel_rad)
        return pos_deg, vel_deg, torques, time.time()

    def _apply_torques(self, torques: np.ndarray, gripper_pos: float):
        """Set joint torques and gripper command."""
        for i, idx in enumerate(self.arm_qvel_idx):
            self.data.qfrc_applied[idx] = torques[i]
        if self.gripper_act_id >= 0:
            self.data.ctrl[self.gripper_act_id] = gripper_pos * 255.0

    def _step(self):
        """Advance physics by one control period."""
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

    def _go_to_joints(self, target_deg: np.ndarray):
        """Teleport to target joint angles (used for homing)."""
        target_rad = self._kinova_deg_to_rad(target_deg)
        for i, idx in enumerate(self.arm_qpos_idx):
            self.data.qpos[idx] = target_rad[i]
        for idx in self.arm_qvel_idx:
            self.data.qvel[idx] = 0.0
        self.data.qfrc_applied[:] = 0
        mujoco.mj_forward(self.model, self.data)

    # ── Message handling ──

    def _handle_message(self, raw: bytes, addr):
        if len(raw) < HEADER_SIZE:
            return

        msg_type, frame_id = unpack_header(raw)

        if msg_type == CMD_TORQUE:
            torques, _, gripper = unpack_torque_payload(raw)
            self._apply_torques(torques, gripper)
            self._step()
            resp = pack_state_resp(frame_id, *self._read_state())
            self.sock.sendto(resp, addr)

        elif msg_type == CMD_READ_STATE:
            resp = pack_state_resp(frame_id, *self._read_state())
            self.sock.sendto(resp, addr)

        elif msg_type == CMD_GO_TO_JOINTS:
            target_deg, _ = unpack_go_to_joints_payload(raw)
            self._go_to_joints(target_deg)
            self.sock.sendto(pack_ack_resp(frame_id, True), addr)

        elif msg_type == CMD_CONNECT:
            print(f"Client connected from {addr}")
            self.sock.sendto(pack_ack_resp(frame_id, True), addr)

        elif msg_type == CMD_DISCONNECT:
            print(f"Client disconnected from {addr}")
            self.sock.sendto(pack_ack_resp(frame_id, True), addr)

    # ── Main loop ──

    def run(self, headless: bool = False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', self.port))

        viewer = None
        if not headless:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)

        print(f"MuJoCo sim | port={self.port} | substeps={self.n_substeps} | dt={self.model.opt.timestep}")

        last_sync = time.monotonic()

        try:
            while True:
                if viewer is not None and not viewer.is_running():
                    break

                ready, _, _ = select.select([self.sock], [], [], 0.001)
                if ready:
                    data, addr = self.sock.recvfrom(4096)
                    self._handle_message(data, addr)

                if viewer is not None:
                    now = time.monotonic()
                    if now - last_sync >= 1.0 / 60.0:
                        viewer.sync()
                        last_sync = now
        except KeyboardInterrupt:
            print("\nShutting down ...")
        finally:
            if viewer is not None:
                viewer.close()
            self.sock.close()
            print("Sim stopped.")


def _resolve_default_mjcf() -> str:
    """Find the default MJCF scene file.

    Prefers the source tree (which has the mesh symlinks intact)
    over the installed share directory.
    """
    # 1. Relative to this source file (works in both source and symlink-install)
    here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    src_path = os.path.join(here, '..', 'assets', 'robots', 'kinova', 'mjcf', 'scene.xml')
    if os.path.isfile(src_path):
        return os.path.realpath(src_path)

    # 2. Via ament index
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg = get_package_share_directory('manipulators')
        return os.path.join(pkg, 'assets', 'robots', 'kinova', 'mjcf', 'scene.xml')
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not find scene.xml. Pass --mjcf explicitly."
    )


def main():
    parser = argparse.ArgumentParser(description='MuJoCo Kinova Gen3 simulation server')
    parser.add_argument('--mjcf', type=str, default=None,
                        help='Path to MJCF XML (default: auto-detect from package)')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'UDP port (default: {DEFAULT_PORT})')
    parser.add_argument('--control-dt', type=float, default=0.0025,
                        help='Control timestep in seconds (default: 0.0025 = 400Hz)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without viewer')

    # ros2 run passes extra args; ignore unknown
    args, _ = parser.parse_known_args()

    mjcf = args.mjcf or _resolve_default_mjcf()
    if not os.path.isfile(mjcf):
        parser.error(f"MJCF file not found: {mjcf}")

    sim = MujocoSim(mjcf, port=args.port, control_dt=args.control_dt)
    sim.run(headless=args.headless)


if __name__ == '__main__':
    main()
