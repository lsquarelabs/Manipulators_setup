"""
MuJoCo hardware interface — drop-in replacement for KinovaHardware.

Communicates with the MuJoCo sim process over UDP using the same
binary protocol, so the control node sees identical latency
characteristics to the real Kortex API.
"""

import socket
import time
import numpy as np

from .hardware import RobotState, NUM_JOINTS
from .sim_protocol import (
    HEADER_SIZE, RESP_STATE,
    pack_torque_cmd, pack_read_state_cmd, pack_go_to_joints_cmd,
    pack_connect_cmd, pack_disconnect_cmd,
    unpack_header, unpack_state_resp, unpack_ack_resp,
)


class MujocoHardware:
    """
    Sim-backed hardware interface with the same API as KinovaHardware.

    All mode-switching / fault-clearing methods are no-ops since
    the sim doesn't have those concepts.
    """

    def __init__(self, ip: str = '127.0.0.1', port: int = 10001, **_kwargs):
        self.ip = ip
        self.port = port
        self._sock = None
        self._frame_id = 0
        self._in_torque_mode = False

        # Not available in sim — guarded by try/except in control_node
        self.base = None
        self.base_cyclic = None
        self.actuator_config = None

    def _next_frame_id(self) -> int:
        self._frame_id = (self._frame_id + 1) % 65536
        return self._frame_id

    def _send_recv(self, packet: bytes, timeout: float = 2.0) -> bytes:
        """Send a UDP packet and wait for the response."""
        self._sock.sendto(packet, (self.ip, self.port))
        self._sock.settimeout(timeout)
        data, _ = self._sock.recvfrom(4096)
        return data

    def _parse_state(self, data: bytes) -> RobotState:
        """Unpack a STATE response into a RobotState."""
        pos_deg, vel_deg, torques, timestamp = unpack_state_resp(data)
        return RobotState(
            positions_deg=pos_deg,
            velocities_deg=vel_deg,
            torques=torques,
            timestamp=timestamp,
        )

    # ── Connection ──

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        resp = self._send_recv(pack_connect_cmd(self._next_frame_id()))
        if not unpack_ack_resp(resp):
            raise RuntimeError("Sim handshake failed")

    def disconnect(self):
        if self._sock is None:
            return
        try:
            self._send_recv(pack_disconnect_cmd(self._next_frame_id()), timeout=1.0)
        except Exception:
            pass
        self._sock.close()
        self._sock = None
        self._in_torque_mode = False

    # ── Mode switching (no-ops in sim) ──

    def set_servoing_mode(self, low_level: bool):
        pass

    def set_torque_mode(self, enabled: bool):
        self._in_torque_mode = enabled

    @property
    def in_torque_mode(self) -> bool:
        return self._in_torque_mode

    # ── Arm state (no-ops in sim) ──

    def clear_faults(self):
        pass

    def stop(self):
        pass

    def is_ready(self) -> bool:
        return self._sock is not None

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        return self._sock is not None

    # ── Real-time I/O ──

    def read_state(self) -> RobotState:
        resp = self._send_recv(pack_read_state_cmd(self._next_frame_id()))
        return self._parse_state(resp)

    def send_torques(
        self,
        torques: np.ndarray,
        positions_deg: np.ndarray,
        gripper_position: float = 0.0,
    ) -> RobotState:
        pkt = pack_torque_cmd(
            self._next_frame_id(), torques, positions_deg, gripper_position,
        )
        resp = self._send_recv(pkt, timeout=0.1)
        return self._parse_state(resp)

    def send_positions(
        self,
        positions_deg: np.ndarray,
        gripper_position: float = 0.0,
    ) -> RobotState:
        return self.send_torques(np.zeros(NUM_JOINTS), positions_deg, gripper_position)

    # ── High-level actions ──

    def go_to_joints(self, target_deg: np.ndarray, duration: float = 8.0) -> bool:
        pkt = pack_go_to_joints_cmd(
            self._next_frame_id(), target_deg, duration,
        )
        resp = self._send_recv(pkt)
        return unpack_ack_resp(resp)

    # ── Gripper (high-level, no-ops for now) ──

    def send_gripper_command(self, position: float):
        pass  # gripper is controlled via send_torques in the loop

    def read_gripper_position(self) -> float:
        return 0.0
