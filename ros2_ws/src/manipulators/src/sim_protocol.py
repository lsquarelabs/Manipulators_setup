"""
Shared UDP protocol for MuJoCo sim <-> hardware client communication.

Binary protocol mimicking the Kortex UDP real-time interface.
All multi-byte values are big-endian (network byte order).

Message layout:  [header (5 bytes)] [payload (variable)]
Header:          msg_type (u8) + frame_id (u32)
"""

import struct
import numpy as np

NUM_JOINTS = 7

# ── Message types (client → server) ──
CMD_TORQUE = 0x01        # apply torques, step physics, return state
CMD_READ_STATE = 0x02    # read current state (no step)
CMD_GO_TO_JOINTS = 0x03  # teleport to joint angles
CMD_CONNECT = 0x04       # handshake
CMD_DISCONNECT = 0x05    # clean shutdown

# ── Response types (server → client) ──
RESP_STATE = 0x81        # robot state (pos, vel, torque, timestamp)
RESP_ACK = 0x82          # acknowledgment (success byte)

# ── Struct formats (network byte order) ──
HEADER_FMT = '!BI'       # u8 msg_type + u32 frame_id
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# 7 torques + 7 positions_deg + 1 gripper = 15 doubles
TORQUE_PAYLOAD_FMT = f'!{NUM_JOINTS}d{NUM_JOINTS}dd'

# 7 pos_deg + 7 vel_deg + 7 torques + 1 timestamp = 22 doubles
STATE_PAYLOAD_FMT = f'!{NUM_JOINTS}d{NUM_JOINTS}d{NUM_JOINTS}dd'

# 7 target_deg + 1 duration = 8 doubles
GO_TO_JOINTS_PAYLOAD_FMT = f'!{NUM_JOINTS}dd'

ACK_PAYLOAD_FMT = '!B'


# ── Packing (client side) ──

def pack_torque_cmd(frame_id: int, torques, positions_deg, gripper: float) -> bytes:
    header = struct.pack(HEADER_FMT, CMD_TORQUE, frame_id)
    payload = struct.pack(TORQUE_PAYLOAD_FMT, *torques, *positions_deg, gripper)
    return header + payload


def pack_read_state_cmd(frame_id: int) -> bytes:
    return struct.pack(HEADER_FMT, CMD_READ_STATE, frame_id)


def pack_go_to_joints_cmd(frame_id: int, target_deg, duration: float) -> bytes:
    header = struct.pack(HEADER_FMT, CMD_GO_TO_JOINTS, frame_id)
    payload = struct.pack(GO_TO_JOINTS_PAYLOAD_FMT, *target_deg, duration)
    return header + payload


def pack_connect_cmd(frame_id: int) -> bytes:
    return struct.pack(HEADER_FMT, CMD_CONNECT, frame_id)


def pack_disconnect_cmd(frame_id: int) -> bytes:
    return struct.pack(HEADER_FMT, CMD_DISCONNECT, frame_id)


# ── Packing (server side) ──

def pack_state_resp(frame_id: int, pos_deg, vel_deg, torques, timestamp: float) -> bytes:
    header = struct.pack(HEADER_FMT, RESP_STATE, frame_id)
    payload = struct.pack(STATE_PAYLOAD_FMT, *pos_deg, *vel_deg, *torques, timestamp)
    return header + payload


def pack_ack_resp(frame_id: int, success: bool) -> bytes:
    header = struct.pack(HEADER_FMT, RESP_ACK, frame_id)
    payload = struct.pack(ACK_PAYLOAD_FMT, 1 if success else 0)
    return header + payload


# ── Unpacking ──

def unpack_header(data: bytes):
    """Returns (msg_type, frame_id)."""
    return struct.unpack(HEADER_FMT, data[:HEADER_SIZE])


def unpack_state_resp(data: bytes):
    """Returns (pos_deg, vel_deg, torques, timestamp)."""
    payload = data[HEADER_SIZE:]
    values = struct.unpack(STATE_PAYLOAD_FMT, payload)
    pos_deg = np.array(values[:NUM_JOINTS])
    vel_deg = np.array(values[NUM_JOINTS:2 * NUM_JOINTS])
    torques = np.array(values[2 * NUM_JOINTS:3 * NUM_JOINTS])
    timestamp = values[3 * NUM_JOINTS]
    return pos_deg, vel_deg, torques, timestamp


def unpack_ack_resp(data: bytes) -> bool:
    payload = data[HEADER_SIZE:]
    return struct.unpack(ACK_PAYLOAD_FMT, payload)[0] == 1


def unpack_torque_payload(data: bytes):
    """Returns (torques, positions_deg, gripper)."""
    payload = data[HEADER_SIZE:]
    values = struct.unpack(TORQUE_PAYLOAD_FMT, payload)
    torques = np.array(values[:NUM_JOINTS])
    positions_deg = np.array(values[NUM_JOINTS:2 * NUM_JOINTS])
    gripper = values[2 * NUM_JOINTS]
    return torques, positions_deg, gripper


def unpack_go_to_joints_payload(data: bytes):
    """Returns (target_deg, duration)."""
    payload = data[HEADER_SIZE:]
    values = struct.unpack(GO_TO_JOINTS_PAYLOAD_FMT, payload)
    target_deg = np.array(values[:NUM_JOINTS])
    duration = values[NUM_JOINTS]
    return target_deg, duration
