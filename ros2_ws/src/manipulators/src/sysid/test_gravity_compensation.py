#!/usr/bin/env python3
"""Test gravity compensation with identified parameters.

Moves the arm to home pose in high-level mode, then switches to
low-level torque control and sends gravity compensation torques
in a real-time loop. The arm should feel weightless / hold its pose
if the identified parameters are correct.

Press Ctrl+C to stop. The arm will return to high-level mode and home.

Usage:
    # Use corrected model (with identified parameters):
    python test_gravity_compensation.py --corrected

    # Use nominal URDF model (no corrections):
    python test_gravity_compensation.py
"""

import os
import sys
import time
import signal
import argparse
import numpy as np
import yaml
from pathlib import Path

import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hardware import KinovaHardware

DEFAULT_URDF = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "assets", "robots", "kinova", "urdf", "gen3_2f85.urdf",
)
DEFAULT_CORRECTIONS = os.path.join(
    os.path.dirname(__file__), "gravity_corrections.yaml",
)

ARM_JOINT_NAMES = [f"gen3_joint_{i}" for i in range(1, 8)]
PARAMS_PER_BODY = 10
HOME_DEG = np.array([0.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])
# HOME_DEG = np.array([0.0, 90.0, 0.0, 90.0, 0.0, 00.0, -90.0])
RATE_HZ = 1000
KP = np.array([0.0]*7)  # Position gains per joint (Nm/rad)
KD = np.array([60.0, 60.0, 60.0, 60.0, 0.0, 0.0, 0.0])  # Damping gains per joint (Nm/(rad/s))
FC = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.3])  # Coulomb friction per joint (Nm)
FV = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 4.0])  # Viscous friction per joint (Nm/(rad/s))


def load_pin_model(urdf_path):
    """Load Pinocchio model and extract arm joint info."""
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    v_idx, q_info = [], []
    for name in ARM_JOINT_NAMES:
        jid = model.getJointId(name)
        joint = model.joints[jid]
        v_idx.append(joint.idx_v)
        q_info.append((joint.idx_q, joint.nq))
    return model, data, np.array(v_idx, dtype=np.intp), q_info


def apply_corrections(model, delta_pi):
    """Update Pinocchio model inertias with identified corrections."""
    n_bodies = len(delta_pi) // PARAMS_PER_BODY
    for i in range(min(n_bodies, model.njoints)):
        dp = delta_pi[i * PARAMS_PER_BODY:(i + 1) * PARAMS_PER_BODY]
        dm, dmx, dmy, dmz = dp[0], dp[1], dp[2], dp[3]
        if abs(dm) < 1e-8 and abs(dmx) < 1e-8 and abs(dmy) < 1e-8 and abs(dmz) < 1e-8:
            continue

        inertia = model.inertias[i]
        new_mass = inertia.mass + dm
        if new_mass < 0.01:
            print(f"  WARNING: {model.names[i]} mass={new_mass:.4f}, skipping")
            continue

        old_fm = inertia.mass * inertia.lever
        new_fm = old_fm + np.array([dmx, dmy, dmz])
        new_lever = new_fm / new_mass
        model.inertias[i] = pin.Inertia(new_mass, new_lever, inertia.inertia)


def set_arm_q(q_full, q_arm, q_info):
    """Set arm joints in Pinocchio q vector."""
    for i, (idx_q, nq) in enumerate(q_info):
        if nq == 1:
            q_full[idx_q] = q_arm[i]
        else:
            q_full[idx_q] = np.cos(q_arm[i])
            q_full[idx_q + 1] = np.sin(q_arm[i])


def compute_gravity_torques(model, data, v_idx, q_info, q_rad):
    """Compute gravity compensation torques for given joint positions."""
    q_full = pin.neutral(model)
    set_arm_q(q_full, q_rad, q_info)
    pin.computeGeneralizedGravity(model, data, q_full)
    return data.g[v_idx].copy()


def kinova_deg_to_rad(positions_deg):
    """Kinova 0-360 degrees to radians (-pi, pi)."""
    signed = positions_deg.copy()
    signed[signed > 180.0] -= 360.0
    return np.deg2rad(signed)


def main():
    parser = argparse.ArgumentParser(
        description="Test gravity compensation with nominal or corrected model."
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="Path to gen3 URDF")
    parser.add_argument("--corrections", default=DEFAULT_CORRECTIONS,
                        help="YAML from identify_gravity.py")
    parser.add_argument("--corrected", action="store_true",
                        help="Use corrected model (apply identified parameters)")
    parser.add_argument("--ip", default="192.168.1.10")
    parser.add_argument("--rate", type=int, default=RATE_HZ,
                        help="Control loop rate in Hz")
    args = parser.parse_args()

    dt = 1.0 / args.rate

    # Load Pinocchio model
    print("Loading URDF model...")
    model, pin_data, v_idx, q_info = load_pin_model(args.urdf)

    # Optionally apply corrections
    if args.corrected:
        print(f"Loading corrections from {args.corrections}...")
        with open(args.corrections, "r") as f:
            corr = yaml.safe_load(f)
        delta_pi = np.array(corr["delta_pi"])
        apply_corrections(model, delta_pi)
        print(f"  Applied {corr['n_active_params']} parameter corrections")
        print(f"  RMSE improvement: {corr['rmse_val_before']:.4f} -> {corr['rmse_val_after']:.4f} Nm")
    else:
        print("Using nominal URDF model (no corrections)")

    # Connect to robot
    hw = KinovaHardware(args.ip)
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nStopping...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("Connecting...")
        hw.connect()
        hw.clear_faults()
        if not hw.wait_until_ready():
            sys.exit("Robot not ready")

        # Go to home pose in high-level mode
        hw.set_servoing_mode(low_level=False)
        print("Going to home pose...")
        hw.go_to_joints(HOME_DEG)
        time.sleep(1.0)

        # Read initial state
        state = hw.read_state()
        positions_deg = state.positions_deg.copy()
        velocities_deg = state.velocities_deg.copy()

        # Switch to low-level torque mode
        print("Switching to low-level torque mode...")
        hw.set_servoing_mode(low_level=True)
        time.sleep(0.5)
        hw.set_torque_mode(enabled=True)

        print(f"Running gravity compensation at {args.rate} Hz with Kp + Kd. Press Ctrl+C to stop.")
        print("-" * 60)

        loop_count = 0
        t_start = time.time()
        q_home_rad = kinova_deg_to_rad(HOME_DEG)

        while running:
            t_loop = time.time()

            # Read current state
            q_rad = kinova_deg_to_rad(positions_deg)
            vel_rad = np.deg2rad(velocities_deg)

            # Compute gravity torques
            g_torques = compute_gravity_torques(
                model, pin_data, v_idx, q_info, q_rad)

            # Position error (wrap to [-pi, pi])
            q_err = q_home_rad - q_rad
            q_err = (q_err + np.pi) % (2 * np.pi) - np.pi

            # Add position torques (pull towards home)
            position_torques = KP * q_err

            # Add damping torques
            damping_torques = -KD * vel_rad

            # Add friction compensation (Coulomb + viscous)
            friction_torques = FC * np.sign(vel_rad) + FV * vel_rad

            total_torques = g_torques + position_torques + damping_torques + friction_torques
            total_torques[5] += 2.0
            # Send torques
            state = hw.send_torques(
                torques=total_torques,
                positions_deg=positions_deg,
            )
            positions_deg = state.positions_deg.copy()
            velocities_deg = state.velocities_deg.copy()
            read_torques = state.torques.copy()

            loop_count += 1
            if loop_count % 10 == 0 or loop_count == 1:
                elapsed = time.time() - t_start
                actual_rate = loop_count / elapsed
                q_signed = positions_deg.copy()
                q_signed[q_signed > 180.0] -= 360.0
                q_err_deg = np.rad2deg(q_err)
                print(f"  rate={actual_rate:.0f}Hz \n g_t=[{', '.join(f'{t:+.2f}' for t in g_torques)}] \n p_t=[{', '.join(f'{t:+.2f}' for t in position_torques)}] \n t_t=[{', '.join(f'{t:+.2f}' for t in total_torques)}] \n r_t=[{', '.join(f'{t:+.2f}' for t in read_torques)}] \n q=[{', '.join(f'{q:.1f}' for q in q_signed)}] \n err=[{', '.join(f'{e:.1f}' for e in q_err_deg)}]deg \n\n")

            # Rate limiting
            dt_elapsed = time.time() - t_loop
            if dt_elapsed < dt:
                time.sleep(dt - dt_elapsed)

    finally:
        # Shutdown sequence (similar to control_node.py)
        print("\nShutting down...")
        try:
            # Disable torque mode
            if hw.in_torque_mode:
                hw.set_torque_mode(enabled=False)
                time.sleep(0.5)

            # Switch back to high-level mode
            hw.set_servoing_mode(low_level=False)
            time.sleep(1.0)

            # Return to home
            hw.clear_faults()
            if hw.wait_until_ready(timeout=5.0):
                print("Returning to home pose...")
                hw.go_to_joints(HOME_DEG)
        except Exception as e:
            print(f"  Warning during shutdown: {e}")

        hw.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
