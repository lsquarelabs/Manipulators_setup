#!/usr/bin/env python3
"""Test gravity compensation with fully identified parameters.

Unlike test_gravity_compensation.py which applies corrections to CAD model,
this script uses the fully identified parameters directly.

Usage:
    python test_identified_model.py --params identified_params.yaml
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
DEFAULT_PARAMS = os.path.join(
    os.path.dirname(__file__), "identified_params.yaml",
)

ARM_JOINT_NAMES = [f"gen3_joint_{i}" for i in range(1, 8)]
PARAMS_PER_BODY = 10
HOME_DEG = np.array([0.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])
RATE_HZ = 1000


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


def apply_identified_params(model, params_yaml):
    """Replace model inertias with fully identified parameters.

    Unlike apply_corrections which adds delta_pi to existing params,
    this replaces the mass and CoM entirely with identified values.
    """
    params = np.array(params_yaml["parameters"])
    n_bodies = len(params) // PARAMS_PER_BODY

    print("Applying identified parameters:")
    for i in range(min(n_bodies, model.njoints)):
        p = params[i * PARAMS_PER_BODY:(i + 1) * PARAMS_PER_BODY]
        m = p[0]
        mx, my, mz = p[1], p[2], p[3]

        if m < 0.001:
            continue

        # Compute CoM from first moments
        lever = np.array([mx, my, mz]) / m

        # Keep original rotational inertia (not identifiable from static data)
        old_inertia = model.inertias[i]

        # Create new inertia with identified mass and CoM
        model.inertias[i] = pin.Inertia(m, lever, old_inertia.inertia)

        body_name = str(model.names[i])
        if abs(m - old_inertia.mass) > 0.001:
            print(f"  {body_name}: m={m:.4f} (was {old_inertia.mass:.4f})")


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
        description="Test gravity compensation with fully identified model."
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="Path to gen3 URDF")
    parser.add_argument("--params", default=DEFAULT_PARAMS,
                        help="YAML from identify_full.py")
    parser.add_argument("--ip", default="192.168.1.10")
    parser.add_argument("--rate", type=int, default=RATE_HZ,
                        help="Control loop rate in Hz")
    args = parser.parse_args()

    dt = 1.0 / args.rate

    # Load Pinocchio model
    print("Loading URDF model...")
    model, pin_data, v_idx, q_info = load_pin_model(args.urdf)

    # Load and apply identified parameters
    print(f"Loading identified parameters from {args.params}...")
    with open(args.params, "r") as f:
        params_yaml = yaml.safe_load(f)

    print(f"  Method: {params_yaml.get('method', 'unknown')}")
    print(f"  RMSE (val): {params_yaml.get('rmse_val', 'N/A'):.4f} Nm")

    apply_identified_params(model, params_yaml)

    # Recreate data after model modification
    pin_data = model.createData()

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

        # Switch to low-level torque mode
        print("Switching to low-level torque mode...")
        hw.set_servoing_mode(low_level=True)
        time.sleep(0.5)
        hw.set_torque_mode(enabled=True)

        print(f"Running gravity compensation at {args.rate} Hz. Press Ctrl+C to stop.")
        print("-" * 60)

        loop_count = 0
        t_start = time.time()

        while running:
            t_loop = time.time()

            # Read current state
            q_rad = kinova_deg_to_rad(positions_deg)

            # Compute gravity torques using identified model
            g_torques = compute_gravity_torques(
                model, pin_data, v_idx, q_info, q_rad)

            # Send gravity torques
            state = hw.send_torques(
                torques=g_torques,
                positions_deg=positions_deg,
            )
            positions_deg = state.positions_deg.copy()

            loop_count += 1
            if loop_count % (args.rate * 2) == 0:
                elapsed = time.time() - t_start
                actual_rate = loop_count / elapsed
                q_signed = positions_deg.copy()
                q_signed[q_signed > 180.0] -= 360.0
                print(f"  rate={actual_rate:.0f}Hz  "
                      f"tau=[{', '.join(f'{t:+.2f}' for t in g_torques)}]  "
                      f"q=[{', '.join(f'{q:.1f}' for q in q_signed)}]")

            # Rate limiting
            dt_elapsed = time.time() - t_loop
            if dt_elapsed < dt:
                time.sleep(dt - dt_elapsed)

    finally:
        # Shutdown sequence
        print("\nShutting down...")
        try:
            if hw.in_torque_mode:
                hw.set_torque_mode(enabled=False)
                time.sleep(0.5)

            hw.set_servoing_mode(low_level=False)
            time.sleep(1.0)

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
