#!/usr/bin/env python3
"""Collect static pose data for gravity identification.

Loads pre-validated poses (from generate_poses.py), moves the arm to
each one via Kinova's built-in planner, holds still, records averaged
(q, tau).

No ROS required. Uses KinovaHardware directly.

Usage:
    python collect_static_poses.py --poses poses.npz --output gravity_data.npz
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hardware import KinovaHardware

HOME_DEG = np.array([90.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])


def kinova_deg_to_rad(positions_deg):
    """Kinova 0-360 degrees to radians (-pi, pi)."""
    signed = positions_deg.copy()
    signed[signed > 180.0] -= 360.0
    return np.deg2rad(signed)


def collect(hw, configs_deg, settle_time, record_time):
    """Move to each config, hold, record averaged (q_rad, tau)."""
    samples_q = []
    samples_tau = []
    n = len(configs_deg)

    for i, target_deg in enumerate(configs_deg):
        print(f"[{i+1}/{n}] -> {np.round(target_deg, 1)}", end="  ", flush=True)

        if not hw.go_to_joints(target_deg, duration=6.0):
            print("SKIP (move failed)")
            continue

        time.sleep(settle_time)

        state = hw.read_state()
        if np.max(np.abs(state.velocities_deg)) > 2.0:
            print("SKIP (not settled)")
            continue

        # Record at ~100 Hz
        q_buf, tau_buf = [], []
        t_end = time.time() + record_time
        while time.time() < t_end:
            state = hw.read_state()
            q_buf.append(state.positions_deg.copy())
            tau_buf.append(state.torques.copy())
            time.sleep(0.01)

        q_avg_deg = np.mean(q_buf, axis=0)
        tau_avg = np.mean(tau_buf, axis=0)
        tau_std = np.std(tau_buf, axis=0)
        q_rad = kinova_deg_to_rad(q_avg_deg)

        samples_q.append(q_rad)
        samples_tau.append(tau_avg)
        print(f"OK  tau=[{', '.join(f'{t:+.2f}' for t in tau_avg)}]  "
              f"std_max={tau_std.max():.3f}")

    return np.array(samples_q), np.array(samples_tau)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", required=True,
                        help=".npz from generate_poses.py")
    parser.add_argument("--ip", default="192.168.1.10")
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--record", type=float, default=1.0)
    parser.add_argument("--output", default="gravity_data.npz")
    args = parser.parse_args()

    poses = np.load(args.poses)
    configs_deg = poses["q_deg_kinova"]
    print(f"Loaded {len(configs_deg)} poses from {args.poses}")

    hw = KinovaHardware(args.ip)
    try:
        print("Connecting...")
        hw.connect()
        hw.clear_faults()
        if not hw.wait_until_ready():
            sys.exit("Robot not ready")

        hw.set_servoing_mode(low_level=False)

        print("Going home...")
        hw.go_to_joints(HOME_DEG)

        q_all, tau_all = collect(hw, configs_deg, args.settle, args.record)

        print("Returning home...")
        hw.go_to_joints(HOME_DEG)
    finally:
        hw.disconnect()

    if len(q_all) == 0:
        sys.exit("No samples collected!")

    np.savez(args.output, q_rad=q_all, tau=tau_all)
    print(f"\nSaved {len(q_all)} samples to {args.output}")


if __name__ == "__main__":
    main()
