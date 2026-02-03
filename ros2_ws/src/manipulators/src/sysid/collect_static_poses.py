#!/usr/bin/env python3
"""Collect static pose data for gravity identification.

Loads sequentially-ordered poses (from generate_poses.py) and moves
directly from one pose to the next (no home detour). The first pose
is always zero (all joints at 0).

Per pose records 5 statistics (mean, min, max, var, std) for:
  - q       (joint position, deg)
  - q_dot   (joint velocity, deg/s)
  - q_ddot  (joint acceleration, deg/s², estimated from velocity)
  - tau     (joint torque, Nm)

With --visualize, a MuJoCo viewer previews each motion before the real
robot executes it. Close the viewer window to abort.

No ROS required. Uses KinovaHardware directly.

Usage:
    python collect_static_poses.py --poses poses.yaml --output gravity_data.yaml
    python collect_static_poses.py --test
    python collect_static_poses.py --test --visualize
"""

import sys
import time
import argparse
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hardware import KinovaHardware

ZERO_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def wait_until_settled(hw, q_tol_deg, vel_tol_deg, acc_tol_deg, timeout):
    """Wait until velocity and acceleration are below thresholds.
    Returns True if settled, False if timed out."""
    t_end = time.time() + timeout
    prev_vel = None
    prev_t = None
    while time.time() < t_end:
        state = hw.read_state()
        now = time.time()
        vel = np.array(state.velocities_deg)

        if prev_vel is not None and prev_t is not None:
            dt = now - prev_t
            if dt > 0:
                acc = np.abs((vel - prev_vel) / dt)
            else:
                acc = np.full(7, np.inf)
        else:
            acc = np.full(7, np.inf)

        prev_vel = vel.copy()
        prev_t = now

        if (np.max(np.abs(vel)) < vel_tol_deg
                and np.max(acc) < acc_tol_deg):
            return True

        time.sleep(0.01)
    return False


def compute_stats(buf):
    """Compute 5 statistics over axis=0: mean, min, max, var, std."""
    arr = np.array(buf)
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "var": np.var(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
    }


def preview_in_mujoco(sim_model, sim_data, viewer, target_rad,
                      max_steps=3000, tol=0.02):
    """Simulate motion to target in MuJoCo viewer. Non-blocking preview."""
    import mujoco
    sim_data.ctrl[:7] = target_rad
    for _ in range(max_steps):
        mujoco.mj_step(sim_model, sim_data)
        viewer.sync()
        if not viewer.is_running():
            return False
        if (np.max(np.abs(sim_data.qpos[:7] - target_rad)) < tol
                and np.max(np.abs(sim_data.qvel[:7])) < 0.05):
            break
    return viewer.is_running()


def collect(hw, configs_deg, speed, settle_timeout,
            q_tol_deg, vel_tol_deg, acc_tol_deg, n_samples,
            sim_model=None, sim_data=None, viewer=None):
    """Move to each config, wait until settled, record stats."""
    records = []
    n = len(configs_deg)

    for i, target_deg in enumerate(configs_deg):
        print(f"[{i+1}/{n}] -> {np.round(target_deg, 1)}", end="  ", flush=True)

        # Preview in MuJoCo first
        if viewer is not None:
            if not viewer.is_running():
                print("\nViewer closed, stopping.")
                break
            target_rad = np.deg2rad(target_deg)
            print("(sim) ", end="", flush=True)
            if not preview_in_mujoco(sim_model, sim_data, viewer, target_rad):
                print("\nViewer closed, stopping.")
                break

        # Execute on real robot
        if not hw.go_to_joints(target_deg, duration=speed):
            print("SKIP (move failed)")
            continue

        if not wait_until_settled(hw, q_tol_deg, vel_tol_deg, acc_tol_deg,
                                  settle_timeout):
            print("SKIP (not settled)")
            continue

        # Check position error (wrap to ±180 to handle Kinova 0-360 range)
        state = hw.read_state()
        err = np.array(state.positions_deg) - target_deg
        err = (err + 180.0) % 360.0 - 180.0
        q_err = np.max(np.abs(err))
        if q_err > q_tol_deg:
            print(f"SKIP (position error {q_err:.2f} > {q_tol_deg:.2f} deg)")
            continue

        # Record n_samples at ~100 Hz
        q_buf, q_dot_buf, tau_buf, t_buf = [], [], [], []
        for _ in range(n_samples):
            state = hw.read_state()
            q_buf.append(state.positions_deg.copy())
            q_dot_buf.append(state.velocities_deg.copy())
            tau_buf.append(state.torques.copy())
            t_buf.append(state.timestamp)
            time.sleep(0.01)

        # Estimate acceleration from velocity
        q_dot_arr = np.array(q_dot_buf)
        t_arr = np.array(t_buf)
        q_ddot_buf = []
        for j in range(1, len(q_dot_arr)):
            dt = t_arr[j] - t_arr[j - 1]
            if dt > 0:
                q_ddot_buf.append((q_dot_arr[j] - q_dot_arr[j - 1]) / dt)
            else:
                q_ddot_buf.append(np.zeros(7))

        record = {
            "commanded_deg": target_deg.tolist(),
            "q": compute_stats(q_buf),
            "q_dot": compute_stats(q_dot_buf),
            "q_ddot": compute_stats(q_ddot_buf),
            "tau": compute_stats(tau_buf),
        }
        records.append(record)

        tau_mean = np.array(record["tau"]["mean"])
        tau_std = np.array(record["tau"]["std"])
        print(f"OK  tau=[{', '.join(f'{t:+.2f}' for t in tau_mean)}]  "
              f"tau_std_max={tau_std.max():.3f}")

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", default="poses.yaml",
                        help=".yaml from generate_poses.py")
    parser.add_argument("--ip", default="192.168.1.10")
    parser.add_argument("--speed", type=float, default=6.0,
                        help="Motion duration per move (seconds)")
    parser.add_argument("--settle-timeout", type=float, default=5.0,
                        help="Max time to wait for settling (seconds)")
    parser.add_argument("--q-tol", type=float, default=2.0,
                        help="Position error threshold (degrees)")
    parser.add_argument("--vel-tol", type=float, default=0.05,
                        help="Velocity threshold to start recording (deg/s)")
    parser.add_argument("--acc-tol", type=float, default=0.5,
                        help="Acceleration threshold to start recording (deg/s^2)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples to record per pose (at ~100 Hz)")
    parser.add_argument("--output", default="gravity_data.yaml")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: run only the first 3 poses, save with _test suffix")
    parser.add_argument("--visualize", action="store_true",
                        help="Preview each motion in MuJoCo before executing on real robot")
    args = parser.parse_args()

    with open(args.poses, "r") as f:
        poses_data = yaml.safe_load(f)
    configs_deg = np.array(poses_data["q_deg"])

    if args.test:
        configs_deg = configs_deg[:3]
        out_path = Path(args.output)
        output_file = str(out_path.with_stem(out_path.stem + "_test"))
        print("TEST MODE: running first 3 poses only")
    else:
        output_file = args.output

    print(f"Loaded {len(configs_deg)} poses from {args.poses}")

    # Set up MuJoCo preview if requested
    sim_model, sim_data, viewer = None, None, None
    if args.visualize:
        import mujoco
        import mujoco.viewer
        from generate_poses import load_scene
        sim_model = load_scene(
            str(Path(__file__).resolve().parent.parent.parent /
                "assets" / "robots" / "kinova" / "mjcf" / "gen3.xml"))
        sim_data = mujoco.MjData(sim_model)
        viewer = mujoco.viewer.launch_passive(sim_model, sim_data)
        print("MuJoCo viewer opened for preview")

    hw = KinovaHardware(args.ip)
    try:
        print("Connecting...")
        hw.connect()
        hw.clear_faults()
        if not hw.wait_until_ready():
            sys.exit("Robot not ready")

        hw.set_servoing_mode(low_level=False)

        print("Going to zero pose...")
        hw.go_to_joints(ZERO_DEG)

        records = collect(
            hw, configs_deg, args.speed, args.settle_timeout,
            args.q_tol, args.vel_tol, args.acc_tol, args.n_samples,
            sim_model=sim_model, sim_data=sim_data, viewer=viewer,
        )

        print("Returning to zero pose...")
        hw.go_to_joints(ZERO_DEG)
    finally:
        hw.disconnect()
        if viewer is not None:
            viewer.close()

    if len(records) == 0:
        sys.exit("No samples collected!")

    output = {
        "n_poses": len(records),
        "n_samples_per_pose": args.n_samples,
        "poses": records,
    }
    with open(output_file, "w") as f:
        yaml.dump(output, f, default_flow_style=None, sort_keys=False)
    print(f"\nSaved {len(records)} poses to {output_file}")


if __name__ == "__main__":
    main()
