#!/usr/bin/env python3
"""Visualize collected gravity sysid data.

Plots per-joint statistics from gravity_data.yaml:
  1. Mean torque vs pose index (with min/max band)
  2. Mean velocity and acceleration at each pose (should be ~0 if settled)
  3. Torque std per joint per pose (data quality)
  4. Joint positions (commanded vs measured mean)

Usage:
    python visualize_gravity_data.py --data gravity_data.yaml
    python visualize_gravity_data.py --data gravity_data_test.yaml
"""

import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt


JOINT_NAMES = [f"J{i+1}" for i in range(7)]


def load_data(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    poses = raw["poses"]
    n = len(poses)

    commanded = np.array([p["commanded_deg"] for p in poses])
    q_mean = np.array([p["q"]["mean"] for p in poses])
    q_dot_mean = np.array([p["q_dot"]["mean"] for p in poses])
    q_dot_max = np.array([p["q_dot"]["max"] for p in poses])
    q_ddot_mean = np.array([p["q_ddot"]["mean"] for p in poses])
    q_ddot_max = np.array([p["q_ddot"]["max"] for p in poses])
    tau_mean = np.array([p["tau"]["mean"] for p in poses])
    tau_min = np.array([p["tau"]["min"] for p in poses])
    tau_max = np.array([p["tau"]["max"] for p in poses])
    tau_std = np.array([p["tau"]["std"] for p in poses])

    return {
        "n": n,
        "commanded": commanded,
        "q_mean": q_mean,
        "q_dot_mean": q_dot_mean,
        "q_dot_max": q_dot_max,
        "q_ddot_mean": q_ddot_mean,
        "q_ddot_max": q_ddot_max,
        "tau_mean": tau_mean,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "tau_std": tau_std,
    }


def plot_torques(d, axes):
    """Mean torque per joint with min/max band."""
    ax = axes
    x = np.arange(d["n"])
    for j in range(7):
        ax.plot(x, d["tau_mean"][:, j], label=JOINT_NAMES[j], linewidth=1)
        ax.fill_between(x, d["tau_min"][:, j], d["tau_max"][:, j], alpha=0.15)
    ax.set_xlabel("Pose index")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Mean torque per joint (shaded = min/max)")
    ax.legend(fontsize=7, ncol=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_settling(d, axes):
    """Velocity and acceleration magnitudes — should be near zero."""
    ax1, ax2 = axes
    x = np.arange(d["n"])

    for j in range(7):
        ax1.plot(x, np.abs(d["q_dot_max"][:, j]), linewidth=0.8)
    ax1.set_ylabel("|vel_max| (deg/s)")
    ax1.set_title("Max velocity during recording (should be ~0)")
    ax1.grid(True, alpha=0.3)

    for j in range(7):
        ax2.plot(x, np.abs(d["q_ddot_max"][:, j]), label=JOINT_NAMES[j], linewidth=0.8)
    ax2.set_xlabel("Pose index")
    ax2.set_ylabel("|acc_max| (deg/s²)")
    ax2.set_title("Max acceleration during recording (should be ~0)")
    ax2.legend(fontsize=7, ncol=7, loc="upper right")
    ax2.grid(True, alpha=0.3)


def plot_torque_quality(d, ax):
    """Torque std per joint — lower is better."""
    x = np.arange(d["n"])
    for j in range(7):
        ax.plot(x, d["tau_std"][:, j], label=JOINT_NAMES[j], linewidth=1)
    ax.set_xlabel("Pose index")
    ax.set_ylabel("Torque std (Nm)")
    ax.set_title("Torque standard deviation per pose")
    ax.legend(fontsize=7, ncol=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_positions(d, ax):
    """Commanded vs measured joint positions."""
    x = np.arange(d["n"])
    # Wrap measured q_mean to signed degrees for comparison
    q_signed = d["q_mean"].copy()
    q_signed[q_signed > 180.0] -= 360.0

    for j in range(7):
        ax.plot(x, d["commanded"][:, j], '--', linewidth=0.8, alpha=0.6)
        ax.plot(x, q_signed[:, j], linewidth=1, label=JOINT_NAMES[j])
    ax.set_xlabel("Pose index")
    ax.set_ylabel("Position (deg)")
    ax.set_title("Joint positions (dashed=commanded, solid=measured)")
    ax.legend(fontsize=7, ncol=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="gravity_data.yaml",
                        help="YAML from collect_static_poses.py")
    args = parser.parse_args()

    d = load_data(args.data)
    print(f"Loaded {d['n']} poses from {args.data}")

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"Gravity Sysid Data — {d['n']} poses", fontsize=14)

    plot_torques(d, axes[0])
    plot_settling(d, (axes[1], axes[2]))
    plot_torque_quality(d, axes[3])
    plot_positions(d, axes[4])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
