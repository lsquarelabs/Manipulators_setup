#!/usr/bin/env python3
"""Gravity-only residual identification.

Loads static pose data, builds the gravity regressor via Pinocchio,
solves regularized LS for parameter corrections delta_pi.

No ROS required.

Usage:
    python identify_gravity.py gravity_data.npz \
        --urdf ../assets/robots/kinova/urdf/gen3_2f85.urdf
"""

import argparse
import numpy as np
import pinocchio as pin

ARM_JOINT_NAMES = [f"gen3_joint_{i}" for i in range(1, 8)]
PARAMS_PER_BODY = 10
PARAM_NAMES = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]


def load_model(urdf_path):
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    v_idx, q_info = [], []
    for name in ARM_JOINT_NAMES:
        jid = model.getJointId(name)
        joint = model.joints[jid]
        v_idx.append(joint.idx_v)
        q_info.append((joint.idx_q, joint.nq))
    return model, data, np.array(v_idx, dtype=np.intp), q_info


def set_arm_q(q_full, q_arm, q_info):
    for i, (idx_q, nq) in enumerate(q_info):
        if nq == 1:
            q_full[idx_q] = q_arm[i]
        else:
            q_full[idx_q] = np.cos(q_arm[i])
            q_full[idx_q + 1] = np.sin(q_arm[i])


def build_gravity_system(model, data, v_idx, q_info, q_all, tau_all):
    """Build stacked regressor Y and residual vector for gravity-only."""
    zeros = np.zeros(model.nv)
    q_full = pin.neutral(model)
    Y_list, res_list = [], []

    for q_arm, tau_meas in zip(q_all, tau_all):
        set_arm_q(q_full, q_arm, q_info)

        pin.computeGeneralizedGravity(model, data, q_full)
        g_cad = data.g[v_idx].copy()

        Y = pin.computeJointTorqueRegressor(model, data, q_full, zeros, zeros)
        Y_list.append(Y[v_idx, :])
        res_list.append(tau_meas - g_cad)

    return np.vstack(Y_list), np.concatenate(res_list)


def solve(Y, tau_res, lambda_reg):
    """Regularized LS on active (non-zero) columns only."""
    col_norms = np.linalg.norm(Y, axis=0)
    active = col_norms > 1e-10
    Ya = Y[:, active]

    A = Ya.T @ Ya + lambda_reg * np.eye(Ya.shape[1])
    delta_active = np.linalg.solve(A, Ya.T @ tau_res)

    delta_pi = np.zeros(Y.shape[1])
    delta_pi[active] = delta_active
    return delta_pi, active


def evaluate(Y, tau_res, delta_pi):
    """RMSE before/after, overall and per-joint."""
    pred = Y @ delta_pi
    err_before = tau_res
    err_after = tau_res - pred
    N = len(tau_res) // 7

    rmse_before = np.sqrt(np.mean(err_before ** 2))
    rmse_after = np.sqrt(np.mean(err_after ** 2))
    pj_before = np.sqrt(np.mean(err_before.reshape(N, 7) ** 2, axis=0))
    pj_after = np.sqrt(np.mean(err_after.reshape(N, 7) ** 2, axis=0))
    return rmse_before, rmse_after, pj_before, pj_after


def print_corrections(model, delta_pi):
    n_bodies = delta_pi.shape[0] // PARAMS_PER_BODY
    print("\n--- Parameter corrections ---")
    for i in range(n_bodies):
        dp = delta_pi[i * PARAMS_PER_BODY:(i + 1) * PARAMS_PER_BODY]
        body = str(model.names[i])
        entries = [(n, v) for n, v in zip(PARAM_NAMES, dp) if abs(v) > 1e-6]
        if entries:
            print(f"  {body}:")
            for n, v in entries:
                print(f"    d{n:3s} = {v:+.6f}")


def apply_corrections(model, delta_pi):
    """Update Pinocchio model inertias (mass + CoM) with identified corrections."""
    for i in range(model.njoints):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help=".npz from collect_static_poses.py")
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="gravity_corrections.npz")
    args = parser.parse_args()

    # Load data
    d = np.load(args.data)
    q_all, tau_all = d["q_rad"], d["tau"]
    N = len(q_all)
    print(f"Loaded {N} samples")

    # Split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    n_val = max(1, int(N * args.val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Model
    model, data, v_idx, q_info = load_model(args.urdf)

    # Build systems
    Y_train, res_train = build_gravity_system(
        model, data, v_idx, q_info, q_all[train_idx], tau_all[train_idx]
    )
    Y_val, res_val = build_gravity_system(
        model, data, v_idx, q_info, q_all[val_idx], tau_all[val_idx]
    )

    # Solve
    delta_pi, active = solve(Y_train, res_train, args.lambda_reg)

    # Evaluate
    rb_t, ra_t, pjb_t, pja_t = evaluate(Y_train, res_train, delta_pi)
    rb_v, ra_v, pjb_v, pja_v = evaluate(Y_val, res_val, delta_pi)

    print(f"\nActive params: {active.sum()}")
    print(f"Cond(Y_active): {np.linalg.cond(Y_train[:, active]):.1f}")
    print(f"\nRMSE (Nm)       Train    Val")
    print(f"  Before:      {rb_t:.4f}  {rb_v:.4f}")
    print(f"  After:       {ra_t:.4f}  {ra_v:.4f}")
    print(f"  Improvement: {(1 - ra_t / rb_t) * 100:.1f}%   {(1 - ra_v / rb_v) * 100:.1f}%")

    print(f"\nPer-joint RMSE (Nm) — validation:")
    print("  Joint:  " + "  ".join(f"  J{j+1}  " for j in range(7)))
    print("  Before: " + "  ".join(f"{v:.4f}" for v in pjb_v))
    print("  After:  " + "  ".join(f"{v:.4f}" for v in pja_v))

    print_corrections(model, delta_pi)

    # Save
    np.savez(args.output, delta_pi=delta_pi, active=active,
             lambda_reg=np.array(args.lambda_reg))
    print(f"\nSaved corrections to {args.output}")


if __name__ == "__main__":
    main()
