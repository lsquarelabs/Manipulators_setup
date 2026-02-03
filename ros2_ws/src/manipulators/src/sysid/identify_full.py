#!/usr/bin/env python3
"""Full system identification (not residual corrections).

Identifies inertial parameters directly from data, without relying on
CAD model accuracy.

For static (gravity) data:
    tau = Y_gravity(q) * pi
    Only mass and first moments (m, m*cx, m*cy, m*cz) are identifiable.

For dynamic data (future):
    tau = Y_full(q, qd, qdd) * pi
    Full inertial parameters (mass, CoM, inertia tensor) identifiable.

Usage:
    python identify_full.py --data gravity_data.yaml --urdf gen3.urdf
"""

import os
import argparse
import numpy as np
import yaml
import pinocchio as pin

DEFAULT_URDF = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "assets", "robots", "kinova", "urdf", "gen3_2f85.urdf",
)

ARM_JOINT_NAMES = [f"gen3_joint_{i}" for i in range(1, 8)]
PARAMS_PER_BODY = 10
PARAM_NAMES = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]


def load_model(urdf_path):
    """Load Pinocchio model for regressor computation."""
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    v_idx, q_info = [], []
    for name in ARM_JOINT_NAMES:
        jid = model.getJointId(name)
        joint = model.joints[jid]
        v_idx.append(joint.idx_v)
        q_info.append((joint.idx_q, joint.nq))
    return model, data, np.array(v_idx, dtype=np.intp), q_info


def load_static_data(yaml_path):
    """Load static pose data, return q_rad (N,7) and tau (N,7)."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    poses = raw["poses"]

    q_deg = np.array([p["q"]["mean"] for p in poses])
    tau = np.array([p["tau"]["mean"] for p in poses])

    # Convert Kinova 0-360 degrees to signed radians
    q_signed = q_deg.copy()
    q_signed[q_signed > 180.0] -= 360.0
    q_rad = np.deg2rad(q_signed)

    # Negate torques: Kinova reports reaction torque, Pinocchio expects applied torque
    tau = -tau

    # Joint torque offset corrections (empirically measured biases)
    # These account for sensor calibration, friction, cable tension etc.
    JOINT_OFFSETS = np.array([
        -0.2093,  # J1
        -0.8741,  # J2
        -0.2108,  # J3
        -0.0029,  # J4
        -0.0086,  # J5
        +1.2611,  # J6
        +0.0385,  # J7
    ])
    tau -= JOINT_OFFSETS

    return q_rad, tau


def set_arm_q(q_full, q_arm, q_info):
    """Set arm joint positions in Pinocchio config vector."""
    for i, (idx_q, nq) in enumerate(q_info):
        if nq == 1:
            q_full[idx_q] = q_arm[i]
        else:
            q_full[idx_q] = np.cos(q_arm[i])
            q_full[idx_q + 1] = np.sin(q_arm[i])


def build_gravity_regressor(model, data, v_idx, q_info, q_all, tau_all):
    """Build regressor for tau = Y * pi (full params, not residual)."""
    zeros = np.zeros(model.nv)
    q_full = pin.neutral(model)
    Y_list, tau_list = [], []

    for q_arm, tau_meas in zip(q_all, tau_all):
        set_arm_q(q_full, q_arm, q_info)

        # Compute full regressor Y such that tau_gravity = Y * pi
        Y = pin.computeJointTorqueRegressor(model, data, q_full, zeros, zeros)
        Y_list.append(Y[v_idx, :])
        tau_list.append(tau_meas)

    return np.vstack(Y_list), np.concatenate(tau_list)


def find_identifiable_params(Y, tol=1e-8):
    """Find which parameter columns are identifiable (have non-zero effect).

    For static/gravity data, only mass and first moments matter.
    Inertia terms (Ixx, etc.) have zero columns since qdd=0.
    """
    col_norms = np.linalg.norm(Y, axis=0)
    active = col_norms > tol
    return active


def solve_with_constraints(Y, tau, lambda_reg=1.0, min_mass=0.01):
    """Solve for parameters with regularization and physical constraints.

    For unconstrained solve: pi = (Y'Y + lambda*I)^-1 Y' tau

    Physical constraints (enforced post-hoc for now):
    - Mass must be positive
    - Inertia must be positive semi-definite (not enforced in static case)
    """
    active = find_identifiable_params(Y)
    Ya = Y[:, active]

    # Regularized least squares
    A = Ya.T @ Ya + lambda_reg * np.eye(Ya.shape[1])
    pi_active = np.linalg.solve(A, Ya.T @ tau)

    # Expand to full parameter vector
    pi = np.zeros(Y.shape[1])
    pi[active] = pi_active

    # Post-hoc mass constraint: clip negative masses
    n_bodies = len(pi) // PARAMS_PER_BODY
    for i in range(n_bodies):
        mass_idx = i * PARAMS_PER_BODY
        if pi[mass_idx] < min_mass:
            # If mass is negative/tiny, set to minimum and zero out first moments
            pi[mass_idx] = min_mass
            pi[mass_idx + 1:mass_idx + 4] = 0.0

    return pi, active


def solve_constrained_qp(Y, tau, lambda_reg=1.0, min_mass=0.01):
    """Solve with explicit mass positivity constraints using QP.

    minimize    ||Y*pi - tau||^2 + lambda*||pi||^2
    subject to  m_i >= min_mass  for all bodies

    Requires cvxpy or scipy optimize.
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("cvxpy not available, falling back to unconstrained solve")
        return solve_with_constraints(Y, tau, lambda_reg, min_mass)

    active = find_identifiable_params(Y)
    n_params = Y.shape[1]
    n_bodies = n_params // PARAMS_PER_BODY

    # Decision variable
    pi = cp.Variable(n_params)

    # Objective: ||Y*pi - tau||^2 + lambda*||pi||^2
    objective = cp.Minimize(
        cp.sum_squares(Y @ pi - tau) + lambda_reg * cp.sum_squares(pi)
    )

    # Constraints: mass >= min_mass for each body
    constraints = []
    for i in range(n_bodies):
        mass_idx = i * PARAMS_PER_BODY
        constraints.append(pi[mass_idx] >= min_mass)

    # Fix non-identifiable params to zero
    for j in range(n_params):
        if not active[j]:
            constraints.append(pi[j] == 0)

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        if problem.status in ['optimal', 'optimal_inaccurate']:
            return pi.value, active
        else:
            print(f"QP solver status: {problem.status}, falling back")
            return solve_with_constraints(Y, tau, lambda_reg, min_mass)
    except Exception as e:
        print(f"QP solve failed: {e}, falling back")
        return solve_with_constraints(Y, tau, lambda_reg, min_mass)


def evaluate(Y, tau, pi):
    """Compute RMSE overall and per-joint."""
    pred = Y @ pi
    err = tau - pred
    N = len(tau) // 7

    rmse = np.sqrt(np.mean(err ** 2))
    per_joint = np.sqrt(np.mean(err.reshape(N, 7) ** 2, axis=0))
    return rmse, per_joint


def extract_inertial_params(model):
    """Extract current inertial parameters from Pinocchio model."""
    params = []
    for i in range(model.njoints):
        inertia = model.inertias[i]
        m = inertia.mass
        lever = inertia.lever
        I = inertia.inertia  # 3x3 symmetric matrix

        # First moments: m * c
        mx, my, mz = m * lever

        # Inertia at CoM (Pinocchio stores inertia at joint frame)
        # For simplicity, extract the 6 unique components
        Ixx, Iyy, Izz = I[0, 0], I[1, 1], I[2, 2]
        Ixy, Ixz, Iyz = I[0, 1], I[0, 2], I[1, 2]

        params.extend([m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz])

    return np.array(params)


def params_to_inertias(pi, n_bodies):
    """Convert parameter vector to list of (mass, lever, inertia_matrix)."""
    inertias = []
    for i in range(n_bodies):
        p = pi[i * PARAMS_PER_BODY:(i + 1) * PARAMS_PER_BODY]
        m = max(p[0], 0.001)  # Ensure positive mass
        mx, my, mz = p[1], p[2], p[3]
        lever = np.array([mx, my, mz]) / m

        Ixx, Ixy, Ixz, Iyy, Iyz, Izz = p[4], p[5], p[6], p[7], p[8], p[9]
        I_mat = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ])

        inertias.append((m, lever, I_mat))
    return inertias


def print_identified_params(pi, model, active):
    """Print identified parameters in readable format."""
    n_bodies = len(pi) // PARAMS_PER_BODY
    print("\n--- Identified Parameters ---")

    for i in range(n_bodies):
        p = pi[i * PARAMS_PER_BODY:(i + 1) * PARAMS_PER_BODY]
        a = active[i * PARAMS_PER_BODY:(i + 1) * PARAMS_PER_BODY]
        body = str(model.names[i]) if i < len(model.names) else f"body_{i}"

        # Only print if any parameter is active
        if not any(a):
            continue

        print(f"\n  {body}:")
        for j, (name, val, is_active) in enumerate(zip(PARAM_NAMES, p, a)):
            if is_active and abs(val) > 1e-8:
                print(f"    {name:3s} = {val:+.6f}")

        # Derived: CoM if mass is identified
        if a[0] and p[0] > 0.01:
            cx = p[1] / p[0] if a[1] else 0
            cy = p[2] / p[0] if a[2] else 0
            cz = p[3] / p[0] if a[3] else 0
            print(f"    CoM = [{cx:+.4f}, {cy:+.4f}, {cz:+.4f}]")


def compare_with_cad(pi_identified, model):
    """Compare identified parameters with CAD model values."""
    pi_cad = extract_inertial_params(model)
    n_bodies = len(pi_identified) // PARAMS_PER_BODY

    print("\n--- Comparison with CAD Model ---")
    print(f"{'Body':<20} {'Param':<5} {'CAD':>10} {'Identified':>10} {'Diff':>10}")
    print("-" * 60)

    for i in range(n_bodies):
        body = str(model.names[i]) if i < len(model.names) else f"body_{i}"
        for j in range(4):  # Only mass and first moments
            idx = i * PARAMS_PER_BODY + j
            cad_val = pi_cad[idx]
            id_val = pi_identified[idx]
            diff = id_val - cad_val

            if abs(cad_val) > 1e-6 or abs(id_val) > 1e-6:
                print(f"{body:<20} {PARAM_NAMES[j]:<5} {cad_val:>10.4f} {id_val:>10.4f} {diff:>+10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Full system identification")
    parser.add_argument("--data", default="gravity_data_combined.yaml",
                        help="Static pose data from collect_static_poses.py")
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--lambda-reg", type=float, default=0.1,
                        help="Regularization strength (lower = more trust in data)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-qp", action="store_true",
                        help="Use QP solver with explicit constraints")
    parser.add_argument("--output", default="identified_params.yaml")
    parser.add_argument("--compare-cad", action="store_true",
                        help="Compare with CAD model parameters")
    args = parser.parse_args()

    # Load data
    q_all, tau_all = load_static_data(args.data)
    N = len(q_all)
    print(f"Loaded {N} static pose samples")

    # Train/val split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    n_val = max(1, int(N * args.val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Load model (for regressor computation, not for parameters)
    model, data, v_idx, q_info = load_model(args.urdf)
    print(f"Model has {model.njoints} joints, {model.njoints * PARAMS_PER_BODY} parameters")

    # Build regressor matrices
    Y_train, tau_train = build_gravity_regressor(
        model, data, v_idx, q_info, q_all[train_idx], tau_all[train_idx]
    )
    Y_val, tau_val = build_gravity_regressor(
        model, data, v_idx, q_info, q_all[val_idx], tau_all[val_idx]
    )

    print(f"Regressor shape: {Y_train.shape}")

    # Identify parameters
    if args.use_qp:
        print("Using QP solver with mass constraints...")
        pi, active = solve_constrained_qp(Y_train, tau_train, args.lambda_reg)
    else:
        print("Using regularized LS...")
        pi, active = solve_with_constraints(Y_train, tau_train, args.lambda_reg)

    print(f"Active (identifiable) parameters: {active.sum()}")

    # Evaluate
    rmse_train, pj_train = evaluate(Y_train, tau_train, pi)
    rmse_val, pj_val = evaluate(Y_val, tau_val, pi)

    print(f"\nRMSE (Nm):")
    print(f"  Train: {rmse_train:.4f}")
    print(f"  Val:   {rmse_val:.4f}")

    print(f"\nPer-joint RMSE (Nm) - Validation:")
    print("  Joint:  " + "  ".join(f"  J{j+1}  " for j in range(7)))
    print("  RMSE:   " + "  ".join(f"{v:.4f}" for v in pj_val))

    # Print identified parameters
    print_identified_params(pi, model, active)

    # Compare with CAD
    if args.compare_cad:
        compare_with_cad(pi, model)

    # Save results
    n_bodies = len(pi) // PARAMS_PER_BODY
    output = {
        "method": "full_sysid",
        "lambda_reg": args.lambda_reg,
        "n_samples": int(N),
        "n_active_params": int(active.sum()),
        "rmse_train": float(rmse_train),
        "rmse_val": float(rmse_val),
        "per_joint_rmse_val": pj_val.tolist(),
        "parameters": pi.tolist(),
        "active": active.tolist(),
        # Also save in structured form for easier use
        "bodies": [],
    }

    for i in range(n_bodies):
        p = pi[i * PARAMS_PER_BODY:(i + 1) * PARAMS_PER_BODY]
        body_name = str(model.names[i]) if i < len(model.names) else f"body_{i}"
        body_params = {
            "name": body_name,
            "mass": float(p[0]),
            "first_moment": [float(p[1]), float(p[2]), float(p[3])],
        }
        if p[0] > 0.01:
            body_params["com"] = [
                float(p[1] / p[0]),
                float(p[2] / p[0]),
                float(p[3] / p[0])
            ]
        output["bodies"].append(body_params)

    with open(args.output, "w") as f:
        yaml.dump(output, f, default_flow_style=None, sort_keys=False)
    print(f"\nSaved identified parameters to {args.output}")


if __name__ == "__main__":
    main()
