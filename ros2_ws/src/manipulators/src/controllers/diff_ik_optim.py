"""
QP-based Differential IK controller with gravity compensation.

Solves a quadratic program at each timestep to find optimal joint displacements
that track a Cartesian target while respecting joint velocity and position limits.

Supports two modes:
  - 'pose': tracks a target pose (position + orientation)
  - 'velocity': tracks a target twist (linear + angular velocity)

QP formulation (decision variable: Δq ∈ ℝ⁷):
  minimize   ½ ||J·Δq - dt·v_des||²_w  +  ½ w_p·||Δq - Δq_ref||²  +  ½ λ·||Δq||²
  subject to  -dt·v_max ≤ Δq ≤ dt·v_max             (velocity limits)
              γ·(q_min - q) ≤ Δq ≤ γ·(q_max - q)    (position limits)

After solving:
  q_desired = q + Δq,  dq_desired = Δq / dt
  τ = Kp·(q_desired - q) + Kd·(dq_desired - dq) + g(q)
"""

import numpy as np
np.set_printoptions(precision=3, suppress=True)
import qpsolvers
from scipy import sparse

from ..robot_model import RobotModel
from ..utility import pose_error
from .base import BaseController


class DiffIKOptimController(BaseController):
    """
    QP-based differential IK controller.

    Solves a constrained QP at each timestep for the optimal joint displacement,
    then converts to torques via PD tracking + gravity compensation.
    """

    def __init__(
        self,
        model: RobotModel,
        kp_task: np.ndarray,
        kp_joint: np.ndarray,
        kd_joint: np.ndarray,
        dt: float,
        task_weight: float = 1.0,
        posture_weight: float = 1e-3,
        damping: float = 1e-6,
        velocity_limit_scale: float = 1.0,
        q_ref: np.ndarray = None,
        config_limit_gain: float = 0.5,
        max_torque: np.ndarray = None,
        solver: str = 'osqp',
        mode: str = 'pose',
    ):
        """
        Args:
            model: RobotModel instance (joint limits read from URDF automatically)
            kp_task: (6,) task-space proportional gains (pose error → desired twist)
            kp_joint: (7,) joint-space P gains for torque computation
            kd_joint: (7,) joint-space D gains for torque computation
            dt: control timestep [s]
            task_weight: scalar weight for the EE tracking cost in the QP
            posture_weight: scalar weight for posture regularization on bounded joints
            damping: Tikhonov regularization for numerical stability
            velocity_limit_scale: multiplier on URDF velocity limits (>1 to allow faster motion)
            q_ref: (7,) reference posture for regularization (bounded joints only)
            config_limit_gain: gain ∈ (0,1] for position limit constraints (0 disables)
            max_torque: (7,) torque limits [Nm]
            solver: QP solver backend ('osqp', 'daqp', 'cvxopt', etc.)
            mode: 'pose' or 'velocity'
        """
        super().__init__(model, mode)

        nq = 7
        self.nq = nq
        self.dt = dt

        # Task-space gains (error → desired twist)
        self.kp_task = np.asarray(kp_task, dtype=float)  # (6,)

        # Joint-space PD gains (for torque output)
        self.kp_joint = np.asarray(kp_joint, dtype=float)  # (7,)
        self.kd_joint = np.asarray(kd_joint, dtype=float)  # (7,)

        # QP cost weights
        self.task_weight = task_weight
        self.posture_weight = posture_weight
        self.damping = damping

        # Joint limits from URDF (via RobotModel), scaled
        self.v_max = model.v_max.copy() * velocity_limit_scale
        self.q_min = model.q_min.copy()                  # (7,) -inf for continuous
        self.q_max = model.q_max.copy()                  # (7,) +inf for continuous
        self.bounded = model.bounded_joints.copy()        # (7,) bool mask
        self.config_limit_gain = config_limit_gain

        # Reference posture for regularization (only applied to bounded joints)
        self.q_ref = (
            np.asarray(q_ref, dtype=float) if q_ref is not None
            else np.zeros(nq)
        )

        # Torque limits
        self.max_torque = (
            np.asarray(max_torque, dtype=float) if max_torque is not None
            else np.full(nq, 50.0)
        )

        self.solver = solver

        # Friction compensation (same values as OSC controller)
        self.FC = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.3])  # Coulomb friction (Nm)
        self.FV = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 4.0])  # Viscous friction (Nm/(rad/s))

        # Bind twist computation at init (no branching in control loop)
        if mode == 'pose':
            self._compute_twist = self._twist_from_pose
        elif mode == 'velocity':
            self._compute_twist = self._twist_from_velocity
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'pose' or 'velocity'.")

        # Pre-build static constraint structure and posture mask
        self._build_constraint_structure()
        self._build_posture_weight_matrix()

    def _build_constraint_structure(self):
        """Pre-compute static parts of the inequality constraint matrices.

        Velocity limits apply to all joints.
        Position limits only apply to bounded (non-continuous) joints from URDF.
        """
        nq = self.nq

        # Velocity limits: [I; -I] Δq ≤ [dt·v_max; dt·v_max]
        G_vel = np.vstack([np.eye(nq), -np.eye(nq)])  # (14, 7)
        self._h_vel = np.tile(self.dt * self.v_max, 2)  # (14,)

        # Position limits: only bounded joints, and only if gain > 0
        self._bounded_idx = np.where(self.bounded)[0]
        self._use_pos_limits = (
            len(self._bounded_idx) > 0 and self.config_limit_gain > 0.0
        )

        if self._use_pos_limits:
            P = np.zeros((len(self._bounded_idx), nq))
            for row, col in enumerate(self._bounded_idx):
                P[row, col] = 1.0
            G_pos = np.vstack([P, -P])
            self._G_full = sparse.csc_matrix(np.vstack([G_vel, G_pos]))
        else:
            self._G_full = sparse.csc_matrix(G_vel)

    def _build_posture_weight_matrix(self):
        """Build diagonal weight matrix for posture regularization (all joints)."""
        self._W_posture = self.posture_weight * np.eye(self.nq)

    def _build_h(self, q: np.ndarray) -> np.ndarray:
        """Build the h vector for G·Δq ≤ h at the current configuration."""
        h = self._h_vel.copy()

        if self._use_pos_limits:
            idx = self._bounded_idx
            gain = self.config_limit_gain
            h_upper = gain * (self.q_max[idx] - q[idx])
            h_lower = gain * (q[idx] - self.q_min[idx])
            h = np.concatenate([h, h_upper, h_lower])

        return h

    def _twist_from_pose(self, target: tuple, q: np.ndarray) -> np.ndarray:
        """Compute desired twist from pose error."""
        target_pos, target_quat = target
        ee_pos, ee_rot = self.model.fk(q)
        error = pose_error(target_pos, target_quat, ee_pos, ee_rot)
        return self.kp_task * error

    def _twist_from_velocity(self, target: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Use target twist directly."""
        return np.asarray(target, dtype=float)

    def compute(
        self,
        target,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> np.ndarray:
        """
        Compute joint torques via QP-based differential IK.

        Args:
            target: depends on mode:
                - 'pose': (target_pos (3,), target_quat_xyzw (4,))
                - 'velocity': target_twist (6,)
            q: current joint positions in radians (7,)
            dq: current joint velocities in rad/s (7,)

        Returns:
            torques: (7,) joint torques in Nm
        """
        nq = self.nq
        dt = self.dt

        # --- Desired twist (method selected at init) ---
        twist_desired = self._compute_twist(target, q)

        # --- Jacobian ---
        J = self.model.jacobian(q)  # (6, 7)

        # --- QP cost: ½ Δqᵀ P Δq + cᵀ Δq ---

        # Task cost: ½ ||J·Δq - b||²  where b = dt * twist_desired
        b = dt * twist_desired
        w = self.task_weight
        P = w * (J.T @ J)          # (7, 7)
        c = -w * (J.T @ b)         # (7,)

        # Posture regularization (bounded joints only — continuous joints are free)
        # ½ (Δq - Δq_ref)ᵀ W_p (Δq - Δq_ref), W_p is zero for continuous joints
        q_err = self.q_ref - q
        q_err = (q_err + np.pi) % (2 * np.pi) - np.pi

        P += self._W_posture
        c += -self._W_posture @ q_err

        # Tikhonov damping
        P += self.damping * np.eye(nq)

        # Symmetrize and convert to sparse (numerical safety + OSQP perf)
        P = 0.5 * (P + P.T)
        P_sparse = sparse.csc_matrix(P)

        # --- QP constraints ---
        h_ineq = self._build_h(q)

        # --- Solve ---
        delta_q = qpsolvers.solve_qp(
            P_sparse, c, self._G_full, h_ineq, solver=self.solver,
        )

        if delta_q is None:
            # Fallback: damped pseudoinverse (unconstrained)
            JJT = J @ J.T + (0.01 ** 2) * np.eye(6)
            delta_q = dt * J.T @ np.linalg.solve(JJT, twist_desired)

        # --- Convert to torques ---
        dq_desired = delta_q / dt
        q_desired = q + delta_q

        friction_torques = self.FC * np.sign(dq) + self.FV * dq

        tau = (self.kp_joint * (q_desired - q)
               + self.kd_joint * (dq_desired - dq)
               + self.model.gravity(q)
               + friction_torques)

        tau = np.clip(tau, -self.max_torque, self.max_torque)

        print(f"\n------------"
              f"\nq: {q}"
              f"\ndq: {dq}"
              f"\ntarget: {target}"
              f"\ntwist_desired: {twist_desired}"
              f"\nb: {b}"
              f"\ndelta_q: {delta_q}"
              f"\ndq_desired: {dq_desired}"
              f"\nq_desired: {q_desired}"
              f"\ntau: {tau}")

        return tau
