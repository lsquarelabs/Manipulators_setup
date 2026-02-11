"""
Operational Space Controller (OSC) with nullspace optimization.

Supports two modes:
  - 'pose': tracks a target pose (position + orientation)
  - 'velocity': tracks a target twist (linear + angular velocity)

tau = J^T @ wrench  +  N @ M @ u_null  +  g(q)

where:
  wrench   = Kp_task * pose_err + Kd_task * vel_err   (pose mode)
           = Kd_task * (target_twist - ee_twist)      (velocity mode)
  Lambda   = (J @ M^-1 @ J^T)^-1                      (task-space inertia)
  J_bar    = M^-1 @ J^T @ Lambda                      (dyn-consistent pseudo-inv)
  N        = I - J^T @ J_bar^T                        (nullspace projector)
  u_null   = Kp_null * (q_ref - q) - Kd_null * dq     (nullspace PD)
"""

import numpy as np

from ..robot_model import RobotModel
from ..utility import pose_error
from .base import BaseController


class OSCController(BaseController):
    """
    Computes joint torques using Operational Space Control
    with dynamically-consistent nullspace.

    Modes:
      - 'pose': target = (pos, quat) -> PD control on pose error
      - 'velocity': target = twist -> D control on velocity error
    """

    def __init__(
        self,
        model: RobotModel,
        kp_task: np.ndarray,
        kd_task: np.ndarray,
        kp_null: np.ndarray,
        kd_null: np.ndarray,
        q_ref: np.ndarray,
        max_joint_torque: np.ndarray,
        max_wrench: np.ndarray,
        mode: str = 'pose',
    ):
        super().__init__(model, mode)
        self.Kp = np.diag(np.asarray(kp_task, dtype=float))       # (6,6)
        self.Kd = np.diag(np.asarray(kd_task, dtype=float))       # (6,6)
        self.kp_null = np.asarray(kp_null, dtype=float)           # (7,)
        self.kd_null = np.asarray(kd_null, dtype=float)           # (7,)
        self.q_ref = np.asarray(q_ref, dtype=float)               # (7,)
        self.q_home = self.q_ref.copy()                                   # (7,)
        self.max_joint_torque = np.asarray(max_joint_torque, dtype=float)  # (7,)
        self.max_wrench = np.asarray(max_wrench, dtype=float)     # (6,)

        # Bind error computation method at init (no branching in control loop)
        if mode == 'pose':
            self._compute_wrench = self._wrench_from_pose
        elif mode == 'velocity':
            self._compute_wrench = self._wrench_from_velocity
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'pose' or 'velocity'.")
        
        self.KD = np.array([60.0, 60.0, 60.0, 60.0, 0.0, 0.0, 0.0])  # Damping gains per joint (Nm/(rad/s))
        self.FC = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.3])  # Coulomb friction per joint (Nm)
        self.FV = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 4.0])  # Viscous friction per joint (Nm/(rad/s))

    def _wrench_from_pose(
        self, target: tuple, q: np.ndarray, dq: np.ndarray, J: np.ndarray
    ) -> np.ndarray:
        """Compute wrench for pose tracking (PD control)."""
        target_pos, target_quat = target
        ee_pos, ee_rot = self.model.fk(q)
        err = pose_error(target_pos, target_quat, ee_pos, ee_rot)
        ee_vel = J @ dq
        vel_err = -ee_vel  # target assumed stationary
        return self.Kp @ err + self.Kd @ vel_err

    def _wrench_from_velocity(
        self, target: np.ndarray, q: np.ndarray, dq: np.ndarray, J: np.ndarray
    ) -> np.ndarray:
        """Compute wrench for velocity tracking (D control)."""
        target_twist = np.asarray(target, dtype=float)
        ee_vel = J @ dq
        vel_err = target_twist - ee_vel
        return self.Kd @ vel_err

    # def compute(
    #     self,
    #     target,
    #     q: np.ndarray,
    #     dq: np.ndarray,
    # ) -> np.ndarray:
    #     """
    #     Compute joint torques to track the target.

    #     Args:
    #         target: depends on mode:
    #             - 'pose': (target_pos (3,), target_quat_xyzw (4,))
    #             - 'velocity': target_twist (6,)
    #         q: current joint positions in radians (7,)
    #         dq: current joint velocities in rad/s (7,)

    #     Returns:
    #         torques: (7,) joint torques in Nm
    #     """
    #     J = self.model.jacobian(q)          # (6,7)
    #     M = self.model.mass_matrix(q)       # (7,7)
    #     M_inv = np.linalg.inv(M)
    #     g = self.model.gravity(q)

    #     # Task-space inertia
    #     Lambda = np.linalg.inv(J @ M_inv @ J.T)

    #     # Compute wrench (method selected at init)
    #     wrench = self._compute_wrench(target, q, dq, J)
    #     # wrench = np.clip(wrench, -self.max_wrench, self.max_wrench)

    #     # Task-space torque
    #     tau_task = J.T @ wrench

    #     # Nullspace projector
    #     N = np.eye(7) - J.T @ Lambda @ J @ M_inv

    #     # Nullspace control (pull toward q_ref)
    #     alpha = 1e-4
    #     self.q_ref += alpha * (N @ (self.q_home - q))
    #     q_err = self.q_ref - q
    #     q_err = (q_err + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    #     # u_null = self.kp_null * q_err - self.kd_null * dq
        
    #     dq_null = N @ dq
    #     u_null = self.kp_null * q_err - self.kd_null * dq_null
    #     tau_null = N @ M @ u_null

        
        
    #     tau = tau_task + g 
    #     # tau = np.clip(tau, -self.max_joint_torque, self.max_joint_torque)
        
    #     # Total torque
    #         # damping_torques = -self.KD * dq
    #         # friction_torques = self.FC * np.sign(dq) + self.FV * dq

    #     return tau

    # def compute(
    #     self,
    #     target,
    #     q: np.ndarray,
    #     dq: np.ndarray,
    # ) -> np.ndarray:
    #     """
    #     Compute joint torques to track the target.

    #     Args:
    #         target: depends on mode:
    #             - 'pose': (target_pos (3,), target_quat_xyzw (4,))
    #             - 'velocity': target_twist (6,)
    #         q: current joint positions in radians (7,)
    #         dq: current joint velocities in rad/s (7,)

    #     Returns:
    #         torques: (7,) joint torques in Nm
    #     """
    #     # ============================================================
    #     # 1. Compute dynamics
    #     # ============================================================
    #     M = self.model.mass_matrix(q)           # (7,7)
    #     M = (M + M.T) * 0.5                     # numerical symmetry
    #     M_inv = np.linalg.inv(M)

    #     nle = self.model.nonlinear_effects(q, dq)  # C(q,dq)*dq + g(q)

    #     # ============================================================
    #     # 2. Jacobians
    #     # ============================================================
    #     J = self.model.jacobian(q)              # (6,7)
    #     Jdot = self.model.jacobian_dot(q, dq)   # (6,7)

    #     # ============================================================
    #     # 3. Task-space inertia
    #     # ============================================================
    #     Lambda = np.linalg.inv(J @ M_inv @ J.T)

    #     # ============================================================
    #     # 4. Task-space nonlinear effects
    #     # ============================================================
    #     mu = Lambda @ (J @ M_inv @ nle - Jdot @ dq)

    #     # ============================================================
    #     # 5. Task-space control law (impedance)
    #     # ============================================================
    #     xdd_des = self._compute_wrench(target, q, dq, J)
    #     F_task = Lambda @ xdd_des + mu

    #     # ============================================================
    #     # 6. Task torque
    #     # ============================================================
    #     tau_task = J.T @ F_task

    #     # ============================================================
    #     # 7. Nullspace projector (dynamically consistent)
    #     # ============================================================
    #     N = np.eye(self.model.nq) - J.T @ Lambda @ J @ M_inv

    #     # ============================================================
    #     # 8. Null-space posture control
    #     # ============================================================
    #     q_err = self.q_ref - q
    #     q_err = (q_err + np.pi) % (2 * np.pi) - np.pi

    #     dq_null = N @ dq
    #     u_null = self.kp_null * q_err - self.kd_null * dq_null
    #     tau_null = N @ M @ u_null

    #     # ============================================================
    #     # 9. Total torque (full dynamic compensation)
    #     # ============================================================
    #     tau = tau_task
    #     tau = np.clip(tau, -self.max_joint_torque, self.max_joint_torque)
    #     return tau