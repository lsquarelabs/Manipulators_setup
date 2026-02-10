"""
Operational Space Controller (OSC) with nullspace optimization.

tau = J^T @ wrench  +  N @ M @ u_null  +  g(q)

where:
  wrench   = Kp_task * pose_err + Kd_task * vel_err   (task-space PD)
  Lambda   = (J @ M^-1 @ J^T)^-1                      (task-space inertia)
  J_bar    = M^-1 @ J^T @ Lambda                      (dyn-consistent pseudo-inv)
  N        = I - J^T @ J_bar^T                         (nullspace projector)
  u_null   = Kp_null * (q_ref - q) - Kd_null * dq     (nullspace PD)
"""

import numpy as np

from .robot_model import RobotModel
from .utility import pose_error


class OSCController:
    """
    Computes joint torques to track a Cartesian pose target using
    Operational Space Control with dynamically-consistent nullspace.

    Each cycle:
      1. FK to get current EE pose
      2. 6D pose error (position + orientation)
      3. EE velocity via Jacobian
      4. Task-space wrench (PD), clipped to safety limits
      5. Joint torque via J^T @ wrench
      6. Nullspace torque to pull joints toward q_ref
      7. Add gravity compensation
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
    ):
        self.model = model
        self.Kp = np.diag(np.asarray(kp_task, dtype=float))       # (6,6)
        self.Kd = np.diag(np.asarray(kd_task, dtype=float))       # (6,6)
        self.kp_null = np.asarray(kp_null, dtype=float)            # (7,)
        self.kd_null = np.asarray(kd_null, dtype=float)            # (7,)
        self.q_ref = np.asarray(q_ref, dtype=float)                # (7,)
        self.max_joint_torque = np.asarray(max_joint_torque, dtype=float)  # (7,)
        self.max_wrench = np.asarray(max_wrench, dtype=float)      # (6,)

    def compute(
        self,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> np.ndarray:
        """
        Compute joint torques to track the target pose.

        Args:
            target_pos: desired EE position (3,)
            target_quat_xyzw: desired EE orientation [x,y,z,w] (4,)
            q: current joint positions in radians (7,)
            dq: current joint velocities in rad/s (7,)

        Returns:
            torques: (7,) joint torques in Nm
        """
        # Current EE pose
        ee_pos, ee_rot = self.model.fk(q)

        # 6D pose error
        err = pose_error(target_pos, target_quat_xyzw, ee_pos, ee_rot)

        # Jacobian, mass matrix, gravity
        J = self.model.jacobian(q)      # (6, 7)
        M = self.model.mass_matrix(q)   # (7, 7)
        g = self.model.gravity(q)       # (7,)

        # EE velocity and velocity error (target assumed stationary)
        ee_vel = J @ dq                 # (6,)
        vel_err = -ee_vel               # (6,)

        # Task-space wrench (PD)
        wrench = self.Kp @ err + self.Kd @ vel_err
        wrench = np.clip(wrench, -self.max_wrench, self.max_wrench)

        # Task-space torque
        tau_task = J.T @ wrench

        # Dynamic consistency
        M_inv = np.linalg.inv(M)
        Lambda = np.linalg.inv(J @ M_inv @ J.T)    # task-space inertia
        J_bar = M_inv @ J.T @ Lambda                # dyn-consistent pseudo-inv

        # Nullspace projector
        N = np.eye(7) - J.T @ J_bar.T

        # Nullspace control (pull toward q_ref)
        q_err = self.q_ref - q
        q_err = (q_err + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        u_null = self.kp_null * q_err - self.kd_null * dq
        tau_null = N @ M @ u_null

        # Total torque
        tau = tau_task + tau_null + g
        tau = np.clip(tau, -self.max_joint_torque, self.max_joint_torque)

        return tau
