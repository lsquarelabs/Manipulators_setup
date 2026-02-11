"""
Differential IK controller with gravity compensation.

Supports two modes:
  - 'pose': tracks a target pose (position + orientation)
  - 'velocity': tracks a target twist (linear + angular velocity)

Pose mode:
  Target EE pose --> pose error --> desired twist --> diff-IK --> dq_desired --> torques

Velocity mode:
  Target twist --> diff-IK --> dq_desired --> torques
"""

import numpy as np

from ..robot_model import RobotModel
from ..utility import pose_error
from .base import BaseController


class DiffIKController(BaseController):
    """
    Computes joint torques to track a Cartesian target.

    Modes:
      - 'pose': target = (pos, quat) -> Kp * pose_error -> desired twist
      - 'velocity': target = twist -> use directly as desired twist
    """

    def __init__(
        self,
        model: RobotModel,
        kp_task: np.ndarray,
        kp_joint: np.ndarray,
        kd_joint: np.ndarray,
        dt: float,
        damping: float = 0.01,
        max_joint_velocity: float = 1.5,
        max_torque: np.ndarray = None,
        mode: str = 'pose',
    ):
        super().__init__(model, mode)
        self.kp_task = np.asarray(kp_task, dtype=float)       # (6,)
        self.kp_joint = np.asarray(kp_joint, dtype=float)     # (7,)
        self.kd_joint = np.asarray(kd_joint, dtype=float)     # (7,)
        self.dt = dt
        self.damping = damping
        self.max_dq = max_joint_velocity
        self.max_torque = (
            np.asarray(max_torque, dtype=float) if max_torque is not None
            else np.full(7, 50.0)
        )

        # Bind twist computation method at init (no branching in control loop)
        if mode == 'pose':
            self._compute_twist = self._twist_from_pose
        elif mode == 'velocity':
            self._compute_twist = self._twist_from_velocity
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'pose' or 'velocity'.")

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
        Compute joint torques to track the target.

        Args:
            target: depends on mode:
                - 'pose': (target_pos (3,), target_quat_xyzw (4,))
                - 'velocity': target_twist (6,)
            q: current joint positions in radians (7,)
            dq: current joint velocities in rad/s (7,)

        Returns:
            torques: (7,) joint torques in Nm
        """
        # Desired twist (method selected at init)
        twist_desired = self._compute_twist(target, q)

        # Jacobian and damped pseudoinverse
        J = self.model.jacobian(q)  # (6, 7)
        JJT = J @ J.T + (self.damping ** 2) * np.eye(6)
        dq_desired = J.T @ np.linalg.solve(JJT, twist_desired)  # (7,)

        # Clamp desired joint velocities
        # dq_scale = np.max(np.abs(dq_desired)) / self.max_dq
        # if dq_scale > 1.0:
        #     dq_desired /= dq_scale

        # Desired joint position (one-step integration)
        q_desired = q + dq_desired * self.dt

        # Joint torques: PD tracking + gravity compensation
        dq_desired *= 0.0
        # self.kp_joint*=0.0
        tau = (self.kp_joint * (q_desired - q)
               + self.kd_joint * (dq_desired - dq)
               + self.model.gravity(q))

        # Clamp torques
        tau = np.clip(tau, -self.max_torque, self.max_torque)

        return tau

