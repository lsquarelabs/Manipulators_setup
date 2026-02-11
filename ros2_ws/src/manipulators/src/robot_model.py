"""
Pinocchio robot model wrapper.

Loads full URDF but uses only 7 arm joints for control.
Gripper joints are kept at neutral position.
"""

import numpy as np
import pinocchio as pin


ARM_JOINT_NAMES = [f"gen3_joint_{i}" for i in range(1, 8)]


class RobotModel:
    """Pinocchio model for the Kinova Gen3 arm (7-DOF control, full URDF)."""

    def __init__(self, urdf_path: str, ee_frame: str = "gen3_end_effector_link"):
        # Load full URDF (includes gripper joints)
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Cache EE frame ID
        self.ee_frame_id = self.model.getFrameId(ee_frame)
        if self.ee_frame_id >= self.model.nframes:
            raise ValueError(f"Frame '{ee_frame}' not found in model")

        # Precompute joint indices for 7 arm joints
        self._v_idx = []   # Velocity indices (for Jacobian/gravity extraction)
        self._q_info = []  # (idx_q, nq) pairs for config assignment
        for name in ARM_JOINT_NAMES:
            jid = self.model.getJointId(name)
            if jid == 0:
                raise ValueError(f"Joint '{name}' not found in URDF")
            joint = self.model.joints[jid]
            self._v_idx.append(joint.idx_v)
            self._q_info.append((joint.idx_q, joint.nq))

        self._v_idx = np.array(self._v_idx, dtype=np.intp)

        # Pre-allocate full config and velocity (reused every call)
        self._q_full = pin.neutral(self.model)
        self._dq_full = np.zeros(self.model.nv)

        self.nq = 7

        # Cache per-joint limits from URDF
        self._build_joint_limits()

    def _build_joint_limits(self):
        """Extract position and velocity limits for the 7 arm joints from the URDF.

        Continuous joints (nq=2, cos/sin) get (-inf, +inf) position limits.
        Bounded joints (nq=1) get the URDF limits directly.
        """
        q_min = np.full(self.nq, -np.inf)
        q_max = np.full(self.nq, np.inf)
        v_max = np.zeros(self.nq)
        bounded = np.zeros(self.nq, dtype=bool)

        for i, (idx_q, nq) in enumerate(self._q_info):
            v_max[i] = self.model.velocityLimit[self._v_idx[i]]
            if nq == 1:
                # Revolute joint with finite range
                q_min[i] = self.model.lowerPositionLimit[idx_q]
                q_max[i] = self.model.upperPositionLimit[idx_q]
                bounded[i] = True
            # else: continuous (nq==2), stays at ±inf

        self.q_min = q_min
        self.q_max = q_max
        self.v_max = v_max
        self.bounded_joints = bounded

    def _set_q(self, q: np.ndarray):
        """Set arm joint angles in full config. Handles nq=2 (cos,sin) joints."""
        for i, (idx_q, nq) in enumerate(self._q_info):
            if nq == 1:
                self._q_full[idx_q] = q[i]
            else:  # nq == 2: unbounded joint stored as (cos, sin)
                self._q_full[idx_q] = np.cos(q[i])
                self._q_full[idx_q + 1] = np.sin(q[i])

    def _set_dq(self, dq: np.ndarray):
        """Set arm joint velocities in full velocity vector."""
        self._dq_full[self._v_idx] = dq

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Compute generalized gravity torques (7,)."""
        self._set_q(q)
        pin.computeGeneralizedGravity(self.model, self.data, self._q_full)
        return self.data.g[self._v_idx]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute EE Jacobian (6x7) in LOCAL_WORLD_ALIGNED frame."""
        self._set_q(q)
        pin.computeJointJacobians(self.model, self.data, self._q_full)
        pin.framesForwardKinematics(self.model, self.data, self._q_full)
        J_full = pin.getFrameJacobian(
            self.model, self.data, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        return J_full[:, self._v_idx]

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute joint-space mass matrix M(q) (7x7)."""
        self._set_q(q)
        pin.crba(self.model, self.data, self._q_full)
        M = self.data.M.copy()
        M = np.triu(M) + np.triu(M, 1).T  # crba only fills upper triangle, not needed even without this its correct but just to be safe we symmetrize it
        return M[np.ix_(self._v_idx, self._v_idx)]

    def nonlinear_effects(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Compute C(q,dq)*dq + g(q) (7,)."""
        self._set_q(q)
        self._set_dq(dq)
        nle = pin.nonLinearEffects(self.model, self.data, self._q_full, self._dq_full)
        return nle[self._v_idx]

    def jacobian_dot(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Compute EE Jacobian time derivative dJ/dt (6x7) in LOCAL_WORLD_ALIGNED frame."""
        self._set_q(q)
        self._set_dq(dq)
        pin.computeJointJacobians(self.model, self.data, self._q_full)
        pin.framesForwardKinematics(self.model, self.data, self._q_full)
        pin.computeJointJacobiansTimeVariation(
            self.model, self.data, self._q_full, self._dq_full
        )
        Jdot_full = pin.getFrameJacobianTimeVariation(
            self.model, self.data, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        return Jdot_full[:, self._v_idx]

    def fk(self, q: np.ndarray):
        """Forward kinematics. Returns (position (3,), rotation (3,3))."""
        self._set_q(q)
        pin.framesForwardKinematics(self.model, self.data, self._q_full)
        oMf = self.data.oMf[self.ee_frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()
