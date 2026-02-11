"""
Abstract base class for Cartesian controllers.

Supports two modes:
  - 'pose': target is (position, quaternion)
  - 'velocity': target is desired twist (6D)
"""

from abc import ABC, abstractmethod
import numpy as np

from ..robot_model import RobotModel


class BaseController(ABC):
    """
    Abstract base for controllers that track Cartesian targets.

    Subclasses must implement `compute()` which returns joint torques.
    The interpretation of the target depends on the mode ('pose' or 'velocity').
    """

    def __init__(self, model: RobotModel, mode: str = 'pose'):
        self.model = model
        self.mode = mode

    @abstractmethod
    def compute(
        self,
        target: tuple,
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
        pass
