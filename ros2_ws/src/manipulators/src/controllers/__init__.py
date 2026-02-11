"""
Controllers for Cartesian tracking.

Available controllers:
  - DiffIKController: Differential IK with damped pseudoinverse
  - DiffIKOptimController: QP-based Differential IK with joint limit constraints
  - OSCController: Operational Space Control with nullspace optimization

All support modes:
  - 'pose': target = (position, quaternion)
  - 'velocity': target = twist (6D)

Factory function:
  - create_controller(controller_type, model, mode='pose', **kwargs) -> BaseController
"""

from .base import BaseController
from .diff_ik import DiffIKController
from .diff_ik_optim import DiffIKOptimController
from .osc import OSCController


def create_controller(
    controller_type: str,
    model,
    mode: str = 'pose',
    **kwargs
) -> BaseController:
    """
    Factory function to create a controller by name.

    Args:
        controller_type: 'diff_ik' or 'osc'
        model: RobotModel instance
        mode: 'pose' or 'velocity'
        **kwargs: controller-specific parameters

    Returns:
        BaseController subclass instance

    Raises:
        ValueError: if controller_type is unknown
    """
    if controller_type == 'diff_ik':
        return DiffIKController(
            model=model,
            kp_task=kwargs['kp_task'],
            kp_joint=kwargs['kp_joint'],
            kd_joint=kwargs['kd_joint'],
            dt=kwargs['dt'],
            damping=kwargs.get('damping', 0.01),
            max_joint_velocity=kwargs.get('max_joint_velocity', 1.5),
            max_torque=kwargs.get('max_torque'),
            mode=mode,
        )
    elif controller_type == 'osc':
        return OSCController(
            model=model,
            kp_task=kwargs['kp_task'],
            kd_task=kwargs['kd_task'],
            kp_null=kwargs['kp_null'],
            kd_null=kwargs['kd_null'],
            q_ref=kwargs['q_ref'],
            max_joint_torque=kwargs['max_joint_torque'],
            max_wrench=kwargs['max_wrench'],
            mode=mode,
        )
    elif controller_type == 'diff_ik_optim':
        return DiffIKOptimController(
            model=model,
            kp_task=kwargs['kp_task'],
            kp_joint=kwargs['kp_joint'],
            kd_joint=kwargs['kd_joint'],
            dt=kwargs['dt'],
            task_weight=kwargs.get('task_weight', 1.0),
            posture_weight=kwargs.get('posture_weight', 1e-3),
            damping=kwargs.get('damping', 1e-6),
            velocity_limit_scale=kwargs.get('velocity_limit_scale', 1.0),
            q_ref=kwargs.get('q_ref'),
            config_limit_gain=kwargs.get('config_limit_gain', 0.5),
            max_torque=kwargs.get('max_torque'),
            solver=kwargs.get('solver', 'osqp'),
            mode=mode,
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}. Use 'diff_ik', 'diff_ik_optim', or 'osc'.")


__all__ = [
    'BaseController',
    'DiffIKController',
    'DiffIKOptimController',
    'OSCController',
    'create_controller',
]
