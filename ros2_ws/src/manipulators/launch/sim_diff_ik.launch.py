"""
Launch control node + keyboard teleop in simulation mode.

Start the MuJoCo sim server first in a separate terminal:
    ros2 run manipulators mujoco_sim

Then launch this:
    ros2 launch manipulators sim_diff_ik.launch.py
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('manipulators')
    config_file = os.path.join(pkg_share, 'config', 'diff_ik_teleop.yaml')

    return LaunchDescription([
        Node(
            package='manipulators',
            executable='control_node',
            name='control_node',
            output='screen',
            parameters=[
                config_file,
                {
                    'use_sim': True,
                    'robot_ip': '127.0.0.1',
                },
            ],
        ),

        Node(
            package='manipulators',
            executable='keyboard_teleop',
            name='keyboard_teleop',
            output='screen',
            prefix='xterm -e',
            parameters=[config_file],
        ),
    ])
