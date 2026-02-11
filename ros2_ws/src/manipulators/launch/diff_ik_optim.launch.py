"""Launch control node with QP-based diff-IK controller and keyboard teleop."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('manipulators')
    config_file = os.path.join(pkg_share, 'config', 'diff_ik_optim_teleop.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_ip',
            default_value='192.168.1.10',
            description='Kinova robot IP address',
        ),

        Node(
            package='manipulators',
            executable='control_node',
            name='control_node',
            output='screen',
            parameters=[
                config_file,
                {'robot_ip': LaunchConfiguration('robot_ip')},
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
