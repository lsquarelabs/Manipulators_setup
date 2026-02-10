"""Launch control node with OSC controller and keyboard teleop."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('manipulators')
    config_file = os.path.join(pkg_share, 'config', 'osc_teleop.yaml')

    realsense_share = get_package_share_directory('realsense2_camera')
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_share, 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'align_depth.enable': 'true',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
        }.items(),
    )

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
        realsense_launch,
    ])

