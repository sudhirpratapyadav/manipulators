"""
Launch file for pick-and-place system.

Launches:
    - control_node: Low-level torque control (400 Hz)
    - RealSense camera
    - object_detection_node: Object detection and 3D localization
    - pick_place_policy: Reactive pick-place state machine

Usage:
    ros2 launch manipulators pick_place.launch.py
    ros2 launch manipulators pick_place.launch.py robot_ip:=192.168.1.10

To start a pick-place cycle:
    ros2 service call /pick_place/start std_srvs/srv/Trigger

To abort:
    ros2 service call /pick_place/abort std_srvs/srv/Trigger
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('manipulators')

    # Config files
    kinova_config = os.path.join(pkg_share, 'config', 'kinova_gen3.yaml')
    detection_config = os.path.join(pkg_share, 'config', 'detection.yaml')
    camera_config = os.path.join(pkg_share, 'config', 'camera.yaml')
    pick_place_config = os.path.join(pkg_share, 'config', 'pick_place.yaml')

    # Launch arguments
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.10',
        description='IP address of the Kinova robot',
    )

    # ── 1. Control node ──────────────────────────────────────────────
    control_node = Node(
        package='manipulators',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[
            kinova_config,
            {'robot_ip': LaunchConfiguration('robot_ip')},
        ],
    )

    # ── 2. RealSense camera ──────────────────────────────────────────
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

    # ── 3. Object detection node ─────────────────────────────────────
    detection_node = Node(
        package='manipulators',
        executable='object_detection_node',
        name='object_detection_node',
        output='screen',
        parameters=[
            detection_config,
            {'camera_calibration_file': camera_config},
        ],
    )

    # ── 4. Pick-place policy node (in separate xterm) ───────────────
    policy_node = Node(
        package='manipulators',
        executable='pick_place_policy',
        name='pick_place_policy',
        output='screen',
        prefix='xterm -e',
        parameters=[pick_place_config],
    )

    return LaunchDescription([
        robot_ip_arg,
        control_node,
        realsense_launch,
        detection_node,
        policy_node,
    ])
