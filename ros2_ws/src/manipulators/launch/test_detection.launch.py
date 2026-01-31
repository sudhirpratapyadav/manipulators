"""
Launch file for testing object detection.

Launches:
    - object_detection_node: Object detection and 3D localization

This is a minimal launch file for testing and tuning the object detection
without running the full robot control system.

Usage:
    ros2 launch manipulators test_detection.launch.py

To visualize detection output:
    ros2 run rqt_image_view rqt_image_view /detection_image

To check detected positions:
    ros2 topic echo /detected_object_point

Note: Requires RealSense camera to be running:
    ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('manipulators')

    # Config files
    detection_config = os.path.join(pkg_share, 'config', 'detection.yaml')
    camera_config = os.path.join(pkg_share, 'config', 'camera.yaml')

    # Object detection node
    # camera.yaml is NOT a ROS2 params file — the node loads it manually
    # via yaml.safe_load(), so pass the path as a parameter instead.
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

    return LaunchDescription([
        detection_node,
    ])
