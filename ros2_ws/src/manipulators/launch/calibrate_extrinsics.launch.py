"""
Launch file for interactive extrinsic calibration.

Launches:
    - control_node  (starts robot at calib_pose)
    - keyboard_teleop
    - RealSense camera
    - extrinsic_calibration_node (OpenCV GUI with trackbars)

Usage:
    ros2 launch manipulators calibrate_extrinsics.launch.py
    ros2 launch manipulators calibrate_extrinsics.launch.py robot_ip:=192.168.1.10
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("manipulators")
    kinova_config = os.path.join(pkg_share, "config", "kinova_gen3.yaml")
    camera_config = os.path.join(pkg_share, "config", "camera.yaml")

    # calib_pose from poses.yaml converted to Kinova degrees (0-360)
    calib_pose_deg = [67.94, 37.13, 227.14, 287.06, 336.63, 277.43, 216.55]

    # ── Launch arguments ─────────────────────────────────────────────
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.1.10",
        description="Kinova robot IP address",
    )

    # ── 1. Control node (goes to calib_pose on startup) ──────────────
    control_node = Node(
        package="manipulators",
        executable="control_node",
        name="control_node",
        output="screen",
        parameters=[
            kinova_config,
            {
                "robot_ip": LaunchConfiguration("robot_ip"),
                "initial_pose_deg": calib_pose_deg,
            },
        ],
    )

    # ── 2. Keyboard teleop ───────────────────────────────────────────
    keyboard_teleop = Node(
        package="manipulators",
        executable="keyboard_teleop",
        name="keyboard_teleop",
        output="screen",
        prefix="xterm -e",
        parameters=[kinova_config],
    )

    # ── 3. RealSense camera ──────────────────────────────────────────
    realsense_share = get_package_share_directory("realsense2_camera")
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_share, "launch", "rs_launch.py")
        ),
        launch_arguments={
            "align_depth.enable": "true",
            "rgb_camera.color_profile": "640x480x30",
            "depth_module.depth_profile": "640x480x30",
        }.items(),
    )

    # ── 4. Extrinsic calibration node ────────────────────────────────
    calibration_node = Node(
        package="manipulators",
        executable="extrinsic_calibration_node",
        name="extrinsic_calibration_node",
        output="screen",
        parameters=[
            {
                "camera_yaml": camera_config,
                "board_size": [5, 3],
                "square_size_mm": 45.0,
                "image_topic": "/camera/camera/color/image_raw",
                "default_xyz": [0.0, 0.0, 0.0],
                "default_rpy": [0.0, 180.0, 0.0],
            }
        ],
    )

    return LaunchDescription([
        robot_ip_arg,
        control_node,
        keyboard_teleop,
        realsense_launch,
        calibration_node,
    ])
