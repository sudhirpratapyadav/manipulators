"""
Object detection node.

Subscribes to RealSense camera topics (via realsense2_camera driver),
runs a pluggable detector, projects 2D detections to 3D using depth
and a calibrated camera-to-robot extrinsic transform, then publishes
the object pose and bounding box.

No GUI — purely detection + publish.
"""

import numpy as np
import cv2
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge

from .detectors import create_detector


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__("object_detection_node")

        # -- Parameters --
        self.declare_parameter("camera_calibration_file", "")
        self.declare_parameter("detector_type", "color")
        self.declare_parameter("hsv_low", [0, 120, 70])
        self.declare_parameter("hsv_high", [10, 255, 255])
        self.declare_parameter("bgr_low", [0, 0, 0])
        self.declare_parameter("bgr_high", [255, 255, 255])
        self.declare_parameter("crop", [0, 0, 640, 480])
        self.declare_parameter("min_area", 100)
        self.declare_parameter("label", "object")
        self.declare_parameter("depth_range", [0.1, 2.0])
        self.declare_parameter("depth_patch_radius", 3)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("object_offset_robot", [0.0, 0.0, 0.0])

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

        self.depth_range = self.get_parameter("depth_range").value
        self.depth_patch_radius = self.get_parameter("depth_patch_radius").value
        self.object_offset_robot = np.array(
            self.get_parameter("object_offset_robot").value, dtype=float
        )

        # -- Load camera extrinsics (camera-to-robot transform) --
        calib_file = self.get_parameter("camera_calibration_file").value
        self.T_robot_camera = np.eye(4)
        if calib_file:
            self._load_extrinsics(calib_file)
        else:
            self.get_logger().warn("No camera calibration file. Using identity transform.")

        # -- Camera intrinsics (from CameraInfo topic) --
        self.cam_mtx = None
        self.cam_dist = None

        # -- Create detector --
        detector_type = self.get_parameter("detector_type").value
        detector_kwargs = {
            "hsv_low": self.get_parameter("hsv_low").value,
            "hsv_high": self.get_parameter("hsv_high").value,
            "bgr_low": self.get_parameter("bgr_low").value,
            "bgr_high": self.get_parameter("bgr_high").value,
            "crop": self.get_parameter("crop").value,
            "min_area": self.get_parameter("min_area").value,
            "label": self.get_parameter("label").value,
        }
        self.detector = create_detector(detector_type, **detector_kwargs)
        self.get_logger().info(f"Detector: {detector_type}")

        # -- CV Bridge --
        self.bridge = CvBridge()

        # -- State --
        self._latest_color = None
        self._latest_depth = None

        # -- Publishers --
        self.object_pose_pub = self.create_publisher(
            PointStamped, "detected_object_point", 10
        )
        self.bbox_pub = self.create_publisher(
            Int32MultiArray, "detection_bbox", 10
        )

        # -- Subscribers --
        color_topic = self.get_parameter("color_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("camera_info_topic").value

        self.create_subscription(Image, color_topic, self._on_color, 1)
        self.create_subscription(Image, depth_topic, self._on_depth, 1)
        self.create_subscription(CameraInfo, info_topic, self._on_camera_info, 1)

        # -- Timer for detection loop --
        rate = self.get_parameter("publish_rate").value
        self.create_timer(1.0 / rate, self._detect_callback)

        self.get_logger().info("Object detection node ready.")

    # ------------------------------------------------------------------
    # Extrinsics loading
    # ------------------------------------------------------------------

    def _load_extrinsics(self, filepath: str):
        """Load camera-to-robot 4x4 transform from YAML."""
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)

            if "extrinsics" in data:
                ext = data["extrinsics"]
                # rvec/tvec encode T_cam_robot (robot→camera).
                # Invert to get T_robot_camera (camera→robot).
                rvec = np.array(ext["rvec"], dtype=float).reshape(3, 1)
                tvec = np.array(ext["tvec"], dtype=float).reshape(3, 1)
                R, _ = cv2.Rodrigues(rvec)
                T_cam_robot = np.eye(4)
                T_cam_robot[:3, :3] = R
                T_cam_robot[:3, 3] = tvec.flatten()
                self.T_robot_camera = np.linalg.inv(T_cam_robot)
                self.get_logger().info(f"Loaded extrinsics from {filepath}")
            else:
                self.get_logger().warn(f"No 'extrinsics' key in {filepath}")
        except Exception as e:
            self.get_logger().error(f"Failed to load extrinsics: {e}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_color(self, msg: Image):
        self._latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def _on_depth(self, msg: Image):
        self._latest_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def _on_camera_info(self, msg: CameraInfo):
        if self.cam_mtx is not None:
            return
        self.cam_mtx = np.array(msg.k, dtype=float).reshape(3, 3)
        self.cam_dist = np.array(msg.d, dtype=float)
        self.get_logger().info("Camera intrinsics received from CameraInfo topic.")

    # ------------------------------------------------------------------
    # Detection loop
    # ------------------------------------------------------------------

    def _detect_callback(self):
        if self._latest_color is None or self._latest_depth is None:
            return
        if self.cam_mtx is None:
            return

        color = self._latest_color
        depth = self._latest_depth

        # Run detector
        detections = self.detector.detect(color)

        if not detections:
            return

        det = detections[0]
        cx, cy = det.center_px

        # Publish bounding box [bx, by, bw, bh, cx, cy]
        bbox_msg = Int32MultiArray()
        if det.contour is not None:
            bx, by, bw, bh = cv2.boundingRect(det.contour)
            bbox_msg.data = [bx, by, bw, bh, cx, cy]
        else:
            bbox_msg.data = [0, 0, 0, 0, cx, cy]
        self.bbox_pub.publish(bbox_msg)

        # Get depth at detection center (median over patch)
        r = self.depth_patch_radius
        h, w = depth.shape[:2]
        y1 = max(0, cy - r)
        y2 = min(h, cy + r + 1)
        x1 = max(0, cx - r)
        x2 = min(w, cx + r + 1)
        patch = depth[y1:y2, x1:x2].astype(float)

        depth_m = np.median(patch) * 0.001
        if depth_m < self.depth_range[0] or depth_m > self.depth_range[1]:
            return

        # Back-project to 3D in camera frame
        fx, fy = self.cam_mtx[0, 0], self.cam_mtx[1, 1]
        ppx, ppy = self.cam_mtx[0, 2], self.cam_mtx[1, 2]

        x_cam = (cx - ppx) * depth_m / fx
        y_cam = (cy - ppy) * depth_m / fy
        z_cam = depth_m

        # Transform to robot frame
        pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        pt_robot = self.T_robot_camera @ pt_cam

        # Apply robot-frame offset
        pt_robot[:3] += self.object_offset_robot

        # Publish 3D point
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.point.x = float(pt_robot[0])
        msg.point.y = float(pt_robot[1])
        msg.point.z = float(pt_robot[2])
        self.object_pose_pub.publish(msg)

    # ------------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
