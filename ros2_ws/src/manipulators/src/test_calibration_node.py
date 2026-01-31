"""
Test calibration node.

Reads saved camera-to-robot extrinsics from YAML and overlays
robot-base and end-effector coordinate axes on the live camera feed.
No chessboard detection — uses the stored T_cam_robot directly.

Press ESC to quit.
"""

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge

from .utility import quat_to_matrix


_WIN = "Test Calibration"


def _project_origin(T_cam_frame, cam_mtx, cam_dist):
    """Return the 2-D pixel of a frame's origin (for label placement)."""
    pt = T_cam_frame[:3, 3].reshape(3, 1).astype(np.float64)
    px, _ = cv2.projectPoints(pt, np.zeros(3), np.zeros(3), cam_mtx, cam_dist)
    return int(px[0, 0, 0]), int(px[0, 0, 1])


def _draw_labeled_axes(frame, cam_mtx, cam_dist, T_cam_frame,
                       length, label, color):
    """Draw coordinate axes + a text label for a given frame."""
    rvec, _ = cv2.Rodrigues(T_cam_frame[:3, :3])
    tvec = T_cam_frame[:3, 3].astype(np.float64)
    cv2.drawFrameAxes(frame, cam_mtx, cam_dist, rvec, tvec, length)
    try:
        px, py = _project_origin(T_cam_frame, cam_mtx, cam_dist)
        cv2.putText(frame, label, (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except Exception:
        pass


class TestCalibrationNode(Node):
    def __init__(self):
        super().__init__("test_calibration_node")

        # -- Parameters --
        self.declare_parameter("camera_yaml", "")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")

        camera_yaml = self.get_parameter("camera_yaml").value
        image_topic = self.get_parameter("image_topic").value

        # -- Load intrinsics + extrinsics from YAML --
        try:
            with open(camera_yaml, "r") as f:
                cam_data = yaml.safe_load(f)

            intr = cam_data["intrinsics"]
            self.cam_mtx = np.array(intr["camera_matrix"], dtype=np.float64).reshape(3, 3)
            self.cam_dist = np.array(intr["dist_coeffs"], dtype=np.float64)

            ext = cam_data["extrinsics"]
            self.T_cam_robot = np.array(ext["T_cam_robot"], dtype=np.float64).reshape(4, 4)
            self.get_logger().info(f"Loaded intrinsics + extrinsics from {camera_yaml}")
        except Exception as e:
            self.get_logger().fatal(f"Cannot load camera YAML {camera_yaml}: {e}")
            raise SystemExit(1)

        # -- State --
        self.bridge = CvBridge()
        self._latest_frame = None
        self._ee_pos = None
        self._ee_rot = None
        self._object_pos = None  # (3,) in robot frame

        # -- Subscriptions --
        self.create_subscription(Image, image_topic, self._on_image, 1)
        self.create_subscription(PoseStamped, "ee_pose", self._on_ee_pose, 1)
        self.create_subscription(
            PointStamped, "detected_object_point", self._on_object, 1
        )

        self.get_logger().info(
            f"Subscribing to {image_topic}, /ee_pose, "
            "/detected_object_point. Press ESC to quit."
        )

        cv2.namedWindow(_WIN)

    def _on_image(self, msg: Image):
        self._latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def _on_ee_pose(self, msg: PoseStamped):
        self._ee_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        quat_xyzw = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        self._ee_rot = quat_to_matrix(quat_xyzw)

    def _on_object(self, msg: PointStamped):
        self._object_pos = np.array([
            msg.point.x, msg.point.y, msg.point.z,
        ])

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.03)

            if self._latest_frame is None:
                continue

            frame = self._latest_frame.copy()

            # 1. Robot base axes
            _draw_labeled_axes(
                frame, self.cam_mtx, self.cam_dist,
                self.T_cam_robot,
                length=0.1,
                label="Robot Base", color=(255, 180, 0),
            )

            # 2. End-effector axes
            if self._ee_pos is not None:
                T_robot_ee = np.eye(4)
                T_robot_ee[:3, :3] = self._ee_rot
                T_robot_ee[:3, 3] = self._ee_pos
                T_cam_ee = self.T_cam_robot @ T_robot_ee
                _draw_labeled_axes(
                    frame, self.cam_mtx, self.cam_dist,
                    T_cam_ee,
                    length=0.05,
                    label="EE", color=(0, 0, 255),
                )

            # 3. Detected object axes (identity rotation in robot frame)
            if self._object_pos is not None:
                T_robot_obj = np.eye(4)
                T_robot_obj[:3, 3] = self._object_pos
                T_cam_obj = self.T_cam_robot @ T_robot_obj
                _draw_labeled_axes(
                    frame, self.cam_mtx, self.cam_dist,
                    T_cam_obj,
                    length=0.03,
                    label="Object", color=(0, 255, 255),
                )

            cv2.putText(
                frame, "ESC=quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

            cv2.imshow(_WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cv2.destroyAllWindows()
        self.get_logger().info("Test calibration node exiting.")


def main(args=None):
    rclpy.init(args=args)
    node = TestCalibrationNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
