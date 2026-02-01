"""
Visualization node.

Unified display combining RGB + depth (side-by-side) with:
  - Detection bounding box + center dot on both images
  - Robot base, end-effector, and object coordinate axes on color image

Subscribes to:
  - Camera color / depth images (from RealSense)
  - ee_pose (PoseStamped from control_node)
  - detected_object_point (PointStamped from object_detection_node)
  - detection_bbox (Int32MultiArray from object_detection_node)

Press ESC to quit.
"""

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge

from .utility import quat_to_matrix


_WIN = "Visualization"


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


class VisualizationNode(Node):
    def __init__(self):
        super().__init__("visualization_node")

        # -- Parameters --
        self.declare_parameter("camera_yaml", "")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic",
                               "/camera/camera/aligned_depth_to_color/image_raw")

        camera_yaml = self.get_parameter("camera_yaml").value
        image_topic = self.get_parameter("image_topic").value
        depth_topic = self.get_parameter("depth_topic").value

        # -- Load intrinsics + extrinsics from YAML --
        try:
            with open(camera_yaml, "r") as f:
                cam_data = yaml.safe_load(f)

            intr = cam_data["intrinsics"]
            self.cam_mtx = np.array(
                intr["camera_matrix"], dtype=np.float64
            ).reshape(3, 3)
            self.cam_dist = np.array(intr["dist_coeffs"], dtype=np.float64)

            ext = cam_data["extrinsics"]
            self.T_cam_robot = np.array(
                ext["T_cam_robot"], dtype=np.float64
            ).reshape(4, 4)
            self.get_logger().info(
                f"Loaded intrinsics + extrinsics from {camera_yaml}"
            )
        except Exception as e:
            self.get_logger().fatal(
                f"Cannot load camera YAML {camera_yaml}: {e}"
            )
            raise SystemExit(1)

        # -- State --
        self.bridge = CvBridge()
        self._latest_color = None
        self._latest_depth = None
        self._ee_pos = None
        self._ee_rot = None
        self._object_pos = None   # (3,) in robot frame
        self._bbox = None         # [bx, by, bw, bh, cx, cy]

        # -- Subscriptions --
        self.create_subscription(Image, image_topic, self._on_color, 1)
        self.create_subscription(Image, depth_topic, self._on_depth, 1)
        self.create_subscription(PoseStamped, "ee_pose", self._on_ee_pose, 1)
        self.create_subscription(
            PointStamped, "detected_object_point", self._on_object, 1
        )
        self.create_subscription(
            Int32MultiArray, "detection_bbox", self._on_bbox, 1
        )

        self.get_logger().info(
            f"Subscribing to {image_topic}, {depth_topic}, "
            "ee_pose, detected_object_point, detection_bbox. ESC to quit."
        )

        cv2.namedWindow(_WIN)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_color(self, msg: Image):
        self._latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def _on_depth(self, msg: Image):
        self._latest_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

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

    def _on_bbox(self, msg: Int32MultiArray):
        if len(msg.data) >= 6:
            self._bbox = list(msg.data)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_bbox(self, frame, bbox):
        """Draw bounding box rectangle and center dot."""
        bx, by, bw, bh, cx, cy = bbox
        if bw > 0 and bh > 0:
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                          (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.03)

            if self._latest_color is None:
                continue

            color = self._latest_color.copy()

            # --- Depth colorization ---
            if self._latest_depth is not None:
                depth_raw = self._latest_depth
                depth_clip = np.clip(depth_raw, 0, 4000)
                depth_norm = (depth_clip / 4000.0 * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            else:
                depth_color = np.zeros_like(color)

            # --- Draw bounding box on both ---
            if self._bbox is not None:
                self._draw_bbox(color, self._bbox)
                self._draw_bbox(depth_color, self._bbox)

            # --- Draw axes on color image ---
            # 1. Robot base
            _draw_labeled_axes(
                color, self.cam_mtx, self.cam_dist,
                self.T_cam_robot,
                length=0.1,
                label="Robot Base", color=(255, 180, 0),
            )

            # 2. End-effector
            if self._ee_pos is not None:
                T_robot_ee = np.eye(4)
                T_robot_ee[:3, :3] = self._ee_rot
                T_robot_ee[:3, 3] = self._ee_pos
                T_cam_ee = self.T_cam_robot @ T_robot_ee
                _draw_labeled_axes(
                    color, self.cam_mtx, self.cam_dist,
                    T_cam_ee,
                    length=0.05,
                    label="EE", color=(0, 0, 255),
                )

            # 3. Detected object (identity rotation in robot frame)
            if self._object_pos is not None:
                T_robot_obj = np.eye(4)
                T_robot_obj[:3, 3] = self._object_pos
                T_cam_obj = self.T_cam_robot @ T_robot_obj
                _draw_labeled_axes(
                    color, self.cam_mtx, self.cam_dist,
                    T_cam_obj,
                    length=0.03,
                    label="Object", color=(0, 255, 255),
                )

            # --- Side-by-side ---
            combined = np.hstack([color, depth_color])

            cv2.putText(
                combined, "ESC=quit",
                (10, combined.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

            cv2.imshow(_WIN, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cv2.destroyAllWindows()
        self.get_logger().info("Visualization node exiting.")


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
