"""
Detector tuner node.

Same detection pipeline as the object detection node, but with live
HSV trackbars for calibrating color thresholds.  Shows color, depth,
and HSV mask side-by-side.  Press SPACE to save tuned values to YAML,
ESC to quit.
"""

import os

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from .detectors.color_detector import ColorDetector


_WIN = "Detector Tuner"


class DetectorTunerNode(Node):
    def __init__(self):
        super().__init__("detector_tuner_node")

        # -- Parameters --
        self.declare_parameter("camera_calibration_file", "")
        self.declare_parameter("output_file", "detector_tuning.yaml")
        self.declare_parameter("hsv_low", [0, 120, 70])
        self.declare_parameter("hsv_high", [10, 255, 255])
        self.declare_parameter("bgr_low", [0, 0, 0])
        self.declare_parameter("bgr_high", [255, 255, 255])
        self.declare_parameter("crop", [0, 0, 640, 480])
        self.declare_parameter("min_area", 100)
        self.declare_parameter("label", "object")
        self.declare_parameter("depth_range", [0.1, 2.0])
        self.declare_parameter("depth_patch_radius", 3)

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

        self.output_file = self.get_parameter("output_file").value
        self.depth_range = self.get_parameter("depth_range").value
        self.depth_patch_radius = self.get_parameter("depth_patch_radius").value

        hsv_low = list(self.get_parameter("hsv_low").value)
        hsv_high = list(self.get_parameter("hsv_high").value)

        # -- Load camera extrinsics --
        self.T_robot_camera = np.eye(4)
        calib_file = self.get_parameter("camera_calibration_file").value
        if calib_file:
            self._load_extrinsics(calib_file)

        # -- Create detector --
        self.detector = ColorDetector(
            hsv_low=hsv_low,
            hsv_high=hsv_high,
            bgr_low=self.get_parameter("bgr_low").value,
            bgr_high=self.get_parameter("bgr_high").value,
            crop=self.get_parameter("crop").value,
            min_area=self.get_parameter("min_area").value,
            label=self.get_parameter("label").value,
        )

        # -- State --
        self.bridge = CvBridge()
        self._latest_color = None
        self._latest_depth = None
        self.cam_mtx = None
        self.cam_dist = None

        # -- Subscriptions --
        color_topic = self.get_parameter("color_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("camera_info_topic").value

        self.create_subscription(Image, color_topic, self._on_color, 1)
        self.create_subscription(Image, depth_topic, self._on_depth, 1)
        self.create_subscription(CameraInfo, info_topic, self._on_camera_info, 1)

        # -- GUI --
        cv2.namedWindow(_WIN, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("H low",  _WIN, hsv_low[0],  179, lambda _: None)
        cv2.createTrackbar("S low",  _WIN, hsv_low[1],  255, lambda _: None)
        cv2.createTrackbar("V low",  _WIN, hsv_low[2],  255, lambda _: None)
        cv2.createTrackbar("H high", _WIN, hsv_high[0], 179, lambda _: None)
        cv2.createTrackbar("S high", _WIN, hsv_high[1], 255, lambda _: None)
        cv2.createTrackbar("V high", _WIN, hsv_high[2], 255, lambda _: None)

        self.get_logger().info(
            f"Detector tuner ready. Adjust HSV trackbars. "
            f"SPACE=save to {self.output_file}, ESC=quit."
        )

    # ------------------------------------------------------------------

    def _load_extrinsics(self, filepath: str):
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            if "extrinsics" in data:
                ext = data["extrinsics"]
                rvec = np.array(ext["rvec"], dtype=float).reshape(3, 1)
                tvec = np.array(ext["tvec"], dtype=float).reshape(3, 1)
                R, _ = cv2.Rodrigues(rvec)
                self.T_robot_camera = np.eye(4)
                self.T_robot_camera[:3, :3] = R
                self.T_robot_camera[:3, 3] = tvec.flatten()
                self.get_logger().info(f"Loaded extrinsics from {filepath}")
        except Exception as e:
            self.get_logger().error(f"Failed to load extrinsics: {e}")

    def _on_color(self, msg: Image):
        self._latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def _on_depth(self, msg: Image):
        self._latest_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def _on_camera_info(self, msg: CameraInfo):
        if self.cam_mtx is not None:
            return
        self.cam_mtx = np.array(msg.k, dtype=float).reshape(3, 3)
        self.cam_dist = np.array(msg.d, dtype=float)
        self.get_logger().info("Camera intrinsics received.")

    # ------------------------------------------------------------------

    def _read_trackbars(self):
        """Read current HSV values from trackbars."""
        h_lo = cv2.getTrackbarPos("H low",  _WIN)
        s_lo = cv2.getTrackbarPos("S low",  _WIN)
        v_lo = cv2.getTrackbarPos("V low",  _WIN)
        h_hi = cv2.getTrackbarPos("H high", _WIN)
        s_hi = cv2.getTrackbarPos("S high", _WIN)
        v_hi = cv2.getTrackbarPos("V high", _WIN)
        return [h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi]

    def _save(self, hsv_low, hsv_high):
        """Save current detector config to YAML."""
        data = {
            "hsv_low": hsv_low,
            "hsv_high": hsv_high,
        }
        with open(self.output_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        self.get_logger().info(f"Saved detector tuning to {self.output_file}")

    # ------------------------------------------------------------------

    def run(self):
        saved = False
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.03)

            if self._latest_color is None or self._latest_depth is None:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                continue

            color = self._latest_color
            depth = self._latest_depth

            # Update detector HSV from trackbars
            hsv_low, hsv_high = self._read_trackbars()
            self.detector.hsv_low = np.array(hsv_low)
            self.detector.hsv_high = np.array(hsv_high)

            # Prepare visualization images
            vis_color = color.copy()
            depth_clipped = np.clip(
                depth, self.depth_range[0] * 1000, self.depth_range[1] * 1000
            )
            depth_norm = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX)
            depth_u8 = depth_norm.astype(np.uint8)
            vis_depth = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

            # Build HSV mask for display
            hsv_img = cv2.cvtColor(
                cv2.GaussianBlur(color, (5, 5), 0), cv2.COLOR_BGR2HSV
            )
            mask = cv2.inRange(hsv_img, np.array(hsv_low), np.array(hsv_high))
            vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Run detector
            detections = self.detector.detect(color)

            # Crop offset for bounding box correction
            crop = self.detector.crop
            ox, oy = 0, 0
            if crop:
                ox, oy = crop[0], crop[1]

            pt_robot = None
            if detections:
                det = detections[0]
                cx, cy = det.center_px

                # Bounding box from contour (offset by crop origin)
                if det.contour is not None:
                    bx, by, bw, bh = cv2.boundingRect(det.contour)
                    bx += ox
                    by += oy
                    cv2.rectangle(vis_color, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.rectangle(vis_depth, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.rectangle(vis_mask, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

                cv2.circle(vis_color, (cx, cy), 5, (0, 255, 255), -1)
                cv2.circle(vis_depth, (cx, cy), 5, (0, 255, 255), -1)

                # 3D projection if we have intrinsics
                if self.cam_mtx is not None:
                    r = self.depth_patch_radius
                    h, w = depth.shape[:2]
                    y1 = max(0, cy - r)
                    y2 = min(h, cy + r + 1)
                    x1 = max(0, cx - r)
                    x2 = min(w, cx + r + 1)
                    patch = depth[y1:y2, x1:x2].astype(float)
                    depth_m = np.median(patch) * 0.001

                    if self.depth_range[0] <= depth_m <= self.depth_range[1]:
                        fx, fy = self.cam_mtx[0, 0], self.cam_mtx[1, 1]
                        ppx, ppy = self.cam_mtx[0, 2], self.cam_mtx[1, 2]
                        x_cam = (cx - ppx) * depth_m / fx
                        y_cam = (cy - ppy) * depth_m / fy
                        pt_cam = np.array([x_cam, y_cam, depth_m, 1.0])
                        pt_robot = self.T_robot_camera @ pt_cam

                if pt_robot is not None:
                    label = (f"{det.label} ({pt_robot[0]:.3f}, "
                             f"{pt_robot[1]:.3f}, {pt_robot[2]:.3f})")
                else:
                    label = f"{det.label}"
                cv2.putText(vis_color, label, (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(vis_depth, label, (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Panel labels
            cv2.putText(vis_color, "Color", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_depth, "Depth", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_mask, "HSV Mask", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # HSV values text on mask panel
            cv2.putText(vis_mask,
                        f"Low:  [{hsv_low[0]}, {hsv_low[1]}, {hsv_low[2]}]",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_mask,
                        f"High: [{hsv_high[0]}, {hsv_high[1]}, {hsv_high[2]}]",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            status = "SPACE=save  ESC=quit"
            cv2.putText(vis_mask, status,
                        (10, vis_mask.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            canvas = np.hstack([vis_color, vis_depth, vis_mask])
            cv2.imshow(_WIN, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                self._save(hsv_low, hsv_high)
                saved = True

        cv2.destroyAllWindows()
        if saved:
            self.get_logger().info("Tuning saved. Exiting.")
        else:
            self.get_logger().info("Exited without saving.")


def main(args=None):
    rclpy.init(args=args)
    node = DetectorTunerNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
