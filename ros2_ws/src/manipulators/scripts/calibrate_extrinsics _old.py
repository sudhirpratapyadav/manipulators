#!/usr/bin/env python3
"""
Camera-to-robot extrinsic calibration.

Places a chessboard at a known position relative to the robot base,
detects it in the camera image, and computes the camera-to-robot transform.

Usage:
  ros2 run manipulators calibrate_extrinsics -- \
    --camera-yaml ./config/camera.yaml \
    --board-size 5 3 \
    --square-size 45.0 \
    --board-to-robot-xyz -0.11 -0.01 0.0 \
    --board-to-robot-rpy 0.0 180.0 0.0

The script:
  1. Subscribes to the color image topic
  2. Detects the chessboard and solves PnP for camera-to-board transform
  3. Combines with the known board-to-robot transform
  4. Saves the camera-to-robot transform to the camera YAML file
"""

import argparse
import sys

import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def rotation_matrix_from_rpy(roll_deg, pitch_deg, yaw_deg):
    """RPY (degrees) to 3x3 rotation matrix."""
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)

    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


class ExtrinsicCalibrator(Node):
    def __init__(self, cam_mtx, cam_dist, board_size, square_size_m,
                 T_board_robot, output_file, topic):
        super().__init__("calibrate_extrinsics")
        self.bridge = CvBridge()
        self.cam_mtx = cam_mtx
        self.cam_dist = cam_dist
        self.board_size = board_size
        self.T_board_robot = T_board_robot
        self.output_file = output_file
        self._latest_frame = None
        self._done = False

        self.square_size_m = square_size_m

        # Chessboard object points
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_m

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.create_subscription(Image, topic, self._on_image, 1)
        self.get_logger().info(
            f"Subscribing to {topic}. Point camera at chessboard, press SPACE to calibrate."
        )

    def _on_image(self, msg: Image):
        self._latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def run(self):
        while rclpy.ok() and not self._done:
            rclpy.spin_once(self, timeout_sec=0.03)

            if self._latest_frame is None:
                continue

            frame = self._latest_frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)

            if ret:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                cv2.drawChessboardCorners(frame, self.board_size, corners, ret)

                # Draw coordinate frame at chessboard origin
                ok, rvec, tvec = cv2.solvePnP(
                    self.objp, corners, self.cam_mtx, self.cam_dist
                )
                if ok:
                    axis_length = 4 * self.square_size_m
                    cv2.drawFrameAxes(frame, self.cam_mtx, self.cam_dist,
                                      rvec, tvec, axis_length)

                cv2.putText(frame, "Chessboard found! Press SPACE to calibrate",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No chessboard detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Extrinsic Calibration", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32 and ret:  # SPACE
                self._compute_and_save(corners)
                self._done = True

        cv2.destroyAllWindows()

    def _compute_and_save(self, corners):
        """Compute camera-to-robot transform and save."""
        # Solve PnP: camera-to-board
        ret, rvec_cb, tvec_cb = cv2.solvePnP(
            self.objp, corners, self.cam_mtx, self.cam_dist
        )
        if not ret:
            self.get_logger().error("solvePnP failed")
            return

        R_cam_board, _ = cv2.Rodrigues(rvec_cb)
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = R_cam_board
        T_cam_board[:3, 3] = tvec_cb.flatten()

        # Camera-to-robot = camera-to-board * board-to-robot
        T_cam_robot = T_cam_board @ self.T_board_robot

        # Convert to rvec/tvec for storage
        rvec_cr, _ = cv2.Rodrigues(T_cam_robot[:3, :3])
        tvec_cr = T_cam_robot[:3, 3]

        self.get_logger().info(f"Camera-to-robot transform:")
        self.get_logger().info(f"  rvec: {rvec_cr.flatten()}")
        self.get_logger().info(f"  tvec: {tvec_cr.flatten()}")

        # Save to YAML (merge with existing)
        data = {}
        try:
            with open(self.output_file, "r") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            pass

        data["extrinsics"] = {
            "rvec": rvec_cr.flatten().tolist(),
            "tvec": tvec_cr.flatten().tolist(),
            "T_cam_robot": T_cam_robot.flatten().tolist(),
        }

        with open(self.output_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        self.get_logger().info(f"Saved extrinsics to {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Camera-to-robot extrinsic calibration")
    parser.add_argument("--camera-yaml", required=True,
                        help="Camera YAML file (intrinsics must already exist)")
    parser.add_argument("--board-size", nargs=2, type=int, default=[7, 4],
                        help="Chessboard inner corners (cols rows)")
    parser.add_argument("--square-size", type=float, default=35.0,
                        help="Chessboard square size in mm")
    parser.add_argument("--board-to-robot-xyz", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help="Board origin position in robot frame (meters)")
    parser.add_argument("--board-to-robot-rpy", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help="Board orientation in robot frame (degrees, roll pitch yaw)")
    parser.add_argument("--topic", default="/camera/camera/color/image_raw")
    args, ros_args = parser.parse_known_args()

    # Load intrinsics
    try:
        with open(args.camera_yaml, "r") as f:
            cam_data = yaml.safe_load(f)
        intr = cam_data["intrinsics"]
        cam_mtx = np.array(intr["camera_matrix"], dtype=float).reshape(3, 3)
        cam_dist = np.array(intr["dist_coeffs"], dtype=float)
    except Exception as e:
        print(f"Error loading intrinsics from {args.camera_yaml}: {e}")
        print("Run calibrate_intrinsics.py first.")
        sys.exit(1)

    # Build board-to-robot transform
    T_board_robot = np.eye(4)
    T_board_robot[:3, :3] = rotation_matrix_from_rpy(*args.board_to_robot_rpy)
    T_board_robot[:3, 3] = args.board_to_robot_xyz

    square_size_m = args.square_size / 1000.0
    board_size = tuple(args.board_size)

    rclpy.init(args=ros_args)
    node = ExtrinsicCalibrator(
        cam_mtx, cam_dist, board_size, square_size_m,
        T_board_robot, args.camera_yaml, args.topic
    )
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
