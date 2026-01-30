#!/usr/bin/env python3
"""
Capture calibration images from a RealSense camera via ROS2 topics.

Usage:
  ros2 run manipulators capture_images -- --output-dir ./calibration_images --count 20

Press SPACE to capture an image, ESC to quit early.
Requires realsense2_camera node to be running.
"""

import argparse
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageCapture(Node):
    def __init__(self, output_dir: str, count: int, topic: str):
        super().__init__("capture_images")
        self.bridge = CvBridge()
        self.output_dir = output_dir
        self.target_count = count
        self.captured = 0
        self._latest_frame = None

        os.makedirs(output_dir, exist_ok=True)

        self.create_subscription(Image, topic, self._on_image, 1)
        self.get_logger().info(
            f"Subscribing to {topic}. Press SPACE to capture, ESC to quit."
        )

    def _on_image(self, msg: Image):
        self._latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def run(self):
        while rclpy.ok() and self.captured < self.target_count:
            rclpy.spin_once(self, timeout_sec=0.03)

            if self._latest_frame is None:
                continue

            display = self._latest_frame.copy()
            cv2.putText(
                display,
                f"Captured: {self.captured}/{self.target_count}  [SPACE=capture, ESC=quit]",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Capture Calibration Images", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                filename = os.path.join(
                    self.output_dir, f"calib_{self.captured:03d}.png"
                )
                cv2.imwrite(filename, self._latest_frame)
                self.captured += 1
                self.get_logger().info(
                    f"Saved {filename} ({self.captured}/{self.target_count})"
                )

        cv2.destroyAllWindows()
        self.get_logger().info(f"Done. Captured {self.captured} images in {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Capture calibration images")
    parser.add_argument("--output-dir", default="./calibration_images")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--topic", default="/camera/camera/color/image_raw")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ImageCapture(args.output_dir, args.count, args.topic)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
