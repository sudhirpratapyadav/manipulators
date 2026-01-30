"""
HSV color-based object detector.

Applies BGR and HSV color filtering, finds contours,
returns the largest contour centroid as a detection.
"""

from typing import List

import cv2
import numpy as np

from .base import Detection, DetectorBase


class ColorDetector:
    """
    Detects objects by color using HSV thresholding.

    Parameters are passed from config/detection.yaml.
    """

    def __init__(
        self,
        hsv_low: list = None,
        hsv_high: list = None,
        bgr_low: list = None,
        bgr_high: list = None,
        crop: list = None,
        min_area: int = 100,
        blur_kernel: int = 5,
        label: str = "object",
    ):
        self.hsv_low = np.array(hsv_low or [0, 120, 70])
        self.hsv_high = np.array(hsv_high or [10, 255, 255])
        self.bgr_low = np.array(bgr_low or [0, 0, 0])
        self.bgr_high = np.array(bgr_high or [255, 255, 255])
        # crop: [x1, y1, x2, y2] or None for full image
        self.crop = crop
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.label = label

    def detect(self, color_image: np.ndarray) -> List[Detection]:
        """Detect colored objects in a BGR image."""
        h_img, w_img = color_image.shape[:2]

        # Crop region
        x1, y1, x2, y2 = 0, 0, w_img, h_img
        if self.crop:
            x1, y1, x2, y2 = self.crop

        roi = color_image[y1:y2, x1:x2]

        # Blur
        if self.blur_kernel > 0:
            roi = cv2.GaussianBlur(roi, (self.blur_kernel, self.blur_kernel), 0)

        # BGR mask
        bgr_mask = cv2.inRange(roi, self.bgr_low, self.bgr_high)
        bgr_filtered = cv2.bitwise_and(roi, roi, mask=bgr_mask)

        # HSV mask
        hsv = cv2.cvtColor(bgr_filtered, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        # Find contours
        contours, _ = cv2.findContours(
            hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            # Centroid in full image coordinates
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1

            detections.append(Detection(
                label=self.label,
                center_px=(cx, cy),
                confidence=min(1.0, area / 5000.0),
                contour=cnt,
            ))

        # Sort by area (largest first)
        detections.sort(
            key=lambda d: cv2.contourArea(d.contour) if d.contour is not None else 0,
            reverse=True,
        )

        return detections
