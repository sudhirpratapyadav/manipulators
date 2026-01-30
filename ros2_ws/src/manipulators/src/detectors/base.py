"""Detector base protocol and shared data types."""

from dataclasses import dataclass, field
from typing import List, Protocol

import numpy as np


@dataclass
class Detection:
    """Single detected object."""
    label: str                        # object class / name
    center_px: tuple                  # (x, y) pixel coordinates in color image
    confidence: float = 1.0           # detection confidence [0, 1]
    contour: np.ndarray = None        # optional contour points
    mask: np.ndarray = None           # optional binary mask


class DetectorBase(Protocol):
    """
    Interface that all detectors must implement.

    To add a new detector (e.g. YOLO):
      1. Create src/detectors/yolo_detector.py
      2. Implement the detect() method
      3. Register it in src/detectors/__init__.py DETECTORS dict
    """

    def detect(self, color_image: np.ndarray) -> List[Detection]:
        """
        Run detection on a color image (BGR, uint8).

        Returns list of Detection objects (may be empty).
        """
        ...
