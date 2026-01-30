from .base import DetectorBase, Detection
from .color_detector import ColorDetector

DETECTORS = {
    "color": ColorDetector,
}


def create_detector(detector_type: str, **kwargs) -> DetectorBase:
    """Factory: create a detector by name."""
    if detector_type not in DETECTORS:
        raise ValueError(
            f"Unknown detector '{detector_type}'. Available: {list(DETECTORS.keys())}"
        )
    return DETECTORS[detector_type](**kwargs)
