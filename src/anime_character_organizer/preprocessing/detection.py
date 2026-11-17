"""
Anime character head and person detection wrappers and selection heuristics.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .geometry import Box, box_area, clamp_box


def detection_to_dict(det: Tuple[Any, Any, float]) -> Dict[str, Any]:
    """Convert detection tuple (box, label, score) to JSON-serializable dictionary."""
    box, label, score = det
    return {
        "box": [int(v) for v in box],
        "label": str(label),
        "score": float(score),
    }


def choose_best_detection(
    detections: List[Tuple[Box, Any, float]],
    image_width: int,
    image_height: int,
) -> Optional[Tuple[Box, Any, float]]:
    """
    Select the best detection based on detection score and relative box area.
    """
    if not detections:
        return None

    image_area = max(1, image_width * image_height)

    def score_detection(det: Tuple[Box, Any, float]) -> float:
        box, _, score = det
        area = box_area(clamp_box(box, image_width, image_height))
        area_ratio = area / image_area
        return float(score) + 0.10 * min(area_ratio, 1.0)

    return max(detections, key=score_detection)


def _run_detector_safe(
    detector_attr: str,
    image: Image.Image,
    conf_threshold: float,
    iou_threshold: float,
) -> List[Tuple[Box, str, float]]:
    """Invoke an imgutils.detect function safely, returning clamped non-empty boxes."""
    try:
        import imgutils.detect as imgutils_detect

        detector: Callable[..., List[Tuple[Any, Any, float]]] = getattr(imgutils_detect, detector_attr)
        raw_dets = detector(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        width, height = image.size
        results: List[Tuple[Box, str, float]] = []
        for det in raw_dets:
            clamped = clamp_box(det[0], width, height)
            if box_area(clamped) > 0:
                results.append((clamped, str(det[1]), float(det[2])))
        return results
    except Exception as exc:
        warnings.warn(
            f"Detector failed ({type(exc).__name__}): {exc}. Crop fallback requires review.",
            RuntimeWarning,
            stacklevel=3,
        )
        return []


def detect_heads_safe(
    image: Image.Image,
    conf_threshold: float = 0.40,
    iou_threshold: float = 0.50,
) -> List[Tuple[Box, str, float]]:
    """Safe wrapper around imgutils.detect.detect_heads."""
    return _run_detector_safe("detect_heads", image, conf_threshold, iou_threshold)


def detect_persons_safe(
    image: Image.Image,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.50,
) -> List[Tuple[Box, str, float]]:
    """Safe wrapper around imgutils.detect.detect_person."""
    return _run_detector_safe("detect_person", image, conf_threshold, iou_threshold)
