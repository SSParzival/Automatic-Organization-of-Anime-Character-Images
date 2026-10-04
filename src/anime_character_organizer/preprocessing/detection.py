"""
Anime character head and person detection wrappers and selection heuristics.
"""

from typing import Any, Dict, List, Optional, Tuple

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


def detect_heads_safe(
    image: Image.Image,
    conf_threshold: float = 0.40,
    iou_threshold: float = 0.50,
) -> List[Tuple[Box, str, float]]:
    """Safe wrapper around imgutils.detect.detect_heads."""
    try:
        from imgutils.detect import detect_heads

        raw_dets = detect_heads(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        width, height = image.size
        return [
            (clamp_box(det[0], width, height), str(det[1]), float(det[2]))
            for det in raw_dets
            if box_area(clamp_box(det[0], width, height)) > 0
        ]
    except Exception:
        return []


def detect_persons_safe(
    image: Image.Image,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.50,
) -> List[Tuple[Box, str, float]]:
    """Safe wrapper around imgutils.detect.detect_person."""
    try:
        from imgutils.detect import detect_person

        raw_dets = detect_person(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        width, height = image.size
        return [
            (clamp_box(det[0], width, height), str(det[1]), float(det[2]))
            for det in raw_dets
            if box_area(clamp_box(det[0], width, height)) > 0
        ]
    except Exception:
        return []
