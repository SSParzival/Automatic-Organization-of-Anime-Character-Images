"""
Bounding box geometry, clamping, expansion, and area calculations.
"""

from typing import Tuple

Box = Tuple[int, int, int, int]


def clamp_box(box: Tuple[float, float, float, float], width: int, height: int) -> Box:
    """Clamp bounding box coordinates to image dimensions [0, width] and [0, height]."""
    x0, y0, x1, y1 = box
    cx0 = max(0, min(int(round(x0)), width - 1))
    cy0 = max(0, min(int(round(y0)), height - 1))
    cx1 = max(cx0 + 1, min(int(round(x1)), width))
    cy1 = max(cy0 + 1, min(int(round(y1)), height))
    return cx0, cy0, cx1, cy1


def expand_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
    padding_ratio: float,
) -> Box:
    """Expand bounding box outward by padding_ratio of its width and height."""
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    px = int(round(bw * padding_ratio))
    py = int(round(bh * padding_ratio))
    return clamp_box((x0 - px, y0 - py, x1 + px, y1 + py), width, height)


def box_area(box: Tuple[float, float, float, float]) -> int:
    """Calculate area of bounding box."""
    x0, y0, x1, y1 = box
    return max(0, int(x1 - x0)) * max(0, int(y1 - y0))
