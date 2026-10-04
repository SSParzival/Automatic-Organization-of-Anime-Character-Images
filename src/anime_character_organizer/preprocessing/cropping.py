"""
Image cropping, priority-based region extraction (head -> person -> fallback), and crop saving.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from PIL import Image, ImageOps

from ..utils.time import now_iso
from .detection import choose_best_detection, detect_heads_safe, detect_persons_safe, detection_to_dict
from .geometry import clamp_box, expand_box

Image.MAX_IMAGE_PIXELS = 300_000_000


def make_crop_id(index_value: int, path_value: Union[str, Path]) -> str:
    """Generate a clean, deterministic identifier for an image crop."""
    stem = Path(str(path_value)).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    safe_stem = safe_stem[:80].strip("_") or "image"
    return f"img_{int(index_value):07d}_{safe_stem}"


def save_rgb_image(
    image: Image.Image, output_path: Union[str, Path], crop_format: str = "JPEG", quality: int = 95
) -> None:
    """Save PIL image as RGB JPEG or PNG with optimization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if crop_format.upper() == "JPEG":
        image.convert("RGB").save(output_path, format="JPEG", quality=quality, optimize=True)
    elif crop_format.upper() == "PNG":
        image.save(output_path, format="PNG", optimize=True)
    else:
        raise ValueError(f"Unsupported crop format: {crop_format}")


def process_image_for_crop(
    row: Dict[str, Any],
    crops_dir: Path,
    head_conf: float = 0.40,
    head_iou: float = 0.50,
    person_conf: float = 0.35,
    person_iou: float = 0.50,
    head_padding: float = 0.20,
    person_padding: float = 0.10,
    crop_format: str = "JPEG",
    crop_quality: int = 95,
    crop_output_size: Optional[Tuple[int, int]] = (512, 512),
    min_crop_side: int = 64,
) -> Dict[str, Any]:
    """
    Process one image row to extract character crop according to priority:
    1. Best head crop (with head_padding)
    2. Best person crop (with person_padding)
    3. Full image fallback (if no detection or detection too small)
    """
    path = Path(row["path"])
    image_index = int(row["image_index"])
    crop_id = make_crop_id(image_index, path)
    crop_extension = ".jpg" if crop_format.upper() == "JPEG" else ".png"
    crop_path = crops_dir / f"{crop_id}{crop_extension}"

    record: Dict[str, Any] = {
        "image_index": image_index,
        "crop_id": crop_id,
        "source_path": path.as_posix(),
        "relative_path": row.get("relative_path"),
        "crop_path": crop_path.as_posix(),
        "status": "unknown",
        "error_type": None,
        "error_message": None,
        "source_width": None,
        "source_height": None,
        "crop_width": None,
        "crop_height": None,
        "selected_region_type": None,
        "selected_score": None,
        "detected_heads_count": 0,
        "detected_persons_count": 0,
        "is_multi_head": False,
        "is_multi_person": False,
        "selected_x0": None,
        "selected_y0": None,
        "selected_x1": None,
        "selected_y1": None,
        "head_detections_json": None,
        "person_detections_json": None,
        "processed_at": now_iso(),
    }

    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img.convert("RGB"))
            width, height = img.size
            record["source_width"] = int(width)
            record["source_height"] = int(height)

            head_detections = detect_heads_safe(img, conf_threshold=head_conf, iou_threshold=head_iou)
            person_detections = detect_persons_safe(img, conf_threshold=person_conf, iou_threshold=person_iou)

            record["detected_heads_count"] = int(len(head_detections))
            record["detected_persons_count"] = int(len(person_detections))
            record["is_multi_head"] = bool(len(head_detections) > 1)
            record["is_multi_person"] = bool(len(person_detections) > 1)
            record["head_detections_json"] = json.dumps(
                [detection_to_dict(det) for det in head_detections], ensure_ascii=False
            )
            record["person_detections_json"] = json.dumps(
                [detection_to_dict(det) for det in person_detections], ensure_ascii=False
            )

            best_head = choose_best_detection(head_detections, width, height)
            best_person = choose_best_detection(person_detections, width, height)

            if best_head is not None:
                selected_box = expand_box(best_head[0], width, height, head_padding)
                selected_type = "head"
                selected_score = float(best_head[2])
            elif best_person is not None:
                selected_box = expand_box(best_person[0], width, height, person_padding)
                selected_type = "person"
                selected_score = float(best_person[2])
            else:
                selected_box = (0, 0, width, height)
                selected_type = "full_image"
                selected_score = None

            x0, y0, x1, y1 = clamp_box(selected_box, width, height)

            if (x1 - x0) < min_crop_side or (y1 - y0) < min_crop_side:
                x0, y0, x1, y1 = 0, 0, width, height
                selected_type = "full_image_small_detection_fallback"
                selected_score = None

            crop = img.crop((x0, y0, x1, y1))

            if crop_output_size is not None:
                crop = crop.resize(crop_output_size, Image.Resampling.LANCZOS)

            save_rgb_image(crop, crop_path, crop_format, crop_quality)

            record["status"] = "ok"
            record["crop_width"] = int(crop.width)
            record["crop_height"] = int(crop.height)
            record["selected_region_type"] = selected_type
            record["selected_score"] = selected_score
            record["selected_x0"] = int(x0)
            record["selected_y0"] = int(y0)
            record["selected_x1"] = int(x1)
            record["selected_y1"] = int(y1)

            return record

    except Exception as exc:
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)
        return record


def add_review_flags(crops_df: pd.DataFrame) -> pd.DataFrame:
    """Classify crop quality and assign review flags and reasons."""
    review_df = crops_df.copy()
    review_df["needs_review"] = False
    review_df["review_reason"] = ""

    review_df.loc[review_df["status"].ne("ok"), "needs_review"] = True
    review_df.loc[review_df["status"].ne("ok"), "review_reason"] = "crop_error"

    review_df.loc[review_df["selected_region_type"].eq("full_image"), "review_reason"] = review_df.loc[
        review_df["selected_region_type"].eq("full_image"), "review_reason"
    ].replace("", "no_detection_full_image_fallback")

    review_df.loc[review_df["selected_region_type"].eq("full_image"), "needs_review"] = True

    review_df.loc[review_df["selected_region_type"].eq("full_image_small_detection_fallback"), "needs_review"] = True

    review_df.loc[review_df["selected_region_type"].eq("full_image_small_detection_fallback"), "review_reason"] = (
        "small_detection_full_image_fallback"
    )

    review_df.loc[review_df["is_multi_head"].astype(bool), "needs_review"] = True

    review_df.loc[review_df["is_multi_head"].astype(bool) & review_df["review_reason"].eq(""), "review_reason"] = (
        "multiple_heads_detected"
    )

    review_df.loc[
        review_df["is_multi_head"].astype(bool) & ~review_df["review_reason"].eq("multiple_heads_detected"),
        "review_reason",
    ] = (
        review_df.loc[
            review_df["is_multi_head"].astype(bool) & ~review_df["review_reason"].eq("multiple_heads_detected"),
            "review_reason",
        ]
        + ";multiple_heads_detected"
    )

    return review_df
