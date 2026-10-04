"""
Dataset scanning, image verification, and cryptographic/perceptual hash computation.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import imagehash
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from ..utils.hashing import imagehash_to_hex, sha256_file
from ..utils.paths import path_to_posix, safe_relative_path
from ..utils.time import now_iso

Image.MAX_IMAGE_PIXELS = 300_000_000


def scan_candidate_files(
    input_dir: Union[str, Path],
    valid_extensions: Set[str],
) -> List[Path]:
    """
    Recursively scan directory for files matching specified extensions.

    Args:
        input_dir: Source folder to scan.
        valid_extensions: Set of valid lowercase file extensions including dot (e.g. {'.jpg', '.png'}).

    Returns:
        Sorted list of matching Path objects.
    """
    input_dir = Path(input_dir).resolve()
    valid_extensions = {ext.lower() for ext in valid_extensions}
    candidates: List[Path] = []

    for root, _, files in os.walk(input_dir):
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() in valid_extensions:
                candidates.append(path)

    return sorted(candidates)


def inspect_image(path: Union[str, Path], input_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Inspect an image file: verify readability, collect dimensions/format,
    and compute SHA-256 and perceptual hashes.
    """
    path = Path(path).resolve()
    input_dir = Path(input_dir).resolve()

    record: Dict[str, Any] = {
        "path": path_to_posix(path),
        "relative_path": safe_relative_path(path, input_dir),
        "filename": path.name,
        "stem": path.stem,
        "extension": path.suffix.lower(),
        "parent": path.parent.as_posix(),
        "file_size_bytes": None,
        "status": "unknown",
        "error_type": None,
        "error_message": None,
        "format": None,
        "width": None,
        "height": None,
        "mode": None,
        "is_animated": None,
        "n_frames": None,
        "sha256": None,
        "phash": None,
        "dhash": None,
        "whash": None,
        "colorhash": None,
        "processed_at": now_iso(),
    }

    try:
        stat = path.stat()
        record["file_size_bytes"] = int(stat.st_size)

        if stat.st_size <= 0:
            record["status"] = "invalid"
            record["error_type"] = "EmptyFile"
            record["error_message"] = "File has zero bytes."
            return record

        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            record["format"] = img.format
            record["width"] = int(img.width)
            record["height"] = int(img.height)
            record["mode"] = img.mode
            record["is_animated"] = bool(getattr(img, "is_animated", False))
            record["n_frames"] = int(getattr(img, "n_frames", 1))

            rgb = ImageOps.exif_transpose(img.convert("RGB"))

            record["sha256"] = sha256_file(path)
            record["phash"] = imagehash_to_hex(imagehash.phash(rgb, hash_size=8))
            record["dhash"] = imagehash_to_hex(imagehash.dhash(rgb, hash_size=8))
            record["whash"] = imagehash_to_hex(imagehash.whash(rgb, hash_size=8))
            record["colorhash"] = imagehash_to_hex(imagehash.colorhash(rgb, binbits=3))

        record["status"] = "valid"
        return record

    except Exception as exc:
        record["status"] = "invalid"
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)
        return record


def audit_images(
    candidate_files: List[Path],
    input_dir: Union[str, Path],
    max_workers: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Audit and hash candidate image files in parallel.

    Returns:
        (metadata_df, valid_df, invalid_df)
    """
    if max_workers is None:
        max_workers = max(1, min(16, (os.cpu_count() or 4)))

    records: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(inspect_image, path, input_dir): path for path in candidate_files}
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Validating and hashing images")

        for future in iterator:
            path = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                records.append(
                    {
                        "path": path_to_posix(path),
                        "relative_path": safe_relative_path(path, input_dir),
                        "filename": path.name,
                        "stem": path.stem,
                        "extension": path.suffix.lower(),
                        "parent": path.parent.as_posix(),
                        "file_size_bytes": None,
                        "status": "invalid",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "format": None,
                        "width": None,
                        "height": None,
                        "mode": None,
                        "is_animated": None,
                        "n_frames": None,
                        "sha256": None,
                        "phash": None,
                        "dhash": None,
                        "whash": None,
                        "colorhash": None,
                        "processed_at": now_iso(),
                    }
                )

    ordered_columns = [
        "path",
        "relative_path",
        "filename",
        "stem",
        "extension",
        "parent",
        "file_size_bytes",
        "status",
        "error_type",
        "error_message",
        "format",
        "width",
        "height",
        "mode",
        "is_animated",
        "n_frames",
        "sha256",
        "phash",
        "dhash",
        "whash",
        "colorhash",
        "processed_at",
    ]

    metadata_df = pd.DataFrame(records).reindex(columns=ordered_columns)
    valid_df = metadata_df[metadata_df["status"].eq("valid")].copy().reset_index(drop=True)
    invalid_df = metadata_df[metadata_df["status"].ne("valid")].copy().reset_index(drop=True)

    return metadata_df, valid_df, invalid_df
