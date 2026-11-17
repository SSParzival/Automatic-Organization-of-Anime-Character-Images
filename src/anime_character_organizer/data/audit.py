"""
Dataset scanning, image verification, and cryptographic/perceptual hash computation.
"""

import hashlib
import io
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import imagehash
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from ..utils.hashing import imagehash_to_hex, sha256_file
from ..utils.paths import checked_path, path_to_posix
from ..utils.time import now_iso

Image.MAX_IMAGE_PIXELS = 300_000_000

_AUDIT_COLUMNS = [
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
_IN_MEMORY_READ_LIMIT_BYTES = 64 * 1024 * 1024
_HASH_THUMBNAIL_MAX_SIDE = 256


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
    input_dir = checked_path(input_dir)
    valid_extensions = {ext.lower() for ext in valid_extensions}
    candidates: List[Path] = []

    for root, dirs, files in os.walk(input_dir, followlinks=False):
        dirs[:] = [d for d in dirs if d != "sandbox" and not (Path(root) / d).is_symlink()]
        root_path = Path(root)
        for name in files:
            path = root_path / name
            if path.suffix.lower() in valid_extensions and not path.is_symlink():
                candidates.append(path)

    return sorted(candidates)


def sample_candidate_files(
    candidates: List[Path],
    max_images: Optional[int],
    strategy: str = "stratified",
    seed: int = 42,
) -> List[Path]:
    """Select a deterministic representative subset from scanned candidate image paths."""
    if max_images is None or len(candidates) <= max_images:
        return candidates
    if max_images <= 0:
        raise ValueError("max_images must be positive.")

    if strategy == "first":
        return candidates[:max_images]
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(candidates, max_images))
    if strategy == "stratified":
        # Order by directory + extension + filename, then sample evenly across buckets.
        bucketed = sorted(candidates, key=lambda p: (p.parent.as_posix(), p.suffix.lower(), p.name))
        total = len(bucketed)
        indices = [int(i * total / max_images) for i in range(max_images)]
        return sorted(bucketed[idx] for idx in indices)
    raise ValueError(f"Unsupported sample_strategy: {strategy}")


def _relative_posix(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def inspect_image(
    path: Union[str, Path],
    input_dir: Union[str, Path],
    *,
    _checked_input_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Inspect an image file: verify readability, collect dimensions/format,
    and compute SHA-256 and perceptual hashes.
    """
    checked_file = checked_path(path)
    base_dir = _checked_input_dir if _checked_input_dir is not None else checked_path(input_dir)

    record: Dict[str, Any] = {
        "path": path_to_posix(checked_file),
        "relative_path": _relative_posix(checked_file, base_dir),
        "filename": checked_file.name,
        "stem": checked_file.stem,
        "extension": checked_file.suffix.lower(),
        "parent": checked_file.parent.as_posix(),
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
        stat = checked_file.stat()
        record["file_size_bytes"] = int(stat.st_size)

        if stat.st_size <= 0:
            record["status"] = "invalid"
            record["error_type"] = "EmptyFile"
            record["error_message"] = "File has zero bytes."
            return record

        if stat.st_size <= _IN_MEMORY_READ_LIMIT_BYTES:
            raw_bytes = checked_file.read_bytes()
            sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.verify()
            open_target: Union[Path, io.BytesIO] = io.BytesIO(raw_bytes)
        else:
            sha256_hex = sha256_file(checked_file)
            with Image.open(checked_file) as img:
                img.verify()
            open_target = checked_file

        with Image.open(open_target) as img:
            record["format"] = img.format
            record["width"] = int(img.width)
            record["height"] = int(img.height)
            record["mode"] = img.mode
            record["is_animated"] = bool(getattr(img, "is_animated", False))
            record["n_frames"] = int(getattr(img, "n_frames", 1))

            rgb = ImageOps.exif_transpose(img).convert("RGB")
            if max(rgb.size) > _HASH_THUMBNAIL_MAX_SIDE:
                rgb.thumbnail((_HASH_THUMBNAIL_MAX_SIDE, _HASH_THUMBNAIL_MAX_SIDE), Image.Resampling.BILINEAR)

            record["sha256"] = sha256_hex
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

    checked_input_dir = checked_path(input_dir)
    records: List[Dict[str, Any]] = []
    worker_count = min(max_workers, max(1, len(candidate_files)))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(inspect_image, path, checked_input_dir): path
            for path in candidate_files
        }
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
                        "relative_path": _relative_posix(Path(path), checked_input_dir),
                        "filename": Path(path).name,
                        "stem": Path(path).stem,
                        "extension": Path(path).suffix.lower(),
                        "parent": Path(path).parent.as_posix(),
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

    metadata_df = (
        pd.DataFrame(records).reindex(columns=_AUDIT_COLUMNS).sort_values("path", kind="stable").reset_index(drop=True)
    )
    valid_df = metadata_df[metadata_df["status"].eq("valid")].copy().reset_index(drop=True)
    invalid_df = metadata_df[metadata_df["status"].ne("valid")].copy().reset_index(drop=True)

    return metadata_df, valid_df, invalid_df
