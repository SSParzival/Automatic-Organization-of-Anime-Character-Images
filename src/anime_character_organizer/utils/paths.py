"""
Filesystem path normalization and portability helpers.
"""

from pathlib import Path
from typing import Union


def normalize_path_string(value: Union[str, Path]) -> str:
    """Resolve and expand a path, returning a normalized POSIX string."""
    return Path(str(value)).expanduser().resolve().as_posix()


def path_to_posix(path: Union[str, Path]) -> str:
    """Convert a path to POSIX string representation."""
    return Path(path).as_posix()


def safe_relative_path(path: Union[str, Path], base_dir: Union[str, Path]) -> str:
    """
    Compute a relative path from base_dir if possible.
    If path is not under base_dir, return normalized POSIX path.
    """
    path = Path(path).resolve()
    base_dir = Path(base_dir).resolve()
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()
