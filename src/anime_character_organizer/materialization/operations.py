"""
Filesystem materialization operations: hard links, symbolic links, and file copies with safety checks.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Union


def remove_existing_file(path: Union[str, Path]) -> None:
    """Safely remove a file or symlink without touching directories."""
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise IsADirectoryError(f"Destination exists and is a directory, not a file: {path}")


def materialize_one_file(
    source_path: Union[str, Path],
    destination_path: Union[str, Path],
    mode: str = "hardlink",
    on_existing: str = "error",
    allow_hardlink_fallback_to_copy: bool = True,
) -> Dict[str, Any]:
    """
    Materialize a single file at destination using the specified mode.

    Modes:
        'hardlink': Creates a hard link pointing to the same inode.
        'copy': Copies file bytes and preserves timestamps (shutil.copy2).
        'symlink': Creates a relative or absolute symbolic link.

    on_existing:
        'error': Raises FileExistsError if destination exists.
        'skip': Skips file if destination exists.
        'overwrite': Deletes destination file/symlink before creating new link/copy.
    """
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    result: Dict[str, Any] = {
        "status": "unknown",
        "operation_used": None,
        "error_type": None,
        "error_message": None,
    }

    try:
        if not source_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")
        if not source_path.is_file():
            raise ValueError(f"Source path is not a regular file: {source_path}")

        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if destination_path.exists() or destination_path.is_symlink():
            if on_existing == "error":
                raise FileExistsError(f"Destination already exists: {destination_path}")
            if on_existing == "skip":
                result["status"] = "skipped_existing"
                result["operation_used"] = "skip"
                return result
            if on_existing == "overwrite":
                remove_existing_file(destination_path)

        if mode == "copy":
            shutil.copy2(source_path, destination_path)
            result["status"] = "ok"
            result["operation_used"] = "copy"
            return result

        if mode == "symlink":
            destination_path.symlink_to(source_path)
            result["status"] = "ok"
            result["operation_used"] = "symlink"
            return result

        if mode == "hardlink":
            try:
                os.link(source_path, destination_path)
                result["status"] = "ok"
                result["operation_used"] = "hardlink"
                return result
            except OSError as hardlink_exc:
                if not allow_hardlink_fallback_to_copy:
                    raise
                shutil.copy2(source_path, destination_path)
                result["status"] = "ok"
                result["operation_used"] = "copy_fallback_from_hardlink"
                result["error_type"] = type(hardlink_exc).__name__
                result["error_message"] = str(hardlink_exc)
                return result

        raise ValueError(f"Unsupported materialization mode: {mode}")

    except Exception as exc:
        result["status"] = "error"
        result["operation_used"] = mode
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        return result
