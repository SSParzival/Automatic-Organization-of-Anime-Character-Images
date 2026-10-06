"""File linking and exclusive copying with source preservation checks."""

import errno
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Union

from ..utils.paths import checked_path


def remove_existing_file(path: Union[str, Path]) -> None:
    """Remove an explicit file or leaf symlink; reject directory aliases."""
    path = checked_path(path, allow_leaf_symlink=True)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise IsADirectoryError(f"Destination is a directory: {path}")


def _copy_exclusive(source: Path, destination: Path) -> None:
    """Create a new file exclusively so concurrent destinations are preserved."""
    created = False
    try:
        with source.open("rb") as src, destination.open("xb") as dst:
            created = True
            shutil.copyfileobj(src, dst)
        shutil.copystat(source, destination)
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise


def materialize_one_file(
    source_path: Union[str, Path],
    destination_path: Union[str, Path],
    mode: str = "hardlink",
    on_existing: str = "error",
    allow_hardlink_fallback_to_copy: bool = True,
) -> Dict[str, Any]:
    """Link/copy a source without moving it, returning an auditable status.

    Invalid policies and source aliases fail before mutation. Overwrite applies
    only to distinct destination files. Hardlink fallback is limited to filesystem
    or permission errors; collisions never trigger copying over another file.
    Hardlinks share bytes with the source: subsequent edits affect both names.
    """
    result = {"status": "unknown", "operation_used": None, "error_type": None, "error_message": None}
    try:
        if mode not in {"copy", "symlink", "hardlink"}:
            raise ValueError(f"Unsupported materialization mode: {mode}")
        if on_existing not in {"error", "skip", "overwrite"}:
            raise ValueError(f"Unsupported existing-file policy: {on_existing}")
        source = checked_path(source_path)
        destination = checked_path(destination_path, allow_leaf_symlink=True)
        if source == destination:
            raise ValueError("Source and destination must differ.")
        if not source.is_file():
            raise FileNotFoundError(f"Source is not a regular file: {source}")
        if destination.exists() and not destination.is_symlink() and os.path.samefile(source, destination):
            raise ValueError("Destination aliases the source inode.")
        if destination.exists() or destination.is_symlink():
            if on_existing == "error":
                raise FileExistsError(f"Destination already exists: {destination}")
            if on_existing == "skip":
                result.update(status="skipped_existing", operation_used="skip")
                return result
            remove_existing_file(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            _copy_exclusive(source, destination)
        elif mode == "symlink":
            destination.symlink_to(source)
        else:
            try:
                os.link(source, destination)
            except OSError as exc:
                fallback_errors = {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EOPNOTSUPP}
                if not allow_hardlink_fallback_to_copy or exc.errno not in fallback_errors:
                    raise
                _copy_exclusive(source, destination)
                result.update(
                    status="ok",
                    operation_used="copy_fallback_from_hardlink",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                return result
        result.update(status="ok", operation_used=mode)
    except Exception as exc:
        result.update(status="error", operation_used=mode, error_type=type(exc).__name__, error_message=str(exc))
    return result
