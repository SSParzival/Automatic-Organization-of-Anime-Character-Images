"""
Pipeline run directory discovery and resolution.
"""

from pathlib import Path
from typing import List, Optional, Union

from ..exceptions import RunNotFoundError
from .paths import checked_path


def find_latest_run(
    project_dir: Union[str, Path],
    stage_prefix: str,
    required_relative_paths: Optional[List[Union[str, Path]]] = None,
) -> Path:
    """
    Locate the latest run directory matching stage_prefix that contains all required files.

    Args:
        project_dir: Path to base project directory (e.g. ./anime_character_pipeline).
        stage_prefix: Directory prefix to match (e.g. '01_dataset_audit_*').
        required_relative_paths: List of relative paths that must exist in a valid run.

    Returns:
        Path to latest valid run directory.

    Raises:
        RunNotFoundError: If no matching and valid run directory is found.
    """
    if "/" in stage_prefix or "\\" in stage_prefix or stage_prefix == "sandbox":
        raise ValueError("Run prefix must not traverse directories.")
    for relative in required_relative_paths or []:
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or "sandbox" in rel.parts:
            raise ValueError("Required artifact paths must remain inside a run.")
    runs_dir = checked_path(checked_path(project_dir) / "runs")
    candidates = sorted(
        [p for p in runs_dir.glob(stage_prefix) if p.name != "sandbox" and not p.is_symlink() and p.is_dir()],
        key=lambda p: p.name,
    )

    if not required_relative_paths:
        if not candidates:
            raise RunNotFoundError(f"No run matching {stage_prefix} found in {runs_dir}.")
        return candidates[-1]

    valid_candidates = []
    for candidate in candidates:
        if all(checked_path(candidate / rel_path).exists() for rel_path in required_relative_paths):
            valid_candidates.append(candidate)

    if not valid_candidates:
        required_text = ", ".join(str(p) for p in required_relative_paths)
        raise RunNotFoundError(
            f"No valid run matching {stage_prefix} found in {runs_dir}. Required files: {required_text}"
        )

    return valid_candidates[-1]


def create_run_directory(project_dir: Union[str, Path], stage: str) -> Path:
    """Reserve a unique run atomically; never reuse a previous run's artifacts."""
    from datetime import datetime
    from uuid import uuid4

    from .paths import checked_path

    if not stage or "/" in stage or "\\" in stage or stage == "sandbox":
        raise ValueError("Stage must be a single non-protected directory prefix.")
    runs = checked_path(project_dir) / "runs"
    checked_path(runs).mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
        run_dir = runs / f"{stage}_{run_id}"
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError("Unable to allocate a unique pipeline run.")
