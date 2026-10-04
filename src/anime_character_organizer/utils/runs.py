"""
Pipeline run directory discovery and resolution.
"""

from pathlib import Path
from typing import List, Optional, Union

from ..exceptions import RunNotFoundError


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
    runs_dir = Path(project_dir).expanduser().resolve() / "runs"
    candidates = sorted(
        [p for p in runs_dir.glob(stage_prefix) if p.is_dir()],
        key=lambda p: p.name,
    )

    if not required_relative_paths:
        if not candidates:
            raise RunNotFoundError(f"No run matching {stage_prefix} found in {runs_dir}.")
        return candidates[-1]

    valid_candidates = []
    for candidate in candidates:
        if all((candidate / rel_path).exists() for rel_path in required_relative_paths):
            valid_candidates.append(candidate)

    if not valid_candidates:
        required_text = ", ".join(str(p) for p in required_relative_paths)
        raise RunNotFoundError(
            f"No valid run matching {stage_prefix} found in {runs_dir}. Required files: {required_text}"
        )

    return valid_candidates[-1]
