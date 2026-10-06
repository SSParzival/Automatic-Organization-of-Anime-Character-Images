"""
Destination planning and collision prevention for folder materialization.
"""

from pathlib import Path
from typing import Any, Dict, List, Set, Union

import pandas as pd

from ..utils.hashing import stable_short_hash
from ..utils.naming import sanitize_filename_component, sanitize_folder_name
from ..utils.paths import checked_path


def make_destination_filename(row: Dict[str, Any], source_path: Union[str, Path]) -> str:
    """Format safe destination filename incorporating embedding row prefix and source stem."""
    source_path = Path(source_path)
    suffix = source_path.suffix
    stem = source_path.stem

    embedding_row = row.get("embedding_row", None)
    try:
        prefix = f"{int(embedding_row):07d}"
    except Exception:
        prefix = stable_short_hash(source_path.as_posix(), length=10)

    safe_stem = sanitize_filename_component(stem, fallback="image", max_length=110)
    return f"{prefix}__{safe_stem}{suffix}"


def ensure_unique_destination_path(destination_path: Path, used_paths: Set[str]) -> Path:
    """Ensure destination path does not collide with existing files or previously planned files."""
    dest_key = destination_path.as_posix()
    if dest_key not in used_paths and not destination_path.exists() and not destination_path.is_symlink():
        used_paths.add(dest_key)
        return destination_path

    parent = destination_path.parent
    stem = destination_path.stem
    suffix = destination_path.suffix
    counter = 1

    while True:
        candidate = parent / f"{stem}__dup{counter:03d}{suffix}"
        candidate_key = candidate.as_posix()
        if candidate_key not in used_paths and not candidate.exists() and not candidate.is_symlink():
            used_paths.add(candidate_key)
            return candidate
        counter += 1


def build_materialization_plan(
    assignment_df: pd.DataFrame,
    output_dir: Path,
    source_column: str = "source_path",
    folder_column: str = "proposed_folder",
) -> pd.DataFrame:
    """
    Build complete materialization plan dataframe specifying destination path for each row.
    """
    output_dir = checked_path(output_dir)
    used_paths: Set[str] = set()
    plan_rows: List[Dict[str, Any]] = []

    for _, row in assignment_df.iterrows():
        source_path = checked_path(row[source_column])
        folder_name = sanitize_folder_name(row[folder_column], fallback="_needs_review")
        dest_filename = make_destination_filename(row, source_path)
        dest_path = checked_path(output_dir / folder_name / dest_filename, allow_leaf_symlink=True)
        unique_dest_path = ensure_unique_destination_path(dest_path, used_paths)

        plan_rows.append(
            {
                "embedding_row": row.get("embedding_row"),
                "cluster_label": row.get("cluster_label"),
                "proposed_folder": folder_name,
                "source_path": source_path.as_posix(),
                "destination_path": unique_dest_path.as_posix(),
                "destination_filename": unique_dest_path.name,
            }
        )

    result_df = assignment_df.copy().reset_index(drop=True)
    result_df["source_path"] = [r["source_path"] for r in plan_rows]
    result_df[folder_column] = [r["proposed_folder"] for r in plan_rows]
    result_df["destination_path"] = [r["destination_path"] for r in plan_rows]
    result_df["destination_filename"] = [r["destination_filename"] for r in plan_rows]
    return result_df
