"""
Validation of materialized output directory files against source files.
"""

from typing import Any, Dict, List

import pandas as pd

from ..utils.paths import checked_path


def validate_materialized_files(materialization_result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that all materialized files exist, are regular files or symlinks,
    and have sizes consistent with the original source files.
    """
    validation_rows: List[Dict[str, Any]] = []

    for _, row in materialization_result_df.iterrows():
        source_path = checked_path(row["source_path"])
        dest_path = checked_path(row["destination_path"], allow_leaf_symlink=True)
        if dest_path.is_symlink():
            target = dest_path.readlink()
            checked_path(target if target.is_absolute() else dest_path.parent / target)
        status = row.get("status")

        dest_exists = dest_path.exists() or dest_path.is_symlink()
        dest_is_file = dest_path.is_file() or dest_path.is_symlink()
        size_matches = False
        error_msg = None

        if dest_exists and source_path.exists():
            try:
                size_matches = dest_path.stat().st_size == source_path.stat().st_size
            except Exception as e:
                error_msg = str(e)

        is_valid = bool(status in ("ok", "skipped_existing") and dest_exists and dest_is_file and size_matches)

        validation_rows.append(
            {
                "embedding_row": row.get("embedding_row"),
                "source_path": source_path.as_posix(),
                "destination_path": dest_path.as_posix(),
                "status": status,
                "dest_exists": dest_exists,
                "dest_is_file": dest_is_file,
                "size_matches": size_matches,
                "is_valid": is_valid,
                "validation_error": error_msg,
            }
        )

    return pd.DataFrame(validation_rows)
