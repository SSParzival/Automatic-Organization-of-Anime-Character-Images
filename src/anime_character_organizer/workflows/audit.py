"""
Workflow orchestrator for Stage 1: Dataset audit, validation, and duplicate detection.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Optional, Set, Union
import pandas as pd

from ..config import AuditConfig, DEFAULT_VALID_EXTENSIONS
from ..data.audit import audit_images, scan_candidate_files
from ..data.duplicates import find_exact_duplicates, find_perceptual_duplicates
from ..exceptions import ConfigurationError
from ..utils.time import now_iso


def run_dataset_audit(
    input_dir: Union[str, Path],
    project_dir: Union[str, Path] = "./anime_character_pipeline",
    valid_extensions: Optional[Set[str]] = None,
    phash_threshold: int = 6,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Execute complete Stage 1 dataset audit workflow.
    
    Returns:
        Dict containing run_dir, tables, summary, and path mappings.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    project_dir = Path(project_dir).expanduser().resolve()

    if not input_dir.exists():
        raise ConfigurationError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ConfigurationError(f"Input path is not a directory: {input_dir}")

    if valid_extensions is None:
        valid_extensions = set(DEFAULT_VALID_EXTENSIONS)

    if max_workers is None:
        max_workers = max(1, min(16, (os.cpu_count() or 4)))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_dir / "runs" / f"01_dataset_audit_{run_id}"
    reports_dir = run_dir / "reports"
    tables_dir = run_dir / "tables"
    logs_dir = run_dir / "logs"

    for d in [run_dir, reports_dir, tables_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    candidate_files = scan_candidate_files(input_dir, valid_extensions)

    scan_report = {
        "input_dir": input_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "candidate_file_count": len(candidate_files),
        "valid_extensions": sorted(list(valid_extensions)),
        "created_at": now_iso(),
    }
    with (reports_dir / "scan_report.json").open("w", encoding="utf-8") as f:
        json.dump(scan_report, f, ensure_ascii=False, indent=2)

    if not candidate_files:
        raise RuntimeError(f"No candidate image files found in {input_dir} with extensions {valid_extensions}")

    metadata_df, valid_df, invalid_df = audit_images(
        candidate_files=candidate_files,
        input_dir=input_dir,
        max_workers=max_workers,
        show_progress=show_progress,
    )

    exact_duplicates_df = find_exact_duplicates(valid_df)
    perceptual_candidates_df, perceptual_groups_df = find_perceptual_duplicates(
        valid_df=valid_df,
        phash_threshold=phash_threshold,
        show_progress=show_progress,
    )

    metadata_path = tables_dir / "image_metadata.csv"
    valid_path = tables_dir / "valid_images.csv"
    invalid_path = tables_dir / "invalid_images.csv"
    exact_duplicates_path = tables_dir / "exact_duplicate_groups.csv"
    perceptual_candidates_path = tables_dir / "perceptual_duplicate_candidate_pairs.csv"
    perceptual_groups_path = tables_dir / "perceptual_duplicate_groups.csv"

    metadata_df.to_csv(metadata_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    invalid_df.to_csv(invalid_path, index=False)
    exact_duplicates_df.to_csv(exact_duplicates_path, index=False)
    perceptual_candidates_df.to_csv(perceptual_candidates_path, index=False)
    perceptual_groups_df.to_csv(perceptual_groups_path, index=False)

    summary = {
        "run_id": run_id,
        "created_at": now_iso(),
        "input_dir": input_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "total_candidate_files": int(len(metadata_df)),
        "valid_images": int(len(valid_df)),
        "invalid_images": int(len(invalid_df)),
        "exact_duplicate_groups": int(exact_duplicates_df["exact_group_id"].nunique()) if not exact_duplicates_df.empty else 0,
        "exact_duplicate_files_involved": int(len(exact_duplicates_df)),
        "perceptual_hash_threshold": int(phash_threshold),
        "perceptual_duplicate_candidate_pairs": int(len(perceptual_candidates_df)),
        "perceptual_duplicate_groups": int(perceptual_groups_df["perceptual_group_id"].nunique()) if not perceptual_groups_df.empty else 0,
        "perceptual_duplicate_files_involved": int(len(perceptual_groups_df)),
        "python": sys.version,
        "platform": platform.platform(),
        "max_workers": int(max_workers),
        "outputs": {
            "metadata": metadata_path.as_posix(),
            "valid_images": valid_path.as_posix(),
            "invalid_images": invalid_path.as_posix(),
            "exact_duplicate_groups": exact_duplicates_path.as_posix(),
            "perceptual_duplicate_candidate_pairs": perceptual_candidates_path.as_posix(),
            "perceptual_duplicate_groups": perceptual_groups_path.as_posix(),
        },
    }

    summary_path = reports_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "metadata_df": metadata_df,
        "valid_df": valid_df,
        "invalid_df": invalid_df,
        "exact_duplicates_df": exact_duplicates_df,
        "perceptual_candidates_df": perceptual_candidates_df,
        "perceptual_groups_df": perceptual_groups_df,
        "summary": summary,
        "summary_path": summary_path,
    }
