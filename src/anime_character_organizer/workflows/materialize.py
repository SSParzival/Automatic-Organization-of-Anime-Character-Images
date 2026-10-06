"""
Workflow orchestrator for Stage 5: Non-destructive folder materialization.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from tqdm.auto import tqdm

from ..config import validate_parameters
from ..exceptions import ConfigurationError, MaterializationError
from ..materialization.operations import materialize_one_file
from ..materialization.planner import build_materialization_plan
from ..materialization.validation import validate_materialized_files
from ..utils.naming import sanitize_folder_name
from ..utils.paths import checked_path, normalize_path_string
from ..utils.runs import create_run_directory, find_latest_run
from ..utils.serialization import json_safe_value, require_columns, safe_bool
from ..utils.time import now_iso
from ..visualization.contact_sheet import create_contact_sheet


def run_folder_materialization(
    project_dir: Union[str, Path] = "./anime_character_pipeline",
    previous_run_dir: Optional[Union[str, Path]] = None,
    final_output_dir: Optional[Union[str, Path]] = None,
    materialization_mode: str = "hardlink",
    allow_hardlink_fallback_to_copy: bool = True,
    on_existing: str = "error",
    source_column: str = "source_path",
    create_cluster_metadata_files: bool = True,
    create_folder_manifest_files: bool = True,
    create_output_readme: bool = True,
    copy_cluster_contact_sheets_to_output: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Execute complete Stage 5 folder materialization workflow.
    """
    validate_parameters(**locals())
    project_dir = checked_path(project_dir)

    if previous_run_dir is None:
        previous_run_dir = find_latest_run(
            project_dir=project_dir,
            stage_prefix="04_hdbscan_clustering_*",
            required_relative_paths=[
                Path("reports") / "summary.json",
                Path("tables") / "folder_assignment_manifest.csv",
                Path("tables") / "cluster_summary.csv",
            ],
        )
    else:
        previous_run_dir = checked_path(previous_run_dir)

    assignment_path = checked_path(previous_run_dir / "tables" / "folder_assignment_manifest.csv")
    summary_path_in = checked_path(previous_run_dir / "tables" / "cluster_summary.csv")

    if not assignment_path.exists():
        raise ConfigurationError(f"Required folder assignment manifest missing: {assignment_path}")

    folder_assignment_df = pd.read_csv(assignment_path)
    require_columns(
        folder_assignment_df,
        {source_column, "proposed_folder", "embedding_row", "cluster_label", "requires_manual_review"},
        "Folder assignment manifest",
    )
    cluster_summary_df = pd.read_csv(summary_path_in) if summary_path_in.exists() else pd.DataFrame()

    run_dir = create_run_directory(project_dir, "05_folder_materialization")
    run_id = run_dir.name.removeprefix("05_folder_materialization_")
    reports_dir = run_dir / "reports"
    tables_dir = run_dir / "tables"
    logs_dir = run_dir / "logs"

    if final_output_dir is None:
        final_output_dir = project_dir / "organized_output" / f"anime_organized_{run_id}"
    else:
        final_output_dir = checked_path(final_output_dir)

    final_output_dir = checked_path(final_output_dir)
    if final_output_dir.exists() and any(final_output_dir.iterdir()):
        raise ConfigurationError("Output directory must be new or empty to protect existing artifacts.")

    for d in [run_dir, reports_dir, tables_dir, logs_dir, final_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    assignment_df = folder_assignment_df.copy()
    assignment_df[source_column] = assignment_df[source_column].map(normalize_path_string)
    assignment_df["proposed_folder"] = assignment_df["proposed_folder"].map(
        lambda v: sanitize_folder_name(v, fallback="_needs_review")
    )
    assignment_df["requires_manual_review"] = assignment_df["requires_manual_review"].map(safe_bool)

    # Validate all source files exist before doing any operations
    missing_sources = assignment_df[~assignment_df[source_column].map(lambda p: Path(p).is_file())]
    if not missing_sources.empty:
        missing_p = tables_dir / "missing_sources_before_materialization.csv"
        missing_sources.to_csv(missing_p, index=False)
        raise FileNotFoundError(f"Missing source files detected before materialization: {missing_p}")

    assignment_df = assignment_df.sort_values(
        ["proposed_folder", "cluster_label", "embedding_row"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    # Build destination plan
    materialization_plan_df = build_materialization_plan(
        assignment_df=assignment_df,
        output_dir=final_output_dir,
        source_column=source_column,
        folder_column="proposed_folder",
    )
    plan_path = tables_dir / "materialization_plan.csv"
    materialization_plan_df.to_csv(plan_path, index=False)

    # Perform materialization
    materialization_rows = []
    mat_iter = (
        tqdm(
            materialization_plan_df.iterrows(), total=len(materialization_plan_df), desc="Materializing organized files"
        )
        if show_progress
        else materialization_plan_df.iterrows()
    )

    for _, row in mat_iter:
        src = Path(row["source_path"])
        dest = Path(row["destination_path"])
        res = materialize_one_file(
            source_path=src,
            destination_path=dest,
            mode=materialization_mode,
            on_existing=on_existing,
            allow_hardlink_fallback_to_copy=allow_hardlink_fallback_to_copy,
        )
        materialization_rows.append(
            {
                "embedding_row": row["embedding_row"],
                "source_path": src.as_posix(),
                "destination_path": dest.as_posix(),
                "proposed_folder": row["proposed_folder"],
                "cluster_label": row.get("cluster_label"),
                "status": res["status"],
                "operation_used": res["operation_used"],
                "error_type": res["error_type"],
                "error_message": res["error_message"],
                "processed_at": now_iso(),
            }
        )

    materialization_result_df = pd.DataFrame(materialization_rows)
    result_path = tables_dir / "materialization_result.csv"
    materialization_result_df.to_csv(result_path, index=False)

    # Validate materialized files
    validation_df = validate_materialized_files(materialization_result_df)
    validation_path = tables_dir / "materialization_validation.csv"
    validation_df.to_csv(validation_path, index=False)

    if not validation_df["is_valid"].all():
        raise MaterializationError(f"Materialization is incomplete; inspect {validation_path}.")

    # Create per-folder manifests, metadata, contact sheets
    unique_folders = materialization_plan_df["proposed_folder"].unique()
    for folder_name in unique_folders:
        folder_dir = final_output_dir / folder_name
        folder_rows = materialization_plan_df[materialization_plan_df["proposed_folder"].eq(folder_name)]

        if create_folder_manifest_files:
            folder_manifest_path = folder_dir / "_folder_manifest.csv"
            folder_rows.to_csv(folder_manifest_path, index=False)

        if create_cluster_metadata_files and not cluster_summary_df.empty:
            label_val = folder_rows["cluster_label"].iloc[0]
            summary_match = cluster_summary_df[cluster_summary_df["cluster_label"].eq(label_val)]
            info = {
                "folder_name": folder_name,
                "cluster_label": int(label_val),
                "image_count": len(folder_rows),
                "summary": summary_match.to_dict(orient="records")[0] if not summary_match.empty else {},
                "created_at": now_iso(),
            }
            with (folder_dir / "_cluster_info.json").open("w", encoding="utf-8") as f:
                json.dump(json_safe_value(info), f, ensure_ascii=False, indent=2, allow_nan=False)

        # Contact sheet per folder
        dest_paths = [Path(p) for p in folder_rows["destination_path"].head(24) if Path(p).exists()]
        create_contact_sheet(
            dest_paths, folder_dir / "_contact_sheet.jpg", title=f"{folder_name} | n={len(folder_rows)}"
        )

    # Copy clustering contact sheets if requested
    if copy_cluster_contact_sheets_to_output:
        prev_sheets_dir = checked_path(previous_run_dir / "contact_sheets")
        if prev_sheets_dir.exists():
            out_sheets_dir = final_output_dir / "_clustering_contact_sheets"
            out_sheets_dir.mkdir(parents=True, exist_ok=True)
            for sheet_file in prev_sheets_dir.glob("*.jpg"):
                shutil.copy2(checked_path(sheet_file), checked_path(out_sheets_dir / sheet_file.name))

    # Global materialization index
    global_index_path = final_output_dir / "_global_materialization_index.csv"
    materialization_plan_df.to_csv(global_index_path, index=False)

    # Output README
    if create_output_readme:
        readme_lines = [
            "# Anime Organized Image Output",
            "",
            f"Generated on {now_iso()} by Stage 5 materialization workflow.",
            "",
            "## Structure",
            "- Each subdirectory contains images grouped by character identity.",
            "- Folders with `_review` require human inspection.",
            "- `_needs_review_noise` contains images that could not be assigned confidently.",
            "- `_global_materialization_index.csv` indexes every materialized image.",
            "- Inside each folder, `_folder_manifest.csv` records source locations and metadata.",
            "",
            f"Total images organized: {len(materialization_plan_df)} across {len(unique_folders)} folders.",
        ]
        with (final_output_dir / "README.md").open("w", encoding="utf-8") as f:
            f.write("\n".join(readme_lines) + "\n")

    folder_counts_df = (
        materialization_plan_df.groupby("proposed_folder")
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count", ascending=False)
        .reset_index(drop=True)
    )
    folder_counts_df.to_csv(tables_dir / "final_folder_counts.csv", index=False)

    operation_counts_df = (
        materialization_result_df.groupby(["status", "operation_used"]).size().reset_index(name="count")
    )
    operation_counts_df.to_csv(tables_dir / "operation_counts.csv", index=False)

    # Consistency checks
    checks = {
        "materialization_plan_exists": plan_path.exists(),
        "materialization_result_exists": result_path.exists(),
        "validation_exists": validation_path.exists(),
        "all_files_valid": bool(validation_df["is_valid"].all()),
        "output_count_matches_input": bool(len(materialization_result_df) == len(assignment_df)),
    }
    with (reports_dir / "consistency_checks.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe_value(checks), f, ensure_ascii=False, indent=2, allow_nan=False)

    summary = {
        "run_id": run_id,
        "created_at": now_iso(),
        "project_dir": project_dir.as_posix(),
        "previous_run_dir": previous_run_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "final_output_dir": final_output_dir.as_posix(),
        "materialization_mode": materialization_mode,
        "total_files": len(materialization_plan_df),
        "total_folders": len(unique_folders),
        "successful_operations": int(materialization_result_df["status"].eq("ok").sum()),
        "skipped_operations": int(materialization_result_df["status"].eq("skipped_existing").sum()),
        "failed_operations": int(materialization_result_df["status"].eq("error").sum()),
        "outputs": {
            "final_output_dir": final_output_dir.as_posix(),
            "materialization_plan": plan_path.as_posix(),
            "materialization_result": result_path.as_posix(),
            "materialization_validation": validation_path.as_posix(),
            "global_index": global_index_path.as_posix(),
        },
    }
    summary_path = reports_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe_value(summary), f, ensure_ascii=False, indent=2, allow_nan=False)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "final_output_dir": final_output_dir,
        "plan_df": materialization_plan_df,
        "result_df": materialization_result_df,
        "validation_df": validation_df,
        "folder_counts_df": folder_counts_df,
        "operation_counts_df": operation_counts_df,
        "summary": summary,
        "summary_path": summary_path,
    }
