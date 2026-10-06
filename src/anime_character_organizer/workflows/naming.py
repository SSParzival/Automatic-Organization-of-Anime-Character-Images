"""
Workflow orchestrator for Stage 6: Semantic cluster naming with anime image taggers.
"""

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..config import validate_parameters
from ..exceptions import ConfigurationError, InvalidImageError, MaterializationError, TaggingError
from ..materialization.operations import materialize_one_file
from ..materialization.planner import build_materialization_plan
from ..materialization.validation import validate_materialized_files
from ..tagging.aggregation import aggregate_cluster_tags
from ..tagging.extraction import extract_tags_with_fallback
from ..tagging.renaming import build_folder_rename_plan
from ..tagging.sampling import select_cluster_samples
from ..utils.naming import sanitize_filename_component
from ..utils.paths import checked_path, normalize_path_string
from ..utils.runs import create_run_directory, find_latest_run
from ..utils.serialization import safe_bool, safe_json_dumps
from ..utils.time import now_iso
from ..utils.validation import validate_image_file
from ..visualization.contact_sheet import create_contact_sheet


def run_cluster_naming(
    project_dir: Union[str, Path] = "./anime_character_pipeline",
    previous_clustering_run_dir: Optional[Union[str, Path]] = None,
    previous_materialization_run_dir: Optional[Union[str, Path]] = None,
    primary_tagger: str = "pixai",
    pixai_model_name: str = "v0.9",
    wd14_model_name: str = "SwinV2_v3",
    use_wd14_fallback: bool = True,
    min_cluster_size_to_name: int = 2,
    max_images_per_cluster_to_tag: int = 12,
    min_character_score: float = 0.70,
    min_name_share: float = 0.35,
    min_weighted_score: float = 0.55,
    min_top_margin: float = 0.08,
    ignore_review_folders_for_naming: bool = False,
    tag_only_non_noise_clusters: bool = True,
    create_rename_plan: bool = True,
    create_named_output: bool = True,
    named_output_mode: str = "hardlink",
    allow_hardlink_fallback_to_copy: bool = True,
    on_existing: str = "error",
    named_output_dir: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Execute complete Stage 6 anime tagging, cluster name suggestion, and optional named materialization workflow.
    """
    validate_parameters(**locals())
    project_dir = checked_path(project_dir)
    named_output_dir = checked_path(named_output_dir) if named_output_dir is not None else None

    if previous_clustering_run_dir is None:
        previous_clustering_run_dir = find_latest_run(
            project_dir=project_dir,
            stage_prefix="04_hdbscan_clustering_*",
            required_relative_paths=[
                Path("reports") / "summary.json",
                Path("tables") / "cluster_manifest.csv",
                Path("tables") / "cluster_summary.csv",
                Path("tables") / "folder_assignment_manifest.csv",
            ],
        )
    else:
        previous_clustering_run_dir = checked_path(previous_clustering_run_dir)

    cluster_manifest_path = checked_path(previous_clustering_run_dir / "tables" / "cluster_manifest.csv")
    folder_assignment_path = checked_path(previous_clustering_run_dir / "tables" / "folder_assignment_manifest.csv")

    if not cluster_manifest_path.exists() or not folder_assignment_path.exists():
        raise ConfigurationError("Clustering run missing required tables.")

    cluster_manifest_df = pd.read_csv(cluster_manifest_path)
    folder_assignment_df = pd.read_csv(folder_assignment_path)

    run_dir = create_run_directory(project_dir, "06_cluster_naming")
    run_id = run_dir.name.removeprefix("06_cluster_naming_")
    reports_dir = run_dir / "reports"
    tables_dir = run_dir / "tables"
    logs_dir = run_dir / "logs"
    contact_sheets_dir = run_dir / "contact_sheets"
    naming_metadata_dir = run_dir / "cluster_naming_metadata"

    for d in [run_dir, reports_dir, tables_dir, logs_dir, contact_sheets_dir, naming_metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if create_named_output:
        if named_output_dir is None:
            named_output_dir = project_dir / "organized_output" / f"anime_named_{run_id}"
        else:
            named_output_dir = checked_path(named_output_dir)
        named_output_dir = checked_path(named_output_dir)
        if named_output_dir.exists() and any(named_output_dir.iterdir()):
            raise ConfigurationError("Output directory must be new or empty to protect existing artifacts.")

        named_output_dir.mkdir(parents=True, exist_ok=True)

    cluster_work_df = cluster_manifest_df.copy()
    cluster_work_df["crop_path"] = cluster_work_df["crop_path"].map(normalize_path_string)
    cluster_work_df["source_path"] = cluster_work_df["source_path"].map(normalize_path_string)
    cluster_work_df["requires_manual_review"] = cluster_work_df["requires_manual_review"].map(safe_bool)
    cluster_work_df["cluster_label"] = cluster_work_df["cluster_label"].astype(int)

    cluster_work_df["crop_exists"] = cluster_work_df["crop_path"].map(lambda p: Path(p).exists())
    cluster_work_df["source_exists"] = cluster_work_df["source_path"].map(lambda p: Path(p).exists())

    if tag_only_non_noise_clusters:
        cluster_work_df = cluster_work_df[cluster_work_df["cluster_label"].ne(-1)].copy()

    if ignore_review_folders_for_naming:
        cluster_work_df = cluster_work_df[~cluster_work_df["requires_manual_review"]].copy()

    cluster_sizes = cluster_work_df.groupby("cluster_label").size().reset_index(name="cluster_size")
    eligible_labels = cluster_sizes[cluster_sizes["cluster_size"].ge(min_cluster_size_to_name)][
        "cluster_label"
    ].tolist()
    cluster_work_df = cluster_work_df[cluster_work_df["cluster_label"].isin(eligible_labels)].copy()

    if cluster_work_df.empty:
        raise ConfigurationError("No clusters eligible for naming under current settings.")

    # Select representative samples per cluster
    sample_rows = []
    for cluster_label, group in cluster_work_df.groupby("cluster_label", dropna=False):
        group = group[group["crop_exists"]].copy()
        if group.empty:
            continue
        selected = select_cluster_samples(group, max_images=max_images_per_cluster_to_tag)
        selected["selected_for_tagging"] = True
        sample_rows.append(selected)

    if not sample_rows:
        raise InvalidImageError("No existing crop files available in eligible clusters. Run crop preparation again.")

    tagging_input_df = pd.concat(sample_rows, axis=0).reset_index(drop=True)
    tagging_input_df.insert(0, "tagging_index", np.arange(len(tagging_input_df), dtype=int))
    tagging_input_path = tables_dir / "tagging_input_manifest.csv"
    tagging_input_df.to_csv(tagging_input_path, index=False)

    # Validate tagging inputs
    validation_rows = []
    val_iter = (
        tqdm(tagging_input_df.iterrows(), total=len(tagging_input_df), desc="Validating tagging images")
        if show_progress
        else tagging_input_df.iterrows()
    )
    for _, row in val_iter:
        is_valid, err_type, err_msg = validate_image_file(row["crop_path"])
        validation_rows.append(
            {
                "tagging_index": int(row["tagging_index"]),
                "cluster_label": int(row["cluster_label"]),
                "crop_path": row["crop_path"],
                "valid_for_tagging": bool(is_valid),
                "validation_error_type": err_type,
                "validation_error_message": err_msg,
            }
        )

    tagging_validation_df = pd.DataFrame(validation_rows)
    tagging_ready_df = tagging_input_df.merge(
        tagging_validation_df, on=["tagging_index", "cluster_label", "crop_path"], how="left"
    )
    tagging_ready_df = (
        tagging_ready_df[tagging_ready_df["valid_for_tagging"].astype(bool)].copy().reset_index(drop=True)
    )

    tagging_validation_df.to_csv(tables_dir / "tagging_validation.csv", index=False)
    tagging_ready_df.to_csv(tables_dir / "tagging_ready_manifest.csv", index=False)

    if tagging_ready_df.empty:
        raise InvalidImageError("No valid images available for tagging.")

    # Tag images
    tagging_rows = []
    tag_iter = (
        tqdm(tagging_ready_df.iterrows(), total=len(tagging_ready_df), desc="Tagging cluster crops")
        if show_progress
        else tagging_ready_df.iterrows()
    )

    for _, row in tag_iter:
        tags, fallback_err_type, fallback_err_msg = extract_tags_with_fallback(
            image_path=row["crop_path"],
            primary_tagger=primary_tagger,
            pixai_model=pixai_model_name,
            wd14_model=wd14_model_name,
            use_wd14_fallback=use_wd14_fallback,
        )

        if tags is None:
            tagging_rows.append(
                {
                    "tagging_index": int(row["tagging_index"]),
                    "embedding_row": int(row["embedding_row"]),
                    "cluster_label": int(row["cluster_label"]),
                    "cluster_folder_stub": row["cluster_folder_stub"],
                    "relative_path": row.get("relative_path"),
                    "crop_path": row["crop_path"],
                    "source_path": row["source_path"],
                    "tagging_status": "error",
                    "tagger_used": None,
                    "error_type": fallback_err_type,
                    "error_message": fallback_err_msg,
                    "character_tags_json": "{}",
                    "general_tags_json": "{}",
                    "top_character_tag": None,
                    "top_character_score": np.nan,
                    "processed_at": now_iso(),
                }
            )
            continue

        char_tags = tags.get("character_tags", {}) or {}
        gen_tags = tags.get("general_tags", {}) or {}
        filtered_char_tags = {k: v for k, v in char_tags.items() if float(v) >= min_character_score}

        top_char_tag, top_char_score = (
            max(filtered_char_tags.items(), key=lambda x: x[1]) if filtered_char_tags else (None, np.nan)
        )
        top_gen_tag, top_gen_score = max(gen_tags.items(), key=lambda x: x[1]) if gen_tags else (None, np.nan)

        tagging_rows.append(
            {
                "tagging_index": int(row["tagging_index"]),
                "embedding_row": int(row["embedding_row"]),
                "cluster_label": int(row["cluster_label"]),
                "cluster_folder_stub": row["cluster_folder_stub"],
                "relative_path": row.get("relative_path"),
                "crop_path": row["crop_path"],
                "source_path": row["source_path"],
                "tagging_status": "ok",
                "tagger_used": tags.get("tagger"),
                "error_type": None,
                "error_message": None,
                "character_tags_json": safe_json_dumps(filtered_char_tags),
                "general_tags_json": safe_json_dumps(gen_tags),
                "top_character_tag": top_char_tag,
                "top_character_score": float(top_char_score) if pd.notna(top_char_score) else np.nan,
                "top_general_tag": top_gen_tag,
                "top_general_score": float(top_gen_score) if pd.notna(top_gen_score) else np.nan,
                "processed_at": now_iso(),
            }
        )

    tagging_manifest_df = pd.DataFrame(tagging_rows)
    tagging_manifest_path = tables_dir / "tagging_manifest.csv"
    tagging_manifest_df.to_csv(tagging_manifest_path, index=False)

    if tagging_manifest_df["tagging_status"].ne("ok").all():
        raise TaggingError(f"All tagger calls failed; inspect {tagging_manifest_path}.")

    # Aggregate character tags by cluster
    cluster_name_suggestions_df = aggregate_cluster_tags(
        tagging_manifest_df=tagging_manifest_df,
        min_character_score=min_character_score,
        min_name_share=min_name_share,
        min_weighted_score=min_weighted_score,
        min_top_margin=min_top_margin,
    )
    cluster_name_suggestions_path = tables_dir / "cluster_name_suggestions.csv"
    cluster_name_suggestions_df.to_csv(cluster_name_suggestions_path, index=False)

    # Folder rename plan
    folder_rename_plan_df = pd.DataFrame()
    folder_rename_preview_df = pd.DataFrame()
    if create_rename_plan:
        folder_rename_plan_df, folder_rename_preview_df = build_folder_rename_plan(
            folder_assignment_df=folder_assignment_df,
            cluster_name_suggestions_df=cluster_name_suggestions_df,
        )
        folder_rename_plan_df.to_csv(tables_dir / "folder_rename_plan.csv", index=False)
        folder_rename_preview_df.to_csv(tables_dir / "folder_rename_preview.csv", index=False)

    # Cluster naming metadata files
    metadata_rows = []
    for _, srow in cluster_name_suggestions_df.iterrows():
        lbl = int(srow["cluster_label"])
        stub = srow["cluster_folder_stub"]
        meta = {
            "cluster_label": lbl,
            "cluster_folder_stub": stub,
            "accepted_name": bool(srow["accepted_name"]),
            "suggested_character_tag": srow["suggested_character_tag"],
            "proposed_named_folder": srow["proposed_named_folder"],
            "tagged_image_count": int(srow["tagged_image_count"]),
            "tagging_error_count": int(srow["tagging_error_count"]),
            "unique_candidate_character_tags": int(srow["unique_candidate_character_tags"]),
            "top_tags": json.loads(srow["top_tags_json"]) if pd.notna(srow["top_tags_json"]) else [],
            "tagger": {
                "primary": primary_tagger,
                "pixai_model_name": pixai_model_name,
                "wd14_model_name": wd14_model_name,
            },
            "created_at": now_iso(),
        }
        meta_p = naming_metadata_dir / f"{sanitize_filename_component(stub)}.json"
        with meta_p.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        metadata_rows.append({"cluster_label": lbl, "cluster_folder_stub": stub, "metadata_path": meta_p.as_posix()})

    metadata_index_path = tables_dir / "cluster_naming_metadata_index.csv"
    metadata_index_df = pd.DataFrame(metadata_rows)
    metadata_index_df.to_csv(metadata_index_path, index=False)

    # Naming contact sheets
    contact_sheet_rows = []
    for _, srow in cluster_name_suggestions_df.head(250).iterrows():
        lbl = int(srow["cluster_label"])
        stub = srow["cluster_folder_stub"]
        pname = srow["proposed_named_folder"]
        grp = cluster_work_df[cluster_work_df["cluster_label"].eq(lbl)]
        grp = select_cluster_samples(grp, max_images=24)
        cpaths = grp["crop_path"].dropna().tolist()
        out_sheet = contact_sheets_dir / f"{sanitize_filename_component(pname, fallback=stub)}.jpg"
        title = f"{pname} | accepted={bool(srow['accepted_name'])} | score={srow['best_weighted_score']:.3f}"
        created = create_contact_sheet(cpaths, out_sheet, title=title)
        contact_sheet_rows.append(
            {
                "cluster_label": lbl,
                "cluster_folder_stub": stub,
                "proposed_named_folder": pname,
                "accepted_name": bool(srow["accepted_name"]),
                "image_count": len(cpaths),
                "created": bool(created),
                "contact_sheet_path": out_sheet.as_posix() if created else None,
            }
        )

    naming_sheets_df = pd.DataFrame(contact_sheet_rows)
    naming_sheets_df.to_csv(tables_dir / "naming_contact_sheets.csv", index=False)

    # Named materialization if requested
    named_materialization_df = pd.DataFrame()
    if create_named_output and not folder_rename_plan_df.empty:
        if named_output_dir is None:
            raise ConfigurationError("Named output directory was not initialized.")
        named_plan = build_materialization_plan(
            folder_rename_plan_df, named_output_dir, folder_column="named_folder", stem_max_length=140
        )
        named_plan.to_csv(tables_dir / "named_materialization_plan.csv", index=False)
        named_plan_rows = []
        for _, row in named_plan.iterrows():
            src_p = Path(row["source_path"])
            folder_n = row["named_folder"]
            dest_p = Path(row["destination_path"])
            res = materialize_one_file(
                source_path=src_p,
                destination_path=dest_p,
                mode=named_output_mode,
                on_existing=on_existing,
                allow_hardlink_fallback_to_copy=allow_hardlink_fallback_to_copy,
            )
            named_plan_rows.append(
                {
                    "embedding_row": row["embedding_row"],
                    "source_path": src_p.as_posix(),
                    "named_destination_path": dest_p.as_posix(),
                    "named_folder": folder_n,
                    "cluster_label": row["cluster_label"],
                    "accepted_name": row["accepted_name"],
                    "suggested_character_tag": row.get("suggested_character_tag"),
                    "status": res["status"],
                    "operation_used": res["operation_used"],
                    "error_type": res["error_type"],
                    "error_message": res["error_message"],
                    "processed_at": now_iso(),
                }
            )

        named_materialization_df = pd.DataFrame(named_plan_rows)
        named_materialization_df.to_csv(tables_dir / "named_materialization_result.csv", index=False)
        named_validation = validate_materialized_files(
            named_materialization_df.rename(columns={"named_destination_path": "destination_path"})
        )
        named_validation.to_csv(tables_dir / "named_materialization_validation.csv", index=False)
        if not named_validation["is_valid"].all():
            raise MaterializationError(
                f"Named output is incomplete; inspect {tables_dir / 'named_materialization_validation.csv'}."
            )

    summary = {
        "run_id": run_id,
        "created_at": now_iso(),
        "project_dir": project_dir.as_posix(),
        "previous_clustering_run_dir": previous_clustering_run_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "primary_tagger": primary_tagger,
        "total_clusters_evaluated": len(cluster_name_suggestions_df),
        "candidate_clusters": len(cluster_name_suggestions_df),
        "accepted_character_names": int(cluster_name_suggestions_df["accepted_name"].sum())
        if not cluster_name_suggestions_df.empty
        else 0,
        "named_clusters": int(cluster_name_suggestions_df["accepted_name"].sum())
        if not cluster_name_suggestions_df.empty
        else 0,
        "unnamed_clusters": len(cluster_name_suggestions_df)
        - (int(cluster_name_suggestions_df["accepted_name"].sum()) if not cluster_name_suggestions_df.empty else 0),
        "total_tagged_images": len(tagging_manifest_df),
        "tagging_errors": int(tagging_manifest_df["tagging_status"].ne("ok").sum()),
        "create_named_output": create_named_output,
        "named_output_dir": named_output_dir.as_posix() if named_output_dir else None,
        "python": sys.version,
        "platform": platform.platform(),
        "outputs": {
            "tagging_manifest": tagging_manifest_path.as_posix(),
            "cluster_name_suggestions": cluster_name_suggestions_path.as_posix(),
            "folder_rename_plan": (tables_dir / "folder_rename_plan.csv").as_posix(),
            "folder_rename_preview": (tables_dir / "folder_rename_preview.csv").as_posix(),
            "naming_contact_sheets": (tables_dir / "naming_contact_sheets.csv").as_posix(),
            "cluster_naming_metadata_index": metadata_index_path.as_posix(),
        },
    }

    summary_path = reports_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "cluster_name_suggestions_df": cluster_name_suggestions_df,
        "name_suggestions_df": cluster_name_suggestions_df,
        "tagging_manifest_df": tagging_manifest_df,
        "folder_rename_plan_df": folder_rename_plan_df,
        "folder_rename_preview_df": folder_rename_preview_df,
        "naming_sheets_df": naming_sheets_df,
        "named_materialization_df": named_materialization_df,
        "named_output_dir": named_output_dir,
        "summary": summary,
        "summary_path": summary_path,
    }
