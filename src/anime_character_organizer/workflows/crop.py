"""
Workflow orchestrator for Stage 2: Representative set selection and crop preparation.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..exceptions import ConfigurationError, RunNotFoundError
from ..preprocessing.cropping import add_review_flags, process_image_for_crop
from ..utils.paths import normalize_path_string
from ..utils.runs import find_latest_run
from ..utils.serialization import safe_read_csv
from ..utils.time import now_iso
from ..visualization.contact_sheet import create_contact_sheet


def run_crop_preparation(
    project_dir: Union[str, Path] = "./anime_character_pipeline",
    previous_run_dir: Optional[Union[str, Path]] = None,
    use_perceptual_representatives: bool = False,
    head_conf_threshold: float = 0.35,
    head_iou_threshold: float = 0.50,
    person_conf_threshold: float = 0.30,
    person_iou_threshold: float = 0.50,
    head_padding_ratio: float = 0.25,
    person_padding_ratio: float = 0.10,
    crop_format: str = "JPEG",
    crop_quality: int = 95,
    crop_output_size: Optional[Tuple[int, int]] = (512, 512),
    min_crop_side: int = 64,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Execute complete Stage 2 crop preparation workflow.
    """
    project_dir = Path(project_dir).expanduser().resolve()

    if previous_run_dir is None:
        previous_run_dir = find_latest_run(
            project_dir=project_dir,
            stage_prefix="01_dataset_audit_*",
            required_relative_paths=[Path("tables") / "valid_images.csv"],
        )
    else:
        previous_run_dir = Path(previous_run_dir).expanduser().resolve()

    valid_images_path = previous_run_dir / "tables" / "valid_images.csv"
    if not valid_images_path.exists():
        raise ConfigurationError(f"Required valid images file missing: {valid_images_path}")

    valid_df = pd.read_csv(valid_images_path)
    exact_duplicates_path = previous_run_dir / "tables" / "exact_duplicate_groups.csv"
    exact_duplicates_df = safe_read_csv(exact_duplicates_path)

    perceptual_groups_path = previous_run_dir / "tables" / "perceptual_duplicate_groups.csv"
    perceptual_groups_df = safe_read_csv(perceptual_groups_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_dir / "runs" / f"02_crop_preparation_{run_id}"
    reports_dir = run_dir / "reports"
    tables_dir = run_dir / "tables"
    logs_dir = run_dir / "logs"
    contact_sheets_dir = run_dir / "contact_sheets"
    crops_dir = run_dir / "crops" / "crops"

    for d in [run_dir, reports_dir, tables_dir, logs_dir, contact_sheets_dir, crops_dir]:
        d.mkdir(parents=True, exist_ok=True)

    working_df = valid_df.copy()
    working_df["path"] = working_df["path"].map(normalize_path_string)

    excluded_exact_paths = set()
    if not exact_duplicates_df.empty and "is_representative" in exact_duplicates_df.columns:
        exact_df = exact_duplicates_df.copy()
        exact_df["path"] = exact_df["path"].map(normalize_path_string)
        non_rep_exact = exact_df[~exact_df["is_representative"].astype(bool)]
        excluded_exact_paths = set(non_rep_exact["path"].tolist())

    excluded_perceptual_paths = set()
    if use_perceptual_representatives and not perceptual_groups_df.empty and "is_representative" in perceptual_groups_df.columns:
        perceptual_df = perceptual_groups_df.copy()
        perceptual_df["path"] = perceptual_df["path"].map(normalize_path_string)
        non_rep_perceptual = perceptual_df[~perceptual_df["is_representative"].astype(bool)]
        excluded_perceptual_paths = set(non_rep_perceptual["path"].tolist())

    working_df["excluded_exact_duplicate"] = working_df["path"].isin(excluded_exact_paths)
    working_df["excluded_perceptual_duplicate"] = working_df["path"].isin(excluded_perceptual_paths)
    working_df["selected_for_detection"] = ~working_df["excluded_exact_duplicate"] & ~working_df["excluded_perceptual_duplicate"]

    selected_df = working_df[working_df["selected_for_detection"]].copy().reset_index(drop=True)
    selected_df.insert(0, "image_index", np.arange(len(selected_df), dtype=int))

    if selected_df.empty:
        raise RuntimeError("No images were selected for detection.")

    if max_workers is None:
        max_workers = max(1, min(8, (os.cpu_count() or 2)))

    crop_records: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_image_for_crop,
                row=dict(row),
                crops_dir=crops_dir,
                head_conf=head_conf_threshold,
                head_iou=head_iou_threshold,
                person_conf=person_conf_threshold,
                person_iou=person_iou_threshold,
                head_padding=head_padding_ratio,
                person_padding=person_padding_ratio,
                crop_format=crop_format,
                crop_quality=crop_quality,
                crop_output_size=crop_output_size,
                min_crop_side=min_crop_side,
            ): row["image_index"]
            for _, row in selected_df.iterrows()
        }

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Extracting character crops")

        for future in iterator:
            try:
                crop_records.append(future.result())
            except Exception as exc:
                idx = futures[future]
                crop_records.append({
                    "image_index": idx,
                    "crop_id": f"img_{idx:07d}_error",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "processed_at": now_iso(),
                })

    crops_df = pd.DataFrame(crop_records).sort_values("image_index").reset_index(drop=True)
    review_df = add_review_flags(crops_df)

    region_counts = review_df["selected_region_type"].value_counts(dropna=False).reset_index()
    region_counts.columns = ["selected_region_type", "count"]

    review_counts = review_df["review_reason"].value_counts(dropna=False).reset_index()
    review_counts.columns = ["review_reason", "count"]

    crops_manifest_path = tables_dir / "crop_manifest.csv"
    review_manifest_path = tables_dir / "crop_manifest_with_review_flags.csv"
    region_counts_path = tables_dir / "selected_region_counts.csv"
    review_counts_path = tables_dir / "review_reason_counts.csv"
    selection_path = tables_dir / "selected_images_for_detection.csv"
    working_path = tables_dir / "valid_images_with_selection_flags.csv"

    crops_df.to_csv(crops_manifest_path, index=False)
    review_df.to_csv(review_manifest_path, index=False)
    region_counts.to_csv(region_counts_path, index=False)
    review_counts.to_csv(review_counts_path, index=False)
    selected_df.to_csv(selection_path, index=False)
    working_df.to_csv(working_path, index=False)

    # Contact sheets
    contact_sheet_rows = []
    ok_review_df = review_df[review_df["status"].eq("ok")].copy()
    groups_for_sheets = []

    for region_type, group in ok_review_df.groupby("selected_region_type", dropna=False):
        groups_for_sheets.append((f"region_{region_type}", group))

    needs_review_group = ok_review_df[ok_review_df["needs_review"].astype(bool)].copy()
    if not needs_review_group.empty:
        groups_for_sheets.append(("needs_review", needs_review_group))

    multi_head_group = ok_review_df[ok_review_df["is_multi_head"].astype(bool)].copy()
    if not multi_head_group.empty:
        groups_for_sheets.append(("multiple_heads_detected", multi_head_group))

    for sheet_name, group in groups_for_sheets[:10]:
        group_sample = group.head(25)
        image_paths = group_sample["crop_path"].dropna().tolist()
        output_path = contact_sheets_dir / f"{sheet_name}.jpg"
        created = create_contact_sheet(
            image_paths=image_paths,
            output_path=output_path,
            thumb_size=180,
            columns=5,
        )
        contact_sheet_rows.append({
            "sheet_name": sheet_name,
            "image_count": len(image_paths),
            "created": bool(created),
            "path": output_path.as_posix() if created else None,
        })

    contact_sheets_df = pd.DataFrame(contact_sheet_rows)
    contact_sheets_path = tables_dir / "contact_sheets.csv"
    contact_sheets_df.to_csv(contact_sheets_path, index=False)

    existing_crop_count = int(review_df["crop_path"].dropna().map(lambda p: Path(p).exists()).sum())
    expected_crop_count = int(review_df["status"].eq("ok").sum())

    verification = {
        "expected_successful_crops": expected_crop_count,
        "existing_crop_files": existing_crop_count,
        "all_successful_crops_exist": bool(existing_crop_count == expected_crop_count),
    }
    with (reports_dir / "crop_file_verification.json").open("w", encoding="utf-8") as f:
        json.dump(verification, f, ensure_ascii=False, indent=2)

    summary = {
        "run_id": run_id,
        "created_at": now_iso(),
        "project_dir": project_dir.as_posix(),
        "previous_run_dir": previous_run_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "use_perceptual_representatives": bool(use_perceptual_representatives),
        "valid_images_loaded": int(len(valid_df)),
        "selected_for_detection": int(len(selected_df)),
        "successful_crops": int(crops_df["status"].eq("ok").sum()),
        "crop_errors": int(crops_df["status"].ne("ok").sum()),
        "head_conf_threshold": float(head_conf_threshold),
        "head_iou_threshold": float(head_iou_threshold),
        "person_conf_threshold": float(person_conf_threshold),
        "person_iou_threshold": float(person_iou_threshold),
        "head_padding_ratio": float(head_padding_ratio),
        "person_padding_ratio": float(person_padding_ratio),
        "crop_format": crop_format,
        "crop_quality": int(crop_quality),
        "crop_output_size": list(crop_output_size) if crop_output_size else None,
        "python": sys.version,
        "platform": platform.platform(),
        "outputs": {
            "selected_images_for_detection": selection_path.as_posix(),
            "valid_images_with_selection_flags": working_path.as_posix(),
            "crop_manifest": crops_manifest_path.as_posix(),
            "crop_manifest_with_review_flags": review_manifest_path.as_posix(),
            "selected_region_counts": region_counts_path.as_posix(),
            "review_reason_counts": review_counts_path.as_posix(),
            "crops_dir": crops_dir.as_posix(),
        },
    }

    summary_path = reports_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "crops_df": crops_df,
        "review_df": review_df,
        "region_counts": region_counts,
        "review_counts": review_counts,
        "contact_sheets_df": contact_sheets_df,
        "summary": summary,
        "summary_path": summary_path,
    }
