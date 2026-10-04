"""
Workflow orchestrator for Stage 3: CCIP embedding extraction and L2 normalization.
"""

from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..embeddings.extraction import extract_all_embeddings, warmup_ccip_model
from ..embeddings.normalization import l2_normalize_matrix
from ..exceptions import ConfigurationError, RunNotFoundError
from ..utils.paths import normalize_path_string
from ..utils.runs import find_latest_run
from ..utils.serialization import safe_bool_series
from ..utils.time import now_iso
from ..utils.validation import validate_image_file


def run_embedding_extraction(
    project_dir: Union[str, Path] = "./anime_character_pipeline",
    previous_run_dir: Optional[Union[str, Path]] = None,
    ccip_model: str = "ccip-caformer-24-randaug-pruned",
    ccip_image_size: int = 384,
    batch_size: int = 16,
    use_review_crops: bool = True,
    require_crop_file_exists: bool = True,
    normalize_embeddings: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Execute complete Stage 3 CCIP embedding extraction workflow.
    """
    project_dir = Path(project_dir).expanduser().resolve()

    if previous_run_dir is None:
        previous_run_dir = find_latest_run(
            project_dir=project_dir,
            stage_prefix="02_crop_preparation_*",
            required_relative_paths=[Path("tables") / "crop_manifest_with_review_flags.csv"],
        )
    else:
        previous_run_dir = Path(previous_run_dir).expanduser().resolve()

    crop_manifest_path = previous_run_dir / "tables" / "crop_manifest_with_review_flags.csv"
    if not crop_manifest_path.exists():
        raise ConfigurationError(f"Required crop manifest not found: {crop_manifest_path}")

    crop_df = pd.read_csv(crop_manifest_path)
    if "crop_path" not in crop_df.columns or "status" not in crop_df.columns:
        raise ConfigurationError("Crop manifest missing required columns 'crop_path' or 'status'.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_dir / "runs" / f"03_ccip_embeddings_{run_id}"
    reports_dir = run_dir / "reports"
    tables_dir = run_dir / "tables"
    arrays_dir = run_dir / "arrays"
    logs_dir = run_dir / "logs"

    for d in [run_dir, reports_dir, tables_dir, arrays_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    embedding_input_df = crop_df.copy()
    embedding_input_df["crop_path"] = embedding_input_df["crop_path"].map(normalize_path_string)

    if "needs_review" in embedding_input_df.columns:
        embedding_input_df["needs_review"] = safe_bool_series(embedding_input_df["needs_review"])
    else:
        embedding_input_df["needs_review"] = False

    embedding_input_df = embedding_input_df[embedding_input_df["status"].eq("ok")].copy()
    if not use_review_crops:
        embedding_input_df = embedding_input_df[~embedding_input_df["needs_review"]].copy()

    if require_crop_file_exists:
        embedding_input_df["crop_file_exists"] = embedding_input_df["crop_path"].map(lambda p: Path(p).exists())
        embedding_input_df = embedding_input_df[embedding_input_df["crop_file_exists"]].copy()

    embedding_input_df = embedding_input_df.reset_index(drop=True)
    embedding_input_df.insert(0, "embedding_index", np.arange(len(embedding_input_df), dtype=int))

    if embedding_input_df.empty:
        raise RuntimeError("No crops were selected for embedding extraction.")

    embedding_input_path = tables_dir / "embedding_input_manifest.csv"
    embedding_input_df.to_csv(embedding_input_path, index=False)

    # Validate crop files before inference
    validation_rows = []
    val_iter = tqdm(embedding_input_df.iterrows(), total=len(embedding_input_df), desc="Validating crop files") if show_progress else embedding_input_df.iterrows()
    for _, row in val_iter:
        is_valid, error_type, error_msg = validate_image_file(row["crop_path"])
        validation_rows.append({
            "embedding_index": int(row["embedding_index"]),
            "crop_path": row["crop_path"],
            "valid_for_embedding": bool(is_valid),
            "validation_error_type": error_type,
            "validation_error_message": error_msg,
        })

    crop_validation_df = pd.DataFrame(validation_rows)
    embedding_ready_df = embedding_input_df.merge(crop_validation_df, on=["embedding_index", "crop_path"], how="left")
    embedding_ready_df = embedding_ready_df[embedding_ready_df["valid_for_embedding"].astype(bool)].copy().reset_index(drop=True)
    embedding_ready_df["embedding_index"] = np.arange(len(embedding_ready_df), dtype=int)
    invalid_crop_val_df = crop_validation_df[~crop_validation_df["valid_for_embedding"].astype(bool)].copy()

    crop_validation_path = tables_dir / "crop_embedding_validation.csv"
    embedding_ready_path = tables_dir / "embedding_ready_manifest.csv"
    invalid_crop_val_path = tables_dir / "invalid_crops_for_embedding.csv"

    crop_validation_df.to_csv(crop_validation_path, index=False)
    embedding_ready_df.to_csv(embedding_ready_path, index=False)
    invalid_crop_val_df.to_csv(invalid_crop_val_path, index=False)

    if embedding_ready_df.empty:
        raise RuntimeError("No valid crop files available for CCIP extraction.")

    # Warmup
    warmup_ccip_model(embedding_ready_df.iloc[0]["crop_path"], model=ccip_model, size=ccip_image_size)

    # Extract all embeddings
    all_paths = embedding_ready_df["crop_path"].tolist()
    feature_records = extract_all_embeddings(
        crop_paths=all_paths,
        model=ccip_model,
        image_size=ccip_image_size,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    feature_status_df = pd.DataFrame([
        {
            "crop_path": rec["path"],
            "embedding_status": rec["status"],
            "embedding_error_type": rec["error_type"],
            "embedding_error_message": rec["error_message"],
        }
        for rec in feature_records
    ])

    ok_records = [r for r in feature_records if r["status"] == "ok"]
    if not ok_records:
        raise RuntimeError("All CCIP embedding extractions failed.")

    ok_paths = [r["path"] for r in ok_records]
    raw_embeddings = np.vstack([r["feature"] for r in ok_records]).astype(np.float32)

    if normalize_embeddings:
        normalized_embeddings = l2_normalize_matrix(raw_embeddings)
    else:
        normalized_embeddings = raw_embeddings.copy()

    successful_embedding_df = pd.DataFrame({
        "embedding_row": np.arange(len(ok_paths), dtype=int),
        "crop_path": ok_paths,
    })

    feature_status_clean = feature_status_df.drop_duplicates(subset=["crop_path"], keep="first").reset_index(drop=True)
    embedding_manifest_df = embedding_ready_df.merge(feature_status_clean, on="crop_path", how="left")
    embedding_manifest_df["embedding_status"] = embedding_manifest_df["embedding_status"].fillna("not_processed")

    successful_manifest_df = embedding_manifest_df[embedding_manifest_df["embedding_status"].eq("ok")].copy()
    successful_manifest_df = successful_manifest_df.merge(successful_embedding_df, on="crop_path", how="left")
    successful_manifest_df = successful_manifest_df.sort_values("embedding_row").reset_index(drop=True)
    successful_manifest_df["embedding_row"] = successful_manifest_df["embedding_row"].astype(int)

    raw_embeddings_path = arrays_dir / "ccip_embeddings_raw.npy"
    normalized_embeddings_path = arrays_dir / "ccip_embeddings_l2.npy"
    compressed_embeddings_path = arrays_dir / "ccip_embeddings_bundle.npz"

    embedding_manifest_path = tables_dir / "embedding_manifest.csv"
    successful_manifest_path = tables_dir / "successful_embedding_manifest.csv"
    feature_status_path = tables_dir / "embedding_status.csv"

    np.save(raw_embeddings_path, raw_embeddings)
    np.save(normalized_embeddings_path, normalized_embeddings)
    np.savez_compressed(
        compressed_embeddings_path,
        raw_embeddings=raw_embeddings,
        normalized_embeddings=normalized_embeddings,
    )

    embedding_manifest_df.to_csv(embedding_manifest_path, index=False)
    successful_manifest_df.to_csv(successful_manifest_path, index=False)
    feature_status_df.to_csv(feature_status_path, index=False)

    summary = {
        "run_id": run_id,
        "created_at": now_iso(),
        "project_dir": project_dir.as_posix(),
        "previous_run_dir": previous_run_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "ccip_model": ccip_model,
        "ccip_image_size": ccip_image_size,
        "batch_size": batch_size,
        "normalize_embeddings": normalize_embeddings,
        "crops_loaded": len(crop_df),
        "crops_selected": len(embedding_input_df),
        "crops_valid_for_embedding": len(embedding_ready_df),
        "successful_embeddings": len(ok_records),
        "failed_embeddings": len(feature_records) - len(ok_records),
        "embedding_dim": int(raw_embeddings.shape[1]),
        "python": sys.version,
        "platform": platform.platform(),
        "outputs": {
            "raw_embeddings": raw_embeddings_path.as_posix(),
            "normalized_embeddings": normalized_embeddings_path.as_posix(),
            "bundle": compressed_embeddings_path.as_posix(),
            "embedding_manifest": embedding_manifest_path.as_posix(),
            "successful_embedding_manifest": successful_manifest_path.as_posix(),
            "embedding_status": feature_status_path.as_posix(),
        },
    }

    summary_path = reports_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "raw_embeddings": raw_embeddings,
        "normalized_embeddings": normalized_embeddings,
        "embeddings": normalized_embeddings,
        "embedding_manifest_df": embedding_manifest_df,
        "successful_manifest_df": successful_manifest_df,
        "summary": summary,
        "summary_path": summary_path,
    }
