"""
Workflow orchestrator for Stage 5: character naming with the anime tagger.
"""

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..config import validate_parameters
from ..exceptions import ConfigurationError, InvalidImageError, TaggingError
from ..tagging.aggregation import aggregate_cluster_tags
from ..tagging.extraction import TAGGER_NAME, TAGGER_REVISION, extract_character_tags, load_character_tagger
from ..tagging.renaming import REVIEW_FOLDER, build_named_assignment
from ..tagging.sampling import select_cluster_samples
from ..utils.paths import checked_path, normalize_path_string
from ..utils.runs import create_run_directory, find_latest_run
from ..utils.serialization import json_safe_value, require_columns, safe_bool, safe_json_dumps
from ..utils.time import now_iso


def tag_crops(rows: pd.DataFrame, show_progress: bool = True, desc: str = "Tagging crops") -> List[Dict[str, Any]]:
    """Tag each crop; per-image failures are recorded instead of aborting the run."""
    records = []
    iterator = tqdm(rows.iterrows(), total=len(rows), desc=desc) if show_progress else rows.iterrows()
    for _, row in iterator:
        record = {
            "embedding_row": int(row["embedding_row"]),
            "cluster_label": int(row["cluster_label"]),
            "relative_path": row.get("relative_path"),
            "crop_path": row["crop_path"],
            "source_path": row["source_path"],
        }
        try:
            tags = extract_character_tags(row["crop_path"])
        except Exception as exc:
            record |= {"tagging_status": "error", "error_type": type(exc).__name__, "error_message": str(exc)}
            tags = {}
        else:
            record |= {"tagging_status": "ok", "error_type": None, "error_message": None}
        top_tag, top_score = next(iter(tags.items()), (None, np.nan))
        record |= {
            "tagger_used": TAGGER_NAME,
            "character_tags_json": safe_json_dumps(tags),
            "top_character_tag": top_tag,
            "top_character_score": top_score,
            "processed_at": now_iso(),
        }
        records.append(record)
    return records


def run_cluster_naming(
    project_dir: Union[str, Path] = "./anime_character_pipeline",
    previous_run_dir: Optional[Union[str, Path]] = None,
    max_images_per_cluster_to_tag: int = 12,
    min_character_score: float = 0.70,
    min_name_share: float = 0.35,
    min_weighted_score: float = 0.55,
    min_top_margin: float = 0.08,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Name every image after the character it shows.

    Representative crops of each cluster vote for a cluster name. Noise images and the remaining
    images of clusters without an accepted name are then tagged individually. Images without a
    confident name are assigned to the review folder.
    """
    validate_parameters(**locals())
    project_dir = checked_path(project_dir)

    if previous_run_dir is None:
        previous_run_dir = find_latest_run(
            project_dir=project_dir,
            stage_prefix="04_hdbscan_clustering_*",
            required_relative_paths=[Path("reports") / "summary.json", Path("tables") / "cluster_manifest.csv"],
        )
    else:
        previous_run_dir = checked_path(previous_run_dir)

    cluster_manifest_path = checked_path(previous_run_dir / "tables" / "cluster_manifest.csv")
    if not cluster_manifest_path.exists():
        raise ConfigurationError(f"Clustering run is missing {cluster_manifest_path}.")
    cluster_df = pd.read_csv(cluster_manifest_path)
    require_columns(
        cluster_df,
        {"cluster_label", "crop_path", "source_path", "embedding_row", "requires_manual_review"},
        "Cluster manifest",
    )
    cluster_df["crop_path"] = cluster_df["crop_path"].map(normalize_path_string)
    cluster_df["source_path"] = cluster_df["source_path"].map(normalize_path_string)
    cluster_df["requires_manual_review"] = cluster_df["requires_manual_review"].map(safe_bool)
    cluster_df["cluster_label"] = cluster_df["cluster_label"].astype(int)
    cluster_df["crop_exists"] = cluster_df["crop_path"].map(lambda p: Path(p).is_file())
    if not cluster_df["crop_exists"].any():
        raise InvalidImageError("No existing crop files available for tagging. Run crop preparation again.")

    try:
        load_character_tagger()
    except Exception as exc:
        raise TaggingError(f"Could not load {TAGGER_NAME}: {exc}") from exc

    run_dir = create_run_directory(project_dir, "05_cluster_naming")
    run_id = run_dir.name.removeprefix("05_cluster_naming_")
    reports_dir, tables_dir = run_dir / "reports", run_dir / "tables"
    for d in [reports_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    taggable = cluster_df[cluster_df["crop_exists"]]
    clustered = taggable[taggable["cluster_label"].ne(-1)]
    samples = [
        select_cluster_samples(group, max_images=max_images_per_cluster_to_tag)
        for _, group in clustered.groupby("cluster_label")
    ]
    sample_df = pd.concat(samples) if samples else clustered
    records = tag_crops(sample_df, show_progress, "Tagging cluster samples")

    cluster_name_suggestions_df = aggregate_cluster_tags(
        pd.DataFrame(records, columns=["cluster_label", "tagging_status", "character_tags_json", "relative_path"]),
        min_character_score=min_character_score,
        min_name_share=min_name_share,
        min_weighted_score=min_weighted_score,
        min_top_margin=min_top_margin,
    )
    named_labels = (
        set(cluster_name_suggestions_df.loc[cluster_name_suggestions_df["accepted_name"], "cluster_label"])
        if not cluster_name_suggestions_df.empty
        else set()
    )
    remaining = taggable[
        ~taggable["cluster_label"].isin(named_labels) & ~taggable["embedding_row"].isin(sample_df["embedding_row"])
    ]
    records += tag_crops(remaining, show_progress, "Tagging unnamed images")

    tagging_manifest_df = pd.DataFrame(records)
    tagging_manifest_path = tables_dir / "tagging_manifest.csv"
    tagging_manifest_df.to_csv(tagging_manifest_path, index=False)
    if tagging_manifest_df["tagging_status"].ne("ok").all():
        raise TaggingError(f"All tagger calls failed; inspect {tagging_manifest_path}.")

    cluster_name_suggestions_path = tables_dir / "cluster_name_suggestions.csv"
    cluster_name_suggestions_df.to_csv(cluster_name_suggestions_path, index=False)

    named_assignment_df = build_named_assignment(
        cluster_df.drop(columns=["crop_exists"]),
        cluster_name_suggestions_df,
        tagging_manifest_df,
        min_character_score=min_character_score,
        min_top_margin=min_top_margin,
    )
    named_assignment_path = tables_dir / "named_folder_assignment.csv"
    named_assignment_df.to_csv(named_assignment_path, index=False)

    folder_preview_df = (
        named_assignment_df.groupby(["named_folder", "naming_source"])
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count", ascending=False)
        .reset_index(drop=True)
    )
    folder_preview_df.to_csv(tables_dir / "named_folder_preview.csv", index=False)

    source_counts = named_assignment_df["naming_source"].value_counts()
    summary = {
        "run_id": run_id,
        "created_at": now_iso(),
        "project_dir": project_dir.as_posix(),
        "previous_run_dir": previous_run_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "tagger": {"name": TAGGER_NAME, "revision": TAGGER_REVISION},
        "clusters_evaluated": len(cluster_name_suggestions_df),
        "clusters_named": len(named_labels),
        "character_folders": int(folder_preview_df["named_folder"].drop_duplicates().ne(REVIEW_FOLDER).sum()),
        "images_named_by_cluster_vote": int(source_counts.get("cluster_vote", 0)),
        "images_named_by_image_tags": int(source_counts.get("image_tags", 0)),
        "images_needing_review": int(source_counts.get("unidentified", 0)),
        "total_tagged_images": len(tagging_manifest_df),
        "tagging_errors": int(tagging_manifest_df["tagging_status"].ne("ok").sum()),
        "python": sys.version,
        "platform": platform.platform(),
        "outputs": {
            "tagging_manifest": tagging_manifest_path.as_posix(),
            "cluster_name_suggestions": cluster_name_suggestions_path.as_posix(),
            "named_folder_assignment": named_assignment_path.as_posix(),
        },
    }
    summary_path = reports_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe_value(summary), f, ensure_ascii=False, indent=2, allow_nan=False)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "cluster_name_suggestions_df": cluster_name_suggestions_df,
        "tagging_manifest_df": tagging_manifest_df,
        "named_assignment_df": named_assignment_df,
        "folder_preview_df": folder_preview_df,
        "summary": summary,
        "summary_path": summary_path,
    }
