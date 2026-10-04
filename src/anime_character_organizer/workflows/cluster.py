"""
Workflow orchestrator for Stage 4: HDBSCAN clustering and review diagnostics.
"""

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..clustering.algorithm import fit_hdbscan
from ..clustering.centroids import compute_cluster_centroids, euclidean_distance_to_centroid
from ..clustering.diagnostics import (
    build_folder_assignment_manifest,
    calculate_cluster_summary,
    flag_cluster_review_items,
    sample_representative_and_boundary_rows,
)
from ..exceptions import ConfigurationError
from ..utils.runs import find_latest_run
from ..utils.time import now_iso
from ..visualization.contact_sheet import create_contact_sheet
from ..visualization.plots import plot_cluster_size_distribution, plot_probability_and_outlier_distributions


def run_clustering(
    project_dir: Union[str, Path] = "./anime_character_pipeline",
    previous_run_dir: Optional[Union[str, Path]] = None,
    embedding_file_name: str = "ccip_embeddings_l2.npy",
    manifest_file_name: str = "successful_embedding_manifest.csv",
    min_cluster_size: int = 2,
    min_samples: int = 1,
    cluster_selection_epsilon: float = 0.50,
    cluster_selection_method: str = "eom",
    metric: str = "euclidean",
    low_probability_threshold: float = 0.35,
    high_outlier_quantile: float = 0.95,
    reassign_noise: bool = True,
    max_reassign_distance: float = 0.55,
    separate_review_folders: bool = False,
    random_seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Execute complete Stage 4 HDBSCAN clustering workflow.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    np.random.seed(random_seed)

    if previous_run_dir is None:
        previous_run_dir = find_latest_run(
            project_dir=project_dir,
            stage_prefix="03_ccip_embeddings_*",
            required_relative_paths=[
                Path("arrays") / embedding_file_name,
                Path("tables") / manifest_file_name,
            ],
        )
    else:
        previous_run_dir = Path(previous_run_dir).expanduser().resolve()

    embeddings_path = previous_run_dir / "arrays" / embedding_file_name
    manifest_path = previous_run_dir / "tables" / manifest_file_name

    if not embeddings_path.exists():
        raise ConfigurationError(f"Embedding matrix not found: {embeddings_path}")
    if not manifest_path.exists():
        raise ConfigurationError(f"Embedding manifest not found: {manifest_path}")

    embedding_manifest_df = pd.read_csv(manifest_path)
    embeddings = np.load(embeddings_path).astype(np.float32)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_dir / "runs" / f"04_hdbscan_clustering_{run_id}"
    reports_dir = run_dir / "reports"
    tables_dir = run_dir / "tables"
    arrays_dir = run_dir / "arrays"
    figures_dir = run_dir / "figures"
    contact_sheets_dir = run_dir / "contact_sheets"
    cluster_metadata_dir = run_dir / "cluster_metadata"
    logs_dir = run_dir / "logs"

    for d in [
        run_dir,
        reports_dir,
        tables_dir,
        arrays_dir,
        figures_dir,
        contact_sheets_dir,
        cluster_metadata_dir,
        logs_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    labels, probabilities, outlier_scores, backend_used = fit_hdbscan(
        embeddings=embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )

    cluster_manifest_df = embedding_manifest_df.copy()
    cluster_manifest_df["cluster_label"] = labels
    cluster_manifest_df["cluster_probability"] = probabilities
    cluster_manifest_df["outlier_score"] = outlier_scores

    cluster_manifest_df, high_outlier_thresh = flag_cluster_review_items(
        cluster_manifest_df=cluster_manifest_df,
        low_prob_threshold=low_probability_threshold,
        high_outlier_quantile=high_outlier_quantile,
    )

    centroids = compute_cluster_centroids(embeddings, labels)
    distance_to_centroid = np.full(embeddings.shape[0], np.nan, dtype=np.float32)
    for label, centroid in centroids.items():
        idx = np.where(labels == label)[0]
        distance_to_centroid[idx] = euclidean_distance_to_centroid(embeddings[idx], centroid).astype(np.float32)

    cluster_manifest_df["distance_to_centroid"] = distance_to_centroid

    # Centroid-based soft reassignment for noise points within distance threshold
    reassigned_count = 0
    if reassign_noise and len(centroids) > 0:
        noise_idx = np.where(labels == -1)[0]
        if len(noise_idx) > 0:
            c_labels = np.array(list(centroids.keys()))
            c_matrix = np.stack([centroids[lbl] for lbl in c_labels])
            for n_i in noise_idx:
                emb = embeddings[n_i].reshape(1, -1)
                dists = np.linalg.norm(c_matrix - emb, axis=1)
                best_idx = int(np.argmin(dists))
                best_dist = float(dists[best_idx])
                if best_dist <= max_reassign_distance:
                    assigned_lbl = int(c_labels[best_idx])
                    labels[n_i] = assigned_lbl
                    prob_val = max(0.5, float(1.0 - best_dist))
                    probabilities[n_i] = prob_val
                    outlier_scores[n_i] = best_dist
                    distance_to_centroid[n_i] = best_dist
                    cluster_manifest_df.loc[n_i, "cluster_label"] = assigned_lbl
                    cluster_manifest_df.loc[n_i, "cluster_probability"] = prob_val
                    cluster_manifest_df.loc[n_i, "outlier_score"] = best_dist
                    cluster_manifest_df.loc[n_i, "distance_to_centroid"] = best_dist
                    cluster_manifest_df.loc[n_i, "is_noise"] = False
                    cluster_manifest_df.loc[n_i, "cluster_review_reason"] = "reassigned_from_noise"
                    cluster_manifest_df.loc[n_i, "requires_manual_review"] = True
                    cluster_manifest_df.loc[n_i, "cluster_folder_stub"] = f"cluster_{assigned_lbl:05d}_unknown"
                    reassigned_count += 1

            if reassigned_count > 0:
                centroids = compute_cluster_centroids(embeddings, labels)
                for label, centroid in centroids.items():
                    idx = np.where(labels == label)[0]
                    final_distances = euclidean_distance_to_centroid(embeddings[idx], centroid).astype(np.float32)
                    distance_to_centroid[idx] = final_distances
                    cluster_manifest_df.loc[idx, "distance_to_centroid"] = final_distances

    cluster_summary_df = calculate_cluster_summary(cluster_manifest_df)
    folder_assignment_df = build_folder_assignment_manifest(
        cluster_manifest_df,
        separate_review_folders=separate_review_folders,
    )

    # Save arrays and tables
    labels_path = arrays_dir / "cluster_labels.npy"
    probabilities_path = arrays_dir / "cluster_probabilities.npy"
    outlier_scores_path = arrays_dir / "outlier_scores.npy"
    centroids_path = arrays_dir / "cluster_centroids.npz"

    cluster_manifest_path = tables_dir / "cluster_manifest.csv"
    cluster_summary_path = tables_dir / "cluster_summary.csv"
    review_manifest_path = tables_dir / "manual_review_manifest.csv"
    noise_manifest_path = tables_dir / "noise_manifest.csv"
    folder_assignment_path = tables_dir / "folder_assignment_manifest.csv"

    np.save(labels_path, labels)
    np.save(probabilities_path, probabilities)
    np.save(outlier_scores_path, outlier_scores)
    np.savez_compressed(
        centroids_path,
        **{f"cluster_{label}": centroid for label, centroid in centroids.items()},
    )

    cluster_manifest_df.to_csv(cluster_manifest_path, index=False)
    cluster_summary_df.to_csv(cluster_summary_path, index=False)
    cluster_manifest_df[cluster_manifest_df["requires_manual_review"]].to_csv(review_manifest_path, index=False)
    cluster_manifest_df[cluster_manifest_df["is_noise"]].to_csv(noise_manifest_path, index=False)
    folder_assignment_df.to_csv(folder_assignment_path, index=False)

    # Review reason summary & folder assignment preview tables
    review_reason_summary_df = (
        cluster_manifest_df.assign(
            cluster_review_reason=lambda df: df["cluster_review_reason"].replace("", "no_cluster_review_reason")
        )
        .groupby("cluster_review_reason", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    review_reason_summary_df.to_csv(tables_dir / "review_reason_summary.csv", index=False)

    folder_assignment_preview_df = (
        folder_assignment_df.groupby(["assignment_category", "proposed_folder"], dropna=False)
        .size()
        .reset_index(name="image_count")
        .sort_values(["assignment_category", "image_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    folder_assignment_preview_df.to_csv(tables_dir / "folder_assignment_preview.csv", index=False)

    # Figures
    cluster_size_plot_path = figures_dir / "cluster_size_distribution.png"
    plot_cluster_size_distribution(cluster_summary_df, output_path=cluster_size_plot_path)

    probability_outlier_plot_path = figures_dir / "probability_outlier_distribution.png"
    plot_probability_and_outlier_distributions(
        cluster_manifest_df,
        low_probability_threshold=low_probability_threshold,
        high_outlier_threshold=high_outlier_thresh,
        output_path=probability_outlier_plot_path,
    )

    # Cluster contact sheets
    contact_sheet_rows = []
    eligible_clusters = (
        cluster_summary_df[(~cluster_summary_df["is_noise_cluster"]) & (cluster_summary_df["image_count"] >= 2)]
        .sort_values(["image_count", "mean_probability"], ascending=[False, False])
        .head(250)
    )

    for _, crow in eligible_clusters.iterrows():
        lbl = int(crow["cluster_label"])
        stub = crow["cluster_folder_stub"]
        c_items = cluster_manifest_df[cluster_manifest_df["cluster_label"].eq(lbl)]
        sampled = sample_representative_and_boundary_rows(c_items, 18, 6)
        cpaths = sampled["crop_path"].dropna().tolist()
        out_sheet = contact_sheets_dir / f"{stub}.jpg"
        created = create_contact_sheet(
            cpaths,
            out_sheet,
            title=f"{stub} | n={len(c_items)} | mean_p={crow['mean_probability']:.3f}",
        )
        contact_sheet_rows.append(
            {
                "cluster_label": lbl,
                "cluster_folder_stub": stub,
                "image_count": len(c_items),
                "sampled_image_count": len(cpaths),
                "created": bool(created),
                "contact_sheet_path": out_sheet.as_posix() if created else None,
            }
        )

    contact_sheets_df = pd.DataFrame(contact_sheet_rows)
    contact_sheets_path = tables_dir / "cluster_contact_sheets.csv"
    contact_sheets_df.to_csv(contact_sheets_path, index=False)

    # Special contact sheets (noise, low prob, high outlier, manual review)
    special_groups = {
        "noise_samples": cluster_manifest_df[cluster_manifest_df["is_noise"]],
        "low_probability_samples": cluster_manifest_df[
            (~cluster_manifest_df["is_noise"]) & cluster_manifest_df["is_low_probability"]
        ],
        "high_outlier_samples": cluster_manifest_df[
            (~cluster_manifest_df["is_noise"]) & cluster_manifest_df["is_high_outlier"]
        ],
        "manual_review_samples": cluster_manifest_df[cluster_manifest_df["requires_manual_review"]],
    }
    special_sheet_rows = []
    for sname, sgroup in special_groups.items():
        if sgroup.empty:
            special_sheet_rows.append(
                {"sheet_name": sname, "image_count": 0, "created": False, "contact_sheet_path": None}
            )
            continue
        sample_paths = sgroup.head(24)["crop_path"].dropna().tolist()
        out_sheet = contact_sheets_dir / f"_{sname}.jpg"
        created = create_contact_sheet(sample_paths, out_sheet, title=f"{sname} | n={len(sgroup)}")
        special_sheet_rows.append(
            {
                "sheet_name": sname,
                "image_count": len(sgroup),
                "sampled_image_count": len(sample_paths),
                "created": bool(created),
                "contact_sheet_path": out_sheet.as_posix() if created else None,
            }
        )

    special_sheets_df = pd.DataFrame(special_sheet_rows)
    special_sheets_path = tables_dir / "special_contact_sheets.csv"
    special_sheets_df.to_csv(special_sheets_path, index=False)

    # Cluster metadata files
    metadata_rows = []
    for _, crow in cluster_summary_df.iterrows():
        lbl = int(crow["cluster_label"])
        stub = crow["cluster_folder_stub"]
        c_items = cluster_manifest_df[cluster_manifest_df["cluster_label"].eq(lbl)]
        meta = {
            "cluster_label": lbl,
            "cluster_folder_stub": stub,
            "is_noise_cluster": bool(crow["is_noise_cluster"]),
            "image_count": int(crow["image_count"]),
            "requires_review_count": int(crow["requires_review_count"]),
            "requires_review_share": float(crow["requires_review_share"]),
            "mean_probability": None if pd.isna(crow["mean_probability"]) else float(crow["mean_probability"]),
            "median_probability": None if pd.isna(crow["median_probability"]) else float(crow["median_probability"]),
            "clustering": {
                "backend": backend_used,
                "algorithm": "HDBSCAN",
                "metric": metric,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": cluster_selection_method,
            },
            "created_at": now_iso(),
            "sample_items": c_items[
                [
                    "embedding_row",
                    "relative_path",
                    "source_path",
                    "crop_path",
                    "cluster_probability",
                    "outlier_score",
                    "distance_to_centroid",
                    "requires_manual_review",
                    "cluster_review_reason",
                ]
            ]
            .head(25)
            .to_dict(orient="records"),
        }
        meta_p = cluster_metadata_dir / f"{stub}.json"
        with meta_p.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        metadata_rows.append({"cluster_label": lbl, "cluster_folder_stub": stub, "metadata_path": meta_p.as_posix()})

    metadata_index_df = pd.DataFrame(metadata_rows)
    metadata_index_path = tables_dir / "cluster_metadata_index.csv"
    metadata_index_df.to_csv(metadata_index_path, index=False)

    cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int(np.sum(labels == -1))

    # Consistency checks
    checks = {
        "labels_file_exists": labels_path.exists(),
        "probabilities_file_exists": probabilities_path.exists(),
        "outlier_scores_file_exists": outlier_scores_path.exists(),
        "cluster_manifest_exists": cluster_manifest_path.exists(),
        "cluster_summary_exists": cluster_summary_path.exists(),
        "folder_assignment_exists": folder_assignment_path.exists(),
        "labels_match_embeddings": bool(len(labels) == embeddings.shape[0]),
        "probabilities_match_embeddings": bool(len(probabilities) == embeddings.shape[0]),
        "outlier_scores_match_embeddings": bool(len(outlier_scores) == embeddings.shape[0]),
        "cluster_manifest_rows_match_embeddings": bool(len(cluster_manifest_df) == embeddings.shape[0]),
        "folder_assignment_rows_match_embeddings": bool(len(folder_assignment_df) == embeddings.shape[0]),
        "no_nan_labels": bool(not pd.Series(labels).isna().any()),
        "cluster_count_non_negative": bool(cluster_count >= 0),
    }
    with (reports_dir / "consistency_checks.json").open("w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)

    summary = {
        "run_id": run_id,
        "created_at": now_iso(),
        "project_dir": project_dir.as_posix(),
        "previous_run_dir": previous_run_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "backend": backend_used,
        "algorithm": "HDBSCAN",
        "metric": metric,
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "cluster_selection_epsilon": float(cluster_selection_epsilon),
        "cluster_selection_method": cluster_selection_method,
        "reassign_noise": bool(reassign_noise),
        "max_reassign_distance": float(max_reassign_distance),
        "reassigned_points": int(reassigned_count),
        "embedding_count": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "estimated_clusters": int(cluster_count),
        "noise_points": int(noise_count),
        "clustered_points": int(len(labels) - noise_count),
        "manual_review_points": int(cluster_manifest_df["requires_manual_review"].sum()),
        "low_probability_threshold": float(low_probability_threshold),
        "high_outlier_quantile": float(high_outlier_quantile),
        "high_outlier_threshold": float(high_outlier_thresh),
        "python": sys.version,
        "platform": platform.platform(),
        "outputs": {
            "cluster_labels": labels_path.as_posix(),
            "cluster_probabilities": probabilities_path.as_posix(),
            "outlier_scores": outlier_scores_path.as_posix(),
            "cluster_centroids": centroids_path.as_posix(),
            "cluster_manifest": cluster_manifest_path.as_posix(),
            "cluster_summary": cluster_summary_path.as_posix(),
            "manual_review_manifest": review_manifest_path.as_posix(),
            "noise_manifest": noise_manifest_path.as_posix(),
            "folder_assignment_manifest": folder_assignment_path.as_posix(),
            "cluster_contact_sheets": contact_sheets_path.as_posix(),
            "special_contact_sheets": special_sheets_path.as_posix(),
            "cluster_metadata_index": metadata_index_path.as_posix(),
            "cluster_size_plot": cluster_size_plot_path.as_posix(),
            "probability_outlier_plot": probability_outlier_plot_path.as_posix(),
            "consistency_checks": (reports_dir / "consistency_checks.json").as_posix(),
        },
    }

    summary_path = reports_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "labels": labels,
        "probabilities": probabilities,
        "outlier_scores": outlier_scores,
        "centroids": centroids,
        "cluster_manifest_df": cluster_manifest_df,
        "cluster_summary_df": cluster_summary_df,
        "folder_assignment_df": folder_assignment_df,
        "contact_sheets_df": contact_sheets_df,
        "special_sheets_df": special_sheets_df,
        "summary": summary,
        "summary_path": summary_path,
    }
