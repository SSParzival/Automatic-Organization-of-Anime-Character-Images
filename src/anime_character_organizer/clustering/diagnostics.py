"""
Cluster diagnostics, quality metrics, manual review flagging, and folder assignment planning.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.naming import sanitize_folder_name
from ..utils.serialization import safe_bool_series


def flag_cluster_review_items(
    cluster_manifest_df: pd.DataFrame,
    low_prob_threshold: float = 0.35,
    high_outlier_quantile: float = 0.95,
) -> Tuple[pd.DataFrame, float]:
    """
    Flag items requiring manual review based on noise status, low probability,
    high outlier score quantile, and prior crop review flags.

    Returns:
        (updated_cluster_manifest_df, computed_high_outlier_threshold)
    """
    df = cluster_manifest_df.copy()

    if "needs_review" in df.columns:
        df["needs_review"] = safe_bool_series(df["needs_review"])
    else:
        df["needs_review"] = False

    finite_outliers = df["outlier_score"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_outliers.empty:
        high_outlier_thresh = 1.0
    else:
        high_outlier_thresh = float(finite_outliers.quantile(high_outlier_quantile))

    df["is_noise"] = df["cluster_label"].eq(-1)
    df["is_low_probability"] = df["cluster_probability"].lt(low_prob_threshold)
    df["is_high_outlier"] = df["outlier_score"].gt(high_outlier_thresh)

    df["cluster_review_reason"] = ""

    df.loc[df["is_noise"], "cluster_review_reason"] = "hdbscan_noise"

    df.loc[
        ~df["is_noise"] & df["is_low_probability"],
        "cluster_review_reason",
    ] = "low_cluster_membership_probability"

    df.loc[
        ~df["is_noise"] & ~df["is_low_probability"] & df["is_high_outlier"],
        "cluster_review_reason",
    ] = "high_local_outlier_score"

    df.loc[
        df["needs_review"] & df["cluster_review_reason"].eq(""),
        "cluster_review_reason",
    ] = "previous_crop_review_flag"

    df["requires_manual_review"] = (
        df["is_noise"] | df["is_low_probability"] | df["is_high_outlier"] | df["needs_review"]
    )

    df["cluster_folder_stub"] = df["cluster_label"].map(
        lambda x: "_needs_review_noise" if int(x) == -1 else f"cluster_{int(x):05d}_unknown"
    )

    return df, high_outlier_thresh


def calculate_cluster_summary(
    cluster_manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate aggregated diagnostic metrics per cluster."""
    rows: List[Dict[str, Any]] = []

    for label in sorted(cluster_manifest_df["cluster_label"].unique()):
        cdf = cluster_manifest_df[cluster_manifest_df["cluster_label"].eq(label)].copy()
        if cdf.empty:
            continue

        is_noise = int(label) == -1
        folder_stub = "_needs_review_noise" if is_noise else f"cluster_{int(label):05d}_unknown"

        distances = cdf["distance_to_centroid"].replace([np.inf, -np.inf], np.nan).dropna()
        probs = cdf["cluster_probability"].replace([np.inf, -np.inf], np.nan).dropna()
        outliers = cdf["outlier_score"].replace([np.inf, -np.inf], np.nan).dropna()

        rows.append(
            {
                "cluster_label": int(label),
                "cluster_folder_stub": folder_stub,
                "is_noise_cluster": bool(is_noise),
                "image_count": int(len(cdf)),
                "requires_review_count": int(cdf["requires_manual_review"].sum()),
                "requires_review_share": float(cdf["requires_manual_review"].mean()),
                "mean_probability": float(probs.mean()) if not probs.empty else np.nan,
                "median_probability": float(probs.median()) if not probs.empty else np.nan,
                "min_probability": float(probs.min()) if not probs.empty else np.nan,
                "mean_outlier_score": float(outliers.mean()) if not outliers.empty else np.nan,
                "median_outlier_score": float(outliers.median()) if not outliers.empty else np.nan,
                "max_outlier_score": float(outliers.max()) if not outliers.empty else np.nan,
                "mean_distance_to_centroid": float(distances.mean()) if not distances.empty else np.nan,
                "median_distance_to_centroid": float(distances.median()) if not distances.empty else np.nan,
                "max_distance_to_centroid": float(distances.max()) if not distances.empty else np.nan,
                "head_crop_count": int(cdf["selected_region_type"].eq("head").sum())
                if "selected_region_type" in cdf.columns
                else 0,
                "person_crop_count": int(cdf["selected_region_type"].eq("person").sum())
                if "selected_region_type" in cdf.columns
                else 0,
                "full_image_count": int(
                    cdf["selected_region_type"].astype(str).str.contains("full_image", na=False).sum()
                )
                if "selected_region_type" in cdf.columns
                else 0,
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["is_noise_cluster", "image_count", "cluster_label"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    return summary_df


def sample_representative_and_boundary_rows(
    cluster_df: pd.DataFrame,
    representative_count: int = 18,
    boundary_count: int = 6,
) -> pd.DataFrame:
    """Sample closest items to centroid (representatives) and farthest items (boundary)."""
    cdf = cluster_df.sort_values("distance_to_centroid", ascending=True).copy()
    representative = cdf.head(representative_count)
    boundary = cdf.sort_values("distance_to_centroid", ascending=False).head(boundary_count)

    sampled = pd.concat([representative, boundary], axis=0)
    sampled = sampled.drop_duplicates(subset=["embedding_row"], keep="first")
    return sampled.sort_values("distance_to_centroid", ascending=True)


def build_folder_assignment_manifest(
    cluster_manifest_df: pd.DataFrame,
    separate_review_folders: bool = False,
) -> pd.DataFrame:
    """
    Build folder assignment plan based on cluster label and review flags.

    If separate_review_folders is False (recommended):
        Non-noise items go to their character cluster folder (e.g. cluster_XXXXX_unknown)
        with review flags recorded in manifests and metadata for non-destructive inspection.
        Only unassigned noise items go to _needs_review_noise.

    If separate_review_folders is True:
        Clean items -> cluster_XXXXX_unknown
        Noise items -> _needs_review_noise
        Uncertain items -> cluster_XXXXX_unknown_review
    """
    df = cluster_manifest_df.copy()
    df["assignment_category"] = "cluster"

    df.loc[df["is_noise"], "assignment_category"] = "needs_review_noise"
    df.loc[
        (~df["is_noise"]) & df["requires_manual_review"],
        "assignment_category",
    ] = "needs_review_cluster_member"

    df["proposed_folder"] = df["cluster_folder_stub"]

    df.loc[
        df["assignment_category"].eq("needs_review_noise"),
        "proposed_folder",
    ] = "_needs_review_noise"

    if separate_review_folders:
        df.loc[
            df["assignment_category"].eq("needs_review_cluster_member"),
            "proposed_folder",
        ] = (
            df.loc[
                df["assignment_category"].eq("needs_review_cluster_member"),
                "cluster_folder_stub",
            ]
            + "_review"
        )

    df["proposed_folder"] = df["proposed_folder"].map(lambda value: sanitize_folder_name(value, "_needs_review"))

    return df
