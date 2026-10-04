"""
Representative crop sample selection for cluster tagging.
"""

import numpy as np
import pandas as pd


def select_cluster_samples(cluster_df: pd.DataFrame, max_images: int = 12) -> pd.DataFrame:
    """
    Select up to max_images representative samples from a cluster for tagging.
    Prioritizes highest probability, lowest outlier score, and closest distance to centroid.
    """
    df = cluster_df.copy()

    if "distance_to_centroid" not in df.columns:
        df["distance_to_centroid"] = np.nan
    if "cluster_probability" not in df.columns:
        df["cluster_probability"] = 1.0
    if "outlier_score" not in df.columns:
        df["outlier_score"] = 0.0

    df["distance_to_centroid_sort"] = df["distance_to_centroid"].fillna(df["distance_to_centroid"].median())
    df["cluster_probability_sort"] = df["cluster_probability"].fillna(0.0)
    df["outlier_score_sort"] = df["outlier_score"].fillna(1.0)

    rep_df = df.sort_values(
        ["cluster_probability_sort", "outlier_score_sort", "distance_to_centroid_sort"],
        ascending=[False, True, True],
    ).head(max_images)

    if len(rep_df) < max_images:
        remaining_df = df[~df["embedding_row"].isin(rep_df["embedding_row"])].copy()
        extra_df = remaining_df.sort_values(
            ["distance_to_centroid_sort", "cluster_probability_sort"],
            ascending=[True, False],
        ).head(max_images - len(rep_df))
        rep_df = pd.concat([rep_df, extra_df], axis=0)

    rep_df = rep_df.drop_duplicates(subset=["embedding_row"], keep="first").head(max_images).copy()

    return rep_df.drop(
        columns=[c for c in ["distance_to_centroid_sort", "cluster_probability_sort", "outlier_score_sort"] if c in rep_df.columns]
    )
