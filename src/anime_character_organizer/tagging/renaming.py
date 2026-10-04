"""
Folder rename planning based on accepted semantic character naming.
"""

from typing import Tuple
import pandas as pd

from ..utils.naming import sanitize_filename_component
from ..utils.serialization import safe_bool


def build_folder_rename_plan(
    folder_assignment_df: pd.DataFrame,
    cluster_name_suggestions_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build folder rename plan incorporating accepted character tag suggestions into folder destinations.
    
    Returns:
        (folder_rename_plan_df, folder_rename_preview_df)
    """
    plan_df = folder_assignment_df.copy()
    plan_df["cluster_label"] = plan_df["cluster_label"].astype(int)

    lookup_cols = [
        "cluster_label",
        "cluster_folder_stub",
        "accepted_name",
        "suggested_character_tag",
        "proposed_named_folder",
        "best_tag_share",
        "best_weighted_score",
        "top_margin",
        "top_tags_json",
    ]
    lookup_df = cluster_name_suggestions_df[lookup_cols].copy()

    plan_df = plan_df.merge(
        lookup_df,
        on="cluster_label",
        how="left",
        suffixes=("", "_naming"),
    )

    plan_df["accepted_name"] = plan_df["accepted_name"].fillna(False).map(safe_bool)
    plan_df["named_folder"] = plan_df["proposed_folder"]

    accepted_mask = (
        plan_df["accepted_name"]
        & plan_df["proposed_named_folder"].notna()
        & plan_df["cluster_label"].ne(-1)
    )

    plan_df.loc[accepted_mask, "named_folder"] = plan_df.loc[accepted_mask, "proposed_named_folder"]
    plan_df["named_folder"] = plan_df["named_folder"].map(
        lambda value: sanitize_filename_component(value, fallback="_needs_review")
    )

    preview_df = (
        plan_df.groupby(
            ["proposed_folder", "named_folder", "accepted_name", "suggested_character_tag"],
            dropna=False,
        )
        .size()
        .reset_index(name="image_count")
        .sort_values(["accepted_name", "image_count"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return plan_df, preview_df
