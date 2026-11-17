"""
Character folder assignment: cluster votes first, then confident per-image tags; the rest goes to review.
"""

import numpy as np
import pandas as pd

from ..utils.naming import sanitize_folder_name
from ..utils.serialization import parse_dict_like
from .aggregation import decide_image_name

REVIEW_FOLDER = "_needs_review"


def build_named_assignment(
    cluster_df: pd.DataFrame,
    cluster_name_suggestions_df: pd.DataFrame,
    tagging_manifest_df: pd.DataFrame,
    min_character_score: float = 0.70,
    min_top_margin: float = 0.08,
) -> pd.DataFrame:
    """
    Assign every image to a character folder.

    Images of clusters with an accepted name take that name. Noise images and images of clusters
    without an accepted name are named from their own tags when confident; otherwise they go to
    the review folder. Clusters that share a name end up in the same folder.
    """
    cluster_names = {}
    if not cluster_name_suggestions_df.empty:
        accepted = cluster_name_suggestions_df[cluster_name_suggestions_df["accepted_name"].astype(bool)]
        cluster_names = dict(zip(accepted["cluster_label"].astype(int), accepted["suggested_character_tag"]))

    tagged = tagging_manifest_df[tagging_manifest_df["tagging_status"].eq("ok")]
    image_names = {
        int(row): decide_image_name(parse_dict_like(tags), min_character_score, min_top_margin)
        for row, tags in zip(tagged["embedding_row"], tagged["character_tags_json"])
    }

    df = cluster_df.copy()
    by_cluster = df["cluster_label"].astype(int).map(cluster_names)
    by_image = df["embedding_row"].astype(int).map(image_names)
    df["character_name"] = by_cluster.where(by_cluster.notna(), by_image)
    df["naming_source"] = np.select(
        [by_cluster.notna(), by_image.notna()], ["cluster_vote", "image_tags"], default="unidentified"
    )
    df["named_folder"] = [
        sanitize_folder_name(name) if isinstance(name, str) else REVIEW_FOLDER for name in df["character_name"]
    ]
    return df
