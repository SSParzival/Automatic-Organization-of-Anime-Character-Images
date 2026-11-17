"""
Character folder assignment: an image's own online match first, then its cluster's name, then its own
confident tags; the rest goes to review.
"""

from typing import Mapping, Tuple

import pandas as pd

from ..utils.naming import sanitize_folder_name

REVIEW_FOLDER = "_needs_review"
ONLINE_MATCH = "online_match"
IMAGE_OVER_CLUSTER = "image_tags_over_cluster"


def build_named_assignment(
    cluster_df: pd.DataFrame,
    cluster_names: Mapping[int, Tuple[str, str]],
    image_names: Mapping[int, Tuple[str, str]],
) -> pd.DataFrame:
    """
    Assign every image to a character folder.

    `cluster_names` maps cluster labels and `image_names` maps embedding rows to (name, source).
    An image's online match, or own tags naming another character (IMAGE_OVER_CLUSTER), override its
    cluster's name, which overrides the image's other own-tag names.
    Images with the same name share a folder; images without one go to the review folder.
    """
    df = cluster_df.copy()
    by_cluster = df["cluster_label"].astype(int).map(cluster_names)
    by_image = df["embedding_row"].astype(int).map(image_names)
    overrides = by_image.map(lambda value: isinstance(value, tuple) and value[1] in {ONLINE_MATCH, IMAGE_OVER_CLUSTER})
    chosen = by_image.where(overrides, by_cluster.where(by_cluster.notna(), by_image))
    df["character_name"] = chosen.map(lambda value: value[0] if isinstance(value, tuple) else None)
    df["naming_source"] = chosen.map(lambda value: value[1] if isinstance(value, tuple) else "unidentified")
    df["named_folder"] = [
        sanitize_folder_name(name) if isinstance(name, str) else REVIEW_FOLDER for name in df["character_name"]
    ]
    return df
