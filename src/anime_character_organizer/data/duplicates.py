"""
Duplicate detection algorithms: exact SHA-256 grouping and perceptual pHash clustering via BKTree.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm.auto import tqdm

from ..utils.hashing import hamming_int, hash_hex_to_int
from ..utils.structures import BKTree, UnionFind

EXACT_DUPLICATE_COLUMNS = [
    "exact_group_id",
    "sha256",
    "group_size",
    "is_representative",
    "representative_path",
    "path",
    "relative_path",
    "file_size_bytes",
    "width",
    "height",
    "format",
]

PERCEPTUAL_CANDIDATE_COLUMNS = [
    "path_a",
    "path_b",
    "phash_distance",
]

PERCEPTUAL_GROUP_COLUMNS = [
    "perceptual_group_id",
    "group_size",
    "is_representative",
    "representative_path",
    "path",
    "relative_path",
    "file_size_bytes",
    "width",
    "height",
    "format",
    "phash",
    "dhash",
    "whash",
    "colorhash",
]


def find_exact_duplicates(valid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group exact duplicates sharing the same SHA-256 digest.

    The file with the lexicographically earliest relative_path is selected as representative.
    """
    exact_duplicate_rows: List[Dict[str, Any]] = []

    if valid_df.empty or "sha256" not in valid_df.columns:
        return pd.DataFrame(columns=EXACT_DUPLICATE_COLUMNS)

    sha_counts = valid_df["sha256"].value_counts(dropna=True)
    duplicate_sha_values = set(sha_counts[sha_counts > 1].index)

    group_id = 0
    for sha_value in sorted(duplicate_sha_values):
        group = valid_df[valid_df["sha256"].eq(sha_value)].sort_values("relative_path").copy()
        representative_path = group.iloc[0]["path"]

        for rank, (_, row) in enumerate(group.iterrows()):
            exact_duplicate_rows.append(
                {
                    "exact_group_id": f"exact_{group_id:06d}",
                    "sha256": sha_value,
                    "group_size": len(group),
                    "is_representative": rank == 0,
                    "representative_path": representative_path,
                    "path": row["path"],
                    "relative_path": row.get("relative_path"),
                    "file_size_bytes": row.get("file_size_bytes"),
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "format": row.get("format"),
                }
            )

        group_id += 1

    return pd.DataFrame(exact_duplicate_rows, columns=EXACT_DUPLICATE_COLUMNS)


def find_perceptual_duplicates(
    valid_df: pd.DataFrame,
    phash_threshold: int = 6,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify near-duplicate image candidate pairs using a BK-Tree over 64-bit pHash values,
    and group connected components using Union-Find.

    Returns:
        (perceptual_candidates_df, perceptual_groups_df)
    """
    if valid_df.empty or "phash" not in valid_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    phash_df = valid_df[valid_df["phash"].notna()].copy()
    if phash_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    phash_df["phash_int"] = phash_df["phash"].map(hash_hex_to_int)
    items = list(zip(phash_df["path"].tolist(), phash_df["phash_int"].tolist()))

    def bk_distance(item_a: Tuple[str, int], item_b: Tuple[str, int]) -> int:
        return hamming_int(item_a[1], item_b[1])

    tree = BKTree(bk_distance)
    candidate_pairs: List[Dict[str, Any]] = []
    seen_pairs = set()

    tree_iter = tqdm(items, desc="Building pHash BK-tree") if show_progress else items
    for item in tree_iter:
        tree.add(item)

    query_iter = tqdm(items, desc="Querying near-duplicate pHash candidates") if show_progress else items
    for item in query_iter:
        matches = tree.query(item, phash_threshold)
        path_a, _ = item

        for matched_item, distance in matches:
            path_b, _ = matched_item
            if path_a == path_b:
                continue

            pair = tuple(sorted([path_a, path_b]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            candidate_pairs.append(
                {
                    "path_a": pair[0],
                    "path_b": pair[1],
                    "phash_distance": int(distance),
                }
            )

    perceptual_candidates_df = pd.DataFrame(candidate_pairs, columns=PERCEPTUAL_CANDIDATE_COLUMNS)
    if not candidate_pairs:
        return perceptual_candidates_df, pd.DataFrame(columns=PERCEPTUAL_GROUP_COLUMNS)

    perceptual_candidates_df = perceptual_candidates_df.sort_values(["phash_distance", "path_a", "path_b"]).reset_index(
        drop=True
    )

    # Group components using UnionFind
    perceptual_group_rows: List[Dict[str, Any]] = []
    paths_in_pairs = sorted(set(perceptual_candidates_df["path_a"]).union(set(perceptual_candidates_df["path_b"])))
    uf = UnionFind(paths_in_pairs)

    for _, row in perceptual_candidates_df.iterrows():
        uf.union(row["path_a"], row["path_b"])

    groups = defaultdict(list)
    for path in paths_in_pairs:
        groups[uf.find(path)].append(path)

    path_to_meta = valid_df.drop_duplicates(subset="path").set_index("path").to_dict(orient="index")

    group_index = 0
    for _, group_paths in sorted(groups.items(), key=lambda item: (-len(item[1]), sorted(item[1])[0])):
        if len(group_paths) <= 1:
            continue

        group_paths = sorted(group_paths)
        representative_path = group_paths[0]

        for rank, path in enumerate(group_paths):
            meta = path_to_meta.get(path, {})
            perceptual_group_rows.append(
                {
                    "perceptual_group_id": f"perceptual_{group_index:06d}",
                    "group_size": len(group_paths),
                    "is_representative": rank == 0,
                    "representative_path": representative_path,
                    "path": path,
                    "relative_path": meta.get("relative_path"),
                    "file_size_bytes": meta.get("file_size_bytes"),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "format": meta.get("format"),
                    "phash": meta.get("phash"),
                    "dhash": meta.get("dhash"),
                    "whash": meta.get("whash"),
                    "colorhash": meta.get("colorhash"),
                }
            )

        group_index += 1

    perceptual_groups_df = pd.DataFrame(perceptual_group_rows, columns=PERCEPTUAL_GROUP_COLUMNS)
    return perceptual_candidates_df, perceptual_groups_df
