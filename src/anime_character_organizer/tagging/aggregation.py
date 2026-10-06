"""
Cluster-level character tag aggregation, weighted voting, confidence margin calculation, and naming decision rules.
"""

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

import numpy as np
import pandas as pd

from ..utils.naming import sanitize_filename_component
from ..utils.serialization import parse_dict_like, safe_json_dumps
from .normalization import display_tag_name, normalize_tag_name


def aggregate_cluster_tags(
    tagging_manifest_df: pd.DataFrame,
    min_character_score: float = 0.70,
    min_name_share: float = 0.35,
    min_weighted_score: float = 0.55,
    min_top_margin: float = 0.08,
) -> pd.DataFrame:
    """
    Aggregate character tags by cluster label and decide whether to accept the top suggested name.

    A suggested name is accepted only if:
        1. best_share >= min_name_share
        2. best_weighted_score >= min_weighted_score
        3. margin (best_weighted - second_weighted) >= min_top_margin
    """
    cluster_name_rows: List[Dict[str, Any]] = []

    for cluster_label, group in tagging_manifest_df.groupby("cluster_label", dropna=False):
        cluster_label = int(cluster_label)
        ok_group = group[group["tagging_status"].eq("ok")].copy()

        cluster_folder_stub = (
            str(group["cluster_folder_stub"].dropna().iloc[0])
            if group["cluster_folder_stub"].notna().any()
            else f"cluster_{cluster_label:05d}_unknown"
        )

        tag_weight: DefaultDict[str, float] = defaultdict(float)
        tag_count: DefaultDict[str, int] = defaultdict(int)
        tag_scores: DefaultDict[str, List[float]] = defaultdict(list)
        tag_examples: DefaultDict[str, List[str]] = defaultdict(list)

        for _, row in ok_group.iterrows():
            character_tags = parse_dict_like(row.get("character_tags_json", {}))

            for tag, score in character_tags.items():
                score = float(score)
                tag = normalize_tag_name(tag)

                if score < min_character_score:
                    continue

                tag_weight[tag] += score
                tag_count[tag] += 1
                tag_scores[tag].append(score)

                if len(tag_examples[tag]) < 5:
                    tag_examples[tag].append(row.get("relative_path", ""))

        total_tagged = int(len(ok_group))
        unique_tags = len(tag_weight)

        if tag_weight:
            ranked_tags = sorted(
                tag_weight.keys(),
                key=lambda t: (
                    tag_weight[t] / max(1, total_tagged),
                    tag_count[t],
                    np.mean(tag_scores[t]),
                    t,
                ),
                reverse=True,
            )

            best_tag = ranked_tags[0]
            second_tag = ranked_tags[1] if len(ranked_tags) > 1 else None

            best_count = int(tag_count[best_tag])
            best_share = float(best_count / max(1, total_tagged))
            best_weighted_score = float(tag_weight[best_tag] / max(1, total_tagged))
            best_mean_score = float(np.mean(tag_scores[best_tag]))
            best_max_score = float(np.max(tag_scores[best_tag]))

            if second_tag is not None:
                second_weighted_score = float(tag_weight[second_tag] / max(1, total_tagged))
                second_count = int(tag_count[second_tag])
                second_share = float(second_count / max(1, total_tagged))
            else:
                second_weighted_score = 0.0
                second_count = 0
                second_share = 0.0

            margin = float(best_weighted_score - second_weighted_score)

            accepted = bool(
                best_share >= min_name_share and best_weighted_score >= min_weighted_score and margin >= min_top_margin
            )

            suggested_name = display_tag_name(best_tag) if accepted else None

            top_tags_payload = []
            for tag in ranked_tags[:10]:
                top_tags_payload.append(
                    {
                        "tag": tag,
                        "count": int(tag_count[tag]),
                        "share": float(tag_count[tag] / max(1, total_tagged)),
                        "weighted_score": float(tag_weight[tag] / max(1, total_tagged)),
                        "mean_score": float(np.mean(tag_scores[tag])),
                        "max_score": float(np.max(tag_scores[tag])),
                        "examples": tag_examples[tag],
                    }
                )

        else:
            best_tag = None
            second_tag = None
            best_count = 0
            best_share = 0.0
            best_weighted_score = 0.0
            best_mean_score = np.nan
            best_max_score = np.nan
            second_weighted_score = 0.0
            second_count = 0
            second_share = 0.0
            margin = 0.0
            accepted = False
            suggested_name = None
            top_tags_payload = []

        proposed_named_folder = (
            sanitize_filename_component(suggested_name, fallback=cluster_folder_stub)
            if suggested_name is not None
            else cluster_folder_stub
        )
        if suggested_name is not None:
            proposed_named_folder = f"{proposed_named_folder}__{cluster_folder_stub}"

        cluster_name_rows.append(
            {
                "cluster_label": cluster_label,
                "cluster_folder_stub": cluster_folder_stub,
                "tagged_image_count": total_tagged,
                "tagging_error_count": int(group["tagging_status"].ne("ok").sum()),
                "unique_candidate_character_tags": int(unique_tags),
                "accepted_name": bool(accepted),
                "suggested_character_tag": suggested_name,
                "proposed_named_folder": sanitize_filename_component(
                    proposed_named_folder, fallback=cluster_folder_stub
                ),
                "best_raw_tag": best_tag,
                "best_tag_count": int(best_count),
                "best_tag_share": float(best_share),
                "best_weighted_score": float(best_weighted_score),
                "best_mean_score": float(best_mean_score) if pd.notna(best_mean_score) else np.nan,
                "best_max_score": float(best_max_score) if pd.notna(best_max_score) else np.nan,
                "second_raw_tag": second_tag,
                "second_tag_count": int(second_count),
                "second_tag_share": float(second_share),
                "second_weighted_score": float(second_weighted_score),
                "top_margin": float(margin),
                "top_tags_json": safe_json_dumps(top_tags_payload),
                "naming_rule": (
                    f"accepted if share >= {min_name_share}, "
                    f"weighted_score >= {min_weighted_score}, "
                    f"margin >= {min_top_margin}"
                ),
            }
        )

    suggestions_df = pd.DataFrame(cluster_name_rows)
    if not suggestions_df.empty:
        suggestions_df = suggestions_df.sort_values(
            ["accepted_name", "best_weighted_score", "best_tag_share", "tagged_image_count"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    return suggestions_df
