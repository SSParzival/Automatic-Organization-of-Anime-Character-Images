"""
Workflow orchestrator for Stage 5: character naming with online identification and the local tagger.
"""

import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..config import validate_parameters
from ..exceptions import ConfigurationError, InvalidImageError, TaggingError
from ..tagging.aggregation import aggregate_cluster_tags, choose_candidate, decide_image_name, vote_online_cluster
from ..tagging.extraction import (
    TAGGER_NAME,
    TAGGER_PRECISION,
    TAGGER_REVISION,
    extract_character_tags,
    load_character_tagger,
)
from ..tagging.identification import OnlineIdentifier
from ..tagging.online import clients_from_environment
from ..tagging.renaming import IMAGE_OVER_CLUSTER, ONLINE_MATCH, REVIEW_FOLDER, build_named_assignment
from ..tagging.resolution import CharacterResolver, canonical_tags, name_key
from ..tagging.sampling import select_cluster_samples
from ..utils.paths import checked_path, normalize_path_string
from ..utils.runs import create_run_directory, find_latest_run
from ..utils.serialization import json_safe_value, parse_dict_like, require_columns, safe_bool, safe_json_dumps
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


def build_online_services(
    project_dir: Path, min_online_similarity: float
) -> Tuple[OnlineIdentifier, CharacterResolver]:
    """Identifier and name resolver using the credentials in the environment, cached under project_dir."""
    clients = clients_from_environment()
    cache_dir = project_dir / "cache"
    identifier = OnlineIdentifier(
        clients["danbooru"], clients["saucenao"], cache_dir / "online_identification.json", min_online_similarity
    )
    return identifier, CharacterResolver(clients["danbooru"], cache_dir / "character_names.json")


def overlapped(
    items: List[Any], fetch: Callable[[Any], Any], show_progress: bool, desc: str
) -> Iterator[Tuple[Any, Any]]:
    """Yield (item, fetch(item)) in order while a background thread fetches the next items.

    Online lookups wait on the network and local tagging on the CPU, so running them side by side
    hides most of the lookup time.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pairs = zip(items, pool.map(fetch, items))
        yield from tqdm(pairs, total=len(items), desc=desc) if show_progress else pairs
    finally:
        pool.shutdown(wait=True, cancel_futures=True)  # an interrupted run does not finish the queued lookups


def search_cluster(identifier: OnlineIdentifier, group: pd.DataFrame, max_images: int) -> Dict[int, Dict[str, Any]]:
    """
    Identify representative images of a cluster online until two of them agree on a character.

    Exact file matches already cached for any image of the cluster count as votes too.
    """
    hits = {
        int(row): result
        for row, path in zip(group["embedding_row"], group["source_path"])
        if (result := identifier.accepted(path))
    }
    representatives = select_cluster_samples(group, max_images)
    for row, path in zip(representatives["embedding_row"], representatives["source_path"]):
        if raw_vote(hits):
            break
        if int(row) not in hits and (result := identifier.identify(path)):
            hits[int(row)] = result
    return hits


def raw_vote(hits: Mapping[int, Mapping[str, Any]]) -> Optional[str]:
    """Cluster vote on spellings only, used to decide while names are not resolved yet."""
    return vote_online_cluster([[name_key(n) for n in hit["characters"]] for hit in hits.values()])


def needs_tags(listed: List[str], preferred: Optional[str], cluster_named: bool) -> bool:
    """Whether an image needs the local tagger: several online candidates to choose from, or none at all."""
    if len(listed) == 1 or (listed and preferred in listed):
        return False
    return bool(listed) or not cluster_named


def doubtful_members(cluster_df: pd.DataFrame) -> pd.Series:
    """Cluster members HDBSCAN placed with low confidence: rescued noise, low probability or outliers."""
    flags = [c for c in ("reassigned_from_noise", "is_low_probability", "is_high_outlier") if c in cluster_df]
    doubtful = pd.Series(False, index=cluster_df.index)
    for column in flags:
        doubtful |= cluster_df[column].map(safe_bool)
    return doubtful & cluster_df["cluster_label"].ne(-1)


def canonical_candidates(result: Optional[Mapping[str, Any]], names: Mapping[str, Optional[str]]) -> List[str]:
    if not result:
        return []
    return list(dict.fromkeys(n for n in (names.get(c) for c in result["characters"]) if n))


def raw_names(online: Mapping[int, Mapping[str, Any]], records: Iterable[Mapping[str, Any]]) -> List[str]:
    from_online = (c for result in online.values() for c in result["characters"])
    from_tags = (t for r in records if r["tagging_status"] == "ok" for t in parse_dict_like(r["character_tags_json"]))
    return list(set(from_online) | set(from_tags))


def resolve_names(resolver: Optional[CharacterResolver], names: Iterable[str]) -> Dict[str, Optional[str]]:
    names = list(names)
    return resolver.resolve(names) if resolver is not None else {n: name_key(n) or None for n in names}


def canonical_records(records: List[Dict[str, Any]], names: Mapping[str, Optional[str]]) -> pd.DataFrame:
    """Tagging records with tag scores merged under canonical character names."""
    df = pd.DataFrame(records, columns=["cluster_label", "tagging_status", "character_tags_json", "relative_path"])
    df["character_tags_json"] = [
        safe_json_dumps(canonical_tags(parse_dict_like(tags), names)) for tags in df["character_tags_json"]
    ]
    return df


def decide_images(
    rows: Iterable[int],
    candidates: Mapping[int, List[str]],
    local: Mapping[int, Mapping[str, float]],
    preferred: Mapping[int, Optional[str]],
    min_character_score: float,
    min_top_margin: float,
) -> Dict[int, Tuple[str, str]]:
    """
    Name images from their online candidates, or from their own tags when nothing was found online.

    Own tags that confidently name another character, without a trace of the cluster's name, mark an
    image the clustering misplaced; that name then overrides the cluster's.
    """
    names: Dict[int, Tuple[str, str]] = {}
    for row in rows:
        tags = local.get(row, {})
        if candidates.get(row):
            choice = choose_candidate(candidates[row], tags, preferred.get(row))
            source = ONLINE_MATCH
        else:
            choice = decide_image_name(tags, min_character_score, min_top_margin)
            cluster = preferred.get(row)
            source = IMAGE_OVER_CLUSTER if cluster and choice != cluster and cluster not in tags else "image_tags"
        if choice:
            names[row] = (choice, source)
    return names


def online_table(
    online: Mapping[int, Mapping[str, Any]],
    candidates: Mapping[int, List[str]],
    image_names: Mapping[int, Tuple[str, str]],
    paths: Mapping[int, str],
) -> pd.DataFrame:
    rows = []
    for row, result in sorted(online.items()):
        name, source = image_names.get(row, (None, None))
        rows.append(
            {
                "embedding_row": row,
                "source_path": paths[row],
                "provider": result["provider"],
                "similarity": result["similarity"],
                "reference": result["reference"],
                "listed_characters": safe_json_dumps(result["characters"]),
                "canonical_characters": safe_json_dumps(candidates.get(row, [])),
                "chosen_character": name if source == ONLINE_MATCH else None,
            }
        )
    columns = ["embedding_row", "source_path", "provider", "similarity", "reference", "listed_characters"]
    return pd.DataFrame(rows, columns=columns + ["canonical_characters", "chosen_character"])


def run_cluster_naming(
    project_dir: Union[str, Path] = "./anime_character_pipeline",
    previous_run_dir: Optional[Union[str, Path]] = None,
    max_images_per_cluster_to_tag: int = 12,
    min_character_score: float = 0.70,
    min_name_share: float = 0.35,
    min_weighted_score: float = 0.55,
    min_top_margin: float = 0.08,
    online_lookup: bool = True,
    max_images_per_cluster_to_search: int = 3,
    min_online_similarity: float = 85.0,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Name every image after the character it shows.

    With online lookup, every image is first matched by file hash on Danbooru, and each cluster is
    identified online from representative images (reverse image search needs credentials); agreeing
    matches name the cluster. Clusters left unnamed are named by a vote of the local tagger over
    representative crops. Unclustered images, images of unnamed clusters, members HDBSCAN placed
    with low confidence and members of clusters named online are identified online one by one, and
    tagged locally when no online match settles them. Lookups run in a background thread while the CPU tags. All names are mapped to
    canonical Danbooru character tags, so aliases and costume variants share a folder. Images without
    a confident name go to review.
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

    timings: Dict[str, float] = {}
    started = time.perf_counter()
    identifier, resolver = build_online_services(project_dir, min_online_similarity) if online_lookup else (None, None)
    online: Dict[int, Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
    clustered = cluster_df[cluster_df["cluster_label"].ne(-1)]
    groups = [group for _, group in clustered.groupby("cluster_label")]
    search: Callable[[pd.DataFrame], Dict[int, Dict[str, Any]]] = lambda group: {}  # noqa: E731
    if identifier is not None:
        identifier.prefetch(cluster_df["source_path"])
        search = partial(search_cluster, identifier, max_images=max_images_per_cluster_to_search)
    for group, hits in overlapped(groups, search, show_progress, "Naming clusters"):
        online |= hits
        if not raw_vote(hits):
            samples = select_cluster_samples(group[group["crop_exists"]], max_images_per_cluster_to_tag)
            records += tag_crops(samples, show_progress=False)
    names = resolve_names(resolver, raw_names(online, records))
    cluster_names: Dict[int, Tuple[str, str]] = {}
    for group in groups:
        votes = [canonical_candidates(online.get(int(r)), names) for r in group["embedding_row"] if int(r) in online]
        name = vote_online_cluster(votes)
        if name:
            cluster_names[int(group["cluster_label"].iloc[0])] = (name, "online_cluster_vote")
    cluster_name_suggestions_df = aggregate_cluster_tags(
        canonical_records([r for r in records if r["cluster_label"] not in cluster_names], names),
        min_character_score=min_character_score,
        min_name_share=min_name_share,
        min_weighted_score=min_weighted_score,
        min_top_margin=min_top_margin,
    )
    if not cluster_name_suggestions_df.empty:
        accepted = cluster_name_suggestions_df[cluster_name_suggestions_df["accepted_name"]]
        cluster_names |= {
            int(label): (name, "cluster_vote")
            for label, name in zip(accepted["cluster_label"], accepted["suggested_character_tag"])
        }
    timings["clusters"] = time.perf_counter() - started

    started = time.perf_counter()
    cluster_of = dict(zip(cluster_df["embedding_row"].astype(int), cluster_df["cluster_label"].astype(int)))
    preferred = {row: cluster_names.get(label, (None,))[0] for row, label in cluster_of.items()}
    crops = cluster_df[cluster_df["crop_exists"]].set_index(cluster_df["embedding_row"].astype(int), drop=False)
    tagged = {r["embedding_row"] for r in records}

    choosing = 0
    # Members of a cluster named online without a match of their own are checked like unclustered images:
    # lookalike characters share clusters, and nothing else confirms their name.
    online_named = [label for label, (_, source) in cluster_names.items() if source == "online_cluster_vote"]
    unconfirmed = cluster_df["cluster_label"].isin(online_named) & ~cluster_df["embedding_row"].astype(int).isin(online)
    trusted = set(cluster_df.loc[~unconfirmed, "embedding_row"].astype(int))

    def tag_if_needed(row: int, listed: List[str]) -> None:
        nonlocal choosing
        cluster_named = preferred[row] is not None and row in trusted
        if row not in tagged and row in crops.index and needs_tags(listed, preferred[row], cluster_named):
            records.extend(tag_crops(crops.loc[[row]], show_progress=False))
            tagged.add(row)
            choosing += bool(listed)

    pending = cluster_df[~cluster_df["cluster_label"].isin(cluster_names) | doubtful_members(cluster_df) | unconfirmed]
    items = list(zip(pending["embedding_row"].astype(int), pending["source_path"]))
    lookup = (lambda item: identifier.identify(item[1])) if identifier is not None else (lambda item: None)
    for (row, _), result in overlapped(items, lookup, show_progress, "Naming images"):
        if result:
            online[row] = result
        tag_if_needed(row, list(dict.fromkeys(name_key(n) for n in (result or {}).get("characters", []))))
    names |= resolve_names(resolver, raw_names(online, records))
    candidates = {row: canonical_candidates(result, names) for row, result in online.items()}
    for row, listed in candidates.items():
        tag_if_needed(row, listed)
    names |= resolve_names(resolver, raw_names({}, records))
    timings["images"] = time.perf_counter() - started

    local = {
        r["embedding_row"]: canonical_tags(parse_dict_like(r["character_tags_json"]), names)
        for r in records
        if r["tagging_status"] == "ok"
    }
    image_names = decide_images(cluster_of, candidates, local, preferred, min_character_score, min_top_margin)

    tagging_manifest_df = pd.DataFrame(records)
    tagging_manifest_path = tables_dir / "tagging_manifest.csv"
    tagging_manifest_df.to_csv(tagging_manifest_path, index=False)
    if records and tagging_manifest_df["tagging_status"].ne("ok").all():
        raise TaggingError(f"All tagger calls failed; inspect {tagging_manifest_path}.")

    cluster_name_suggestions_path = tables_dir / "cluster_name_suggestions.csv"
    cluster_name_suggestions_df.to_csv(cluster_name_suggestions_path, index=False)

    paths = dict(zip(cluster_df["embedding_row"].astype(int), cluster_df["source_path"]))
    online_df = online_table(online, candidates, image_names, paths)
    online_path = tables_dir / "online_identification.csv"
    online_df.to_csv(online_path, index=False)

    named_assignment_df = build_named_assignment(cluster_df.drop(columns=["crop_exists"]), cluster_names, image_names)
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

    if identifier is not None:
        identifier.save()
    source_counts = named_assignment_df["naming_source"].value_counts()
    summary = {
        "run_id": run_id,
        "created_at": now_iso(),
        "project_dir": project_dir.as_posix(),
        "previous_run_dir": previous_run_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "tagger": {"name": TAGGER_NAME, "revision": TAGGER_REVISION, "precision": TAGGER_PRECISION},
        "online_lookup": {
            "enabled": identifier is not None,
            "reverse_image_search": bool(identifier and identifier.can_search_images),
            "queries": dict(identifier.queries) if identifier else {},
            "errors": dict(identifier.errors) if identifier else {},
            "disabled_services": sorted(identifier.disabled) if identifier else [],
            "images_matched": len(online),
        },
        "clusters_evaluated": int(clustered["cluster_label"].nunique()),
        "clusters_named": len(cluster_names),
        "clusters_named_online": sum(1 for _, source in cluster_names.values() if source == "online_cluster_vote"),
        "character_folders": int(folder_preview_df["named_folder"].drop_duplicates().ne(REVIEW_FOLDER).sum()),
        "images_named_online": int(source_counts.get(ONLINE_MATCH, 0)),
        "images_named_by_online_cluster_vote": int(source_counts.get("online_cluster_vote", 0)),
        "images_named_by_cluster_vote": int(source_counts.get("cluster_vote", 0)),
        "images_named_by_image_tags": int(
            source_counts.get("image_tags", 0) + source_counts.get(IMAGE_OVER_CLUSTER, 0)
        ),
        "images_moved_out_of_their_cluster": int(source_counts.get(IMAGE_OVER_CLUSTER, 0)),
        "images_needing_review": int(source_counts.get("unidentified", 0)),
        "total_tagged_images": len(tagging_manifest_df),
        "images_tagged_to_choose_among_online_characters": choosing,
        "tagging_errors": int(tagging_manifest_df["tagging_status"].ne("ok").sum()) if records else 0,
        "seconds": {phase: round(seconds, 1) for phase, seconds in timings.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "outputs": {
            "tagging_manifest": tagging_manifest_path.as_posix(),
            "cluster_name_suggestions": cluster_name_suggestions_path.as_posix(),
            "online_identification": online_path.as_posix(),
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
        "online_identification_df": online_df,
        "named_assignment_df": named_assignment_df,
        "folder_preview_df": folder_preview_df,
        "summary": summary,
        "summary_path": summary_path,
    }
