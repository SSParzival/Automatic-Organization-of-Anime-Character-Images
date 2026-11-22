#!/usr/bin/env python3
"""Re-fit HDBSCAN variants on a previous run's embeddings and score them against independent names.

The oracle is the character each image matched itself to on Danbooru during a completed naming run,
which does not depend on how that run's clustering grouped the images. Purity is the share of oracle
images whose name equals the modal name of their cluster; noise is the HDBSCAN noise share.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from anime_character_organizer.clustering.algorithm import fit_hdbscan


def find_run(project_dir: Path, prefix: str) -> Path:
    runs = sorted(project_dir.glob(f"runs/{prefix}*"))
    if not runs:
        raise SystemExit(f"No run matching runs/{prefix}* under {project_dir}")
    return runs[-1]


def oracle_from_naming(naming_dir: Path) -> dict[int, str]:
    matched = pd.read_csv(naming_dir / "tables" / "online_identification.csv")
    own = matched[matched["chosen_character"].notna()]
    return dict(zip(own["embedding_row"].astype(int), own["chosen_character"]))


def score(labels: np.ndarray, oracle: dict[int, str]) -> dict:
    names = np.array([oracle.get(i) for i in range(len(labels))], dtype=object)
    df = pd.DataFrame({"label": labels, "name": names})
    df = df[df["name"].notna()]
    total = len(df)
    modal = df.groupby("label")["name"].transform(lambda s: s.mode().iloc[0])
    misplaced = int((df["name"] != modal).sum())
    return {
        "clusters": int(len(set(labels.tolist()) - {-1})),
        "noise_pct": round(100 * float(np.mean(labels == -1)), 1),
        "oracle_images": int(total),
        "oracle_purity_pct": round(100 * (1 - misplaced / max(1, total)), 2),
        "oracle_misplaced": misplaced,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default="anime_character_pipeline")
    parser.add_argument("--cse", type=float, action="append", help="cluster_selection_epsilon to try")
    parser.add_argument("--sample", type=int, help="Evaluate a random subset of the embeddings")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    embeddings = np.load(find_run(project_dir, "03_ccip_embeddings") / "arrays" / "ccip_embeddings_l2.npy")
    oracle = oracle_from_naming(find_run(project_dir, "05_cluster_naming"))
    if args.sample:
        indices = np.sort(np.random.default_rng(42).choice(len(embeddings), args.sample, replace=False))
        embeddings = embeddings[indices]
        oracle = {pos: oracle[row] for pos, row in enumerate(indices) if int(row) in oracle}

    rows = []
    for cse in args.cse or [0.50, 0.45, 0.40, 0.35, 0.30]:
        started = time.perf_counter()
        labels, *_ = fit_hdbscan(embeddings, min_cluster_size=2, min_samples=1, cluster_selection_epsilon=cse)
        row = {"cse": cse, "seconds": round(time.perf_counter() - started, 1), **score(labels, oracle)}
        rows.append(row)
        print(row, flush=True)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
