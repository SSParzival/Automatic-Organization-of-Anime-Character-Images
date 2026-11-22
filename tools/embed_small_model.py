#!/usr/bin/env python3
"""Score the small caformer-6 CCIP model against the same oracle used by evaluate_cluster_quality.py.

Re-embeds every crop of a previous embed run with the small model, clusters the result, and prints
the same quality metrics so the speed/quality trade-off is measurable before touching defaults.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from anime_character_organizer.clustering.algorithm import fit_hdbscan
from anime_character_organizer.embeddings.extraction import extract_all_embeddings

MODEL = "ccip-caformer-6-randaug-pruned_fp32"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_cluster_quality import find_run, oracle_from_naming, score  # noqa: E402


def main() -> None:
    project_dir = Path("anime_character_pipeline")
    emb_run = find_run(project_dir, "03_ccip_embeddings")
    manifest = pd.read_csv(emb_run / "tables" / "successful_embedding_manifest.csv")
    paths = manifest["crop_path"].astype(str).tolist()
    rows = manifest["embedding_row"].astype(int).tolist()

    started = time.perf_counter()
    records = extract_all_embeddings(paths, model=MODEL, image_size=384, batch_size=16, show_progress=False)
    kept = [(row, rec) for row, rec in zip(rows, records) if rec["status"] == "ok"]
    kept_rows = [row for row, _ in kept]
    features = np.stack([rec["feature"] for _, rec in kept])
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    extract_seconds = time.perf_counter() - started
    print(f"model={MODEL} kept={len(kept)}/{len(paths)} extract_seconds={extract_seconds:.0f}", flush=True)

    out_dir = project_dir / "ab_caformer6"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "rows.npy", np.asarray(kept_rows))
    np.save(out_dir / "embeddings_l2.npy", features.astype(np.float32))

    oracle = {row: name for row, name in oracle_from_naming(find_run(project_dir, "05_cluster_naming")).items()}
    oracle = {pos: oracle[row] for pos, row in enumerate(kept_rows) if row in oracle}

    for cse in (0.50, 0.40):
        started = time.perf_counter()
        labels, *_ = fit_hdbscan(features, min_cluster_size=2, min_samples=1, cluster_selection_epsilon=cse)
        row = {"cse": cse, "seconds": round(time.perf_counter() - started, 1), **score(labels, oracle)}
        print(row, flush=True)


if __name__ == "__main__":
    main()
