#!/usr/bin/env python3
"""
CLI script to run Stage 4: HDBSCAN Clustering and Cluster Diagnostics.
"""

import argparse
from pathlib import Path
import sys

from anime_character_organizer.workflows.cluster import run_clustering


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Perform unsupervised HDBSCAN clustering over CCIP embeddings."
    )
    parser.add_argument(
        "--project-dir",
        "-p",
        default=Path("./anime_character_pipeline"),
        type=Path,
        help="Base pipeline directory (default: ./anime_character_pipeline).",
    )
    parser.add_argument(
        "--previous-run-dir",
        default=None,
        type=Path,
        help="Path to previous 03_ccip_embeddings_* run directory (default: latest).",
    )
    parser.add_argument(
        "--min-cluster-size",
        default=5,
        type=int,
        help="HDBSCAN min_cluster_size parameter (default: 5).",
    )
    parser.add_argument(
        "--min-samples",
        default=4,
        type=int,
        help="HDBSCAN min_samples parameter (default: 4).",
    )
    parser.add_argument(
        "--cluster-selection-method",
        default="eom",
        choices=["eom", "leaf"],
        help="HDBSCAN cluster selection method (default: eom).",
    )

    args = parser.parse_args()

    try:
        res = run_clustering(
            project_dir=args.project_dir,
            previous_run_dir=args.previous_run_dir,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_method=args.cluster_selection_method,
        )
        print("\nClustering completed successfully!")
        print(f"Run directory: {res['run_dir']}")
        print(f"Estimated clusters: {res['summary']['estimated_clusters']}")
        print(f"Noise points: {res['summary']['noise_points']}")
        print(f"Clustered points: {res['summary']['clustered_points']}")
        print(f"Manual review points: {res['summary']['manual_review_points']}")
        return 0
    except Exception as exc:
        print(f"Error during clustering: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
