#!/usr/bin/env python3
"""
CLI script to run Stage 4: HDBSCAN Clustering and Cluster Diagnostics.
"""

import argparse
import sys
from pathlib import Path

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
        default=2,
        type=int,
        help="HDBSCAN min_cluster_size parameter (default: 2, allowing character pairs).",
    )
    parser.add_argument(
        "--min-samples",
        default=1,
        type=int,
        help="HDBSCAN min_samples parameter (default: 1, reducing reachability penalty).",
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        default=0.50,
        type=float,
        help="HDBSCAN cluster selection epsilon merge threshold (default: 0.50).",
    )
    parser.add_argument(
        "--cluster-selection-method",
        default="eom",
        choices=["eom", "leaf"],
        help="HDBSCAN cluster selection method (default: eom).",
    )
    parser.add_argument(
        "--no-reassign-noise",
        action="store_true",
        help="Disable centroid-based reassignment of borderline noise points.",
    )
    parser.add_argument(
        "--max-reassign-distance",
        default=0.55,
        type=float,
        help="Maximum distance to centroid for noise reassignment (default: 0.55).",
    )
    parser.add_argument(
        "--separate-review-folders",
        action="store_true",
        help="Split review items into separate _review folders rather than keeping unified folders.",
    )

    args = parser.parse_args()

    try:
        res = run_clustering(
            project_dir=args.project_dir,
            previous_run_dir=args.previous_run_dir,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            cluster_selection_method=args.cluster_selection_method,
            reassign_noise=not args.no_reassign_noise,
            max_reassign_distance=args.max_reassign_distance,
            separate_review_folders=args.separate_review_folders,
        )
        print("\nClustering completed successfully!")
        print(f"Run directory: {res['run_dir']}")
        print(f"Estimated clusters: {res['summary']['estimated_clusters']}")
        print(f"Noise points: {res['summary']['noise_points']}")
        print(f"Clustered points: {res['summary']['clustered_points']}")
        print(f"Reassigned from noise: {res['summary'].get('reassigned_points', 0)}")
        print(f"Manual review points: {res['summary']['manual_review_points']}")
        return 0
    except Exception as exc:
        print(f"Error during clustering: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
