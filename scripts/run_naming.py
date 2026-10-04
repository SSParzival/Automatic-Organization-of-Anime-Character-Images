#!/usr/bin/env python3
"""
CLI script to run Stage 6: Semantic Cluster Naming with Anime Image Taggers.
"""

import argparse
from pathlib import Path
import sys

from anime_character_organizer.workflows.naming import run_cluster_naming


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Tag representative cluster images and generate semantic rename plan."
    )
    parser.add_argument(
        "--project-dir",
        "-p",
        default=Path("./anime_character_pipeline"),
        type=Path,
        help="Base pipeline directory (default: ./anime_character_pipeline).",
    )
    parser.add_argument(
        "--previous-clustering-run",
        default=None,
        type=Path,
        help="Path to previous 04_hdbscan_clustering_* run directory (default: latest).",
    )
    parser.add_argument(
        "--primary-tagger",
        default="pixai",
        choices=["pixai", "wd14"],
        help="Primary anime image tagger model family (default: pixai).",
    )
    parser.add_argument(
        "--create-named-output",
        action="store_true",
        help="Materialize an additional named folder tree in organized_output/.",
    )
    parser.add_argument(
        "--min-score",
        default=0.70,
        type=float,
        help="Minimum tag confidence score (default: 0.70).",
    )
    parser.add_argument(
        "--min-share",
        default=0.35,
        type=float,
        help="Minimum tag frequency share within cluster (default: 0.35).",
    )

    args = parser.parse_args()

    try:
        res = run_cluster_naming(
            project_dir=args.project_dir,
            previous_clustering_run_dir=args.previous_clustering_run,
            primary_tagger=args.primary_tagger,
            min_character_score=args.min_score,
            min_name_share=args.min_share,
            create_named_output=args.create_named_output,
        )
        print("\nCluster naming completed successfully!")
        print(f"Run directory: {res['run_dir']}")
        print(f"Evaluated clusters: {res['summary']['total_clusters_evaluated']}")
        print(f"Accepted names: {res['summary']['accepted_character_names']}")
        return 0
    except Exception as exc:
        print(f"Error during cluster naming: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
