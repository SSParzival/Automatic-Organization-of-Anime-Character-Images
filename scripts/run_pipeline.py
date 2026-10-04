#!/usr/bin/env python3
"""
CLI script to run the complete end-to-end anime character organization pipeline.
"""

import argparse
import sys
from pathlib import Path

from anime_character_organizer.workflows.pipeline import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run complete Automatic Organization of Anime Character Images pipeline."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        type=Path,
        help="Path to folder containing source images to organize.",
    )
    parser.add_argument(
        "--project-dir",
        "-p",
        default=Path("./anime_character_pipeline"),
        type=Path,
        help="Base pipeline workspace directory (default: ./anime_character_pipeline).",
    )
    parser.add_argument(
        "--mode",
        "--link-mode",
        "-m",
        dest="mode",
        default="hardlink",
        choices=["hardlink", "copy", "symlink"],
        help="Materialization link mode (default: hardlink).",
    )
    parser.add_argument(
        "--min-cluster-size",
        default=2,
        type=int,
        help="HDBSCAN min_cluster_size parameter (default: 2).",
    )
    parser.add_argument(
        "--min-samples",
        default=1,
        type=int,
        help="HDBSCAN min_samples parameter (default: 1).",
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        default=0.50,
        type=float,
        help="HDBSCAN cluster selection epsilon merge threshold (default: 0.50).",
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
        help="Split review items into separate _review folders.",
    )
    parser.add_argument(
        "--with-tagging",
        action="store_true",
        help="Also execute stage 6 semantic naming with anime tagger.",
    )

    args = parser.parse_args()

    try:
        results = run_full_pipeline(
            input_dir=args.input_dir,
            project_dir=args.project_dir,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            reassign_noise=not args.no_reassign_noise,
            max_reassign_distance=args.max_reassign_distance,
            separate_review_folders=args.separate_review_folders,
            run_tagging=args.with_tagging,
            materialization_mode=args.mode,
        )
        print("\n=======================================================")
        print("Pipeline finished successfully!")
        print(f"Organized output: {results['materialize']['final_output_dir']}")
        print(f"Total organized files: {results['materialize']['summary']['total_files']}")
        print(f"Total folders created: {results['materialize']['summary']['total_folders']}")
        print("=======================================================")
        return 0
    except Exception as exc:
        print(f"\nPipeline execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
