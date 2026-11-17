#!/usr/bin/env python3
"""
CLI script to run the complete end-to-end anime character organization pipeline.
"""

import argparse
import os
import sys
from pathlib import Path

from anime_character_organizer.utils.paths import load_dotenv
from anime_character_organizer.workflows.pipeline import run_full_pipeline


def main(argv=None):
    load_dotenv()
    env_input = os.environ.get("ANIME_PIPELINE_INPUT_DIR")
    parser = argparse.ArgumentParser(
        description="Run complete Automatic Organization of Anime Character Images pipeline."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=env_input is None,
        default=Path(env_input) if env_input else None,
        type=Path,
        help="Folder with source images (default: ANIME_PIPELINE_INPUT_DIR from .env).",
    )
    parser.add_argument(
        "--project-dir",
        "-p",
        default=Path(os.environ.get("ANIME_PIPELINE_PROJECT_DIR", "./anime_character_pipeline")),
        type=Path,
        help="Pipeline workspace (default: ANIME_PIPELINE_PROJECT_DIR or ./anime_character_pipeline).",
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
        "--max-images",
        type=int,
        default=None,
        help="Process only a representative sample of this many images (default: all).",
    )
    parser.add_argument(
        "--sample-strategy",
        default="stratified",
        choices=["stratified", "first", "random"],
        help="Sampling strategy used with --max-images (default: stratified).",
    )

    args = parser.parse_args(argv)

    try:
        results = run_full_pipeline(
            input_dir=args.input_dir,
            project_dir=args.project_dir,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            reassign_noise=not args.no_reassign_noise,
            max_reassign_distance=args.max_reassign_distance,
            materialization_mode=args.mode,
            max_images=args.max_images,
            sample_strategy=args.sample_strategy,
        )
        print("\n=======================================================")
        print("Pipeline finished successfully!")
        print(f"Organized output: {results['materialize']['final_output_dir']}")
        naming = results["naming"]["summary"]
        print(f"Character folders: {naming['character_folders']}")
        print(f"Images named: {naming['images_named_by_cluster_vote'] + naming['images_named_by_image_tags']}")
        print(f"Images needing review: {naming['images_needing_review']}")
        print("=======================================================")
        return 0
    except Exception as exc:
        print(f"\nPipeline execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
