#!/usr/bin/env python3
"""
CLI script to run Stage 1: Dataset Audit and Duplicate Detection.
"""

import argparse
from pathlib import Path
import sys

from anime_character_organizer.workflows.audit import run_dataset_audit


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Scan image directory, validate files, and identify duplicates."
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
        help="Pipeline directory where runs and tables are saved (default: ./anime_character_pipeline).",
    )
    parser.add_argument(
        "--phash-threshold",
        default=6,
        type=int,
        help="Maximum Hamming distance for perceptual duplicate grouping (default: 6).",
    )
    parser.add_argument(
        "--max-workers",
        default=None,
        type=int,
        help="Worker threads for parallel file inspection (default: CPU count).",
    )

    args = parser.parse_args()

    try:
        res = run_dataset_audit(
            input_dir=args.input_dir,
            project_dir=args.project_dir,
            phash_threshold=args.phash_threshold,
            max_workers=args.max_workers,
        )
        print("\nAudit completed successfully!")
        print(f"Run directory: {res['run_dir']}")
        print(f"Total files: {res['summary']['total_candidate_files']}")
        print(f"Valid images: {res['summary']['valid_images']}")
        print(f"Exact duplicate groups: {res['summary']['exact_duplicate_groups']}")
        print(f"Perceptual duplicate groups: {res['summary']['perceptual_duplicate_groups']}")
        return 0
    except Exception as exc:
        print(f"Error during audit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
