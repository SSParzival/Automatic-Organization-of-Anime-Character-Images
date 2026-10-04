#!/usr/bin/env python3
"""
CLI script to run the complete end-to-end anime character organization pipeline.
"""

import argparse
from pathlib import Path
import sys

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
        "-m",
        default="hardlink",
        choices=["hardlink", "copy", "symlink"],
        help="Materialization link mode (default: hardlink).",
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
