"""Execute every notebook in a fresh kernel against controlled local fixtures.

Run: python -m tests.notebook_smoke --output-dir .audit/notebooks
Model inference is substituted by default; --live attempts actual inference.
Executed copies are evidence artifacts and are never written over source notebooks.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient
from PIL import Image

from anime_character_organizer.utils.paths import checked_path

NOTEBOOK_NAMES = (
    "01_dataset_audit.ipynb",
    "02_crop_preparation.ipynb",
    "03_ccip_embedding_extraction.ipynb",
    "04_hdbscan_clustering.ipynb",
    "05_non_destructive_folder_materialization.ipynb",
    "06_cluster_naming_with_anime_tagger.ipynb",
    "pipeline_master.ipynb",
)


def execute_notebooks(output_dir: Path, *, live: bool = False, timeout: int = 90) -> list[dict]:
    """Execute exact known notebook paths with isolated workspaces and model mode."""
    root = Path(__file__).absolute().parents[1]
    output_dir = checked_path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Execution evidence directory must be new or empty.")
    output_dir.mkdir(parents=True, exist_ok=True)
    images = output_dir / "images"
    images.mkdir()
    for index in range(10):
        Image.new("RGB", (96, 96), (index * 20, 50, 100)).save(images / f"image{index:02d}.png")
    fixture = root / "tests/fixtures/model_substitutes.py"
    results = []
    for name in NOTEBOOK_NAMES:
        source = root / "notebooks" / name
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        workspace = output_dir / ("master_workspace" if name == "pipeline_master.ipynb" else "staged_workspace")
        bootstrap = (
            "import os\n"
            + f"os.environ['ANIME_PIPELINE_PROJECT_DIR'] = {str(workspace)!r}\n"
            + f"os.environ['ANIME_PIPELINE_INPUT_DIR'] = {str(images)!r}\n"
        )
        if not live:
            bootstrap += f"import runpy\n_fixture = runpy.run_path({str(fixture)!r})\n_model_patches = _fixture['install_substitutes']()\n"
        notebook.cells.insert(0, nbformat.v4.new_code_cell(bootstrap))
        manager = KernelManager(kernel_name="python3")
        # Explicitly use the invoking environment instead of an unrelated registered kernel.
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(
            notebook,
            timeout=timeout,
            km=manager,
            resources={"metadata": {"path": str(output_dir)}},
            record_timing=False,
        )
        record = {"notebook": name, "mode": "live" if live else "synthetic_model_substitutes"}
        try:
            client.execute()
            record["result"] = "passed"
        except Exception as exc:
            record.update(result="failed", error_type=type(exc).__name__, error=str(exc)[-2000:])
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
        nbformat.write(notebook, output_dir / name)
        results.append(record)
        print(f"{name}: {record['result']} ({record['mode']})", flush=True)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--live", action="store_true", help="Attempt actual model inference; weights may require downloads."
    )
    parser.add_argument("--timeout", type=int, default=90, help="Maximum seconds per cell.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    results = execute_notebooks(args.output_dir, live=args.live, timeout=args.timeout)
    return int(any(row["result"] != "passed" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
