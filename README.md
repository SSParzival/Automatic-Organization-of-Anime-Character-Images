# Automatic Organization of Anime Character Images

A local research pipeline that groups anime illustrations by visual similarity and optionally proposes character names. It supports dataset curators and researchers preparing unlabelled collections. Human review of clusters and suggested names is required.

Status: professionally organized with documented limitations. Local unit/integration checks and all seven notebooks pass with deterministic model substitutes. All seven also passed bounded execution with actual downloaded models on synthetic images. Character identity quality on representative anime data remains unvalidated. See the [audit report](docs/audit-report.md), [discovery plan](docs/discovery-plan.md), and [validation guide](docs/validation.md).

## Workflow and contract

1. **Audit:** discover images, verify decoding, compute SHA-256 and perceptual hashes, and identify duplicate candidates.
2. **Crop:** exclude exact duplicate non-representatives, detect heads/persons, and select a padded head → person → full-image fallback crop. Review flags preserve uncertainty.
3. **Embed:** extract CCIP features, validate finite nonzero vectors, normalize rows, and persist arrays with an explicit row-to-image manifest.
4. **Cluster:** run HDBSCAN using Euclidean distance on normalized vectors, compute centroids and review diagnostics, and optionally reassign noise within a heuristic distance bound.
5. **Materialize:** create organized folders from original images using hardlinks, copies, or symlinks, then verify accessibility and file sizes.
6. **Name, optional:** sample cluster crops, infer tags, vote on names, and create an additional named tree. Naming does not split or repair mixed clusters.

Visual grouping precedes naming. The system proposes identities; it does not establish ground truth. Model inference is delegated to [dghs-imgutils](https://github.com/deepghs/imgutils). CCIP means Contrastive Character Image Pretraining; its model-specific distance tools differ from this project's normalized Euclidean clustering heuristic. See the [upstream CCIP interface](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html).

Inputs are local images and preceding stage artifacts. Supported discovery extensions are `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`, `.tif`, `.tiff`, and `.avif`; decoding depends on Pillow codecs. Animated files contribute the first frame. Hidden files with matching extensions are eligible. Directories named `sandbox` and all symlink inputs/directories are excluded.

Each execution creates a uniquely reserved directory under `PROJECT_DIR/runs/<stage>_<timestamp>_<suffix>/`. Artifacts include CSV manifests, JSON summaries, crops, NumPy arrays, contact sheets and plots. Stages 5/6 create additional trees under `PROJECT_DIR/organized_output/`. Exact duplicate non-representatives and failed crops/embeddings do not reach materialization. Source images are never moved or intentionally overwritten.

## Architecture

```text
src/anime_character_organizer/
  config.py, exceptions.py    settings and domain errors
  data/                      scanning, inspection and duplicate detection
  preprocessing/             geometry, detection and crop selection
  embeddings/                model extraction and normalization
  clustering/                HDBSCAN, centroids and review diagnostics
  materialization/           shared destination plans, file operations, validation
  tagging/                   inference, sample selection, voting and naming plans
  visualization/             contact sheets and distribution plots
  utils/                     narrow path, run, hashing and serialization routines
  workflows/                 six stages and run_full_pipeline
scripts/                     argparse entry points using package workflows
notebooks/                   seven complete analytical/demonstrative workflows
tests/                       unit, integration and synthetic notebook execution
docs/                        discovery, migration and validation evidence
.github/workflows/ci.yml      configured Python 3.10/3.11 checks
```

Dependencies point from scripts/notebooks to workflows, then domain modules and narrow configuration/utilities. Reusable implementation already lives in the package; notebooks retain experiment parameters, intermediate checks, plots, methodology and interpretation. Ignored historical runs and generated egg-info are local artifacts, not source inputs to CI.

## Installation

Python **3.10+** is declared; local validation used **3.11.14 on Linux**. Python 3.10 and other operating systems require separate validation. CPU inference is supported. GPU use requires a compatible ONNX Runtime provider and drivers; it is not guaranteed by hardware availability alone.

From a checkout of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# Developer tools, tests, notebook execution, build and typing:
python -m pip install -e '.[dev]'
# Interactive JupyterLab and notebook kernels:
python -m pip install -e '.[notebook]'
python -m pip check
```

There is no lockfile. Existing lower-bound dependencies are retained; exact model/provider/dependency versions should be recorded with experimental results. Network access is needed for installation and first model downloads unless dependencies and weights are cached.

## Configuration

The package exposes `AuditConfig`, `CropConfig`, `EmbeddingConfig`, `ClusteringConfig`, `MaterializationConfig`, and `TaggingConfig` dataclasses. They validate supported settings at construction. Workflow functions accept explicit keyword arguments, not a config object. Function signatures and dataclass defaults are synchronized for supported settings. Compatibility fields `CropConfig.min_crop_box_area_ratio` and `MaterializationConfig.create_output_contact_sheet_index` are retained but are not workflow switches; do not pass them as workflow keywords.

Paths supplied to Python/CLI functions are resolved relative to the caller's working directory unless absolute. Explicit `previous_run_dir` takes precedence; otherwise a stage selects the latest matching run containing its required artifacts and `reports/summary.json`. Use explicit prior runs when comparing datasets. Custom output directories must be new or empty. The workspace must be outside the input dataset.

Notebook paths are independent of the launch directory:

- `ANIME_PIPELINE_PROJECT_DIR`: required absolute writable notebook workspace.
- `ANIME_PIPELINE_INPUT_DIR`: absolute notebook input directory; defaults to `input_images` beside the workspace.
- `HF_HOME`: optional upstream model cache directory.

CLI scripts use explicit arguments; they do not read notebook path variables. [.env.example](.env.example) documents examples; `.env` files are not automatically loaded. No application credentials are required.

Defaults such as HDBSCAN epsilon `0.50` and noise distance `0.55` are **uncalibrated project heuristics**. Lower noise rates are not evidence of better identity purity. Library tag acceptance defaults are score/share/weighted/margin `0.70/0.35/0.55/0.08`; naming notebooks retain visible experimental overrides `0.65/0.30/0.50/0.05`.

## Python usage

```python
from pathlib import Path
from anime_character_organizer import run_full_pipeline

results = run_full_pipeline(
    input_dir=Path("/absolute/path/to/images"),
    project_dir=Path("/absolute/path/to/pipeline_workspace"),
    materialization_mode="copy",
    run_tagging=False,
)
print(results["materialize"]["final_output_dir"])
```

Public stage functions are `run_dataset_audit`, `run_crop_preparation`, `run_embedding_extraction`, `run_clustering`, `run_folder_materialization`, and `run_cluster_naming`. Results contain `run_dir`, `summary`, `summary_path`, plus stage-specific tables and arrays. Full-pipeline keys are `audit`, `crop`, `embed`, `cluster`, `materialize`, and optional `naming`.

## Operational scripts

Run scripts from the repository root, or supply their absolute paths from elsewhere after installation. Each supports `--help` and returns zero on success or one on workflow errors. Failed materialization or entirely failed tagging are errors; image-level exclusions remain visible in manifests.

```bash
python scripts/run_pipeline.py --input-dir /path/to/images --project-dir /path/to/workspace --mode copy
python scripts/run_audit.py --input-dir /path/to/images --project-dir /path/to/workspace --phash-threshold 6
python scripts/run_crop_preparation.py --project-dir /path/to/workspace --crop-size 512 --crop-format JPEG
python scripts/run_embeddings.py --project-dir /path/to/workspace --batch-size 16
python scripts/run_clustering.py --project-dir /path/to/workspace --min-cluster-size 2 --min-samples 1 --cluster-selection-epsilon 0.50
python scripts/run_materialization.py --project-dir /path/to/workspace --mode copy
python scripts/run_naming.py --project-dir /path/to/workspace --apply-rename --min-score 0.70 --min-share 0.35
```

Stages 2–5 accept `--previous-run-dir`; naming accepts `--previous-clustering-run`. `--link-mode` aliases `--mode` in materialization and full pipeline; `--apply-rename` aliases `--create-named-output`. These aliases create an additional named tree, never rename source folders. `--with-tagging` enables naming in the full CLI pipeline.

## Notebooks

Before starting Jupyter, from the repository root:

```bash
export ANIME_PIPELINE_INPUT_DIR="/absolute/path/to/images"
export ANIME_PIPELINE_PROJECT_DIR="$PWD/anime_character_pipeline"
python -m ipykernel install --prefix "$PWD/.venv" --name anime-organizer --display-name "Anime organizer"
jupyter lab
```

Select the installed environment's kernel. Run each notebook top to bottom from a fresh kernel:

| Notebook | Role and principal output |
|---|---|
| [01 dataset audit](notebooks/01_dataset_audit.ipynb) | integrity/duplicate inspection; valid_images.csv |
| [02 crop preparation](notebooks/02_crop_preparation.ipynb) | crop quality review; crop_manifest_with_review_flags.csv |
| [03 CCIP embeddings](notebooks/03_ccip_embedding_extraction.ipynb) | vector/norm checks; arrays and successful_embedding_manifest.csv |
| [04 HDBSCAN clustering](notebooks/04_hdbscan_clustering.ipynb) | diagnostic plots/review; folder_assignment_manifest.csv |
| [05 folder materialization](notebooks/05_non_destructive_folder_materialization.ipynb) | accessible organized originals and validation report |
| [06 cluster naming](notebooks/06_cluster_naming_with_anime_tagger.ipynb) | weighted votes, rename plan and optional named output |
| [Master pipeline](notebooks/pipeline_master.ipynb) | explicitly chained alternative to running stages separately |

Stage 6 reads Stage 4 artifacts and originals; Stage 5 is not an implementation prerequisite. Rerunning creates new runs. Notebook outputs in the repository are not evidence of completed inference; the master retains a historical user configuration output.

## Development and validation

```bash
mkdir -p .audit/tmp
export TMPDIR="$PWD/.audit/tmp"
python -m pytest tests --ignore-glob='**/sandbox/**'
python -m ruff check src scripts tests notebooks --force-exclude --exclude sandbox
python -m ruff format --check src scripts tests notebooks --force-exclude --exclude sandbox
python -m mypy --cache-dir .audit/mypy-cache
python -m tests.notebook_smoke --output-dir .audit/new-notebook-smoke
```

Notebook smoke execution validates structures and executes all seven source notebooks in fresh kernels with synthetic images and substituted model inference; it writes executed copies and `results.json` to a new/empty evidence directory. It does not validate model predictions. Use `--live` only when weights are available and downloading is acceptable. The type check covers the package with third-party imports treated permissively; it does not provide fully typed contracts for pandas/NumPy integrations. Build validation and exact commands are documented in the [validation guide](docs/validation.md).

Contribution requirements: protect sources, add regression tests for confirmed defects, keep notebook narratives accurate, format affected files and review the diff. Every directory named `sandbox` is excluded from discovery, testing, formatting, staging and modification. Never run broad cleanup commands. CI is configured but has not been executed on the remote service during this audit.

## Limitations, troubleshooting and privacy

- Multi-character scenes select one region. Stylized art, twin designs and costume changes can merge/split identities. No labelled benchmark or accuracy claim is supplied.
- Exact duplicate non-representatives are intentionally excluded; pHash candidates are retained by default. Transitive pHash groups can contain endpoints farther apart than the threshold. Candidate counts can be quadratic; the embedding matrix remains resident in memory despite batched inference.
- Detector failures emit warnings and use reviewable full-image fallbacks. Inspect those warnings; crop completion alone does not prove working detectors. Missing CCIP weights raise `EmbeddingError`; run preceding stages or select an explicit completed run to address `RunNotFoundError`.
- Hardlinks share image bytes with originals, and symlinks depend on source availability. Use copy mode for editable independent output. Hardlink fallback is limited to filesystem/permission errors and recorded in results.
- Validation checks existence and byte size, not cryptographic byte equality. Invalid output operations raise an error with retained reports. This is not an adversarial filesystem security boundary; avoid concurrently modifying input/output paths.
- The sklearn fallback adjusts min_samples to preserve the contrib backend convention. It estimates outliers from membership scores instead of computing GLOSH, so diagnostics still differ by backend.
- Model downloads contact upstream hosts. Inference is designed to run locally; no image-upload integration is implemented. Manifests and notebook outputs can reveal personal paths, filenames and image-derived features. Keep generated runs private and review artifacts before sharing.
- Pillow verification does not make arbitrary image decoding safe. Use trusted datasets and keep dependencies current. Explicit protected paths and symlink directory aliases are rejected.

License metadata declares MIT, but this checkout does not contain a standalone LICENSE text. The maintainer should confirm/distribute the intended license before publishing. Users must hold appropriate rights to process or redistribute source images; outputs are research proposals requiring human review.
