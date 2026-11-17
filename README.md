# Automatic Organization of Anime Character Images

Organizes a folder of anime illustrations into one folder per character. Images are grouped by visual similarity, and each folder can then be named after the character it contains. Source images are never moved, modified or deleted. Review the results by hand: clusters and names are proposals.

## Install

Python 3.10+ is required.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .            # add '.[dev]' for tests, '.[notebook]' for Jupyter
```

Models are downloaded from Hugging Face the first time they are used. Set `HF_HOME` to change the cache location.

## Configure

```bash
cp .env.example .env
```

Edit `.env`:

- `ANIME_PIPELINE_INPUT_DIR`: folder with your images. It is scanned recursively and only read.
- `ANIME_PIPELINE_PROJECT_DIR`: workspace for outputs. It must be outside the input folder.

`.env` is git-ignored. Command-line flags and exported variables override it.

## Run

```bash
python scripts/run_pipeline.py --mode copy --with-tagging
```

Key flags:

| Flag | Effect |
|---|---|
| `--max-images N` | process a representative sample of N images (stratified by folder); `--sample-strategy first\|random` changes how they are picked |
| `--mode hardlink\|copy\|symlink` | how organized files reference the originals (default `hardlink`; `copy` gives independent files) |
| `--with-tagging` | name the folders after the detected characters |
| `--input-dir`, `--project-dir` | override the `.env` paths |
| `--min-cluster-size`, `--cluster-selection-epsilon`, `--max-reassign-distance` | clustering sensitivity (defaults `2`, `0.50`, `0.55` are heuristics, not calibrated values) |

Run `python scripts/run_pipeline.py --help` for every flag. Individual stages can also be run with `scripts/run_<stage>.py`, and each of those scripts supports `--help`.

## Outputs

All outputs are written under `PROJECT_DIR`:

- `organized_output/anime_organized_<run>/cluster_XXXXX_unknown/`: one folder per visual cluster. Images the clustering could not place go to `_needs_review_noise/`.
- `organized_output/anime_named_<run>/<character_name>/`: written with `--with-tagging`. These are the same clusters, named after their characters.
- `runs/<stage>_<run>/`: manifests (CSV), summaries (JSON), crops, embeddings, contact sheets and plots for each stage. Every execution creates new runs, so earlier results are never overwritten.

## How it works

1. **Audit**: finds the images, verifies that they decode, and hashes them. Exact duplicates are reduced to one copy.
2. **Crop**: detects the character's head, falling back to the whole person and then to the full image, and saves a normalized crop.
3. **Embed**: extracts [CCIP](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html) character-identity vectors from the crops.
4. **Cluster**: groups the vectors with HDBSCAN. Borderline outliers are reassigned to the nearest cluster when they are close enough.
5. **Materialize**: builds the per-cluster folders from the original images and verifies them.
6. **Name** (`--with-tagging`): runs an anime tagger (PixAI, with WD14 as fallback) on sample crops from each cluster. A character name is accepted only when it wins the vote by enough score, share and margin. Clusters without an accepted name keep their `cluster_XXXXX_unknown` name. Naming does not split clusters that contain several characters.

Inference runs locally with [dghs-imgutils](https://github.com/deepghs/imgutils), on CPU by default. GPU inference needs a compatible ONNX Runtime provider.

## Python API

```python
from anime_character_organizer import run_full_pipeline

results = run_full_pipeline("/path/to/images", "/path/to/workspace", materialization_mode="copy", run_tagging=True)
print(results["materialize"]["final_output_dir"])
```

The six stages are also available individually, as `run_dataset_audit`, `run_crop_preparation`, `run_embedding_extraction`, `run_clustering`, `run_folder_materialization` and `run_cluster_naming`. The `notebooks/` folder walks through each stage interactively; the notebooks read `ANIME_PIPELINE_INPUT_DIR` and `ANIME_PIPELINE_PROJECT_DIR` from the environment.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check src scripts tests && python -m ruff format --check src scripts tests
```

Tests use synthetic images and model stand-ins, so they need no downloads. [docs/validation.md](docs/validation.md) covers notebook smoke runs and type checking.

## Limitations

- Each image yields one crop, so in multi-character scenes only one character is used.
- Similar character designs or heavy costume changes can merge or split clusters. No accuracy benchmark is provided.
- With `hardlink` mode, organized files share their bytes with the originals; with `symlink` mode, they break if the source folder moves.
- Manifests contain file paths and image-derived features. Keep the workspace private.
