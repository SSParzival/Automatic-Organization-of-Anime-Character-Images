# Automatic Organization of Anime Character Images

Organizes a folder of anime illustrations, fan art included, into one folder per character, named after that character. Images are grouped by visual similarity and every group is named with an anime character tagger; tagging is always part of a run. Source images are never moved, modified or deleted. Review the results by hand: names are model predictions.

## Install

Python 3.10+ is required.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .            # add '.[dev]' for tests
```

Models are downloaded from Hugging Face the first time they are used (about 2.2 GB, 1.9 GB of it for the tagger). Set `HF_HOME` to change the cache location.

## Configure

```bash
cp .env.example .env
```

Edit `.env`:

- `ANIME_PIPELINE_INPUT_DIR`: folder with your images. It is scanned recursively and only read.
- `ANIME_PIPELINE_PROJECT_DIR`: workspace for outputs. It must be outside the input folder.

`.env` is git-ignored. Command-line flags and exported variables override it.

## Run

`scripts/run_pipeline.py` is the only entry point:

```bash
python scripts/run_pipeline.py --mode copy
```

Key flags:

| Flag | Effect |
|---|---|
| `--max-images N` | process a representative sample of N images (stratified by folder); `--sample-strategy first\|random` changes how they are picked |
| `--mode hardlink\|copy\|symlink` | how organized files reference the originals (default `hardlink`; `copy` gives independent files) |
| `--input-dir`, `--project-dir` | override the `.env` paths |
| `--min-cluster-size`, `--cluster-selection-epsilon`, `--max-reassign-distance` | clustering sensitivity (defaults `2`, `0.50`, `0.55` are heuristics, not calibrated values) |

Run `python scripts/run_pipeline.py --help` for every flag.

## Outputs

All outputs are written under `PROJECT_DIR`:

- `organized_output/anime_organized_<run>/<character_name>/`: one folder per character, named with its Danbooru tag (for example `hatsune_miku` or `rem_(re_zero)`). Each folder holds `_folder_manifest.csv` (source paths and naming evidence), `_folder_info.json` and `_contact_sheet.jpg`.
- `organized_output/anime_organized_<run>/_needs_review/`: images without a confident character name.
- `runs/<stage>_<run>/`: manifests (CSV), summaries (JSON), crops, embeddings, tagger scores, contact sheets and plots for each stage. Every execution creates new runs, so earlier results are never overwritten.

## How it works

1. **Audit**: finds the images, verifies that they decode, and hashes them. Exact duplicates are reduced to one copy.
2. **Crop**: detects the character's head, falling back to the whole person and then to the full image, and saves a normalized crop.
3. **Embed**: extracts [CCIP](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html) character-identity vectors from the crops.
4. **Cluster**: groups the vectors with HDBSCAN. Borderline outliers are reassigned to the nearest cluster when they are close enough.
5. **Name**: tags crops with [PixAI Tagger v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0) and decides a name for every image:
   - up to 12 representative crops per cluster vote; the cluster takes the winning character when it scores at least 0.70 on its images, appears in at least 35 % of them, averages 0.55 or more and leads the runner-up by 0.08;
   - images that HDBSCAN left unclustered, and every image of a cluster whose vote failed, are tagged one by one and take their own top character when it scores at least 0.70 and leads the next candidate by 0.08. A cluster that mixes several characters is split this way;
   - anything else goes to `_needs_review`. Clusters that receive the same name share one folder.
6. **Materialize**: builds the character folders from the original images and verifies them.

Inference runs locally on CPU by default: crops and embeddings use [dghs-imgutils](https://github.com/deepghs/imgutils), the tagger uses ONNX Runtime. GPU inference needs a compatible ONNX Runtime provider.

## Tagger choice

PixAI Tagger v1.0 replaced PixAI v0.9 (with WD14 SwinV2 v3 as fallback) because it recognizes more characters and recognizes them more accurately:

| Tagger | Character tags | Data cutoff | Character micro-F1 (PixAI benchmark) |
|---|---:|---|---:|
| PixAI Tagger v1.0 | 8,308 | May 2026 | 0.924 |
| PixAI Tagger v0.9 | 3,720 | not stated | 0.820 |
| WD14 SwinV2 v3 | 2,751 | Feb 2024 | not evaluated |

The F1 values come from the [PixAI v1.0 model card](https://huggingface.co/pixai-labs/pixai-tagger-v1.0#common-tag-benchmark) (2,099 shared character tags, 50,416 images). AnimeTIMM SigLIP Giant scores 0.927 there, within 0.3 points, but has 12,476 tags in total and requires a gated Hugging Face account. On 200 random fan art crops, v1.0 named 135 confidently against 101 for v0.9, and the two agreed on 75 of the 85 crops both named ([details](docs/validation.md#tagger-comparison)).

The tagger runs from an ONNX export of the official weights ([`noaione/pixai-tagger-v1.0-onnx`](https://huggingface.co/noaione/pixai-tagger-v1.0-onnx)), pinned to commit `68e8f4f`, so no PyTorch install is needed. Its outputs were checked against the official PyTorch model (see [docs/validation.md](docs/validation.md#tagger-comparison)).

## Python API

```python
from anime_character_organizer import run_full_pipeline

results = run_full_pipeline("/path/to/images", "/path/to/workspace", materialization_mode="copy")
print(results["materialize"]["final_output_dir"])
```

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check src scripts tests && python -m ruff format --check src scripts tests
```

Tests use synthetic images and model stand-ins, so they need no downloads. [docs/validation.md](docs/validation.md) covers type checking, the tagger comparison and package builds.

## Limitations

- The tagger only names characters in its vocabulary. Original characters, very obscure characters and characters that appeared after May 2026 go to `_needs_review`.
- Tagging is the slowest stage: on an Intel i5-14600KF it takes about 9 s per tagged crop, and a run peaks at about 6 GB of RAM. Each image is tagged at most once, and named clusters only tag up to 12 crops. Use `--max-images` to try settings on a sample first.
- Each image yields one crop, so in multi-character scenes only one character is used.
- Similar character designs or heavy costume changes can merge or split clusters. A merged cluster is split by per-image tags only when its vote fails.
- With `hardlink` mode, organized files share their bytes with the originals; with `symlink` mode, they break if the source folder moves.
- Manifests contain file paths and image-derived features. Keep the workspace private.
