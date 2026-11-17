# Automatic Organization of Anime Character Images

Organizes a folder of anime illustrations, fan art included, into one folder per character, named after that character. Images are grouped by visual similarity and every group is named with an anime character tagger; tagging is always part of a run. Source images are never moved, modified or deleted. Review the results by hand: names are model predictions.

## Install

Python 3.10+ is required.

The project runs on the system Python (checked with 3.14). On distributions that mark it as externally managed, install into the user site:

```bash
python -m pip install --user --break-system-packages -e ".[dev]"   # '.[dev]' adds the test tools
python -m pip uninstall --break-system-packages -y numpy           # only if the system has NumPy 2
```

dghs-imgutils declares `numpy<2`, so pip puts NumPy 1.26 in the user site, where it shadows the system NumPy 2 and breaks system packages built for it. The pipeline and its tests pass with NumPy 2.5, hence the second command.

Models are downloaded from Hugging Face the first time they are used (about 2.2 GB, 1.9 GB of it for the tagger). Set `HF_HOME` to change the cache location. The first run also builds an int8 copy of the tagger (about 15 s and 6.5 GB of RAM, once) in `~/.cache/anime-character-organizer/` (500 MB; `XDG_CACHE_HOME` moves it).

## Configure

```bash
cp .env.example .env
```

Edit `.env`:

- `ANIME_PIPELINE_INPUT_DIR`: folder with your images. It is scanned recursively and only read.
- `ANIME_PIPELINE_PROJECT_DIR`: workspace for outputs. It must be outside the input folder.

Optional, for reverse image search (free accounts; see [Online identification](#online-identification)):

- `DANBOORU_LOGIN`, `DANBOORU_API_KEY`: Danbooru username and the API key from your Danbooru profile.
- `SAUCENAO_API_KEY`: API key from your SauceNAO account.

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
| `--offline` | name characters with the local tagger only; nothing is sent to Danbooru or SauceNAO |

Run `python scripts/run_pipeline.py --help` for every flag.

## Outputs

All outputs are written under `PROJECT_DIR`:

- `organized_output/anime_organized_<run>/<character_name>/`: one folder per character, named with its Danbooru tag (for example `hatsune_miku` or `rem_(re_zero)`). Each folder holds `_folder_manifest.csv` (source paths and naming evidence), `_folder_info.json` and `_contact_sheet.jpg`.
- `organized_output/anime_organized_<run>/_needs_review/`: images without a confident character name.
- `runs/<stage>_<run>/`: manifests (CSV), summaries (JSON), crops, embeddings, tagger scores, contact sheets and plots for each stage. Every execution creates new runs, so earlier results are never overwritten.
- `cache/`: online answers kept across runs (`online_identification.json`, by file MD5, and `character_names.json`), so no image is sent to the same service twice. Delete it to query again.

## How it works

1. **Audit**: finds the images, verifies that they decode, and hashes them. Exact duplicates are reduced to one copy.
2. **Crop**: detects the character's head, falling back to the whole person and then to the full image, and saves a normalized crop.
3. **Embed**: extracts [CCIP](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html) character-identity vectors from the crops.
4. **Cluster**: groups the vectors with HDBSCAN. Borderline outliers are reassigned to the nearest cluster when they are close enough.
5. **Name**: decides a character for every image, trying the cheapest reliable source first:
   - the MD5 of every image is looked up on Danbooru, 100 images per request and without uploading anything. An unmodified copy of a Danbooru post takes that post's character tags, which people assign by hand;
   - each cluster is identified online from up to 3 representative images (exact matches, then reverse image search when [credentials](#online-identification) are set). The cluster takes a character listed by at least two of its identified images and by two thirds of them;
   - clusters without an online name are named by [PixAI Tagger v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0): up to 12 representative crops vote, and the winning character needs at least 0.70 on its images, presence in 35 % of them, an average of 0.55 and a lead of 0.08 over the runner-up;
   - images that HDBSCAN left unclustered, images of unnamed clusters, cluster members HDBSCAN placed with low confidence and members of clusters named online without a match of their own are identified online one by one. A match listing one character names the image; when it lists several (a group picture), the image's cluster character wins if listed, otherwise the one the local tagger scores highest on the crop. An image's own online match overrides its cluster's name. Images without an online match take their own top tag when it scores at least 0.70 and leads the next by 0.08. Such a tag also moves an image out of a named cluster when the cluster's character is absent from the image's tags;
   - every name is mapped to its canonical Danbooru character tag, so spellings, renamed tags and costume variants share one folder (`saber` and `saber_(fate)` become `artoria_pendragon_(fate)`, `hatsune_miku_(append)` becomes `hatsune_miku`);
   - anything else goes to `_needs_review`. Online lookups run in a background thread while the CPU tags, and their answers are cached, so a second run sends nothing already sent.
6. **Materialize**: builds the character folders from the original images and verifies them.

Inference runs locally on CPU by default: crops and embeddings use [dghs-imgutils](https://github.com/deepghs/imgutils), the tagger uses ONNX Runtime. GPU inference needs a compatible ONNX Runtime provider.

## Online identification

On a random sample of 300 images, looking images up before tagging them named 82 % of the images instead of 72 %, made no error on the 81 images with Danbooru labels (the previous version made 5) and tagged 27 % fewer crops ([details](docs/validation.md#online-identification-and-naming-strategies)). In the whole 23,847-image collection, 23 % of the files are unmodified Danbooru posts, found by hash in 3 minutes.

Danbooru's hash lookup and name resolution need no account. Reverse image search needs free credentials in `.env`:

| Service | Variables | Searches | What it finds |
|---|---|---|---|
| Danbooru IQDB | `DANBOORU_LOGIN`, `DANBOORU_API_KEY` | about 1 per second | Danbooru posts, with their character tags |
| SauceNAO | `SAUCENAO_API_KEY` | 4 per 30 s and 100 per day on a free account, shared by every free account on the same IP | Danbooru, Gelbooru, pixiv, X and other indexes; pixiv and X results are read through Danbooru |

SauceNAO is queried only when Danbooru finds nothing. A service that runs out of quota is skipped for the rest of the run, and its images are retried on the next one. A match names an image when its similarity is at least 85 (IQDB score or SauceNAO similarity); on real images, IQDB matches either score above 70 and agree with the tagger, or below 40 and are unrelated ([details](docs/validation.md#reverse-image-search)).

The Danbooru credentials are checked once per run. When Danbooru rejects them, the run warns, drops them and keeps the anonymous hash lookups. The SauceNAO key is sent in the request body, so it never appears in a URL or an error message. Keep `.env` readable only by your user (`chmod 600 .env`).

To check the credentials in `.env`, run `pytest -m online`: it sends one hash lookup, one profile request and one search per configured service with a synthetic image, and skips the services without credentials.

**Privacy.** With online lookup on, the MD5 of every image and the character names found are sent to Danbooru. With credentials, a JPEG of at most 768 px of each searched image is uploaded to Danbooru IQDB and, when Danbooru finds nothing, to SauceNAO. `--offline` sends nothing.

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

- The local tagger only names characters in its vocabulary. Other characters are named only when an online match finds them; original characters and art that is not on Danbooru or in SauceNAO's indexes go to `_needs_review`.
- Online identification depends on Danbooru and SauceNAO being reachable and on their quotas. When a service fails or runs out, the run goes on with the local tagger and the next run retries those images.
- Names follow Danbooru's tag relations: costume and form tags join their character, and the few player-character roles Danbooru links the same way share one folder (Stelle and Caelus become `trailblazer_(honkai:_star_rail)`).
- Tagging is the slowest stage: on an Intel i5-14600KF the int8 tagger takes about 5 s per tagged crop (8.9 s in float32), and a run peaks at about 6 GB of RAM. Each image is tagged at most once; clusters named online only tag the members no online match confirms, and other named clusters tag up to 12. Use `--max-images` to try settings on a sample first.
- Each image yields one crop, so in multi-character scenes only one character is used. When an online match lists several characters, the one the tagger scores highest on the crop is chosen.
- Similar character designs or heavy costume changes can merge or split clusters. A merged cluster is split by per-image names only when its vote fails; an online cluster name needs two thirds of the identified images to agree, and images with their own online match keep it.
- With `hardlink` mode, organized files share their bytes with the originals; with `symlink` mode, they break if the source folder moves.
- Manifests contain file paths and image-derived features. Keep the workspace private.
