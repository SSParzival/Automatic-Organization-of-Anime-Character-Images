# Validation and reproducibility

This guide explains how to reproduce checks without accessing protected directories. Run commands from the repository root after installing `.[dev]`.

## Local checks

```bash
mkdir -p .audit/tmp
export TMPDIR="$PWD/.audit/tmp"
export MPLBACKEND=Agg
python -m pip check
python -m pytest tests --ignore-glob='**/sandbox/**'
python -m ruff check src scripts tests --force-exclude --exclude sandbox
python -m ruff format --check src scripts tests --force-exclude --exclude sandbox
python -m mypy --cache-dir .audit/mypy-cache
```

Pytest also excludes every directory named `sandbox` through `norecursedirs`. Ruff uses both configured exclusions and explicit protected-directory exclusion. MyPy excludes `(^|/)sandbox/`. Do not supply explicit protected files to any tool. Package discovery lists exactly the ten supported packages and does not recursively discover arbitrary packages.

The suite combines unittest characterization tests with pytest regression/integration tests. Use pytest to run the complete suite; unittest discovery alone does not collect pytest functions. Model inference in tests uses deterministic fixtures, not downloaded models or production data. CLI tests run from unrelated temporary working directories. Model substitutes return two known synthetic embedding groups and one fixed character tag per group; those values are not recommendations, predictions or accuracy measurements. The tagger itself is covered by unit tests of its preprocessing and of how its output block is decoded. Online services are replaced in every test by a client that fails like an unreachable network; tests of online identification use an in-memory stand-in shaped like Danbooru's API, so no test of the default run sends a request. Tests marked `online` (`tests/live/`) query the real services with the credentials in `.env`; they are deselected unless pytest runs with `-m online`.

MyPy covers all 50 package source files. Missing third-party stubs are ignored and imports followed silently, so a successful run does not establish fully typed pandas/NumPy interfaces.

## Tagger comparison

The naming stage uses PixAI Tagger v1.0 instead of the previous PixAI v0.9 with WD14 fallback. The checks below ran on 2026-11-17 on an Intel Core i5-14600KF (CPU only, 16 GB RAM).

**Vocabulary.** From the published tag lists: PixAI v1.0 has 8,308 character tags, PixAI v0.9 3,720 and WD14 v3 (SwinV2 and EVA02-Large) 2,751. Only 81 of v0.9's characters are missing from v1.0. On the model card's benchmark, AnimeTIMM SigLIP Giant scores 0.927 character micro-F1 against 0.924 for v1.0, but AnimeTIMM repositories are gated behind a Hugging Face login.

**Local fan art sample.** 200 images drawn at random from a private collection of 23,853 fan art images (`--max-images 200 --sample-strategy random`), cropped by the crop stage, then tagged by each model on the same 512 px crops. "Named" uses the per-image rule of the naming stage (top character at least 0.70, ahead of the next by 0.08).

| | PixAI v0.9 | PixAI v1.0 |
|---|---:|---:|
| Crops named | 101 (50 %) | 135 (68 %) |
| Median time per crop | 1.08 s | 9.20 s |
| Peak memory | 2.25 GB | 4.80 GB |

Both models named 85 crops and agreed on 75. Of the 50 crops only v1.0 named, 42 were characters absent from v0.9's vocabulary; so were v1.0's answers in 8 of the 10 disagreements. The largest group among them is Zenless Zone Zero (released 2024). Of the 16 crops only v0.9 named, a visual review found several clear errors (a Steven Universe character tagged `harime_nui`, a pig sketch tagged `doraemon_(character)`) and several plausible names that v1.0 scored between 0.45 and 0.61. v1.0 scores are bimodal: lowering the 0.70 threshold to 0.50 would add only 4 named crops, so the threshold was kept. No ground-truth labels exist for this sample, so these figures measure coverage and agreement, not accuracy.

**Export fidelity.** The pipeline runs a third-party ONNX export (`noaione/pixai-tagger-v1.0-onnx`, commit `68e8f4f`) with its own preprocessing. On 12 of the sample crops, its character probabilities matched the official PyTorch pipeline (`pixai-labs/pixai-tagger-v1.0`, commit `9fe10ad`, transformers 4.57.6): same top character on 12/12, largest absolute difference 0.012.

**Memory.** With ONNX Runtime's default CPU memory arena, resident memory reached 7.9 GB after the second 1008 px inference and stayed there. With the arena and memory pattern disabled, as the pipeline does, it returns to 2.1 GB between inferences with a 4.8 GB peak (14 consecutive inferences). Before naming, the pipeline also releases the detector and CCIP sessions that imgutils keeps cached (about 2 GB). A full run on 40 images of the collection then peaked at 5.8 GB and took 6 min 50 s: 19 character folders, 25 images named (7 by cluster vote, 18 by their own tags), 15 sent to `_needs_review`.

Every image is tagged at most once, so on this CPU a full run spends at most about 9 s per image in the naming stage; clusters with an accepted name only tag up to 12 representative crops.

## Online identification and naming strategies

The checks below ran on 2026-11-17 on the same machine and collection (23,847 readable images).

**Exact matches over the whole collection.** Hashing every image took 56 s and looking the digests up on Danbooru 118 s (100 digests per request, nothing uploaded). 5,416 images (22.7 %) are unmodified copies of Danbooru posts: 2,788 list one character, 2,548 several (group pictures, or a character with a mascot) and 80 none.

**Name resolution.** Resolving the tagger's 8,308 character names against Danbooru maps 1,272 of them to another tag, mostly costume and form tags (`akemi_homura_(magical_girl)` to `akemi_homura`, `shigure_kai_ni_(kancolle)` to `shigure_(kancolle)`) and renamed tags (`saber_(fate)` to `artoria_pendragon_(fate)`); 7 cannot be resolved and keep their spelling. The 8,308 names become 7,106 characters. About 100 mappings share no word with their target; reviewing them found alternate identities (`superman` to `clark_kent`, `gorgon_(fate)` to `medusa_(fate)`) and three player-character roles that merge distinct designs (`stelle` and `caelus` to `trailblazer_(honkai:_star_rail)`, the `hildr`, `ortlinde` and `thrud` Valkyries, the two Dragon Quest III heroes).

**Strategies.** 300 images drawn at random (`--max-images 300 --sample-strategy random`) went through stages 1 to 4 (60 clusters, 73 unclustered images). Every crop was tagged once (8.8 s per crop on average) and the recorded tags were replayed through each naming strategy, so all strategies see the same tagger output. Reference labels are the characters Danbooru users tagged on the 81 images that are unmodified Danbooru posts; a name counts as correct when it is one of them after name resolution.

| Strategy | Images named | Wrong on the 81 references | Precision on references | Recall on references | Crops tagged |
|---|---:|---:|---:|---:|---:|
| Previous version (local tagger only) | 217 (72.3 %) | 5 of 57 | 91.2 % | 64.2 % | 297 |
| `--offline` | 217 (72.3 %) | 3 of 57 | 94.7 % | 66.7 % | 297 |
| Local tagger, canonical names | 226 (75.3 %) | 3 of 61 | 95.1 % | 71.6 % | 297 |
| Local tagger first, hash lookup for unnamed images | 243 (81.0 %) | 3 of 78 | 96.2 % | 92.6 % | 297 |
| Hash lookup first (default without credentials) | 246 (82.0 %) | 0 of 78 | 100 % | 96.3 % | 216 |

Looking images up first is both faster and more accurate: exact matches name whole clusters without tagging their crops (81 fewer crops, 12 minutes of tagger time on this sample), and they correct images the clustering misplaced. All five errors of the previous version were images grouped with a lookalike of another character and named by the cluster's vote. Two of them carried a confident tag of their real character and no trace of the cluster's, which `--offline` now uses to move them out; the exact matches fix all five. The references cover only images posted on Danbooru, so the precision of names given by the local tagger alone, on the remaining images, is not measured here.

**Where the local tagger fails.** On the 81 references, the per-image rule of the tagger named 51 images, all correctly. It missed 30:

| Cause | Images |
|---|---:|
| Character absent from the tagger's vocabulary | 9 |
| Group picture: two listed characters within 0.08 of each other | 8 |
| Costume or alias split, merged by name resolution | 5 |
| Character in the vocabulary, top score below 0.70 | 5 |
| Character in the vocabulary, no score of at least 0.27 | 3 |

The tagger's confident names are reliable; its misses are new or obscure characters, group pictures and stylized drawings. Online matches cover the first two, and name resolution the costume splits.

## Reverse image search

Checked on 2026-11-17 with a free Danbooru account and a free SauceNAO account, on the 600 images of `--max-images 600 --sample-strategy random`.

**IQDB score threshold.** The 370 images of that sample that the tagger had tagged and that are not unmodified Danbooru posts were searched on Danbooru IQDB (1.5 to 1.7 s per search, uploads of at most 768 px). Scores split in two groups: 166 images scored at least 70 and 185 below 35, with 19 in between. Of the images scoring at least 70 whose tagger top score was at least 0.70, the IQDB match listed the tagger's character on 128 of 128. Between 40 and 70, 1 of 6 such images agreed: those matches are other pictures of a similar style or composition. Thresholds of 60 to 85 accept the same images within 3 (158 to 161); 90 drops 11 correct matches. The pipeline keeps 85.

On 40 unmodified Danbooru posts cropped by 4 % on each side and re-encoded, IQDB found the original post 37 times and a post of the same characters 40 times, with scores of 40 to 70; uploads of 384 px and 768 px gave the same results and times. Such edits fall below the threshold and are left to the tagger: real reposts in the collection score far higher than this test.

**SauceNAO.** The free account allows 4 searches per 30 s and 100 per day (`short_limit`, `long_limit` in its responses); the daily limit applies to the IP address, shared by every free account on it. On 16 images that IQDB could not match and the tagger named with a score of at least 0.90, SauceNAO found nothing at a similarity of 60 or more: its best results were unrelated images from other sites. One search failed with a non-zero status and no message, and succeeded when repeated. Failed searches are not cached, so the image is searched again on a later run; a positive status (some SauceNAO indexes unavailable) with results keeps those results.

**Naming with reverse image search.** Stage 5 ran three times on the same clustering of the 600 images (107 clusters); each run tagged its crops anew:

| Run | Online matches | Images named | `_needs_review` | Crops tagged | Stage 5 time |
|---|---:|---:|---:|---:|---:|
| Hash lookup only (no credentials) | 141 | 480 | 120 | 444 | 67 min |
| IQDB and SauceNAO, cluster names inherited by unmatched members | 300 | 509 | 91 | 347 | 51 min |
| IQDB and SauceNAO, unmatched members of online-named clusters checked (current) | 345 | 509 | 91 | 411 | 60 min |

The baseline run shared the CPU with a tagger benchmark for a few minutes, so its time is slightly inflated. Of 72 SauceNAO searches in the second run, 1 gave a match (similarity 88.7); the rest of the quota went to images no index had. In the second run, 109 images took their cluster's online name without any check; 20 of them had been tagged in the baseline run, and 3 of those carried a confident tag of another character without the cluster's (sisters `tatsumaki` and `fubuki_(one-punch_man)`, and two images grouped with `kasane_teto`). The current rule searches those members online and tags the ones still unmatched: it costs 89 IQDB searches and 64 crops (about 9 minutes here) and changed 8 names, 5 to the image's own online match and 3 to its own confident tag.

## Package build with protected inputs excluded

Build from a snapshot containing only approved metadata and package source. This prevents build-system traversal of any protected directory in the working checkout:

```bash
python - <<'PY'
from pathlib import Path
import shutil
import subprocess

root = Path.cwd()
snapshot = root / ".audit/new-build-source"
snapshot.mkdir(parents=True, exist_ok=False)
files = ["pyproject.toml", "README.md"] + subprocess.check_output(
    ["git", "ls-files", "--", "src", ":!sandbox/**", ":!**/sandbox/**"],
    text=True,
).splitlines()
for name in files:
    source = root / name
    if "sandbox" in source.parts or source.is_symlink():
        raise ValueError("Unsafe build input")
    if any(parent.is_symlink() for parent in source.parents):
        raise ValueError("Symlink build input ancestor")
    target = snapshot / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
PY
python -m build --no-isolation .audit/new-build-source --outdir .audit/new-dist
```

Install build requirements first, including `setuptools>=77` and wheel, when using `--no-isolation`. This snapshot builds the library; the pipeline script and documentation remain repository deliverables. Local validation built both sdist and wheel, checked archive member names for protected paths, installed the wheel into `.audit/wheel-install` with `--no-deps`, and imported it ahead of the editable package using the already validated dependency environment. A clean-machine wheel installation across all supported Python/OS combinations was not performed.

## Evidence and limits

Local private artifacts under `.audit/` can include test logs, installation/build logs and archives. These files are ignored and must not be committed because reports can contain personal paths.

Remote CI is configured for Python 3.10/3.11. No coverage-percentage target, adversarial race-resistance certification, CVE/database vulnerability audit, labelled identity accuracy benchmark, Windows/macOS run or hardware-provider validation is claimed.
