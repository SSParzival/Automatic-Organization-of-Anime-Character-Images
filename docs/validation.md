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

The suite combines unittest characterization tests with pytest regression/integration tests. Use pytest to run the complete suite; unittest discovery alone does not collect pytest functions. Model inference in tests uses deterministic fixtures, not downloaded models or production data. CLI tests run from unrelated temporary working directories. Model substitutes return two known synthetic embedding groups and one fixed character tag per group; those values are not recommendations, predictions or accuracy measurements. The tagger itself is covered by unit tests of its preprocessing and of how its output block is decoded.

MyPy covers all 47 package source files. Missing third-party stubs are ignored and imports followed silently, so a successful run does not establish fully typed pandas/NumPy interfaces.

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
