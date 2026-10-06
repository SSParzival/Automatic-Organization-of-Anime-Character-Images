# Validation and reproducibility

This guide explains how to reproduce checks without accessing protected directories. Run commands from the repository root after installing `.[dev,notebook]`. The full audit findings and readiness boundaries are in [audit-report.md](audit-report.md); the original inventory and migration plan are in [discovery-plan.md](discovery-plan.md).

## Local checks

```bash
mkdir -p .audit/tmp
export TMPDIR="$PWD/.audit/tmp"
export MPLBACKEND=Agg
python -m pip check
python -m pytest tests --ignore-glob='**/sandbox/**'
python -m ruff check src scripts tests notebooks --force-exclude --exclude sandbox
python -m ruff format --check src scripts tests notebooks --force-exclude --exclude sandbox
python -m mypy --cache-dir .audit/mypy-cache
```

Pytest also excludes every directory named `sandbox` through `norecursedirs`. Ruff uses both configured exclusions and explicit protected-directory exclusion. MyPy excludes `(^|/)sandbox/`. Do not supply explicit protected files to any tool. Package discovery lists exactly the ten supported packages and does not recursively discover arbitrary packages.

The suite combines unittest characterization tests with pytest regression/integration tests. Use pytest to run the complete suite; unittest discovery alone does not collect pytest functions. Model inference in tests uses deterministic fixtures, not downloaded models or production data. CLI tests run from unrelated temporary working directories. Model substitutes return two known synthetic embedding groups and one fixed character tag; those values are not recommendations, predictions or accuracy measurements.

MyPy covers all 47 package source files. Missing third-party stubs are ignored and imports followed silently, so a successful run does not establish fully typed pandas/NumPy interfaces.

## Fresh notebook kernels

```bash
export IPYTHONDIR="$PWD/.audit/ipython"
export JUPYTER_RUNTIME_DIR="$PWD/.audit/jupyter-runtime"
export MPLCONFIGDIR="$PWD/.audit/matplotlib"
python -m tests.notebook_smoke --output-dir .audit/new-notebook-evidence
```

The evidence directory must be new or empty. This command:

1. Reads and structurally validates the exact seven known notebook paths; no recursive notebook traversal is used.
2. Parses every code cell and creates ten synthetic images.
3. Executes each notebook top to bottom in a fresh kernel using the invoking Python interpreter.
4. Sets absolute input/workspace paths independent of the kernel working directory.
5. Substitutes only head/person detection, CCIP extraction/warmup and tagger inference. Audit, validation, normalization, real HDBSCAN, plotting, planning, file operations and report persistence remain real.
6. Chains stages 1–6 through one workspace and executes the master notebook with a separate workspace.
7. Shuts down each kernel, writes executed copies and a `results.json`, and returns nonzero if any notebook fails.

Source notebooks are never overwritten by execution. The master source notebook retains its original user configuration stdout/count/kernel metadata; other source notebooks have no embedded outputs. Executed copies are private evidence artifacts and may contain local paths and synthetic contact sheets.

For actual inference, remove substitution explicitly:

```bash
export HF_HOME="$PWD/.audit/model-cache"
export TORCH_HOME="$PWD/.audit/torch-cache"
export XDG_CACHE_HOME="$PWD/.audit/cache"
HF_HUB_OFFLINE=0 python -m tests.notebook_smoke --live --timeout 45 --output-dir .audit/new-live-evidence
```

This can download model weights. A cell timeout or unavailable model is a failure/blocker, not a passing inference check. With `HF_HUB_OFFLINE=1` and uncached weights, crop preparation can complete using warning-bearing fallbacks, while CCIP raises `EmbeddingError` and downstream stages report missing completed runs. Even successful inference on synthetic solid-color images does not establish useful character clustering accuracy. A representative labelled collection and manually reviewed contact sheets are required for that assessment.

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

Install build requirements first, including `setuptools>=77` and wheel, when using `--no-isolation`. This snapshot builds the library; operational scripts, notebooks and audit documentation remain repository deliverables. Local validation built both sdist and wheel, checked archive member names for protected paths, installed the wheel into `.audit/wheel-install` with `--no-deps`, and imported it ahead of the editable package using the already validated dependency environment. A clean-machine wheel installation across all supported Python/OS combinations was not performed.

## Evidence and failure history

Local private artifacts under `.audit/` include baseline metadata, separated master-notebook upgrade diff, focused/full test logs, initial/final notebook execution copies, offline/live results, installation/build logs and archives. These files are ignored and must not be committed because reports can contain personal paths.

Validation deliberately began with failing regressions: fourteen filesystem/order/schema cases reproduced defects before correction. The first broad type pass identified ten annotations/contracts to correct. A CLI test initially expected a literal `Error` prefix from the pipeline while the CLI emitted an actionable `failed` message with status one; the assertion was corrected to accept both documented error forms. Initial notebook lint found 52 issues; all were fixed with cell-aware Ruff. Intermediate syntax/lint/type failures were corrected and affected checks rerun.

The bounded network-enabled attempt subsequently passed all seven notebooks with actual models on synthetic input; detector failure warnings were absent, CCIP returned ten 768D vectors, and tagging processed ten samples with zero errors and zero accepted names. These observations establish installed interface compatibility, not identity quality.

Remote CI is configured for Python 3.10/3.11, but was not triggered in this session. No coverage-percentage target, adversarial race-resistance certification, CVE/database vulnerability audit, representative identity accuracy benchmark, Windows/macOS run or hardware-provider validation is claimed.
