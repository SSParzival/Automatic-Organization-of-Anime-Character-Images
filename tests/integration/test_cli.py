"""Installed scripts work from an unrelated directory and fail actionably."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[2]
SCRIPTS = [
    "run_audit.py",
    "run_crop_preparation.py",
    "run_embeddings.py",
    "run_clustering.py",
    "run_materialization.py",
    "run_naming.py",
    "run_pipeline.py",
]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_help_from_another_directory(name, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_audit_cli_persists_reports(tmp_path):
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (16, 16), "red").save(images / "fixture.png")
    workspace = tmp_path / "workspace"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_audit.py"),
            "--input-dir",
            str(images),
            "--project-dir",
            str(workspace),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Valid images: 1" in result.stdout
    # Stage prefix is explicit and cannot descend into a protected directory.
    runs = list((workspace / "runs").glob("01_dataset_audit_*/reports/summary.json"))
    assert len(runs) == 1


@pytest.mark.parametrize("name", SCRIPTS)
def test_cli_missing_input_or_upstream_returns_error(name, tmp_path):
    args = [sys.executable, str(ROOT / "scripts" / name), "--project-dir", str(tmp_path / "workspace")]
    if name in {"run_audit.py", "run_pipeline.py"}:
        args.extend(["--input-dir", str(tmp_path / "missing")])
    result = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "error" in result.stderr.lower() or "failed" in result.stderr.lower()
