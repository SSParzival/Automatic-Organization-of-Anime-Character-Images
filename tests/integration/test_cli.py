"""The pipeline script works from an unrelated directory, always names folders and fails actionably."""

import runpy
import subprocess
import sys
from pathlib import Path

from PIL import Image

from tests.fixtures.model_substitutes import install_substitutes

ROOT = Path(__file__).absolute().parents[2]
SCRIPT = ROOT / "scripts" / "run_pipeline.py"


def run_script(*args, cwd):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True, timeout=30)


def test_only_the_pipeline_script_is_shipped():
    assert sorted(p.name for p in (ROOT / "scripts").glob("*.py")) == ["run_pipeline.py"]


def test_script_help_from_another_directory(tmp_path):
    result = run_script("--help", cwd=tmp_path)
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "tagging" not in result.stdout


def test_tagging_cannot_be_disabled_or_requested(tmp_path):
    result = run_script("--with-tagging", "--input-dir", str(tmp_path), cwd=tmp_path)
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_cli_missing_input_returns_error(tmp_path):
    result = run_script(
        "--project-dir", str(tmp_path / "workspace"), "--input-dir", str(tmp_path / "missing"), cwd=tmp_path
    )
    assert result.returncode == 1
    assert "failed" in result.stderr.lower()


def test_cli_runs_every_stage_and_reports_named_folders(tmp_path, monkeypatch, capsys):
    source = tmp_path / "images"
    source.mkdir()
    for i in range(10):
        Image.new("RGB", (64, 64), (i * 20, 40, 90)).save(source / f"image{i:02d}.png")
    monkeypatch.chdir(tmp_path)
    main = runpy.run_path(str(SCRIPT))["main"]
    patches = install_substitutes()
    try:
        code = main(["--input-dir", str(source), "--project-dir", str(tmp_path / "workspace"), "--mode", "copy"])
    finally:
        for handle in patches:
            handle.stop()
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Character folders: 2" in out
    assert "Images needing review: 0" in out
    output = next((tmp_path / "workspace" / "organized_output").iterdir())
    assert {p.name for p in output.iterdir() if p.is_dir()} == {"character_a", "character_b"}
