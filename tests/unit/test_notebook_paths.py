"""Notebook configuration is independent of the kernel working directory."""

import pytest

from anime_character_organizer.exceptions import ConfigurationError
from anime_character_organizer.utils.paths import notebook_paths


def test_explicit_notebook_paths_survive_cwd_change(tmp_path, monkeypatch):
    source = tmp_path / "input"
    project = tmp_path / "workspace"
    monkeypatch.setenv("ANIME_PIPELINE_INPUT_DIR", str(source))
    monkeypatch.setenv("ANIME_PIPELINE_PROJECT_DIR", str(project))
    monkeypatch.chdir(tmp_path.parent)
    assert notebook_paths() == (source, project)


def test_missing_workspace_fails_clearly(monkeypatch):
    monkeypatch.delenv("ANIME_PIPELINE_PROJECT_DIR", raising=False)
    with pytest.raises(ConfigurationError, match="ANIME_PIPELINE_PROJECT_DIR"):
        notebook_paths()


def test_relative_workspace_is_rejected(monkeypatch):
    monkeypatch.setenv("ANIME_PIPELINE_PROJECT_DIR", "relative/workspace")
    with pytest.raises(ConfigurationError, match="absolute"):
        notebook_paths()
