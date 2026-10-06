"""Regression cases for confirmed filesystem, ordering and configuration defects."""

import errno
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from anime_character_organizer.data.audit import audit_images, scan_candidate_files
from anime_character_organizer.data.duplicates import find_perceptual_duplicates
from anime_character_organizer.embeddings.extraction import batch_iterable
from anime_character_organizer.materialization.operations import materialize_one_file
from anime_character_organizer.materialization.planner import build_materialization_plan
from anime_character_organizer.utils.serialization import safe_bool_series
from anime_character_organizer.utils.structures import UnionFind


def test_overwrite_cannot_delete_source(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("keep me")
    result = materialize_one_file(source, source, on_existing="overwrite")
    assert result["status"] == "error"
    assert source.read_text() == "keep me"


@pytest.mark.parametrize("mode,policy", [("invalid", "overwrite"), ("copy", "invalid")])
def test_invalid_policy_has_no_side_effect(tmp_path, mode, policy):
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("source")
    target.write_text("preserve")
    assert materialize_one_file(source, target, mode=mode, on_existing=policy)["status"] == "error"
    assert target.read_text() == "preserve"


def test_relative_source_symlink_is_valid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("source.txt").write_text("source")
    result = materialize_one_file("source.txt", "out/link.txt", mode="symlink")
    assert result["status"] == "ok"
    assert Path("out/link.txt").read_text() == "source"


def test_hardlink_fallback_does_not_overwrite_collision(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_text("source")

    def race(*args):
        target.write_text("concurrent writer")
        raise FileExistsError(errno.EEXIST, "destination created concurrently")

    with patch("anime_character_organizer.materialization.operations.os.link", side_effect=race):
        result = materialize_one_file(source, target)
    assert result["status"] == "error"
    assert target.read_text() == "concurrent writer"


def test_destination_directory_symlink_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.write_text("source")
    outside = tmp_path / "other"
    outside.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "cluster").symlink_to(outside, target_is_directory=True)
    assignment = pd.DataFrame([{"embedding_row": 0, "proposed_folder": "cluster", "source_path": str(source)}])
    with pytest.raises(ValueError):
        build_materialization_plan(assignment, output)
    assert list(outside.iterdir()) == []


def test_scan_prunes_protected_directory_without_reading_it(tmp_path):
    # Simulated os.walk: no sandbox directory or contents are created or accessed.
    dirs = ["sandbox", "images"]

    def walk(root, **kwargs):
        yield str(tmp_path), dirs, ["first.png"]
        assert "sandbox" not in dirs

    with patch("anime_character_organizer.data.audit.os.walk", side_effect=walk):
        assert scan_candidate_files(tmp_path, {".png"}) == [tmp_path / "first.png"]


def test_scan_skips_file_symlinks(tmp_path):
    real = tmp_path / "real.png"
    real.write_bytes(b"fixture")
    (tmp_path / "linked.png").symlink_to(real)
    assert scan_candidate_files(tmp_path, {".png"}) == [real]


def test_parallel_audit_has_stable_path_order(tmp_path):
    paths = [tmp_path / "z.png", tmp_path / "a.png"]
    with patch(
        "anime_character_organizer.data.audit.inspect_image",
        side_effect=lambda p, _: {"path": str(p), "status": "valid"},
    ):
        metadata, _, _ = audit_images(paths, tmp_path, show_progress=False)
    assert metadata["path"].tolist() == sorted(str(p) for p in paths)


def test_empty_perceptual_results_keep_schema():
    pairs, groups = find_perceptual_duplicates(pd.DataFrame(), show_progress=False)
    assert "phash_distance" in pairs.columns
    assert "perceptual_group_id" in groups.columns


def test_union_find_accepts_one_shot_iterables():
    uf = UnionFind(iter(["a", "b"]))
    uf.union("a", "b")
    assert uf.find("a") == uf.find("b")


def test_boolean_scalar_and_series_whitespace_agree():
    assert safe_bool_series(pd.Series([" true ", " yes ", " false "])).tolist() == [True, True, False]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_rejected(batch_size):
    with pytest.raises(ValueError, match="positive"):
        list(batch_iterable([1, 2], batch_size))
