"""Boundary and reproducibility contracts for staged workflows."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from anime_character_organizer.config import AuditConfig, EmbeddingConfig, TaggingConfig, validate_parameters
from anime_character_organizer.exceptions import ConfigurationError
from anime_character_organizer.preprocessing.detection import detect_heads_safe
from anime_character_organizer.utils.paths import checked_path
from anime_character_organizer.utils.runs import create_run_directory
from anime_character_organizer.workflows.audit import run_dataset_audit
from anime_character_organizer.workflows.cluster import run_clustering


@pytest.mark.parametrize(
    "settings",
    [
        {"batch_size": 0},
        {"max_workers": -1},
        {"min_samples": 0},
        {"crop_output_size": (0, 512)},
        {"phash_threshold": 65},
        {"max_reassign_distance": float("nan")},
        {"min_name_share": 1.1},
        {"create_named_output": True, "create_rename_plan": False},
        {"manifest_file_name": "../table.csv"},
    ],
)
def test_invalid_configuration(settings):
    with pytest.raises(ConfigurationError):
        validate_parameters(**settings)


def test_dataclass_defaults_match_public_functions(tmp_path):
    import inspect

    from anime_character_organizer.workflows.embed import run_embedding_extraction
    from anime_character_organizer.workflows.naming import run_cluster_naming

    for config, function in [(EmbeddingConfig(), run_embedding_extraction), (TaggingConfig(), run_cluster_naming)]:
        defaults = inspect.signature(function).parameters
        for key, value in vars(config).items():
            if key in defaults and key != "project_dir":
                assert value == defaults[key].default
    with pytest.raises(ConfigurationError):
        AuditConfig(tmp_path, phash_threshold=-1)


def test_protected_explicit_path_rejected_without_filesystem_calls():
    with patch("pathlib.Path.is_symlink", side_effect=AssertionError("must reject lexically")):
        with pytest.raises(ValueError, match="protected"):
            checked_path("a/sandbox/image.png")


def test_run_allocation_never_reuses_directory(tmp_path):
    first = create_run_directory(tmp_path, "01_dataset_audit")
    (first / "sentinel").write_text("keep")
    second = create_run_directory(tmp_path, "01_dataset_audit")
    assert first != second
    assert (first / "sentinel").read_text() == "keep"


def test_audit_refuses_output_inside_input(tmp_path):
    with pytest.raises(ConfigurationError, match="outside"):
        run_dataset_audit(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_detector_failure_is_visible():
    with patch.dict("sys.modules", {"imgutils.detect": None}):
        with pytest.warns(RuntimeWarning, match="Detector failed"):
            assert detect_heads_safe(None) == []


def make_embedding_run(tmp_path, rows=(0, 1, 2)):
    run = tmp_path / "previous"
    (run / "arrays").mkdir(parents=True)
    (run / "tables").mkdir()
    matrix = np.array([[1, 0], [0.99, 0.1], [0.8, 0.2]], dtype=np.float32)
    manifest = pd.DataFrame(
        {
            "embedding_row": rows,
            "source_path": ["absent.png"] * 3,
            "crop_path": ["absent.png"] * 3,
            "relative_path": ["absent.png"] * 3,
        }
    )
    manifest.to_csv(run / "tables/successful_embedding_manifest.csv", index=False)
    np.save(run / "arrays/ccip_embeddings_l2.npy", matrix)
    return run


def test_cluster_aligns_rows_preserves_rng_and_closes_saved_figures(tmp_path):
    run = make_embedding_run(tmp_path, (2, 0, 1))
    before = np.random.get_state()
    open_figures = plt.get_fignums()
    with patch(
        "anime_character_organizer.workflows.cluster.fit_hdbscan",
        return_value=(np.array([0, 0, -1]), np.array([0.9, 0.9, 0.0]), np.array([0.1, 0.1, 1.0]), "test"),
    ):
        result = run_clustering(tmp_path, run, max_reassign_distance=1.0, show_progress=False)
    assert result["cluster_manifest_df"]["embedding_row"].tolist() == [0, 1, 2]
    assert result["cluster_manifest_df"]["hdbscan_label"].tolist() == [0, 0, -1]
    assert result["cluster_manifest_df"]["reassigned_from_noise"].tolist() == [False, False, True]
    assert result["cluster_manifest_df"].iloc[2]["requires_manual_review"]
    assert not result["cluster_manifest_df"].iloc[2]["is_low_probability"]
    assert plt.get_fignums() == open_figures
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("rows", [(0, 0, 2), (0, 1, 4)])
def test_cluster_refuses_invalid_row_mapping_before_creating_run(tmp_path, rows):
    run = make_embedding_run(tmp_path, rows)
    with pytest.raises(ConfigurationError, match="exactly once"):
        run_clustering(tmp_path, run)
    assert not (tmp_path / "runs").exists()
