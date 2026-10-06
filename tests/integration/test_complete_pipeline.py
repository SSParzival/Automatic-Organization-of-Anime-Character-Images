"""All six real workflows exercised with deterministic model substitutes."""

import pytest
from PIL import Image

from anime_character_organizer.exceptions import (
    ConfigurationError,
    InvalidImageError,
    MaterializationError,
    TaggingError,
)
from anime_character_organizer.workflows.materialize import run_folder_materialization
from anime_character_organizer.workflows.naming import run_cluster_naming
from anime_character_organizer.workflows.pipeline import run_full_pipeline
from tests.fixtures.model_substitutes import install_substitutes


@pytest.fixture
def completed_pipeline(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    for i in range(10):
        Image.new("RGB", (96, 96), (i * 20, 50, 100)).save(source / f"image{i:02d}.png")
    patches = install_substitutes()
    try:
        return run_full_pipeline(
            source, tmp_path / "workspace", run_tagging=True, materialization_mode="copy", show_progress=False
        )
    finally:
        for handle in patches:
            handle.stop()


def test_all_six_workflows_materialize_and_name(completed_pipeline):
    result = completed_pipeline
    assert list(result) == ["audit", "crop", "embed", "cluster", "materialize", "naming"]
    assert result["embed"]["normalized_embeddings"].shape == (10, 8)
    assert result["materialize"]["validation_df"]["is_valid"].all()
    assert result["naming"]["summary"]["accepted_character_names"] == 2
    assert (result["naming"]["run_dir"] / "tables/named_materialization_validation.csv").exists()
    assert len(result["naming"]["named_materialization_df"]) == 10


def test_naming_missing_crops_is_actionable(completed_pipeline, tmp_path):
    for path in completed_pipeline["crop"]["crops_df"]["crop_path"]:
        from pathlib import Path

        Path(path).unlink()
    with pytest.raises(InvalidImageError, match="existing crop"):
        run_cluster_naming(
            tmp_path / "other", completed_pipeline["cluster"]["run_dir"], create_named_output=False, show_progress=False
        )


def test_naming_all_model_failures_are_not_success(completed_pipeline, tmp_path):
    from unittest.mock import patch

    with patch(
        "anime_character_organizer.workflows.naming.extract_tags_with_fallback",
        return_value=(None, "ModelUnavailable", "synthetic outage"),
    ):
        with pytest.raises(TaggingError, match="All tagger"):
            run_cluster_naming(
                tmp_path / "other",
                completed_pipeline["cluster"]["run_dir"],
                create_named_output=False,
                show_progress=False,
            )


def test_materialization_failure_is_not_success(completed_pipeline, tmp_path):
    from unittest.mock import patch

    failure = {
        "status": "error",
        "operation_used": "copy",
        "error_type": "OSError",
        "error_message": "synthetic failure",
    }
    with patch("anime_character_organizer.workflows.materialize.materialize_one_file", return_value=failure):
        with pytest.raises(MaterializationError, match="incomplete"):
            run_folder_materialization(
                tmp_path / "other", completed_pipeline["cluster"]["run_dir"], show_progress=False
            )


def test_existing_output_is_preserved(completed_pipeline, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("user artifact")
    with pytest.raises(ConfigurationError, match="new or empty"):
        run_folder_materialization(
            tmp_path / "other", completed_pipeline["cluster"]["run_dir"], final_output_dir=output, show_progress=False
        )
    assert marker.read_text() == "user artifact"
