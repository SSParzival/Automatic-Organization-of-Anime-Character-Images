"""All six real workflows exercised with deterministic model substitutes."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from anime_character_organizer.exceptions import (
    ConfigurationError,
    InvalidImageError,
    MaterializationError,
    TaggingError,
)
from anime_character_organizer.tagging.renaming import IMAGE_OVER_CLUSTER, ONLINE_MATCH
from anime_character_organizer.workflows.materialize import run_folder_materialization
from anime_character_organizer.workflows.naming import run_cluster_naming
from anime_character_organizer.workflows.pipeline import run_full_pipeline
from tests.fixtures.model_substitutes import image_index, install_substitutes, synthetic_tags


@pytest.fixture
def completed_pipeline(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    for i in range(10):
        Image.new("RGB", (96, 96), (i * 20, 50, 100)).save(source / f"image{i:02d}.png")
    patches = install_substitutes()
    try:
        return run_full_pipeline(source, tmp_path / "workspace", materialization_mode="copy", show_progress=False)
    finally:
        for handle in patches:
            handle.stop()


def folder_contents(output_dir):
    return {
        folder.name: len([f for f in folder.iterdir() if f.suffix == ".png"])
        for folder in Path(output_dir).iterdir()
        if folder.is_dir()
    }


def test_pipeline_always_names_every_folder(completed_pipeline):
    result = completed_pipeline
    assert list(result) == ["audit", "crop", "embed", "cluster", "naming", "materialize"]
    assert result["embed"]["normalized_embeddings"].shape == (10, 8)
    assert result["materialize"]["validation_df"]["is_valid"].all()
    assert result["naming"]["summary"]["clusters_named"] == 2
    assert result["naming"]["summary"]["images_needing_review"] == 0
    assert folder_contents(result["materialize"]["final_output_dir"]) == {"character_a": 5, "character_b": 5}
    info = json.loads((result["materialize"]["final_output_dir"] / "character_a/_folder_info.json").read_text())
    assert info["character_name"] == "character_a"
    assert info["naming_sources"] == {"cluster_vote": 5}


def mixed_tags(path):
    """Cluster b mixes four characters and one unknown, so no name wins its vote."""
    index = image_index(path)
    if index < 5:
        return {"character_a": 0.95}
    return {5: {"character_b": 0.9}, 6: {"character_c": 0.9}, 7: {"character_d": 0.9}, 8: {"character_e": 0.9}}.get(
        index, {}
    )


def test_unnamed_cluster_is_split_by_image_tags(completed_pipeline, tmp_path):
    workspace = tmp_path / "other"
    with (
        patch("anime_character_organizer.workflows.naming.extract_character_tags", side_effect=mixed_tags),
        patch("anime_character_organizer.workflows.naming.load_character_tagger"),
    ):
        naming = run_cluster_naming(
            workspace, completed_pipeline["cluster"]["run_dir"], max_images_per_cluster_to_tag=3, show_progress=False
        )
    # Three samples per cluster, plus the two untagged images of the rejected cluster
    assert naming["summary"]["total_tagged_images"] == 8
    assert naming["summary"]["clusters_named"] == 1
    output = run_folder_materialization(workspace, naming["run_dir"], materialization_mode="copy", show_progress=False)
    assert folder_contents(output["final_output_dir"]) == {
        "character_a": 5,
        "character_b": 1,
        "character_c": 1,
        "character_d": 1,
        "character_e": 1,
        "_needs_review": 1,
    }
    assert not any("cluster_" in folder for folder in folder_contents(output["final_output_dir"]))


def test_naming_missing_crops_is_actionable(completed_pipeline, tmp_path):
    for path in completed_pipeline["crop"]["crops_df"]["crop_path"]:
        Path(path).unlink()
    with pytest.raises(InvalidImageError, match="existing crop"):
        run_cluster_naming(tmp_path / "other", completed_pipeline["cluster"]["run_dir"], show_progress=False)


def test_naming_all_model_failures_are_not_success(completed_pipeline, tmp_path):
    with (
        patch(
            "anime_character_organizer.workflows.naming.extract_character_tags", side_effect=RuntimeError("bad image")
        ),
        patch("anime_character_organizer.workflows.naming.load_character_tagger"),
    ):
        with pytest.raises(TaggingError, match="All tagger"):
            run_cluster_naming(tmp_path / "other", completed_pipeline["cluster"]["run_dir"], show_progress=False)


def test_unavailable_tagger_fails_before_creating_a_run(completed_pipeline, tmp_path):
    with patch("anime_character_organizer.workflows.naming.load_character_tagger", side_effect=OSError("offline")):
        with pytest.raises(TaggingError, match="Could not load"):
            run_cluster_naming(tmp_path / "other", completed_pipeline["cluster"]["run_dir"], show_progress=False)
    assert not (tmp_path / "other" / "runs").exists()


def test_materialization_failure_is_not_success(completed_pipeline, tmp_path):
    failure = {
        "status": "error",
        "operation_used": "copy",
        "error_type": "OSError",
        "error_message": "synthetic failure",
    }
    with patch("anime_character_organizer.workflows.materialize.materialize_one_file", return_value=failure):
        with pytest.raises(MaterializationError, match="incomplete"):
            run_folder_materialization(tmp_path / "other", completed_pipeline["naming"]["run_dir"], show_progress=False)


def test_existing_output_is_preserved(completed_pipeline, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("user artifact")
    with pytest.raises(ConfigurationError, match="new or empty"):
        run_folder_materialization(
            tmp_path / "other", completed_pipeline["naming"]["run_dir"], final_output_dir=output, show_progress=False
        )
    assert marker.read_text() == "user artifact"


@pytest.mark.parametrize("mode", ["copy", "hardlink", "symlink"])
def test_materialization_workflow_supports_all_documented_modes(completed_pipeline, tmp_path, mode):
    output = run_folder_materialization(
        tmp_path / "mode_workspace",
        completed_pipeline["naming"]["run_dir"],
        materialization_mode=mode,
        show_progress=False,
    )
    assert output["validation_df"]["is_valid"].all()
    assert len(output["result_df"]) == 10
    for path in output["plan_df"]["destination_path"]:
        assert Path(path).exists()


def online_workspace(posts, tmp_path):
    from tests.fixtures.online_substitutes import FakeDanbooru

    client = FakeDanbooru(posts=posts)
    client.can_search_images = False
    workspace = tmp_path / "online"
    patches = (
        patch(
            "anime_character_organizer.workflows.naming.clients_from_environment",
            return_value={"danbooru": client, "saucenao": None},
        ),
        patch("anime_character_organizer.workflows.naming.extract_character_tags", side_effect=synthetic_tags),
        patch("anime_character_organizer.workflows.naming.load_character_tagger"),
    )
    for handle in patches:
        handle.start()
    return patches, workspace


def test_unanimous_online_vote_trusts_the_cluster_name(completed_pipeline, tmp_path):
    """
    Two exact Danbooru matches name cluster a; a costume tag and its base character share one name.

    Every identified representative agrees, so the cluster's other members keep its name without an
    individual lookup and only the unnamed cluster b reaches the local tagger.
    """
    from anime_character_organizer.tagging.online import file_md5

    sources = completed_pipeline["audit"]["valid_df"]["path"].sort_values().tolist()
    patches, workspace = online_workspace(
        [
            {"id": 1, "md5": file_md5(sources[0]), "tag_string_character": "hatsune_miku_(append)"},
            {"id": 2, "md5": file_md5(sources[1]), "tag_string_character": "hatsune_miku hatsune_miku_(append)"},
            {"id": 3, "md5": file_md5(sources[9]), "tag_string_character": "saber"},
        ],
        tmp_path,
    )
    try:
        naming = run_cluster_naming(workspace, completed_pipeline["cluster"]["run_dir"], show_progress=False)
    finally:
        for handle in patches:
            handle.stop()
    summary = naming["summary"]
    assert summary["clusters_named_online"] == 1
    assert summary["clusters_named_online_contested"] == 0
    assert summary["images_individually_checked"] == 1  # only the member HDBSCAN flagged as an outlier
    assert summary["total_tagged_images"] == 5  # only the unnamed cluster b is tagged
    assigned = naming["named_assignment_df"].set_index("source_path").loc[sources]
    assert assigned["character_name"].tolist() == ["hatsune_miku"] * 5 + ["character_b"] * 4 + [
        "artoria_pendragon_(fate)"
    ]
    assert assigned["naming_source"].tolist() == [ONLINE_MATCH] * 2 + ["online_cluster_vote"] * 3 + [
        "cluster_vote"
    ] * 4 + [ONLINE_MATCH]
    assert (workspace / "cache" / "online_identification.json").is_file()
    assert json.loads((workspace / "cache" / "character_names.json").read_text())["saber"] == "artoria_pendragon_(fate)"


def test_contested_online_vote_checks_the_unmatched_members(completed_pipeline, tmp_path):
    """
    A third exact match lists another character, so the cluster's vote is contested.

    The majority still names the cluster, but its unmatched members are checked one by one; their
    confident tag of another character, without a trace of the cluster's, moves them out as it would
    for a lookalike.
    """
    from anime_character_organizer.tagging.online import file_md5

    sources = completed_pipeline["audit"]["valid_df"]["path"].sort_values().tolist()
    patches, workspace = online_workspace(
        [
            {"id": 1, "md5": file_md5(sources[0]), "tag_string_character": "hatsune_miku_(append)"},
            {"id": 2, "md5": file_md5(sources[1]), "tag_string_character": "hatsune_miku hatsune_miku_(append)"},
            {"id": 3, "md5": file_md5(sources[2]), "tag_string_character": "saber"},
            {"id": 4, "md5": file_md5(sources[9]), "tag_string_character": "saber"},
        ],
        tmp_path,
    )
    try:
        naming = run_cluster_naming(workspace, completed_pipeline["cluster"]["run_dir"], show_progress=False)
    finally:
        for handle in patches:
            handle.stop()
    summary = naming["summary"]
    assert summary["clusters_named_online"] == 1
    assert summary["clusters_named_online_contested"] == 1
    assert summary["images_individually_checked"] == 2  # the two unmatched members of cluster a
    assert summary["total_tagged_images"] == 7  # the cluster b samples plus those two members
    assigned = naming["named_assignment_df"].set_index("source_path").loc[sources]
    assert assigned["character_name"].tolist() == [
        "hatsune_miku",
        "hatsune_miku",
        "artoria_pendragon_(fate)",
        "character_a",
        "character_a",
    ] + ["character_b"] * 4 + ["artoria_pendragon_(fate)"]
    assert assigned["naming_source"].tolist() == [ONLINE_MATCH] * 3 + [IMAGE_OVER_CLUSTER] * 2 + [
        "cluster_vote"
    ] * 4 + [ONLINE_MATCH]
