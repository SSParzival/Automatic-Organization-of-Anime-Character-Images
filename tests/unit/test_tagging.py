"""
Unit tests for tagging: tagger I/O, normalization, sample selection, cluster votes and folder assignment.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from anime_character_organizer.tagging.aggregation import (
    aggregate_cluster_tags,
    choose_candidate,
    decide_image_name,
    vote_online_cluster,
)
from anime_character_organizer.tagging.extraction import extract_character_tags, preprocess_image
from anime_character_organizer.tagging.normalization import display_tag_name, normalize_tag_name
from anime_character_organizer.tagging.renaming import (
    IMAGE_OVER_CLUSTER,
    ONLINE_MATCH,
    REVIEW_FOLDER,
    build_named_assignment,
)
from anime_character_organizer.tagging.sampling import select_cluster_samples


class TestTagging(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_tag_name("hatsune\\miku"), "hatsunemiku")
        self.assertEqual(normalize_tag_name("rem_(re:zero)"), "rem_(re:zero)")
        self.assertEqual(normalize_tag_name(" rem / ram "), "rem___ram")
        self.assertEqual(display_tag_name("asuka_langley"), "asuka_langley")

    def test_sample_selection(self):
        df = pd.DataFrame(
            {
                "embedding_row": [1, 2, 3, 4],
                "cluster_probability": [0.5, 0.9, 0.8, 0.7],
                "outlier_score": [0.5, 0.1, 0.2, 0.3],
                "distance_to_centroid": [0.5, 0.1, 0.2, 0.3],
            }
        )

        samples = select_cluster_samples(df, max_images=2)
        self.assertEqual(len(samples), 2)
        # Should select highest prob / lowest outlier first (row 2)
        self.assertIn(2, samples["embedding_row"].tolist())

    def test_cluster_tag_aggregation_and_rules(self):
        # 3 tagged images for cluster 0
        tagging_df = pd.DataFrame(
            [
                {
                    "cluster_label": 0,
                    "tagging_status": "ok",
                    "character_tags_json": json.dumps({"hatsune_miku": 0.95, "kagamine_rin": 0.10}),
                    "relative_path": "img1.png",
                },
                {
                    "cluster_label": 0,
                    "tagging_status": "ok",
                    "character_tags_json": json.dumps({"hatsune_miku": 0.90}),
                    "relative_path": "img2.png",
                },
                {
                    "cluster_label": 0,
                    "tagging_status": "ok",
                    "character_tags_json": json.dumps({"hatsune_miku": 0.85}),
                    "relative_path": "img3.png",
                },
                # Cluster 1: contested tags (low margin)
                {
                    "cluster_label": 1,
                    "tagging_status": "ok",
                    "character_tags_json": json.dumps({"tag_a": 0.80}),
                    "relative_path": "img4.png",
                },
                {
                    "cluster_label": 1,
                    "tagging_status": "ok",
                    "character_tags_json": json.dumps({"tag_b": 0.79}),
                    "relative_path": "img5.png",
                },
            ]
        )

        suggestions_df = aggregate_cluster_tags(
            tagging_manifest_df=tagging_df,
            min_character_score=0.70,
            min_name_share=0.35,
            min_weighted_score=0.55,
            min_top_margin=0.08,
        )

        self.assertEqual(len(suggestions_df), 2)
        # Cluster 0 should be accepted as hatsune_miku
        row0 = suggestions_df[suggestions_df["cluster_label"].eq(0)].iloc[0]
        self.assertTrue(row0["accepted_name"])
        self.assertEqual(row0["suggested_character_tag"], "hatsune_miku")

        # Cluster 1 should be rejected due to low margin
        row1 = suggestions_df[suggestions_df["cluster_label"].eq(1)].iloc[0]
        self.assertFalse(row1["accepted_name"])

    def test_decide_image_name(self):
        self.assertEqual(decide_image_name({"rem_(re:zero)": 0.92, "ram_(re:zero)": 0.31}), "rem_(re:zero)")
        self.assertIsNone(decide_image_name({}))
        self.assertIsNone(decide_image_name({"rem_(re:zero)": 0.69}))
        # Two confident candidates too close to call
        self.assertIsNone(decide_image_name({"rem_(re:zero)": 0.85, "ram_(re:zero)": 0.80}))
        self.assertEqual(decide_image_name({"a": 0.60}, min_character_score=0.5, min_top_margin=0.6), "a")

    def test_build_named_assignment(self):
        cluster_df = pd.DataFrame(
            {
                "embedding_row": [0, 1, 2, 3, 4, 5, 6],
                "cluster_label": [0, 0, 1, 1, -1, -1, 2],
                "source_path": [f"/src/{i}.png" for i in range(7)],
                "requires_manual_review": [False] * 7,
            }
        )
        cluster_names = {0: ("hatsune_miku", "cluster_vote"), 2: ("hatsune_miku", "cluster_vote")}
        image_names = {
            0: ("kagamine_rin", "image_tags"),  # cluster vote wins over a single image's tags
            2: ("kagamine_rin", "image_tags"),
            4: ("megurine_luka", "image_tags"),
        }

        df = build_named_assignment(cluster_df, cluster_names, image_names)
        self.assertEqual(
            df["named_folder"].tolist(),
            [
                "hatsune_miku",
                "hatsune_miku",
                "kagamine_rin",
                REVIEW_FOLDER,
                "megurine_luka",
                REVIEW_FOLDER,
                "hatsune_miku",
            ],
        )
        self.assertEqual(
            df["naming_source"].tolist(),
            [
                "cluster_vote",
                "cluster_vote",
                "image_tags",
                "unidentified",
                "image_tags",
                "unidentified",
                "cluster_vote",
            ],
        )
        self.assertTrue(df.loc[df["named_folder"].eq(REVIEW_FOLDER), "character_name"].isna().all())

    def test_online_match_and_contradicting_tags_override_cluster_name(self):
        cluster_df = pd.DataFrame({"embedding_row": [0, 1, 2, 3, 4], "cluster_label": [0, 0, -1, 0, 0]})
        df = build_named_assignment(
            cluster_df,
            {0: ("hatsune_miku", "online_cluster_vote")},
            {
                0: ("gumi", IMAGE_OVER_CLUSTER),
                1: ("kagamine_rin", ONLINE_MATCH),
                2: ("megurine_luka", ONLINE_MATCH),
                4: ("kaito", "image_tags"),
            },
        )
        self.assertEqual(
            df["character_name"].tolist(), ["gumi", "kagamine_rin", "megurine_luka", "hatsune_miku", "hatsune_miku"]
        )
        self.assertEqual(
            df["naming_source"].tolist(),
            [IMAGE_OVER_CLUSTER, ONLINE_MATCH, ONLINE_MATCH, "online_cluster_vote", "online_cluster_vote"],
        )

    def test_named_assignment_sanitizes_names_and_handles_no_clusters(self):
        cluster_df = pd.DataFrame({"embedding_row": [0], "cluster_label": [-1], "source_path": ["/s.png"]})
        df = build_named_assignment(cluster_df, {}, {0: ("rem_(re:zero)", "image_tags")})
        self.assertEqual(df.iloc[0]["character_name"], "rem_(re:zero)")
        self.assertEqual(df.iloc[0]["named_folder"], "rem_(re_zero)")

    def test_online_cluster_vote(self):
        self.assertEqual(vote_online_cluster([["miku"], ["miku", "rin"], ["luka"]]), "miku")
        self.assertIsNone(vote_online_cluster([["miku"]]))  # one image is not enough
        self.assertIsNone(vote_online_cluster([["miku", "rin"], ["miku", "rin"]]))  # tie
        self.assertIsNone(vote_online_cluster([["miku"], ["miku"], ["a"], ["b"], ["c"]]))  # under two thirds
        self.assertIsNone(vote_online_cluster([["miku"]] * 6 + [["rin"]] * 5))  # a cluster mixing two characters
        self.assertIsNone(vote_online_cluster([]))

    def test_choose_candidate(self):
        self.assertEqual(choose_candidate(["miku"], {}), "miku")
        self.assertIsNone(choose_candidate([], {"miku": 0.9}))
        self.assertEqual(choose_candidate(["miku", "rin"], {}, preferred="rin"), "rin")
        self.assertEqual(choose_candidate(["miku", "rin"], {"rin": 0.5, "miku": 0.45}), "rin")  # both are present
        self.assertIsNone(choose_candidate(["miku", "rin"], {"rin": 0.5, "miku": 0.5}))
        self.assertIsNone(choose_candidate(["miku", "rin"], {"luka": 0.9}, preferred="luka"))


class TestCharacterTagger(unittest.TestCase):
    def test_preprocess_matches_upstream_rescale_pad(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.png"
            Image.new("RGBA", (200, 100), (255, 0, 0, 0)).save(path)
            array = preprocess_image(path, size=100)
        self.assertEqual(array.shape, (1, 3, 100, 100))
        self.assertEqual(array.dtype, np.float32)
        # Transparent pixels become white (+1); vertical padding is black (-1)
        np.testing.assert_allclose(array[0, :, 50, 50], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(array[0, :, 0, 50], [-1.0, -1.0, -1.0])
        np.testing.assert_allclose(array[0, :, 99, 50], [-1.0, -1.0, -1.0])

    def test_extract_character_tags_reads_character_block(self):
        class Session:
            def get_inputs(self):
                return [type("Input", (), {"name": "pixel_values"})()]

            def run(self, outputs, feeds):
                assert feeds["pixel_values"].shape == (1, 3, 1008, 1008)
                # Two general logits, then four character logits
                return [np.array([[9.0, 9.0, 3.0, -9.0, 0.0, 5.0]], dtype=np.float32)]

        tagger = (Session(), 2, 4, ["miku", "rin", "luka", "rem_(re:zero)"])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("anime_character_organizer.tagging.extraction.load_character_tagger", return_value=tagger),
        ):
            path = Path(tmp) / "img.png"
            Image.new("RGB", (64, 64)).save(path)
            tags = extract_character_tags(path)
        self.assertEqual(list(tags), ["rem_(re:zero)", "miku", "luka"])
        self.assertAlmostEqual(tags["luka"], 0.5)
        self.assertNotIn("rin", tags)


if __name__ == "__main__":
    unittest.main()


def test_decide_images_prefers_online_candidates_then_own_tags():
    from anime_character_organizer.workflows.naming import decide_images

    candidates = {0: ["rem_(re:zero)"], 1: ["rem_(re:zero)", "ram_(re:zero)"], 2: ["a", "b"]}
    local = {
        1: {"ram_(re:zero)": 0.4},
        2: {"c": 0.99},
        3: {"emilia_(re:zero)": 0.95},
        4: {"x": 0.5},
        5: {"emilia_(re:zero)": 0.95},
        6: {"emilia_(re:zero)": 0.95, "rem_(re:zero)": 0.3},
        7: {"rem_(re:zero)": 0.95},
    }
    preferred = {2: "b", 5: "rem_(re:zero)", 6: "rem_(re:zero)", 7: "rem_(re:zero)"}
    names = decide_images(range(8), candidates, local, preferred, min_character_score=0.7, min_top_margin=0.08)
    assert names == {
        0: ("rem_(re:zero)", ONLINE_MATCH),
        1: ("ram_(re:zero)", ONLINE_MATCH),
        2: ("b", ONLINE_MATCH),  # the cluster's name among the listed characters
        3: ("emilia_(re:zero)", "image_tags"),
        5: ("emilia_(re:zero)", IMAGE_OVER_CLUSTER),  # no trace of the cluster's character: misplaced
        6: ("emilia_(re:zero)", "image_tags"),  # the cluster's character is also present
        7: ("rem_(re:zero)", "image_tags"),
    }


def test_doubtful_members_are_flagged_cluster_members_only():
    from anime_character_organizer.workflows.naming import doubtful_members

    df = pd.DataFrame(
        {
            "cluster_label": [0, 0, 0, -1],
            "reassigned_from_noise": ["False", "True", "False", "True"],
            "is_low_probability": [False, False, True, False],
        }
    )
    assert doubtful_members(df).tolist() == [False, True, True, False]


def test_overlapped_keeps_order_and_stops_queued_work():
    import time

    from anime_character_organizer.workflows.naming import overlapped

    calls = []

    def fetch(item):
        calls.append(item)
        time.sleep(0.01)
        return item * 10

    assert list(overlapped([3, 1, 2], fetch, show_progress=False, desc="")) == [(3, 30), (1, 10), (2, 20)]
    calls.clear()
    for item, _ in overlapped(list(range(50)), fetch, show_progress=False, desc=""):
        break
    assert len(calls) < 50  # leaving early cancels lookups that have not started


def test_needs_tags():
    from anime_character_organizer.workflows.naming import needs_tags

    assert not needs_tags(["miku"], None, cluster_named=False)
    assert not needs_tags(["miku", "rin"], "rin", cluster_named=True)
    assert needs_tags(["miku", "rin"], "luka", cluster_named=True)
    assert needs_tags([], None, cluster_named=False)
    assert not needs_tags([], "luka", cluster_named=True)
