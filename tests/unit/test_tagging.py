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

from anime_character_organizer.tagging.aggregation import aggregate_cluster_tags, decide_image_name
from anime_character_organizer.tagging.extraction import extract_character_tags, preprocess_image
from anime_character_organizer.tagging.normalization import display_tag_name, normalize_tag_name
from anime_character_organizer.tagging.renaming import REVIEW_FOLDER, build_named_assignment
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
        suggestions_df = pd.DataFrame(
            [
                {"cluster_label": 0, "accepted_name": True, "suggested_character_tag": "hatsune_miku"},
                {"cluster_label": 1, "accepted_name": False, "suggested_character_tag": None},
                {"cluster_label": 2, "accepted_name": True, "suggested_character_tag": "hatsune_miku"},
            ]
        )
        tagging_df = pd.DataFrame(
            {
                "embedding_row": [0, 2, 3, 4, 5],
                "tagging_status": ["ok", "ok", "ok", "ok", "error"],
                "character_tags_json": [
                    json.dumps({"kagamine_rin": 0.99}),  # cluster vote wins over a single image
                    json.dumps({"kagamine_rin": 0.95}),
                    json.dumps({}),
                    json.dumps({"megurine_luka": 0.90}),
                    json.dumps({}),
                ],
            }
        )

        df = build_named_assignment(cluster_df, suggestions_df, tagging_df)
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

    def test_named_assignment_sanitizes_names_and_handles_no_clusters(self):
        cluster_df = pd.DataFrame({"embedding_row": [0], "cluster_label": [-1], "source_path": ["/s.png"]})
        tagging_df = pd.DataFrame(
            {
                "embedding_row": [0],
                "tagging_status": ["ok"],
                "character_tags_json": [json.dumps({"rem_(re:zero)": 0.9})],
            }
        )
        df = build_named_assignment(cluster_df, pd.DataFrame(), tagging_df)
        self.assertEqual(df.iloc[0]["character_name"], "rem_(re:zero)")
        self.assertEqual(df.iloc[0]["named_folder"], "rem_(re_zero)")


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
