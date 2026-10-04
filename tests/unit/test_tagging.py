"""
Unit tests for tagging: normalization, sample selection, cluster tag aggregation, and renaming.
"""

import json
import unittest
import pandas as pd

from anime_character_organizer.tagging.aggregation import aggregate_cluster_tags
from anime_character_organizer.tagging.normalization import display_tag_name, normalize_tag_name
from anime_character_organizer.tagging.renaming import build_folder_rename_plan
from anime_character_organizer.tagging.sampling import select_cluster_samples


class TestTagging(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_tag_name("hatsune\\miku"), "hatsunemiku")
        self.assertEqual(normalize_tag_name("rem_(re:zero)"), "rem_(re:zero)")
        self.assertEqual(normalize_tag_name(" rem / ram "), "rem___ram")
        self.assertEqual(display_tag_name("asuka_langley"), "asuka_langley")

    def test_sample_selection(self):
        df = pd.DataFrame({
            "embedding_row": [1, 2, 3, 4],
            "cluster_probability": [0.5, 0.9, 0.8, 0.7],
            "outlier_score": [0.5, 0.1, 0.2, 0.3],
            "distance_to_centroid": [0.5, 0.1, 0.2, 0.3],
        })

        samples = select_cluster_samples(df, max_images=2)
        self.assertEqual(len(samples), 2)
        # Should select highest prob / lowest outlier first (row 2)
        self.assertIn(2, samples["embedding_row"].tolist())

    def test_cluster_tag_aggregation_and_rules(self):
        # 3 tagged images for cluster 0
        tagging_df = pd.DataFrame([
            {
                "cluster_label": 0,
                "cluster_folder_stub": "cluster_00000_unknown",
                "tagging_status": "ok",
                "character_tags_json": json.dumps({"hatsune_miku": 0.95, "kagamine_rin": 0.10}),
                "relative_path": "img1.png",
            },
            {
                "cluster_label": 0,
                "cluster_folder_stub": "cluster_00000_unknown",
                "tagging_status": "ok",
                "character_tags_json": json.dumps({"hatsune_miku": 0.90}),
                "relative_path": "img2.png",
            },
            {
                "cluster_label": 0,
                "cluster_folder_stub": "cluster_00000_unknown",
                "tagging_status": "ok",
                "character_tags_json": json.dumps({"hatsune_miku": 0.85}),
                "relative_path": "img3.png",
            },
            # Cluster 1: contested tags (low margin)
            {
                "cluster_label": 1,
                "cluster_folder_stub": "cluster_00001_unknown",
                "tagging_status": "ok",
                "character_tags_json": json.dumps({"tag_a": 0.80}),
                "relative_path": "img4.png",
            },
            {
                "cluster_label": 1,
                "cluster_folder_stub": "cluster_00001_unknown",
                "tagging_status": "ok",
                "character_tags_json": json.dumps({"tag_b": 0.79}),
                "relative_path": "img5.png",
            },
        ])

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
        self.assertIn("hatsune_miku__cluster_00000_unknown", row0["proposed_named_folder"])

        # Cluster 1 should be rejected due to low margin
        row1 = suggestions_df[suggestions_df["cluster_label"].eq(1)].iloc[0]
        self.assertFalse(row1["accepted_name"])

    def test_build_folder_rename_plan(self):
        assignment_df = pd.DataFrame([
            {
                "embedding_row": 1,
                "cluster_label": 0,
                "proposed_folder": "cluster_00000_unknown",
                "source_path": "/path/img1.png",
            },
            {
                "embedding_row": 2,
                "cluster_label": -1,
                "proposed_folder": "_needs_review_noise",
                "source_path": "/path/img2.png",
            },
        ])

        suggestions_df = pd.DataFrame([
            {
                "cluster_label": 0,
                "cluster_folder_stub": "cluster_00000_unknown",
                "accepted_name": True,
                "suggested_character_tag": "hatsune_miku",
                "proposed_named_folder": "hatsune_miku__cluster_00000_unknown",
                "best_tag_share": 1.0,
                "best_weighted_score": 0.9,
                "top_margin": 0.9,
                "top_tags_json": "[]",
            },
            {
                "cluster_label": -1,
                "cluster_folder_stub": "_needs_review_noise",
                "accepted_name": False,
                "suggested_character_tag": None,
                "proposed_named_folder": "_needs_review_noise",
                "best_tag_share": 0.0,
                "best_weighted_score": 0.0,
                "top_margin": 0.0,
                "top_tags_json": "[]",
            },
        ])

        plan_df, preview_df = build_folder_rename_plan(assignment_df, suggestions_df)
        self.assertEqual(len(plan_df), 2)
        # Cluster 0 should have named_folder updated
        row0 = plan_df[plan_df["cluster_label"].eq(0)].iloc[0]
        self.assertEqual(row0["named_folder"], "hatsune_miku__cluster_00000_unknown")
        # Noise should keep original proposed_folder
        row_noise = plan_df[plan_df["cluster_label"].eq(-1)].iloc[0]
        self.assertEqual(row_noise["named_folder"], "_needs_review_noise")


if __name__ == "__main__":
    unittest.main()
