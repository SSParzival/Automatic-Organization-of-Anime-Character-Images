"""
Unit tests for clustering: algorithm, centroids, diagnostics, and review flagging.
"""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd

from anime_character_organizer.clustering.algorithm import fit_hdbscan
from anime_character_organizer.clustering.centroids import (
    compute_cluster_centroids,
    euclidean_distance_to_centroid,
)
from anime_character_organizer.clustering.diagnostics import (
    build_folder_assignment_manifest,
    calculate_cluster_summary,
    flag_cluster_review_items,
    sample_representative_and_boundary_rows,
)
from anime_character_organizer.workflows.cluster import run_clustering


class TestClustering(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create 2 synthetic clusters of 20 points each in 4D space
        cluster1 = np.random.normal(loc=[1.0, 1.0, 0.0, 0.0], scale=0.05, size=(20, 4)).astype(np.float32)
        cluster2 = np.random.normal(loc=[-1.0, -1.0, 0.0, 0.0], scale=0.05, size=(20, 4)).astype(np.float32)
        self.embeddings = np.vstack([cluster1, cluster2])

    def test_fit_hdbscan(self):
        labels, probs, outliers, backend = fit_hdbscan(
            self.embeddings,
            min_cluster_size=5,
            min_samples=3,
        )
        self.assertEqual(len(labels), 40)
        self.assertEqual(len(probs), 40)
        self.assertEqual(len(outliers), 40)
        self.assertIn(backend, ["hdbscan", "sklearn"])
        # Should detect at least 2 distinct clusters
        non_noise_labels = set(labels) - {-1}
        self.assertGreaterEqual(len(non_noise_labels), 1)

    def test_centroids_and_distances(self):
        labels = np.array([0] * 20 + [1] * 20)
        centroids = compute_cluster_centroids(self.embeddings, labels)
        self.assertIn(0, centroids)
        self.assertIn(1, centroids)
        # Centroids should be unit-normalized
        self.assertAlmostEqual(float(np.linalg.norm(centroids[0])), 1.0, places=5)

        dists = euclidean_distance_to_centroid(self.embeddings[:20], centroids[0])
        self.assertEqual(len(dists), 20)
        self.assertTrue(np.all(dists >= 0))

    def test_diagnostics_and_review_flags(self):
        df = pd.DataFrame({
            "embedding_row": np.arange(4),
            "cluster_label": [0, 0, -1, 1],
            "cluster_probability": [0.95, 0.10, 0.0, 0.90],
            "outlier_score": [0.05, 0.20, 1.0, 0.99],
            "needs_review": [False, False, False, False],
            "distance_to_centroid": [0.1, 0.5, np.nan, 0.2],
        })

        flagged_df, thresh = flag_cluster_review_items(
            df,
            low_prob_threshold=0.35,
            high_outlier_quantile=0.90,
        )

        self.assertFalse(flagged_df.iloc[0]["requires_manual_review"])
        # Low prob should trigger review
        self.assertTrue(flagged_df.iloc[1]["requires_manual_review"])
        self.assertEqual(flagged_df.iloc[1]["cluster_review_reason"], "low_cluster_membership_probability")
        # Noise should trigger review
        self.assertTrue(flagged_df.iloc[2]["requires_manual_review"])
        self.assertEqual(flagged_df.iloc[2]["cluster_review_reason"], "hdbscan_noise")

        # Folder assignment planning
        assignment_df = build_folder_assignment_manifest(flagged_df)
        self.assertEqual(assignment_df.iloc[0]["assignment_category"], "cluster")
        self.assertEqual(assignment_df.iloc[1]["assignment_category"], "needs_review_cluster_member")
        self.assertEqual(assignment_df.iloc[2]["assignment_category"], "needs_review_noise")
        self.assertEqual(assignment_df.iloc[2]["proposed_folder"], "_needs_review_noise")

    def test_calculate_cluster_summary(self):
        df = pd.DataFrame({
            "embedding_row": [0, 1, 2],
            "cluster_label": [0, 0, 0],
            "cluster_folder_stub": ["cluster_00000_unknown"] * 3,
            "cluster_probability": [0.8, 0.9, 0.85],
            "outlier_score": [0.1, 0.2, 0.15],
            "distance_to_centroid": [0.1, 0.2, 0.15],
            "requires_manual_review": [False, False, False],
            "selected_region_type": ["head", "head", "person"],
        })

        summary_df = calculate_cluster_summary(df)
        self.assertEqual(len(summary_df), 1)
        self.assertEqual(summary_df.iloc[0]["image_count"], 3)
        self.assertEqual(summary_df.iloc[0]["head_crop_count"], 2)

    def test_fit_hdbscan_with_epsilon(self):
        # Small clusters of 2 points each
        c1 = np.array([[1.0, 0.0], [0.99, 0.05]], dtype=np.float32)
        c2 = np.array([[-1.0, 0.0], [-0.99, -0.05]], dtype=np.float32)
        X = np.vstack([c1, c2])
        labels, probs, outliers, _ = fit_hdbscan(
            X,
            min_cluster_size=2,
            min_samples=1,
            cluster_selection_epsilon=0.50,
        )
        self.assertEqual(len(labels), 4)
        # Should identify 2 clusters without noise
        self.assertEqual(len(set(labels) - {-1}), 2)

    def test_separate_review_folders_toggle(self):
        df = pd.DataFrame({
            "embedding_row": [0, 1],
            "cluster_label": [0, 0],
            "cluster_folder_stub": ["cluster_00000_unknown", "cluster_00000_unknown"],
            "cluster_probability": [0.9, 0.2],
            "outlier_score": [0.1, 0.9],
            "needs_review": [False, True],
            "distance_to_centroid": [0.1, 0.4],
            "is_noise": [False, False],
            "requires_manual_review": [False, True],
        })

        # By default (separate_review_folders=False), both stay in unified cluster folder
        unified = build_folder_assignment_manifest(df, separate_review_folders=False)
        self.assertEqual(unified.iloc[0]["proposed_folder"], "cluster_00000_unknown")
        self.assertEqual(unified.iloc[1]["proposed_folder"], "cluster_00000_unknown")

        # When separate_review_folders=True, review items get _review suffix
        split = build_folder_assignment_manifest(df, separate_review_folders=True)
        self.assertEqual(split.iloc[0]["proposed_folder"], "cluster_00000_unknown")
        self.assertEqual(split.iloc[1]["proposed_folder"], "cluster_00000_unknown_review")

    def test_noise_reassignment_updates_all_final_centroid_distances(self):
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
        labels = np.array([0, 0, -1])
        probabilities = np.array([0.9, 0.9, 0.0], dtype=np.float64)
        outliers = np.array([0.1, 0.1, 1.0], dtype=np.float64)

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            embed_run = base / "runs" / "03_ccip_embeddings_test"
            (embed_run / "arrays").mkdir(parents=True)
            (embed_run / "tables").mkdir()
            np.save(embed_run / "arrays" / "ccip_embeddings_l2.npy", embeddings)
            pd.DataFrame({
                "embedding_row": [0, 1, 2],
                "crop_path": ["missing0.png", "missing1.png", "missing2.png"],
                "source_path": ["source0.png", "source1.png", "source2.png"],
                "relative_path": ["0.png", "1.png", "2.png"],
                "needs_review": [False, False, False],
                "selected_region_type": ["head", "head", "head"],
            }).to_csv(embed_run / "tables" / "successful_embedding_manifest.csv", index=False)

            with patch(
                "anime_character_organizer.workflows.cluster.fit_hdbscan",
                return_value=(labels, probabilities, outliers, "test"),
            ):
                result = run_clustering(
                    project_dir=base,
                    previous_run_dir=embed_run,
                    max_reassign_distance=1.0,
                    show_progress=False,
                )

        final_centroid = result["centroids"][0]
        expected_distances = np.linalg.norm(embeddings - final_centroid, axis=1)
        np.testing.assert_allclose(
            result["cluster_manifest_df"]["distance_to_centroid"],
            expected_distances,
        )


if __name__ == "__main__":
    unittest.main()
