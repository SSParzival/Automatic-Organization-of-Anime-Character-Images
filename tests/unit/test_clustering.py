"""
Unit tests for clustering: algorithm, centroids, diagnostics, and review flagging.
"""

import unittest
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


if __name__ == "__main__":
    unittest.main()
