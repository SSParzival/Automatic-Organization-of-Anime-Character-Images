"""
Unit tests for embeddings modules: batching, validation, and normalization.
"""

import unittest
import numpy as np

from anime_character_organizer.embeddings.extraction import batch_iterable
from anime_character_organizer.embeddings.normalization import l2_normalize_matrix


class TestEmbeddings(unittest.TestCase):
    def test_batch_iterable(self):
        items = list(range(10))
        batches = list(batch_iterable(items, batch_size=3))
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0], [0, 1, 2])
        self.assertEqual(batches[-1], [9])

    def test_l2_normalization(self):
        mat = np.array([
            [3.0, 4.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        norm_mat = l2_normalize_matrix(mat)
        self.assertEqual(norm_mat.shape, (3, 2))
        self.assertAlmostEqual(float(np.linalg.norm(norm_mat[0])), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(norm_mat[1])), 1.0, places=5)
        self.assertAlmostEqual(norm_mat[0, 0], 0.6, places=5)
        self.assertAlmostEqual(norm_mat[0, 1], 0.8, places=5)

    def test_l2_normalization_invalid_dims(self):
        with self.assertRaises(ValueError):
            l2_normalize_matrix(np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
