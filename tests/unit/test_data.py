"""
Unit tests for data audit and duplicate detection modules.
"""

from pathlib import Path
import tempfile
import unittest
import pandas as pd
from PIL import Image

from anime_character_organizer.data.audit import audit_images, inspect_image, scan_candidate_files
from anime_character_organizer.data.duplicates import find_exact_duplicates, find_perceptual_duplicates


class TestDataAuditAndDuplicates(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

        # Create 3 test images:
        # img1 and img2 are identical (exact duplicates)
        # img3 has slight variation (near duplicate)
        # img4 is different
        self.img1_path = self.base_dir / "img1.png"
        self.img2_path = self.base_dir / "sub" / "img2.png"
        self.img2_path.parent.mkdir(parents=True, exist_ok=True)
        self.img3_path = self.base_dir / "img3.png"

        im1 = Image.new("RGB", (64, 64), color=(255, 0, 0))
        im1.save(self.img1_path)
        im1.save(self.img2_path)

        # Slightly modify one pixel for im3
        im3 = Image.new("RGB", (64, 64), color=(254, 0, 0))
        im3.save(self.img3_path)

        # Corrupted file
        self.corrupt_path = self.base_dir / "corrupt.png"
        with self.corrupt_path.open("wb") as f:
            f.write(b"not a real image")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scan_candidate_files(self):
        candidates = scan_candidate_files(self.base_dir, valid_extensions={".png"})
        candidate_names = [p.name for p in candidates]
        self.assertIn("img1.png", candidate_names)
        self.assertIn("img2.png", candidate_names)
        self.assertIn("img3.png", candidate_names)
        self.assertIn("corrupt.png", candidate_names)

    def test_inspect_image(self):
        rec_valid = inspect_image(self.img1_path, self.base_dir)
        self.assertEqual(rec_valid["status"], "valid")
        self.assertEqual(rec_valid["width"], 64)
        self.assertEqual(rec_valid["height"], 64)
        self.assertIsNotNone(rec_valid["sha256"])
        self.assertIsNotNone(rec_valid["phash"])

        rec_corrupt = inspect_image(self.corrupt_path, self.base_dir)
        self.assertEqual(rec_corrupt["status"], "invalid")
        self.assertIsNotNone(rec_corrupt["error_type"])

    def test_audit_images(self):
        candidates = scan_candidate_files(self.base_dir, valid_extensions={".png"})
        meta_df, valid_df, invalid_df = audit_images(
            candidate_files=candidates,
            input_dir=self.base_dir,
            show_progress=False,
        )
        self.assertEqual(len(meta_df), 4)
        self.assertEqual(len(valid_df), 3)
        self.assertEqual(len(invalid_df), 1)

    def test_exact_duplicates(self):
        candidates = [self.img1_path, self.img2_path, self.img3_path]
        _, valid_df, _ = audit_images(candidates, self.base_dir, show_progress=False)

        exact_df = find_exact_duplicates(valid_df)
        self.assertFalse(exact_df.empty)
        self.assertEqual(exact_df["exact_group_id"].nunique(), 1)
        self.assertEqual(len(exact_df), 2)
        # Check representative flag
        self.assertEqual(exact_df["is_representative"].sum(), 1)

    def test_perceptual_duplicates(self):
        candidates = [self.img1_path, self.img2_path, self.img3_path]
        _, valid_df, _ = audit_images(candidates, self.base_dir, show_progress=False)

        pairs_df, groups_df = find_perceptual_duplicates(valid_df, phash_threshold=6, show_progress=False)
        self.assertFalse(pairs_df.empty)
        self.assertFalse(groups_df.empty)
        # All 3 red images should be clustered into 1 perceptual group
        self.assertEqual(groups_df["perceptual_group_id"].nunique(), 1)
        self.assertEqual(len(groups_df), 3)


if __name__ == "__main__":
    unittest.main()
