"""
Unit tests for preprocessing, geometry, detection selection, and cropping.
"""

from pathlib import Path
import tempfile
import unittest
import pandas as pd
from PIL import Image

from anime_character_organizer.preprocessing.cropping import (
    add_review_flags,
    make_crop_id,
    process_image_for_crop,
    save_rgb_image,
)
from anime_character_organizer.preprocessing.detection import (
    choose_best_detection,
    detection_to_dict,
)
from anime_character_organizer.preprocessing.geometry import (
    box_area,
    clamp_box,
    expand_box,
)


class TestPreprocessing(unittest.TestCase):
    def test_geometry(self):
        # Clamping
        box = (-10, -5, 120, 250)
        clamped = clamp_box(box, width=100, height=200)
        self.assertEqual(clamped, (0, 0, 100, 200))

        # Area
        self.assertEqual(box_area((10, 10, 30, 40)), 20 * 30)

        # Expansion
        exp = expand_box((50, 50, 70, 70), width=200, height=200, padding_ratio=0.5)
        # width = 20, px = 10 -> (40, 40, 80, 80)
        self.assertEqual(exp, (40, 40, 80, 80))

    def test_detection_selection(self):
        det1 = ((10, 10, 50, 50), "head", 0.8)
        det2 = ((10, 10, 80, 80), "head", 0.85)
        best = choose_best_detection([det1, det2], 100, 100)
        self.assertEqual(best, det2)

        det_dict = detection_to_dict(det1)
        self.assertEqual(det_dict["label"], "head")
        self.assertEqual(det_dict["score"], 0.8)

    def test_make_crop_id(self):
        cid = make_crop_id(42, "path/to/my_image.png")
        self.assertEqual(cid, "img_0000042_my_image")

    def test_cropping_and_saving(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            crops_dir = base / "crops"
            crops_dir.mkdir()

            test_img = base / "test.png"
            Image.new("RGB", (200, 200), color="green").save(test_img)

            row = {
                "image_index": 1,
                "path": test_img.as_posix(),
                "relative_path": "test.png",
            }

            rec = process_image_for_crop(
                row=row,
                crops_dir=crops_dir,
                crop_output_size=(100, 100),
            )

            self.assertEqual(rec["status"], "ok")
            self.assertEqual(rec["crop_width"], 100)
            self.assertEqual(rec["crop_height"], 100)
            self.assertTrue(Path(rec["crop_path"]).exists())

    def test_review_flags(self):
        df = pd.DataFrame([
            {
                "status": "ok",
                "selected_region_type": "head",
                "is_multi_head": False,
            },
            {
                "status": "ok",
                "selected_region_type": "full_image",
                "is_multi_head": False,
            },
            {
                "status": "ok",
                "selected_region_type": "head",
                "is_multi_head": True,
            },
            {
                "status": "error",
                "selected_region_type": None,
                "is_multi_head": False,
            },
        ])

        rev_df = add_review_flags(df)
        self.assertFalse(rev_df.iloc[0]["needs_review"])
        self.assertTrue(rev_df.iloc[1]["needs_review"])
        self.assertIn("full_image", rev_df.iloc[1]["review_reason"])
        self.assertTrue(rev_df.iloc[2]["needs_review"])
        self.assertIn("multiple_heads_detected", rev_df.iloc[2]["review_reason"])
        self.assertTrue(rev_df.iloc[3]["needs_review"])


if __name__ == "__main__":
    unittest.main()
