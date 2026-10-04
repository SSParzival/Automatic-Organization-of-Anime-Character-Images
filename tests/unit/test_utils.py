"""
Unit tests for utils modules.
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image

from anime_character_organizer.exceptions import RunNotFoundError
from anime_character_organizer.utils.hashing import (
    hamming_int,
    hash_hex_to_int,
    imagehash_to_hex,
    sha256_file,
    stable_short_hash,
)
from anime_character_organizer.utils.naming import (
    sanitize_filename_component,
    sanitize_folder_name,
)
from anime_character_organizer.utils.paths import (
    normalize_path_string,
    path_to_posix,
    safe_relative_path,
)
from anime_character_organizer.utils.runs import find_latest_run
from anime_character_organizer.utils.serialization import (
    dataframe_to_json_records,
    parse_dict_like,
    safe_bool,
    safe_bool_series,
    safe_float,
    safe_int,
    safe_json_dumps,
)
from anime_character_organizer.utils.structures import BKTree, UnionFind
from anime_character_organizer.utils.time import now_iso
from anime_character_organizer.utils.validation import validate_image_file


class TestUtils(unittest.TestCase):
    def test_now_iso(self):
        t = now_iso()
        self.assertIsInstance(t, str)
        self.assertIn("T", t)

    def test_paths(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            sub = base / "a" / "b.txt"
            sub.parent.mkdir(parents=True)
            sub.touch()

            self.assertEqual(path_to_posix(sub), sub.as_posix())
            self.assertEqual(safe_relative_path(sub, base), "a/b.txt")
            self.assertEqual(normalize_path_string(sub), sub.resolve().as_posix())

    def test_hashing(self):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(b"test data 123")
            tf_path = tf.name

        try:
            h = sha256_file(tf_path)
            self.assertEqual(len(h), 64)
            # Short hash
            sh = stable_short_hash("test", length=8)
            self.assertEqual(len(sh), 8)
            # Int conversion and Hamming
            hex_val = "000000000000000f"
            int_val = hash_hex_to_int(hex_val)
            self.assertEqual(int_val, 15)
            self.assertEqual(hamming_int(15, 0), 4)
            self.assertEqual(hamming_int(15, 14), 1)
        finally:
            Path(tf_path).unlink()

    def test_union_find(self):
        uf = UnionFind(["a", "b", "c", "d"])
        self.assertNotEqual(uf.find("a"), uf.find("b"))
        uf.union("a", "b")
        self.assertEqual(uf.find("a"), uf.find("b"))
        uf.union("c", "d")
        uf.union("b", "c")
        self.assertEqual(uf.find("a"), uf.find("d"))

    def test_bktree(self):
        def dist(a, b):
            return hamming_int(a, b)

        tree = BKTree(dist)
        tree.add(0b0000)
        tree.add(0b0001)
        tree.add(0b0011)
        tree.add(0b1111)

        matches = tree.query(0b0000, threshold=1)
        match_vals = [m[0] for m in matches]
        self.assertIn(0b0000, match_vals)
        self.assertIn(0b0001, match_vals)
        self.assertNotIn(0b1111, match_vals)

    def test_serialization(self):
        self.assertTrue(safe_bool("True"))
        self.assertTrue(safe_bool(1))
        self.assertFalse(safe_bool("false"))
        self.assertFalse(safe_bool(None))

        s = pd.Series(["yes", "no", "1", "0", True, False])
        bool_s = safe_bool_series(s)
        self.assertListEqual(bool_s.tolist(), [True, False, True, False, True, False])

        self.assertEqual(safe_float("1.25"), 1.25)
        self.assertTrue(np.isnan(safe_float("invalid")))

        self.assertEqual(safe_int("42"), 42)
        self.assertEqual(safe_int(None, default=99), 99)

        d = {"b": 2, "a": 1}
        self.assertEqual(safe_json_dumps(d), '{"a": 1, "b": 2}')
        self.assertEqual(parse_dict_like('{"x": 10}'), {"x": 10})
        self.assertEqual(parse_dict_like(None), {})

        df = pd.DataFrame([{"a": 1, "b": np.nan}])
        records = dataframe_to_json_records(df)
        self.assertEqual(records, [{"a": 1, "b": None}])

    def test_naming(self):
        self.assertEqual(sanitize_filename_component("hello/world:test*"), "hello_world_test_")
        self.assertEqual(sanitize_folder_name(""), "_needs_review")
        self.assertEqual(sanitize_folder_name("  folder.. "), "folder")

    def test_validation(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(tf.name)
            img_path = tf.name

        try:
            valid, err_t, err_m = validate_image_file(img_path)
            self.assertTrue(valid)
            self.assertIsNone(err_t)

            # Test invalid image (empty file)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as empty_tf:
                empty_path = empty_tf.name

            try:
                valid_empty, err_t, _ = validate_image_file(empty_path)
                self.assertFalse(valid_empty)
                self.assertEqual(err_t, "EmptyFile")
            finally:
                Path(empty_path).unlink()

        finally:
            Path(img_path).unlink()

    def test_runs_discovery(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            runs = base / "runs"
            runs.mkdir()

            r1 = runs / "01_dataset_audit_20260101_000000"
            r1.mkdir()
            (r1 / "tables").mkdir()
            (r1 / "tables" / "valid_images.csv").touch()

            r2 = runs / "01_dataset_audit_20260102_000000"
            r2.mkdir()
            (r2 / "tables").mkdir()
            (r2 / "tables" / "valid_images.csv").touch()

            latest = find_latest_run(base, "01_dataset_audit_*", [Path("tables") / "valid_images.csv"])
            self.assertEqual(latest, r2)

            with self.assertRaises(RunNotFoundError):
                find_latest_run(base, "09_non_existent_*")


if __name__ == "__main__":
    unittest.main()
