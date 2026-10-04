"""
Unit tests for folder materialization, destination planning, file operations, and validation.
"""

from pathlib import Path
import tempfile
import unittest
import pandas as pd

from anime_character_organizer.materialization.operations import (
    materialize_one_file,
    remove_existing_file,
)
from anime_character_organizer.materialization.planner import (
    build_materialization_plan,
    ensure_unique_destination_path,
    make_destination_filename,
)
from anime_character_organizer.materialization.validation import validate_materialized_files


class TestMaterialization(unittest.TestCase):
    def test_destination_naming_and_uniqueness(self):
        row = {"embedding_row": 5}
        fn = make_destination_filename(row, "path/to/my_image.png")
        self.assertEqual(fn, "0000005__my_image.png")

        used = set()
        dest = Path("/tmp/folder/0000005__my_image.png")
        p1 = ensure_unique_destination_path(dest, used)
        self.assertEqual(p1, dest)
        self.assertIn(dest.as_posix(), used)

        # Second call with same name produces dup suffix
        p2 = ensure_unique_destination_path(dest, used)
        self.assertEqual(p2.name, "0000005__my_image__dup001.png")

    def test_materialize_operations(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src = base / "source.txt"
            src.write_text("hello anime")

            dest_copy = base / "out_copy" / "copied.txt"
            dest_hard = base / "out_hard" / "linked.txt"
            dest_sym = base / "out_sym" / "symlinked.txt"

            # Copy mode
            res_c = materialize_one_file(src, dest_copy, mode="copy")
            self.assertEqual(res_c["status"], "ok")
            self.assertTrue(dest_copy.exists())
            self.assertEqual(dest_copy.read_text(), "hello anime")

            # Hardlink mode
            res_h = materialize_one_file(src, dest_hard, mode="hardlink")
            self.assertEqual(res_h["status"], "ok")
            self.assertTrue(dest_hard.exists())

            # Symlink mode
            res_s = materialize_one_file(src, dest_sym, mode="symlink")
            self.assertEqual(res_s["status"], "ok")
            self.assertTrue(dest_sym.is_symlink())

            # Test on_existing='skip'
            res_skip = materialize_one_file(src, dest_copy, mode="copy", on_existing="skip")
            self.assertEqual(res_skip["status"], "skipped_existing")

            # Test on_existing='error'
            res_err = materialize_one_file(src, dest_copy, mode="copy", on_existing="error")
            self.assertEqual(res_err["status"], "error")

            # Remove file
            remove_existing_file(dest_sym)
            self.assertFalse(dest_sym.exists())

    def test_validation(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src = base / "source.txt"
            src.write_text("content 123")

            dest = base / "dest.txt"
            dest.write_text("content 123")

            res_df = pd.DataFrame([{
                "embedding_row": 1,
                "source_path": src.as_posix(),
                "destination_path": dest.as_posix(),
                "status": "ok",
            }])

            val_df = validate_materialized_files(res_df)
            self.assertTrue(val_df.iloc[0]["is_valid"])


if __name__ == "__main__":
    unittest.main()
