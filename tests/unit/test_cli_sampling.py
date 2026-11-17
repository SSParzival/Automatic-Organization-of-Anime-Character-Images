"""
Tests for representative sampling, .env loading and output-dir guards.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from anime_character_organizer.data.audit import sample_candidate_files
from anime_character_organizer.exceptions import ConfigurationError
from anime_character_organizer.utils.paths import ensure_empty_output_dir, load_dotenv


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.paths = [Path(f"/d{i % 3}/img{i:03d}.png") for i in range(30)]

    def test_no_cap_returns_all(self):
        self.assertEqual(sample_candidate_files(self.paths, None), self.paths)
        self.assertEqual(sample_candidate_files(self.paths, 100), self.paths)

    def test_stratified_covers_all_directories(self):
        sample = sample_candidate_files(self.paths, 6)
        self.assertEqual(len(sample), 6)
        self.assertEqual({p.parent for p in sample}, {Path("/d0"), Path("/d1"), Path("/d2")})

    def test_random_is_deterministic_and_first_is_prefix(self):
        a = sample_candidate_files(self.paths, 5, "random", seed=1)
        self.assertEqual(a, sample_candidate_files(self.paths, 5, "random", seed=1))
        self.assertEqual(sample_candidate_files(self.paths, 4, "first"), self.paths[:4])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            sample_candidate_files(self.paths, 0)
        with self.assertRaises(ValueError):
            sample_candidate_files(self.paths, 3, "bogus")


class TestDotenvAndOutputGuard(unittest.TestCase):
    def test_load_dotenv_does_not_override_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text('# c\nexport AOC_A="one"\nAOC_B=two\nAOC_EMPTY=\nnoequals\n', encoding="utf-8")
            with mock.patch.dict(os.environ, {"AOC_B": "kept"}, clear=False):
                self.assertEqual(load_dotenv(env), env.resolve())
                self.assertEqual(os.environ["AOC_A"], "one")
                self.assertEqual(os.environ["AOC_B"], "kept")
                self.assertNotIn("AOC_EMPTY", os.environ)

    def test_load_dotenv_missing_explicit_file(self):
        with self.assertRaises(ConfigurationError):
            load_dotenv("/nonexistent/.env")

    def test_ensure_empty_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(ensure_empty_output_dir(root / "new"), (root / "new").resolve())
            (root / "full").mkdir()
            (root / "full" / "x").write_text("x")
            with self.assertRaises(ConfigurationError):
                ensure_empty_output_dir(root / "full")
            with self.assertRaises(ConfigurationError):
                ensure_empty_output_dir(root / "full" / "x")


if __name__ == "__main__":
    unittest.main()
