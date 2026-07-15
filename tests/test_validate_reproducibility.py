from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_reproducibility import MANIFEST_PATH, validate_manifest, validate_repository


class ReproducibilityValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((ROOT / MANIFEST_PATH).read_text(encoding="utf-8"))

    def test_committed_repository_is_valid(self) -> None:
        self.assertEqual(validate_repository(ROOT), [])

    def test_duplicate_experiment_id_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["experiments"][1]["id"] = manifest["experiments"][0]["id"]
        errors = validate_manifest(ROOT, manifest)
        self.assertTrue(any("ids must be unique" in error for error in errors))

    def test_incorrect_test_image_count_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["dataset"]["test_images"] = 1
        errors = validate_manifest(ROOT, manifest)
        self.assertTrue(any("1,887-image test set" in error for error in errors))

    def test_result_path_outside_repository_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["experiments"][0]["results_file"] = "../outside.csv"
        errors = validate_manifest(ROOT, manifest)
        self.assertTrue(any("invalid or missing results_file" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
