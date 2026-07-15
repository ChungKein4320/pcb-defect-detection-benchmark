from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_dataset_integrity import audit_dataset


class DatasetIntegrityAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        for shade, split in enumerate(("train", "valid", "test"), start=1):
            (self.temp_dir / split / "images").mkdir(parents=True)
            (self.temp_dir / split / "labels").mkdir(parents=True)
            self._write_sample(split, f"DeepPCB__{split}_jpg.rf.abc123", shade)
        (self.temp_dir / "data.yaml").write_text(
            "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _write_sample(self, split: str, stem: str, shade: int) -> None:
        image_path = self.temp_dir / split / "images" / f"{stem}.jpg"
        Image.new("RGB", (8, 8), color=(shade, shade, shade)).save(image_path)
        (self.temp_dir / split / "labels" / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )

    def test_valid_dataset_passes(self) -> None:
        report = audit_dataset(self.temp_dir)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["cross_split"]["byte_identical_sha256_groups"], 0)
        self.assertEqual(report["within_split"]["byte_identical_sha256_groups"], 0)
        self.assertEqual(len(report["dataset_fingerprint_sha256"]), 64)

    def test_committed_aggregate_report_is_consistent(self) -> None:
        report = json.loads(
            (ROOT / "reports" / "dataset_integrity_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["errors"], [])
        self.assertEqual(
            {split: values["images"] for split, values in report["splits"].items()},
            {"train": 6624, "valid": 937, "test": 1887},
        )
        self.assertTrue(
            all(value == 0 for value in report["cross_split"].values())
        )
        self.assertTrue(
            all(value == 0 for value in report["within_split"].values())
        )
        self.assertEqual(len(report["dataset_fingerprint_sha256"]), 64)
        self.assertIn(
            "absence of all possible data leakage", report["scope"]["not_claimed"]
        )

    def test_byte_identical_cross_split_image_fails(self) -> None:
        train_image = next((self.temp_dir / "train" / "images").iterdir())
        valid_image = next((self.temp_dir / "valid" / "images").iterdir())
        shutil.copyfile(train_image, valid_image)
        report = audit_dataset(self.temp_dir)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cross_split"]["byte_identical_sha256_groups"], 1)

    def test_repeated_pre_export_source_id_fails(self) -> None:
        old_image = next((self.temp_dir / "valid" / "images").iterdir())
        old_label = next((self.temp_dir / "valid" / "labels").iterdir())
        new_stem = "DeepPCB__train_jpg.rf.def456"
        old_image.rename(old_image.with_name(f"{new_stem}.jpg"))
        old_label.rename(old_label.with_name(f"{new_stem}.txt"))
        report = audit_dataset(self.temp_dir)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cross_split"]["normalized_source_id_groups"], 1)

    def test_invalid_yolo_row_fails(self) -> None:
        label = next((self.temp_dir / "test" / "labels").iterdir())
        label.write_text("6 1.2 0.5 0.0 0.2\n", encoding="utf-8")
        report = audit_dataset(self.temp_dir)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["splits"]["test"]["invalid_label_rows"], 1)

    def test_posix_absolute_data_path_warns_on_windows(self) -> None:
        (self.temp_dir / "data.yaml").write_text(
            "path: /kaggle/working/dataset\n", encoding="utf-8"
        )
        report = audit_dataset(self.temp_dir)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["data_yaml_absolute_path"])
        self.assertTrue(any("absolute path" in item for item in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
