from __future__ import annotations

import base64
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_semidetr_public_safety import (
    ARCHIVE_FIGURE_SOURCE,
    EVIDENCE_NOTEBOOK,
    RESULTS,
    audit_notebook,
    matches_test_ap_table,
)


class SemiDetrDisclosureTests(unittest.TestCase):
    def audit(self, name: str, cells: list[dict], **options) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
            return audit_notebook(path, **options)

    def test_detr_analysis_rejects_saved_outputs(self) -> None:
        cell = {
            "id": "historical-test-ap",
            "source": [],
            "outputs": [],
        }
        self.assertEqual(self.audit(EVIDENCE_NOTEBOOK, [cell]), [])
        cell.update({
            "outputs": [{
                "output_type": "display_data",
                "data": {"image/png": base64.b64encode(b"previous DETR figure").decode("ascii")},
                "metadata": {},
            }],
        })
        errors = self.audit(EVIDENCE_NOTEBOOK, [cell])
        self.assertTrue(any("execution record" in error for error in errors))

    def test_detr_table_rejects_a_changed_score(self) -> None:
        with RESULTS.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue(matches_test_ap_table(readme, rows))
        self.assertFalse(matches_test_ap_table(readme.replace("0.5422", "0.5423"), rows))

    def test_training_output_and_credential_assignment_are_rejected(self) -> None:
        cell = {
            "source": ['os.environ["HF_TOKEN"] = "example-value"\n', "PCB_SEMIDETR_DATA_ROOT\n"],
            "outputs": [{"output_type": "stream", "name": "stdout", "text": ["run log"]}],
        }
        errors = self.audit("01_supervised_deformable_detr.ipynb", [cell])
        self.assertTrue(any("credential-like text" in error for error in errors))
        self.assertTrue(any("execution record" in error for error in errors))

    def test_kaggle_archive_requires_reviewed_png_and_display_only_cell(self) -> None:
        name = "07_compare_results.ipynb"
        png = b"reviewed Kaggle figure"
        archive = {
            "id": "historical-evidence",
            "cell_type": "markdown",
            "source": "## Archived Historical Execution Evidence\n\nOriginal validation plot.",
        }
        figure = {
            "id": "historical-figures",
            "cell_type": "code",
            "source": ARCHIVE_FIGURE_SOURCE,
            "execution_count": None,
            "outputs": [{
                "output_type": "display_data",
                "data": {"image/png": base64.b64encode(png).decode("ascii")},
                "metadata": {},
            }],
        }
        allowed = (hashlib.sha256(png).hexdigest(),)
        self.assertEqual(self.audit(name, [archive, figure], benchmark_figures=allowed), [])
        for invalid in (
            {**figure, "execution_count": 1},
            {**figure, "source": "model.train()"},
            {**figure, "outputs": [{"output_type": "stream", "text": "training log"}]},
        ):
            with self.subTest(invalid=invalid):
                errors = self.audit(name, [archive, invalid], benchmark_figures=allowed)
                self.assertTrue(any("display-only cell" in error for error in errors))
        errors = self.audit(name, [archive, figure], benchmark_figures=("0" * 64,))
        self.assertTrue(any("approved archived PNGs" in error for error in errors))
        errors = self.audit("02_train_yolo11s.ipynb", [archive, figure], benchmark_figures=())
        self.assertTrue(any("display-only cell" in error for error in errors))
        archive["source"] += '\nOutput path: /kaggle/input/example'
        errors = self.audit(name, [archive, figure], benchmark_figures=allowed)
        self.assertTrue(any("local path" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
