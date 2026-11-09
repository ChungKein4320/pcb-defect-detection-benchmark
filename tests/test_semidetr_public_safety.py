from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_semidetr_public_safety import EVIDENCE_NOTEBOOK, audit_notebook


class SemiDetrDisclosureTests(unittest.TestCase):
    def audit(self, name: str, cell: dict, approved_png: bytes = b"") -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_text(json.dumps({"cells": [cell]}), encoding="utf-8")
            return audit_notebook(path, approved_png)

    def test_evidence_output_must_match_the_allowlisted_png(self) -> None:
        approved_png = b"example approved asset"
        cell = {
            "id": "historical-test-ap",
            "source": [],
            "outputs": [{
                "output_type": "display_data",
                "data": {"image/png": base64.b64encode(approved_png).decode("ascii")},
                "metadata": {},
            }],
        }
        self.assertEqual(self.audit(EVIDENCE_NOTEBOOK, cell, approved_png), [])
        errors = self.audit(EVIDENCE_NOTEBOOK, cell, b"other asset")
        self.assertTrue(any("approved historical test AP figure" in error for error in errors))

        cell["outputs"][0]["data"]["text/plain"] = "Additional output"
        errors = self.audit(EVIDENCE_NOTEBOOK, cell, approved_png)
        self.assertTrue(any("approved historical test AP figure" in error for error in errors))

    def test_training_output_and_credential_assignment_are_rejected(self) -> None:
        cell = {
            "source": ['os.environ["HF_TOKEN"] = "example-value"\n', "PCB_SEMIDETR_DATA_ROOT\n"],
            "outputs": [{"output_type": "stream", "name": "stdout", "text": ["run log"]}],
        }
        errors = self.audit("01_supervised_deformable_detr.ipynb", cell)
        self.assertTrue(any("credential-like text" in error for error in errors))
        self.assertTrue(any("execution record" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
