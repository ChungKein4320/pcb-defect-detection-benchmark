"""Fail when public Semi-DETR notebooks contain private or embedded artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
from pathlib import Path

if __package__:
    from .audit_semidetr_evidence import EXPERIMENTS, REPORTS, RESULTS
else:
    from audit_semidetr_evidence import EXPERIMENTS, REPORTS, RESULTS

EVIDENCE_NOTEBOOK = "06_historical_results_and_analysis.ipynb"
TEST_AP_FIGURE = REPORTS / "figures" / "historical_test_ap.png"
# Review the regenerated chart before changing this allowlist. Epoch-level figures stay local.
TEST_AP_FIGURE_SHA256 = "6dae89436fb1b3c678cb853fe2c250ac7dda7e98e308d056dde4b3917f880fc0"
PRIVATE_CONTENT = re.compile(
    r"/kaggle/input/|\b[A-Za-z]:[/\\]|/(?:Users|home)/|data:image/|"
    r'hf_[A-Za-z0-9]{20,}|"(?:dataSources|widgets|papermill)"|'
    r"""(?i:(?:api[_-]?key|access[_-]?token|hf_token|password|client[_-]?secret)["']?\]?\s*=\s*["'][^"'${}<>]+["'])"""
)


def audit_notebook(path: Path, approved_png: bytes) -> list[str]:
    errors: list[str] = []
    try:
        raw_text = path.read_text(encoding="utf-8")
        notebook = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path.name}: invalid notebook JSON ({type(exc).__name__})"]

    if set(notebook.get("metadata", {})) - {"kernelspec", "language_info"}:
        errors.append(f"{path.name}: unexpected notebook metadata")
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    if PRIVATE_CONTENT.search(sources):
        errors.append(f"{path.name}: embedded media, local path, or credential-like text found")

    is_evidence = path.name == EVIDENCE_NOTEBOOK
    expected_image = base64.b64encode(approved_png).decode("ascii")
    evidence_cells = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        location = f"{path.name}: cell {index}"
        if cell.get("attachments") or cell.get("metadata"):
            errors.append(f"{location} contains attachments or metadata")
        if is_evidence and cell.get("id") == "historical-test-ap":
            evidence_cells += 1
            outputs = cell.get("outputs", [])
            output = outputs[0] if len(outputs) == 1 else {}
            image = output.get("data", {})
            if (
                set(output) != {"output_type", "data", "metadata"}
                or output["output_type"] != "display_data"
                or output["metadata"] != {}
                or set(image) != {"image/png"}
                or "".join(image["image/png"]).replace("\n", "") != expected_image
            ):
                errors.append(f"{location} must contain only the approved historical test AP figure")
        elif cell.get("outputs") or cell.get("execution_count") is not None:
            errors.append(f"{location} contains an execution record")

    if is_evidence and evidence_cells != 1:
        errors.append(f"{path.name}: expected one historical test AP figure cell")
    if not is_evidence and "PCB_SEMIDETR_DATA_ROOT" not in raw_text:
        errors.append(f"{path.name}: dataset configuration is missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit sanitized Semi-DETR notebooks before committing them."
    )
    parser.add_argument(
        "--notebook-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "notebooks" / "semi_supervised_research",
    )
    args = parser.parse_args()

    notebooks = sorted(args.notebook_dir.glob("*.ipynb"))
    if len(notebooks) != len(EXPERIMENTS) + 1 or not (args.notebook_dir / EVIDENCE_NOTEBOOK).is_file():
        print("Expected five implementation notebooks and one evidence notebook.")
        return 1

    with RESULTS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_keys = [(str(ratio), experiment, method) for ratio in (5, 10) for experiment, method, _ in EXPERIMENTS]
    if [(row["labeled_percent"], row["experiment"], row["method"]) for row in rows] != expected_keys:
        print("Historical results must list the five methods for each labeled ratio.")
        return 1
    for row in rows:
        if any(not re.fullmatch(r"[01]\.\d{4}", row[f"{split}_{metric}"])
               or not 0 <= float(row[f"{split}_{metric}"]) <= 1
               for split in ("val", "test") for metric in ("AP", "AP50", "AP75", "APs")):
            print("Historical results contain an invalid AP field.")
            return 1
        if row["seed"] != "42" or not re.fullmatch(r"\d{1,3}", row["best_epoch"]) or not 1 <= int(row["best_epoch"]) <= 150:
            print("Historical results contain invalid run metadata.")
            return 1

    if not TEST_AP_FIGURE.is_file():
        print("Approved historical test AP figure is missing.")
        return 1
    approved_png = TEST_AP_FIGURE.read_bytes()
    if hashlib.sha256(approved_png).hexdigest() != TEST_AP_FIGURE_SHA256:
        print("Historical test AP figure differs from the reviewed PNG.")
        return 1

    errors = [error for path in notebooks for error in audit_notebook(path, approved_png)]
    if errors:
        print("Semi-DETR public-safety audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Semi-DETR public-safety audit passed: {len(notebooks)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
