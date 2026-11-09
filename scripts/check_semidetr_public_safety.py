"""Audit public notebooks and allow only reviewed historical figure outputs."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
from pathlib import Path

if __package__:
    from .audit_semidetr_evidence import EXPERIMENTS, REPORTS, RESULTS, ROOT
else:
    from audit_semidetr_evidence import EXPERIMENTS, REPORTS, RESULTS, ROOT

EVIDENCE_NOTEBOOK = "06_historical_results_and_analysis.ipynb"
TEST_AP_FIGURE = REPORTS / "figures" / "historical_test_ap.png"
# Review the regenerated chart before changing this allowlist. Epoch-level figures stay local.
TEST_AP_FIGURE_SHA256 = "6dae89436fb1b3c678cb853fe2c250ac7dda7e98e308d056dde4b3917f880fc0"
ARCHIVE_FIGURE_SOURCE = "# Display-only historical figures. See the source references above."
# Exact SHA-256 fingerprints of the reviewed PNG outputs from the public Kaggle archive at 4f3eb23.
BENCHMARK_ARCHIVES = {
    "01_prepare_dataset.ipynb": (),
    "02_train_yolo11s.ipynb": ("2e94210d3012fe23ee064c43eab3ece5e0ae7a5c309047916315663b7883dd81",),
    "03_train_rtdetr_l.ipynb": ("d8551cfe4af86871919a83afd9fb602397d8b63bfbea21548419584c03609369",),
    "04_train_faster_rcnn.ipynb": ("fc7817d092c5c782632322635582605a54041e0f3b7e9665d810e01bed18d29c",),
    "05_train_custom_yolo11s.ipynb": ("b8ae79a2cacf263769d0066a30dad57d5e38aab59af6a93e3bce78147fc9b472",),
    "06_train_rtdetr_hybridopt.ipynb": ("7b6db8a2b37557aa62aa02f15047c47bcc8bbae5fcf674c57cddce28d5700fbf",),
    "07_compare_results.ipynb": (
        "8b54baef9ed46561a07ea088cdd93b399fa6073564cdf76c054857a119788555",
        "eaddca5ba410be6ae7a65719b3f35ae68489751bf61714de39e2d8d34d91c7f8",
    ),
}
PRIVATE_CONTENT = re.compile(
    r"\b[A-Za-z]:[/\\]|/(?:Users|home)/|data:image/|"
    r'\b(?:hf_[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|'
    r'github_pat_[A-Za-z0-9_]{40,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b|'
    r'-----BEGIN [A-Z ]*PRIVATE KEY-----|"(?:dataSources|widgets|papermill)"|'
    r"""(?i:(?:api[_-]?key|access[_-]?token|hf_token|kaggle_key|password|client[_-]?secret)["']?\]?\s*=\s*["'][^"'${}<>]+["'])"""
)


def png_fingerprints(outputs: list[dict]) -> tuple[str, ...] | None:
    hashes = []
    for output in outputs:
        data = output.get("data", {})
        if (
            set(output) != {"output_type", "data", "metadata"}
            or output["output_type"] != "display_data"
            or output["metadata"] != {}
            or set(data) != {"image/png"}
        ):
            return None
        try:
            png = base64.b64decode("".join(data["image/png"]).replace("\n", ""), validate=True)
        except (TypeError, ValueError):
            return None
        hashes.append(hashlib.sha256(png).hexdigest())
    return tuple(hashes)


def audit_notebook(
    path: Path, approved_png: bytes, *, benchmark_figures: tuple[str, ...] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        raw_text = path.read_text(encoding="utf-8")
        notebook = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path.name}: invalid notebook JSON ({type(exc).__name__})"]

    if set(notebook.get("metadata", {})) - {"kernelspec", "language_info"}:
        errors.append(f"{path.name}: unexpected notebook metadata")
    benchmark = benchmark_figures is not None
    is_evidence = not benchmark and path.name == EVIDENCE_NOTEBOOK
    evidence_cells = 0
    archive_sections = 0
    archive_figures = 0
    cells = notebook.get("cells", [])
    for index, cell in enumerate(cells):
        location = f"{path.name}: cell {index}"
        cell_id = cell.get("id")
        source = "".join(cell.get("source", []))
        if PRIVATE_CONTENT.search(source) or (
            (not benchmark or cell_id == "historical-evidence") and "/kaggle/input/" in source
        ):
            errors.append(f"{location}: embedded media, local path, or credential-like text found")
        if cell.get("attachments") or cell.get("metadata"):
            errors.append(f"{location} contains attachments or metadata")
        if cell_id == "historical-evidence":
            archive_sections += 1
            if cell.get("cell_type") != "markdown" or not source.startswith("## Archived Historical Execution Evidence\n"):
                errors.append(f"{location} must be labeled historical evidence")
        if is_evidence and cell_id == "historical-test-ap":
            evidence_cells += 1
            if png_fingerprints(cell.get("outputs", [])) != (hashlib.sha256(approved_png).hexdigest(),):
                errors.append(f"{location} must contain only the approved historical test AP figure")
        elif benchmark and cell_id == "historical-figures":
            archive_figures += 1
            if (
                not benchmark_figures
                or cell.get("cell_type") != "code"
                or cell.get("execution_count") is not None
                or source != ARCHIVE_FIGURE_SOURCE
                or index == 0 or cells[index - 1].get("id") != "historical-evidence"
                or png_fingerprints(cell.get("outputs", [])) != benchmark_figures
            ):
                errors.append(f"{location} must contain only the approved archived PNGs in a display-only cell")
        elif cell.get("outputs") or cell.get("execution_count") is not None:
            errors.append(f"{location} contains an execution record")

    if benchmark and (archive_sections != 1 or archive_figures != int(bool(benchmark_figures))):
        errors.append(f"{path.name}: missing or duplicate historical evidence section")
    if is_evidence and evidence_cells != 1:
        errors.append(f"{path.name}: expected one historical test AP figure cell")
    if not benchmark and not is_evidence and "PCB_SEMIDETR_DATA_ROOT" not in raw_text:
        errors.append(f"{path.name}: dataset configuration is missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--notebook-dir",
        type=Path,
        default=ROOT / "notebooks" / "semi_supervised_research",
    )
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "notebooks" / "public_benchmark")
    args = parser.parse_args()

    notebooks = sorted(args.notebook_dir.glob("*.ipynb"))
    if len(notebooks) != len(EXPERIMENTS) + 1 or not (args.notebook_dir / EVIDENCE_NOTEBOOK).is_file():
        print("Expected five implementation notebooks and one evidence notebook.")
        return 1
    benchmark_notebooks = {path.name: path for path in args.benchmark_dir.glob("*.ipynb")}
    if set(benchmark_notebooks) != set(BENCHMARK_ARCHIVES):
        print("Public benchmark notebook set differs from the evidence allowlist.")
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
    for name, path in benchmark_notebooks.items():
        errors.extend(audit_notebook(path, approved_png, benchmark_figures=BENCHMARK_ARCHIVES[name]))
    if errors:
        print("Notebook public-safety audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Notebook public-safety audit passed: {len(notebooks) + len(benchmark_notebooks)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
