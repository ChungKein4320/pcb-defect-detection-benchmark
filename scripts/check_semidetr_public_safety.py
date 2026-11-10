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
    from .audit_semidetr_evidence import EXPERIMENTS, RESULTS, ROOT
else:
    from audit_semidetr_evidence import EXPERIMENTS, RESULTS, ROOT

EVIDENCE_NOTEBOOK = "06_historical_results_and_analysis.ipynb"
ARCHIVE_FIGURE_SOURCE = "# Display-only historical figures. See the source references above."
# Only the comparison notebook may show the reviewed original Kaggle figures from 4f3eb23.
BENCHMARK_ARCHIVES = {
    "01_prepare_dataset.ipynb": (),
    "02_train_yolo11s.ipynb": (),
    "03_train_rtdetr_l.ipynb": (),
    "04_train_faster_rcnn.ipynb": (),
    "05_train_custom_yolo11s.ipynb": (),
    "06_train_rtdetr_hybridopt.ipynb": (),
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


def audit_notebook(path: Path, *, benchmark_figures: tuple[str, ...] | None = None) -> list[str]:
    errors: list[str] = []
    try:
        raw_text = path.read_text(encoding="utf-8")
        notebook = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path.name}: invalid notebook JSON ({type(exc).__name__})"]

    if set(notebook.get("metadata", {})) - {"kernelspec", "language_info"}:
        errors.append(f"{path.name}: unexpected notebook metadata")
    benchmark = benchmark_figures is not None
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
        if benchmark and cell_id == "historical-figures":
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
    if not benchmark and path.name != EVIDENCE_NOTEBOOK and "PCB_SEMIDETR_DATA_ROOT" not in raw_text:
        errors.append(f"{path.name}: dataset configuration is missing")
    return errors


def matches_test_ap_table(markdown: str, rows: list[dict[str, str]]) -> bool:
    """Compare the five displayed methods and both label budgets with the results CSV."""
    scores = {(row["labeled_percent"], row["method"]): row["test_AP"] for row in rows}
    expected = [
        (method, scores["5", method], scores["10", method])
        for _, method, _ in EXPERIMENTS
    ]
    displayed = [
        (re.sub(r"^\[([^\]]+)\]\([^)]+\)$", r"\1", method), at_5, at_10)
        for method, at_5, at_10 in re.findall(
            r"^\| ([^|\n]+) \| (\d\.\d{4}) \| (\d\.\d{4}) \|$", markdown, re.MULTILINE,
        )
    ]
    return displayed == expected


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

    errors = [error for path in notebooks for error in audit_notebook(path)]
    for name, path in benchmark_notebooks.items():
        errors.extend(audit_notebook(path, benchmark_figures=BENCHMARK_ARCHIVES[name]))
    if errors:
        print("Notebook public-safety audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    analysis = json.loads((args.notebook_dir / EVIDENCE_NOTEBOOK).read_text(encoding="utf-8"))
    markdown = "\n".join("".join(cell["source"]) for cell in analysis["cells"] if cell["cell_type"] == "markdown")
    for label, text in (("README", (ROOT / "README.md").read_text(encoding="utf-8")), (EVIDENCE_NOTEBOOK, markdown)):
        if not matches_test_ap_table(text, rows):
            print(f"{label}: test AP table differs from the historical results CSV.")
            return 1

    print(f"Notebook public-safety audit passed: {len(notebooks) + len(benchmark_notebooks)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
