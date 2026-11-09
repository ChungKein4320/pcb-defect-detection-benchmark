"""Verify aggregate DETR results against the local coursework notebooks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "semi_supervised_research"
RESULTS = REPORTS / "historical_results.csv"
SPLIT_AUDIT = REPORTS / "split_audit.json"
EXPERIMENTS = (
    ("supervised", "Supervised Deformable DETR", "02-supervised-deformable-detr.ipynb"),
    ("vanilla", "Vanilla DETR-SSOD", "03a-vanilla-detr-ssod-baseline.ipynb"),
    ("shm", "SHM approximation", "04-detr-ssod-shm.ipynb"),
    ("shm_cqc", "SHM + CQC proxy", "05-semi-detr-ablation-shm-cqc.ipynb"),
    ("shm_cqc_cpm", "SHM + CQC + CPM", "06-full-semi-detr-shm-cqc-cpm.ipynb"),
)
METRICS = ("AP", "AP50", "AP75", "APs")
AP_PATTERN = " ".join(fr"{metric}=(\d\.\d{{4}})" for metric in METRICS)
EXPORT_SUFFIX = re.compile(r"_(?:jpg|jpeg|bmp|png)\.rf\.[0-9a-f]+$", re.IGNORECASE)
VARIANT_SUFFIX = re.compile(r"_-\d+-$")


def stream_text(output: dict) -> str:
    return "".join(output["text"])


def extract_run(path: Path, ratio: int, experiment: str, method: str) -> tuple[dict, list[dict]]:
    raw = path.read_bytes()
    cells = json.loads(raw)["cells"]
    config = "".join(cells[4]["source"])
    seed = int(re.search(r"\bseed\s*=\s*(\d+)", config).group(1))
    configured_ratio = float(re.search(r"\blabeled_ratio\s*=\s*(\d\.\d+)", config).group(1))
    if configured_ratio != ratio / 100:
        raise ValueError("labeled ratio differs from the notebook configuration")

    eval_cell = 8 if experiment == "supervised" else 9
    final = stream_text(cells[eval_cell]["outputs"][0])
    epoch = int(re.search(r"^Loaded checkpoint: epoch=(\d+)\b", final, re.MULTILINE).group(1))
    eval_suffix = "_student" if experiment == "vanilla" else ""
    row = {
        "labeled_percent": ratio,
        "experiment": experiment,
        "method": method,
        "seed": seed,
        "best_epoch": epoch,
    }
    for split in ("val", "test"):
        match = re.search(fr"^\[final_{split}{eval_suffix}\] {AP_PATTERN}\s+\|", final, re.MULTILINE)
        row.update({f"{split}_{metric}": value for metric, value in zip(METRICS, match.groups())})

    training_cell = cells[eval_cell - 1]
    history = []
    val_ap = {}
    for output in training_cell["outputs"]:
        if output["output_type"] != "stream":
            continue
        columns = []
        training_log = stream_text(output)
        for ep, ap in re.findall(fr"^\[val{eval_suffix}_ep(\d+)\] AP=(\d\.\d{{4}}) ", training_log, re.MULTILINE):
            val_ap[int(ep)] = ap
        for line in training_log.splitlines():
            values = line.split()
            if values[:1] == ["Ep"] and "loss" in values:
                columns = values
            if not columns or not values or not values[0].isdigit():
                continue
            if values[-1] == "*":
                values.pop()
            if len(values) != len(columns):
                raise ValueError("unexpected training log row width")
            record = dict(zip(columns, values))
            history.append({
                "labeled_percent": ratio,
                "experiment": experiment,
                "epoch": int(record["Ep"]),
                "stage": record.get("stage", ""),
                "loss": float(record["loss"]),
                "supervised_loss": float(record["sup"]) if "sup" in record else "",
                "unsupervised_loss": float(record["unsup"]) if "unsup" in record else "",
                "consistency_loss": float(record["cqc"]) if "cqc" in record else "",
                "val_AP": val_ap.get(int(record["Ep"]), ""),
            })
    if [item["epoch"] for item in history] != list(range(1, 151)):
        raise ValueError("expected one saved training row for each of 150 epochs")
    # Validate the numeric fields before exporting anything from the training stream.
    for item in history:
        if item["stage"] and not re.fullmatch(r"[a-zA-Z0-9_+]+", item["stage"]):
            raise ValueError("unexpected stage label")
    if sorted(val_ap) != list(range(5, 151, 5)):
        raise ValueError("expected validation AP every five epochs")
    row.update({
        "selected_stage": history[epoch - 1]["stage"],
        "source_notebook": f"notebooks/semidetr_v4_{ratio}pct/{path.name}",
        "evaluation_cell_0based": eval_cell,
        "evaluation_output_0based": 0,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    })
    return row, history


def audit_split_overlap(data_root: Path) -> dict:
    splits = ("train", "valid", "test")
    files = {split: list((data_root / split / "images").glob("*.jpg")) for split in splits}
    export_ids = {split: {EXPORT_SUFFIX.sub("", file.stem) for file in files[split]} for split in splits}
    family_ids = {split: {VARIANT_SUFFIX.sub("", name) for name in export_ids[split]} for split in splits}
    hashes = {split: {hashlib.sha256(file.read_bytes()).digest() for file in files[split]} for split in splits}
    return {
        "dataset": "historical_five_class_yolo_export",
        "image_counts": {split: len(files[split]) for split in splits},
        "family_rule": "Strip the Roboflow export suffix and then a terminal _-<digits>- variant suffix.",
        "cross_split": {
            f"{left}_{right}": {
                "byte_identical_groups": len(hashes[left] & hashes[right]),
                "shared_export_ids": len(export_ids[left] & export_ids[right]),
                "candidate_family_ids": len(family_ids[left] & family_ids[right]),
            }
            for left, right in combinations(splits, 2)
        },
    }


def csv_text(rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coursework_root", type=Path)
    parser.add_argument("--export-history", action="store_true", help="Write numeric epoch data to ignored .local/")
    args = parser.parse_args()
    results, history = [], []
    for ratio in (5, 10):
        for experiment, method, filename in EXPERIMENTS:
            source = args.coursework_root / "notebooks" / f"semidetr_v4_{ratio}pct" / filename
            try:
                row, run_history = extract_run(source, ratio, experiment, method)
            except (OSError, ValueError, KeyError, IndexError, AttributeError) as exc:
                parser.exit(1, f"Could not verify {ratio}% {experiment} ({type(exc).__name__}).\n")
            results.append(row)
            history.extend(run_history)

    with RESULTS.open(encoding="utf-8", newline="") as handle:
        committed = list(csv.DictReader(handle))
    if list(csv.DictReader(io.StringIO(csv_text(results)))) != committed:
        parser.exit(1, "Historical results differ from the repository CSV.\n")
    split_audit = audit_split_overlap(args.coursework_root / "data" / "YOLO_format")
    if split_audit != json.loads(SPLIT_AUDIT.read_text(encoding="utf-8")):
        parser.exit(1, "Historical dataset overlap differs from the repository audit.\n")

    if args.export_history:
        output = ROOT / ".local" / "semidetr_epoch_metrics.csv"
        output.parent.mkdir(exist_ok=True)
        output.write_text(csv_text(history), encoding="utf-8", newline="\n")
        print("Numeric histories saved to .local/semidetr_epoch_metrics.csv.")
    print(f"Coursework evidence verified for {len(results)} runs and three dataset splits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
