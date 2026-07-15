"""Validate committed PCB benchmark configuration and result provenance offline."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


MANIFEST_PATH = Path("configs/experiments/public_benchmark.json")
REQUIRED_SOURCES = {"Merged", "DeepPCB", "DsPCBSD", "HRIPCB"}
REQUIRED_DEPENDENCIES = {
    "albumentations",
    "ipython",
    "jupyterlab",
    "timm",
    "torch",
    "torchvision",
    "transformers",
    "ultralytics",
}
REQUIRED_CORE_PINS = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "transformers": "4.40.2",
    "ultralytics": "8.4.30",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _dependency_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"[A-Za-z0-9_.-]+", line)
        if match:
            names.add(match.group(0).lower())
    return names


def _exact_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def _repo_file(root: Path, relative_value: str) -> Path | None:
    relative = Path(relative_value)
    if relative.is_absolute():
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def validate_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("manifest schema_version must be 1")
    if manifest.get("status") != "historical_experiment_manifest":
        errors.append("manifest status must disclose that it is historical")

    dataset = manifest.get("dataset", {})
    classes = dataset.get("classes", [])
    if dataset.get("num_classes") != 6 or len(classes) != 6 or len(set(classes)) != 6:
        errors.append("dataset must define six unique classes")
    if dataset.get("test_images") != 1887:
        errors.append("dataset test_images must match the committed 1,887-image test set")
    if set(dataset.get("test_sources", [])) != REQUIRED_SOURCES:
        errors.append("dataset test_sources do not match the source-wise benchmark")

    experiments = manifest.get("experiments", [])
    if len(experiments) != 5:
        errors.append("manifest must contain five public benchmark experiments")
    ids = [item.get("id") for item in experiments]
    display_names = [item.get("display_name") for item in experiments]
    if len(ids) != len(set(ids)):
        errors.append("experiment ids must be unique")
    if len(display_names) != len(set(display_names)):
        errors.append("experiment display names must be unique")

    required_fields = {
        "id",
        "display_name",
        "model_name",
        "family",
        "notebook",
        "results_file",
        "image_size",
        "epochs",
        "train_batch",
        "eval_batch",
        "train_device",
        "eval_device",
        "seed",
        "evaluator",
    }
    for item in experiments:
        label = item.get("id", "<missing-id>")
        missing = sorted(required_fields - set(item))
        if missing:
            errors.append(f"{label}: missing fields: {', '.join(missing)}")
            continue
        for key in ("notebook", "results_file"):
            if _repo_file(root, item[key]) is None:
                errors.append(f"{label}: invalid or missing {key}: {item[key]}")
        for key in ("image_size", "epochs", "train_batch", "eval_batch"):
            if not isinstance(item[key], int) or item[key] <= 0:
                errors.append(f"{label}: {key} must be a positive integer")
        if item["seed"] != 42:
            errors.append(f"{label}: committed historical seed must be 42")

        results_path = _repo_file(root, item["results_file"])
        if results_path is not None:
            rows = _read_csv(results_path)
            if {row.get("test_source") for row in rows} != REQUIRED_SOURCES:
                errors.append(f"{label}: result sources do not match {sorted(REQUIRED_SOURCES)}")
            if {row.get("experiment") for row in rows} != {item["id"]}:
                errors.append(f"{label}: result experiment id mismatch")
            if {row.get("model_name") for row in rows} != {item["model_name"]}:
                errors.append(f"{label}: result model_name mismatch")

    evaluation = manifest.get("evaluation", {})
    comparison_path = root / evaluation.get("comparison_table", "")
    timing_path = root / evaluation.get("timing_context", "")
    if not comparison_path.is_file():
        errors.append("comparison table is missing")
    if not timing_path.is_file():
        errors.append("historical timing context is missing")
    if comparison_path.is_file() and timing_path.is_file():
        comparison = {row["model_display_name"]: row for row in _read_csv(comparison_path)}
        timing = {row["model_display_name"]: row for row in _read_csv(timing_path)}
        expected = set(display_names)
        if set(comparison) != expected or set(timing) != expected:
            errors.append("manifest/comparison/timing model sets differ")
        for name in set(comparison) & set(timing):
            try:
                result_ms = float(comparison[name]["inference_ms_per_image"])
                result_fps = float(comparison[name]["FPS"])
                timing_ms = float(timing[name]["recorded_inference_ms_per_image"])
                timing_fps = float(timing[name]["recorded_fps"])
            except (KeyError, ValueError) as exc:
                errors.append(f"{name}: invalid timing value: {exc}")
                continue
            if abs(result_ms - timing_ms) > 1e-12 or abs(result_fps - timing_fps) > 1e-12:
                errors.append(f"{name}: timing context does not match comparison table")
            if abs(timing_fps - 1000.0 / timing_ms) > 1e-10:
                errors.append(f"{name}: recorded FPS formula is inconsistent")

    return errors


def validate_repository(root: Path) -> list[str]:
    errors: list[str] = []
    manifest_file = root / MANIFEST_PATH
    if not manifest_file.is_file():
        return [f"missing manifest: {MANIFEST_PATH}"]
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid manifest JSON: {exc}"]
    errors.extend(validate_manifest(root, manifest))

    requirements = root / "requirements.txt"
    if not requirements.is_file():
        errors.append("requirements.txt is missing")
    else:
        missing = sorted(REQUIRED_DEPENDENCIES - _dependency_names(requirements))
        if missing:
            errors.append(f"requirements.txt is missing: {', '.join(missing)}")
        pins = _exact_pins(requirements)
        for package, expected in REQUIRED_CORE_PINS.items():
            if pins.get(package) != expected:
                errors.append(f"requirements.txt must pin {package}=={expected}")

    required_docs = (
        "docs/benchmark_protocol.md",
        "docs/docs_data_sources.md",
        "docs/reproducibility.md",
    )
    for relative in required_docs:
        if not (root / relative).is_file():
            errors.append(f"missing documentation: {relative}")
    data_readme = root / "data/README.md"
    if data_readme.is_file() and "docs/data_sources.md" in data_readme.read_text(encoding="utf-8"):
        errors.append("data/README.md still references the old data-source guide path")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate experiment manifest, result provenance, dependencies, and docs."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    errors = validate_repository(root)
    if errors:
        print("Reproducibility validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Reproducibility validation passed: 5 experiments, 4 sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
