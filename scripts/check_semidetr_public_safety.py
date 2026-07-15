"""Fail when public Semi-DETR notebooks contain private or embedded artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


OWNER_SPECIFIC_KAGGLE_PATH = "/kaggle/input/datasets/"
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*=\s*"
    r"[\"'][^\"'${}<>]+[\"']"
)


def audit_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: invalid notebook JSON: {exc}"]

    kaggle_metadata = notebook.get("metadata", {}).get("kaggle", {})
    if kaggle_metadata.get("dataSources"):
        errors.append(f"{path}: Kaggle dataSources metadata must be removed")

    all_source: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else str(source)
        all_source.append(source_text)

        if cell.get("outputs"):
            errors.append(f"{path}: cell {index} contains outputs")
        if cell.get("attachments"):
            errors.append(f"{path}: cell {index} contains attachments")
        if cell.get("metadata", {}).get("execution"):
            errors.append(f"{path}: cell {index} contains execution metadata")

    source_text = "\n".join(all_source)
    if OWNER_SPECIFIC_KAGGLE_PATH in source_text:
        errors.append(f"{path}: owner-specific Kaggle dataset path found")
    if SECRET_ASSIGNMENT.search(source_text):
        errors.append(f"{path}: credential-like assignment found")
    if "PCB_SEMIDETR_DATA_ROOT" not in source_text:
        errors.append(f"{path}: PCB_SEMIDETR_DATA_ROOT configuration is missing")

    raw_text = path.read_text(encoding="utf-8")
    for media_type in ('"image/png"', '"image/jpeg"'):
        if media_type in raw_text:
            errors.append(f"{path}: embedded media type found: {media_type}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit sanitized Semi-DETR notebooks before committing them."
    )
    parser.add_argument(
        "--notebook-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "notebooks" / "semidetr",
    )
    args = parser.parse_args()

    notebooks = sorted(args.notebook_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found under {args.notebook_dir}")
        return 1

    errors = [error for path in notebooks for error in audit_notebook(path)]
    if errors:
        print("Semi-DETR public-safety audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Semi-DETR public-safety audit passed: {len(notebooks)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
