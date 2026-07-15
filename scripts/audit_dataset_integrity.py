"""Audit a local YOLO dataset without publishing the dataset itself."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from PIL import Image


SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ROBOFLOW_SUFFIX = re.compile(r"_jpg\.rf\.[0-9a-f]+$", re.IGNORECASE)
AUDIT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_id(path: Path) -> str:
    """Recover the pre-export identifier used by the committed dataset names."""
    return ROBOFLOW_SUFFIX.sub("", path.stem)


def _source_name(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _is_absolute_on_common_platforms(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _duplicate_group_counts(
    groups: Iterable[list[tuple[str, str]]],
) -> tuple[int, int]:
    cross_split = 0
    within_split = 0
    for entries in groups:
        if len(entries) < 2:
            continue
        if len({split for split, _ in entries}) > 1:
            cross_split += 1
        else:
            within_split += 1
    return cross_split, within_split


def audit_dataset(dataset_root: Path, class_count: int = 6) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    split_reports: dict[str, Any] = {}
    byte_hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    source_ids: dict[str, list[tuple[str, str]]] = defaultdict(list)
    stems: dict[str, list[tuple[str, str]]] = defaultdict(list)
    fingerprint_records: list[tuple[str, str, str]] = []

    for split in SPLITS:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"{split}: expected images/ and labels/ directories")
            continue

        images = _image_files(image_dir)
        labels = sorted(label_dir.glob("*.txt"))
        image_by_stem = {path.stem: path for path in images}
        label_by_stem = {path.stem: path for path in labels}
        missing_labels = sorted(set(image_by_stem) - set(label_by_stem))
        missing_images = sorted(set(label_by_stem) - set(image_by_stem))
        if missing_labels:
            errors.append(f"{split}: {len(missing_labels)} images have no matching label")
        if missing_images:
            errors.append(f"{split}: {len(missing_images)} labels have no matching image")
        if len(image_by_stem) != len(images):
            errors.append(f"{split}: multiple image files share a filename stem")

        class_boxes: Counter[int] = Counter()
        source_images: Counter[str] = Counter()
        corrupt_images = 0
        invalid_label_lines = 0

        for path in images:
            entry = (split, path.name)
            digest = _sha256(path)
            byte_hashes[digest].append(entry)
            fingerprint_records.append((split, path.name, digest))
            source_ids[_source_id(path)].append(entry)
            stems[path.stem].append(entry)
            source_images[_source_name(path)] += 1
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                corrupt_images += 1

        if corrupt_images:
            errors.append(f"{split}: {corrupt_images} images failed Pillow verification")

        for path in labels:
            for line_number, raw_line in enumerate(
                path.read_text(encoding="utf-8-sig").splitlines(), start=1
            ):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                valid = True
                if len(parts) != 5:
                    valid = False
                else:
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                    except ValueError:
                        valid = False
                    else:
                        valid = (
                            0 <= class_id < class_count
                            and 0 <= x_center <= 1
                            and 0 <= y_center <= 1
                            and 0 < width <= 1
                            and 0 < height <= 1
                        )
                        if valid:
                            class_boxes[class_id] += 1
                if not valid:
                    invalid_label_lines += 1
                    if invalid_label_lines <= 5:
                        errors.append(
                            f"{split}: invalid YOLO row {path.name}:{line_number}"
                        )

        if invalid_label_lines > 5:
            errors.append(
                f"{split}: {invalid_label_lines - 5} additional invalid YOLO rows"
            )

        split_reports[split] = {
            "images": len(images),
            "labels": len(labels),
            "class_box_counts": {
                str(class_id): class_boxes.get(class_id, 0)
                for class_id in range(class_count)
            },
            "source_image_counts": dict(sorted(source_images.items())),
            "missing_labels": len(missing_labels),
            "missing_images": len(missing_images),
            "corrupt_images": corrupt_images,
            "invalid_label_rows": invalid_label_lines,
        }

    exact_duplicates, within_split_exact_duplicates = _duplicate_group_counts(
        byte_hashes.values()
    )
    repeated_source_ids, within_split_source_ids = _duplicate_group_counts(
        source_ids.values()
    )
    repeated_stems, within_split_stems = _duplicate_group_counts(stems.values())
    if exact_duplicates:
        errors.append(
            f"{exact_duplicates} byte-identical SHA-256 groups cross dataset splits"
        )
    if repeated_source_ids:
        errors.append(
            f"{repeated_source_ids} normalized source IDs cross dataset splits"
        )
    if repeated_stems:
        errors.append(f"{repeated_stems} filename stems cross dataset splits")
    if within_split_exact_duplicates:
        errors.append(
            f"{within_split_exact_duplicates} byte-identical groups occur within a split"
        )
    if within_split_source_ids:
        errors.append(
            f"{within_split_source_ids} normalized source IDs repeat within a split"
        )
    if within_split_stems:
        errors.append(f"{within_split_stems} filename stems repeat within a split")

    dataset_fingerprint = hashlib.sha256()
    for split, name, digest in sorted(fingerprint_records):
        dataset_fingerprint.update(f"{split}\0{name}\0{digest}\n".encode("utf-8"))

    data_yaml = dataset_root / "data.yaml"
    data_yaml_absolute_path = False
    if data_yaml.is_file():
        for raw_line in data_yaml.read_text(encoding="utf-8-sig").splitlines():
            if raw_line.strip().lower().startswith("path:"):
                value = raw_line.split(":", 1)[1].strip().strip("'\"")
                data_yaml_absolute_path = _is_absolute_on_common_platforms(value)
                break
    else:
        warnings.append("data.yaml was not found")
    if data_yaml_absolute_path:
        warnings.append(
            "data.yaml uses an environment-specific absolute path; use a local config"
        )

    return {
        "schema_version": AUDIT_VERSION,
        "status": "passed" if not errors else "failed",
        "scope": {
            "checks": [
                "image-label pairing",
                "Pillow image verification",
                "YOLO label syntax and normalized bounds",
                "cross-split filename stems",
                "cross-split normalized source IDs",
                "cross-split byte-identical SHA-256 images",
            ],
            "not_claimed": [
                "absence of visually similar images",
                "absence of same-PCB or same-template overlap",
                "absence of all possible data leakage",
            ],
        },
        "splits": split_reports,
        "cross_split": {
            "identical_filename_stem_groups": repeated_stems,
            "normalized_source_id_groups": repeated_source_ids,
            "byte_identical_sha256_groups": exact_duplicates,
        },
        "within_split": {
            "identical_filename_stem_groups": within_split_stems,
            "normalized_source_id_groups": within_split_source_ids,
            "byte_identical_sha256_groups": within_split_exact_duplicates,
        },
        "dataset_fingerprint_sha256": dataset_fingerprint.hexdigest(),
        "data_yaml_absolute_path": data_yaml_absolute_path,
        "warnings": warnings,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--class-count", type=int, default=6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit_dataset(args.dataset_root, args.class_count)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    totals = report["splits"]
    print(f"Dataset integrity audit: {report['status']}")
    for split in SPLITS:
        if split in totals:
            print(
                f"{split}: {totals[split]['images']} images, "
                f"{totals[split]['labels']} labels"
            )
    cross = report["cross_split"]
    print(
        "Cross-split groups: "
        f"SHA-256={cross['byte_identical_sha256_groups']}, "
        f"source-ID={cross['normalized_source_id_groups']}, "
        f"filename-stem={cross['identical_filename_stem_groups']}"
    )
    for warning in report["warnings"]:
        print(f"Warning: {warning}")
    for error in report["errors"]:
        print(f"Error: {error}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
