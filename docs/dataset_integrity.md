# Dataset Integrity Audit

## Purpose

The public benchmark preserves the source datasets' documented train, validation, and test assignments while standardizing labels into one 6-class YOLO format. "Cleaned" in this repository means taxonomy and annotation-format cleaning; it does not by itself prove that every possible form of data leakage is absent.

The repository therefore includes a local audit that checks:

- one-to-one image and label pairing;
- image decodability with Pillow;
- YOLO row syntax, class IDs, and normalized bounding-box values;
- repeated filename stems across splits;
- repeated pre-export source IDs after removing the Roboflow export suffix;
- byte-identical images across splits using SHA-256.

The audit publishes aggregate counts only. It does not publish the dataset, image hashes, or image-level records.

## Run the Audit

From the repository root:

```powershell
python scripts\audit_dataset_integrity.py `
  data\processed\DataPCB_Final_Clean_6cls `
  --output reports\dataset_integrity_summary.json
```

The command exits with a non-zero status when it finds invalid files, invalid labels, missing pairs, repeated source identifiers, or byte-identical images crossing dataset splits.

## Audited Local Snapshot

The committed `reports/dataset_integrity_summary.json` was generated from the local processed snapshot used by this project. It records:

- 6,624 train images and labels;
- 937 validation images and labels;
- 1,887 test images and labels;
- zero invalid image/label records;
- zero cross-split filename-stem groups;
- zero cross-split normalized source-ID groups;
- zero cross-split byte-identical SHA-256 groups.
- zero within-split repeated source-ID or byte-identical groups.

The report also includes one aggregate SHA-256 fingerprint derived from the sorted image names and hashes. It identifies the audited snapshot without exposing per-image hashes. These results support the narrow statement that no exact file copies or repeated exported source IDs were found within or across the three splits in that snapshot.

## Boundaries

This audit does **not** establish that all leakage is absent. In particular, it does not identify:

- differently cropped, resized, or re-encoded views of the same board;
- different captures of the same physical PCB or manufacturing template;
- semantic overlap that cannot be recovered from filenames;
- errors inherited from the original datasets' split policies.

A trial 64-bit difference-hash scan was intentionally not used as pass/fail evidence: highly repetitive PCB imagery produced many low-distance candidates and therefore requires a domain-aware manual or feature-based review. Future work should add board/template identifiers where source metadata permits group-aware splitting.
