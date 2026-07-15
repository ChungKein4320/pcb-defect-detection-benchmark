# Reproducibility Guide

## What This Repository Can Reproduce

The public 6-class benchmark includes the data-preparation logic, training/evaluation notebooks, source-wise result tables, figures, and an auditable experiment manifest.

There are two different reproducibility levels:

- **Offline artifact validation:** available without a GPU or dataset. It checks experiment configuration, committed result provenance, timing arithmetic, required dependencies, and documentation links.
- **Local dataset integrity audit:** requires the processed public dataset and checks image/label validity plus exact cross-split identity indicators.
- **Training/evaluation rerun:** requires the public-source PCB datasets, the processed dataset layout, Kaggle or equivalent GPU resources, and model downloads.

The separate confidential-data Semi-DETR study is code-auditable but not publicly data-reproducible. See `docs/semidetr_private_research_summary.md`.

## Historical Environment Evidence

The following core versions are preserved in executed notebook output or explicit notebook setup code:

| Component | Recorded version | Evidence status |
|---|---|---|
| Python | 3.12.12 | Ultralytics run output |
| PyTorch | 2.10.0+cu128 | notebook output |
| Torchvision | 0.25.0+cu128 | Faster R-CNN notebook output |
| Ultralytics | 8.4.30 | Ultralytics run output |
| Transformers | 4.40.2 | Semi-DETR setup code |
| GPU | 2× Tesla T4 visible | notebook output |

`requirements.txt` pins the four core Python package versions above where a normal PyPI version is meaningful. The CUDA build suffix depends on the PyTorch wheel index and host driver, so install the appropriate PyTorch build for the target machine. Packages whose historical versions were not recorded remain unpinned rather than being guessed.

This is a partial historical pin, not a full lockfile.

## Offline Validation

From the repository root:

```powershell
python scripts\validate_reproducibility.py
python scripts\check_semidetr_public_safety.py
python -m unittest discover -s tests -v
```

Expected output includes:

```text
Reproducibility validation passed: 5 experiments, 4 sources
Semi-DETR public-safety audit passed: 7 notebooks
```

These commands do not train a model, download data, or require a GPU.

GitHub Actions runs these offline checks on every push to `master` and every pull request using `.github/workflows/repository-checks.yml`. The local dataset integrity audit is intentionally excluded from CI because the image archive is not committed.

## Experiment Manifest

The machine-readable historical manifest is:

```text
configs/experiments/public_benchmark.json
```

It records the five experiment IDs, notebook paths, result files, model families, image sizes, epochs, batch sizes, devices, seeds, evaluator backends, dataset classes, and historical core environment evidence.

The manifest was extracted from the executed notebooks and committed result files. It is an audit and validation surface; the current notebooks do not automatically consume it as a training configuration. This limitation is stated explicitly to avoid overstating reproducibility.

## Dataset Setup

Use the source and preprocessing documentation in `docs/docs_data_sources.md`. For local layout, start from:

```text
configs/data/local_datapcb_final_clean_6cls.example.yaml
```

Expected processed structure:

```text
data/processed/DataPCB_Final_Clean_6cls/
├── train/images
├── train/labels
├── valid/images
├── valid/labels
├── test/images
├── test/labels
└── data.yaml
```

The committed preparation notebook reports 6,624 training images, 937 validation images, and 1,887 test images.

After preparing or extracting the dataset, run:

```powershell
python scripts\audit_dataset_integrity.py `
  data\processed\DataPCB_Final_Clean_6cls `
  --output reports\dataset_integrity_summary.json
```

See `docs/dataset_integrity.md` for the exact checks, audited aggregate result, and limitations.

## Recommended Execution Order

1. Run `notebooks/01_prepare_final_datapcb_clean_6cls_sourcewise.ipynb` or use the documented processed archive.
2. Run one or more model notebooks `02` through `06`.
3. Preserve each notebook's source-wise CSV outputs.
4. Run `notebooks/07_compare_sourcewise_benchmark_results.ipynb` to consolidate reports.
5. Run the offline validation commands above before committing regenerated artifacts.

## Known Boundaries

- Heavy training was originally performed on Kaggle and has not been rerun locally end-to-end.
- Model weights are intentionally excluded from Git.
- The public benchmark uses two evaluator implementations; see `docs/benchmark_protocol.md`.
- Historical FPS values are not a standardized latency benchmark.
- The experiment manifest validates recorded settings but does not yet drive the notebooks.
- The exact-identity dataset audit cannot rule out same-board, near-duplicate, or same-template overlap.
- Full bit-for-bit environment recreation is not possible because a complete historical package lock was not captured.
