# Reproducibility

Run commands from the repository root. Use Python 3.12 for the closest match to the historical benchmark. The offline checks also run in [GitHub Actions](../.github/workflows/repository-checks.yml).

## Inspect results on CPU

```sh
python -m pip install Pillow
python -m unittest discover -s tests -v
python scripts/validate_reproducibility.py
python scripts/check_semidetr_public_safety.py
```

The validators compare the benchmark manifest to the model CSVs and the DETR tables to the historical results CSV. The safety check rejects saved outputs in the training notebooks and allows only the reviewed original figures in the six-class comparison notebook. The tests include synthetic dataset integrity checks.

Install the notebook tools separately from the training packages.

```sh
python -m pip install jupyterlab pandas matplotlib
python -m jupyterlab
```

- [Six-class comparison](../notebooks/public_benchmark/07_compare_results.ipynb) shows the saved merged and source-wise comparison figures. Its executable cells read the manifest's five pairs of result files and write regenerated artifacts under `.local/public_benchmark/`.
- [Historical DETR results](../notebooks/semi_supervised_research/06_historical_results_and_analysis.ipynb) contains a verified test AP table and interpretation for GitHub viewing without a kernel, GPU access, or coursework files.

## Audit and rerun the six-class benchmark

Get the three historical YOLO exports or an authorized prepared copy. Run the audit and save the report under `.local/`. See the [dataset provenance and class mapping](public_benchmark.md) and [local data layout](../data/README.md).

```sh
python scripts/audit_dataset_integrity.py data/processed/DataPCB_Final_Clean_6cls --output .local/public_dataset_audit.json
```

To rebuild locally, run [01 · Prepare the dataset](../notebooks/public_benchmark/01_prepare_dataset.ipynb) with JupyterLab, pandas, matplotlib, Pillow, and PyYAML installed. From within the repository it reads `data/raw/datapcb-project/` and writes to `.local/datapcb_final_clean_6cls/` by default. Set `PCB_BENCHMARK_RAW_ROOT` and `PCB_BENCHMARK_PREP_ROOT` for a different input or output location. Use a dedicated preparation directory because the notebook rebuilds its generated subdirectories on each run.

For a GPU rerun, use a Kaggle notebook with the required data attached and a writable `/kaggle/working`. Check the path cell in [01 · Prepare the dataset](../notebooks/public_benchmark/01_prepare_dataset.ipynb) and in the chosen [training notebook](public_benchmark.md#detector-configurations). Point the preparation cell to the directory containing the three exports. Attach the prepared dataset for training and add its mount to the notebook's candidate paths. Check that the model's device and batch settings match the hardware before running. Use a fresh working directory so cached datasets and results from another run cannot be mistaken for the intended snapshot.

The historical runs used Python 3.12.12, PyTorch 2.10.0+cu128, Torchvision 0.25.0+cu128 for Faster R-CNN, and Ultralytics 8.4.30. Two Tesla T4s were visible. Per-model settings and timing scopes are in the [manifest](../configs/experiments/public_benchmark.json) and [protocol](benchmark_protocol.md). The manifest records settings for inspection. The training notebooks use their own config cells.

Preserve any new run's configuration, dependency versions, evaluation backend, checkpoint selection, and data audit separately. The comparison notebook defaults to the committed historical inputs. To compare a new run, work on a separate copy and set the manifest entries to the new CSVs.

## Set up the five-class DETR dataset

Open the DetectionProjects project in Roboflow Universe, select dataset version 1, and download a YOLOv8 bounding-box export if available to your account. Extract it locally. Check that `data.yaml` defines class IDs `0`–`4` and that `train/`, `valid/`, and `test/` each contain `images/` and `labels/`. The historical split sizes were 2,518, 840, and 838. The six-class benchmark uses separate source exports and configuration. [Roboflow Universe listing](https://universe.roboflow.com/detectionprojects-3nwgx/electronic_component_failure) · [Dataset provenance and attribution](semi_supervised_research.md#access-attribution-and-verification)

In an authorized GPU environment, install the appropriate PyTorch/Torchvision build and the packages in [requirements.txt](../requirements.txt). Set `PCB_SEMIDETR_DATA_ROOT` to the extracted dataset root and `PCB_SEMIDETR_OUTPUT_ROOT` to a private writable directory before starting Jupyter. Run the selected sanitized 10% recipe from the [implementation list](semi_supervised_research.md#read-the-implementation).

Exact reproduction of the saved scores remains unverified. Historical checkpoints, exact split manifests, and the complete training environment are outside the release. The historical setup cell requests Transformers 4.40.2 if absent. The installed version and complete dependency lock were not captured.

## Verify the historical DETR runs

Set `PCB_COURSEWORK_ROOT` to the local coursework folder containing `notebooks/` and `data/YOLO_format/`. The verification script reads the originals and compares ten results and the split audit to the repository. The optional flag exports selected numeric histories.

```powershell
python scripts/audit_semidetr_evidence.py "$env:PCB_COURSEWORK_ROOT" --export-history
python scripts/plot_semidetr_evidence.py --local-history
```

The export writes 150 loss rows and 30 validation measurements per run to the ignored `.local/semidetr_epoch_metrics.csv`. The local chart in `.local/historical_validation_ap.png` plots the saved observations without smoothing. It shows late plateaus at both label budgets and marks the target-duplication window for the three SHM recipes. The trajectories reflect each recipe's full schedule and objective. Total losses combine different terms across recipes and are kept as numeric diagnostics. The epoch-level files require coursework disclosure clearance before release.

Use the original artifacts to verify historical runs. The sanitized implementations preserve the training logic, use configurable data and output paths, and have had their credential assignment removed. See the [DETR research guide](semi_supervised_research.md) for selection and split limitations.
