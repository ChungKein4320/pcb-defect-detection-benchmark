# PCB Defect Detection Benchmark

A computer vision project for detecting PCB manufacturing defects using object detection models.

This project focuses on building a clean 6-class PCB defect dataset from multiple sources, training several object detection models, and analyzing per-class detection performance.

## Project Status

Current stage:

- [x] Dataset structure organized
- [x] Data preprocessing strategy finalized
- [x] Final 6-class dataset preparation notebook added
- [ ] YOLOv8n baseline training
- [ ] YOLOv8s training
- [ ] Faster R-CNN training
- [ ] Transformer-based detector training
- [ ] Model comparison
- [ ] Error analysis for weak classes

## Problem

PCB defect detection is a practical industrial computer vision task. The goal is to detect and localize common PCB defects from image data.

Final target classes:

- missing_hole
- mouse_bite
- open_circuit
- short
- spur
- spurious_copper

## Dataset Sources

This project uses three PCB defect datasets:

- DeepPCB
- DsPCBSD+
- HRIPCB

The datasets are not committed to this repository because of size and licensing constraints.

Expected Kaggle input structure:

    /kaggle/input/datasets/chungkein/datapcb-project
    ├── DeepPCB
    ├── DsPCBSD
    └── HRIPCB

Local backup structure:

    data/raw/datapcb-project/
    ├── DeepPCB/
    ├── DsPCBSD/
    └── HRIPCB/

## Data Preprocessing

The final dataset is standardized into a 6-class YOLO detection format.

### DeepPCB

The original labels are mapped into the final 6-class taxonomy.

Example mapping:

- pin-hole -> missing_hole
- mousebite -> mouse_bite
- open -> open_circuit
- short -> short
- spur -> spur
- copper -> spurious_copper

### HRIPCB

The original labels are mapped into the same 6-class taxonomy.

Example mapping:

- Missing_hole -> missing_hole
- Mouse_bite -> mouse_bite
- Open_circuit -> open_circuit
- Short -> short
- Spur -> spur
- Spurious_copper -> spurious_copper

### DsPCBSD+

DsPCBSD+ contains 9 defect classes. This project maps only the 6 classes compatible with the final taxonomy:

- HB -> missing_hole
- MB -> mouse_bite
- OP -> open_circuit
- SH -> short
- SP -> spur
- SC -> spurious_copper

The following DsPCBSD+ classes are removed as boxes:

- CS
- CFO
- BMFO

After removing ignored boxes:

- Images with at least one valid 6-class object are kept.
- Images that become empty are dropped.
- Spur-only images are still kept.
- Original train/valid/test splits are preserved.
- No hard balancing is applied.
- No downsampling is applied.
- No oversampling is applied.
- No offline crop augmentation is applied.

## Why No Hard Balancing?

Several balancing strategies were investigated before finalizing the preprocessing policy.

Hard balancing improved the visual class histogram but did not consistently improve detection performance. In particular, aggressive repeat-based balancing reduced mAP50-95 in confirmation experiments.

Therefore, the final preprocessing strategy prioritizes clean label standardization and preserves the natural data distribution. Class imbalance is handled through per-class evaluation and model comparison rather than physical resampling.

## Final Dataset

Final processed dataset name:

    DataPCB_Final_Clean_6cls

Expected Kaggle output path:

    /kaggle/working/datapcb_final_clean_6cls/DataPCB_Final_Clean_6cls

Expected local backup path:

    data/processed/DataPCB_Final_Clean_6cls/

YOLO data config on Kaggle:

    /kaggle/working/datapcb_final_clean_6cls/DataPCB_Final_Clean_6cls/data.yaml

## Repository Structure

    pcb-defect-detection-benchmark/
    ├── configs/
    │   └── data/
    │       ├── kaggle_datapcb_final_clean_6cls.yaml
    │       └── local_datapcb_final_clean_6cls.example.yaml
    │
    ├── notebooks/
    │   └── 01_prepare_final_datapcb_clean_6cls.ipynb
    │
    ├── reports/
    │   ├── tables/
    │   └── figures/
    │
    ├── assets/
    │   ├── samples/
    │   ├── predictions/
    │   └── README_images/
    │
    ├── data/
    │   └── README.md
    │
    ├── weights/
    │   └── README.md
    │
    ├── runs/
    │   └── README.md
    │
    ├── requirements.txt
    ├── .gitignore
    └── README.md

## Notebooks

Current notebook:

- notebooks/01_prepare_final_datapcb_clean_6cls.ipynb

Planned notebooks:

- notebooks/02_train_yolov8n_baseline.ipynb
- notebooks/03_train_yolov8s_baseline.ipynb
- notebooks/04_train_faster_rcnn.ipynb
- notebooks/05_train_rtdetr.ipynb
- notebooks/06_compare_results.ipynb
- notebooks/07_error_analysis.ipynb

## Model Benchmark Plan

The project will compare multiple detector families:

| Category | Model | Purpose |
|---|---|---|
| One-stage detector | YOLOv8n | Lightweight baseline |
| One-stage detector | YOLOv8s | Capacity comparison |
| Two-stage detector | Faster R-CNN ResNet50-FPN | Strong localization baseline |
| Transformer-based detector | RT-DETR or DETR | Context-aware detection comparison |

## Evaluation Metrics

The main metrics are:

- Precision
- Recall
- mAP50
- mAP50-95
- Per-class mAP50-95

Per-class results are especially important because PCB defect classes have different visual difficulty levels.

Key classes to monitor:

- spur
- mouse_bite
- short
- spurious_copper

## Expected Result Tables

Model comparison table:

| Model | Precision | Recall | mAP50 | mAP50-95 | Notes |
|---|---:|---:|---:|---:|---|
| YOLOv8n | TBD | TBD | TBD | TBD | Baseline |
| YOLOv8s | TBD | TBD | TBD | TBD | Larger YOLO baseline |
| Faster R-CNN | TBD | TBD | TBD | TBD | Two-stage detector |
| RT-DETR / DETR | TBD | TBD | TBD | TBD | Transformer-based detector |

Per-class mAP50-95 table:

| Model | missing_hole | mouse_bite | open_circuit | short | spur | spurious_copper |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8s | TBD | TBD | TBD | TBD | TBD | TBD |
| Faster R-CNN | TBD | TBD | TBD | TBD | TBD | TBD |
| RT-DETR / DETR | TBD | TBD | TBD | TBD | TBD | TBD |

## How to Reproduce

Training is performed on Kaggle.

### Step 1: Prepare final dataset

Run:

    notebooks/01_prepare_final_datapcb_clean_6cls.ipynb

This creates the final processed dataset:

    /kaggle/working/datapcb_final_clean_6cls/DataPCB_Final_Clean_6cls

### Step 2: Train baseline models

Run the training notebooks in order:

    notebooks/02_train_yolov8n_baseline.ipynb
    notebooks/03_train_yolov8s_baseline.ipynb
    notebooks/04_train_faster_rcnn.ipynb
    notebooks/05_train_rtdetr.ipynb

### Step 3: Compare models

Run:

    notebooks/06_compare_results.ipynb

### Step 4: Error analysis

Run:

    notebooks/07_error_analysis.ipynb

The focus will be on weak classes such as spur and mouse_bite.

## Git Policy

The following files are not committed:

- raw datasets
- processed datasets
- zip archives
- model weights
- training runs
- large experiment artifacts

Only selected notebooks, configuration files, figures, tables, and documentation are committed.

## Current Focus

The immediate next step is to train the YOLOv8n baseline on the final clean merged dataset and inspect per-class mAP.