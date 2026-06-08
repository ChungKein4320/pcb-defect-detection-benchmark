# Release Notes — v0.1 Source-wise PCB Defect Detection Benchmark

## Version

`v0.1-sourcewise-benchmark`

## Summary

This release finalizes the first source-wise PCB defect detection benchmark.

The project builds a cleaned 6-class PCB defect dataset from multiple public PCB defect sources and benchmarks several object detection model families under the same evaluation protocol.

The benchmark compares one-stage, two-stage, transformer-based, and custom small-object-oriented detection approaches.

## Main Goals

This release focuses on four goals:

1. Build a clean and consistent 6-class PCB defect dataset.
2. Standardize labels and dataset format across multiple PCB sources.
3. Benchmark multiple object detection model families under the same protocol.
4. Analyze whether weak performance comes from model choice, class difficulty, or dataset-source/domain differences.

## Dataset

Final processed dataset:

```text
DataPCB_Final_Clean_6cls
```

Raw dataset sources:

```text
DeepPCB
DsPCBSD
HRIPCB
```

Final class set:

| ID | Class           |
| -: | --------------- |
|  0 | missing_hole    |
|  1 | mouse_bite      |
|  2 | open_circuit    |
|  3 | short           |
|  4 | spur            |
|  5 | spurious_copper |

## Data Processing

The final dataset was created using the following strategy:

* Standardized class names across different PCB defect sources.
* Converted labels into a consistent YOLO detection format.
* Removed unsupported DsPCBSD classes that were not shared by the other sources.
* Removed invalid or empty labels after class filtering.
* Preserved the original train/validation/test split structure.
* Avoided hard class balancing after experiments showed that count balancing alone did not solve weak-class performance.

## Benchmark Models

The benchmark includes five main detection models:

| Model                          | Type                            | Role                             |
| ------------------------------ | ------------------------------- | -------------------------------- |
| YOLOv11s                       | One-stage detector              | Fast baseline                    |
| RT-DETR-L                      | Transformer-based detector      | Strong accuracy baseline         |
| Faster R-CNN ResNet50-FPN      | Two-stage detector              | Classical two-stage comparison   |
| YOLOv11s-CBAMLite-BiFPNLite-P2 | Custom YOLO variant             | Small-object-oriented experiment |
| PCBNet-RTDETR-HybridOpt        | RT-DETR optimization experiment | Training/optimization variant    |

## Evaluation Protocol

All models were trained on the merged training split and evaluated on:

```text
Merged test
DeepPCB test only
DsPCBSD test only
HRIPCB test only
```

This source-wise evaluation was used to check whether each model generalizes across dataset sources or performs well only on specific source domains.

## Overall Merged-test Results

| Model                          | Precision | Recall |     F1 |  mAP50 | mAP50-95 |     FPS |
| ------------------------------ | --------: | -----: | -----: | -----: | -------: | ------: |
| YOLOv11s                       |    0.8529 | 0.8053 | 0.8285 | 0.8788 |   0.5761 | 90.7908 |
| RT-DETR-L                      |    0.9215 | 0.8970 | 0.9090 | 0.9315 |   0.6595 | 23.7442 |
| Faster R-CNN                   |    0.7823 | 0.9004 | 0.8372 | 0.8924 |   0.5918 | 11.2345 |
| YOLOv11s-CBAMLite-BiFPNLite-P2 |    0.8428 | 0.8146 | 0.8284 | 0.8818 |   0.5764 | 61.0634 |
| PCBNet-RTDETR-HybridOpt        |    0.8945 | 0.8767 | 0.8855 | 0.9177 |   0.6112 | 20.9106 |

## Best Model per Source

| Test Source | Best Model | mAP50-95 |  mAP50 | Precision | Recall |
| ----------- | ---------- | -------: | -----: | --------: | -----: |
| Merged      | RT-DETR-L  |   0.6595 | 0.9315 |    0.9215 | 0.8970 |
| DeepPCB     | RT-DETR-L  |   0.8345 | 0.9862 |    0.9875 | 0.9705 |
| DsPCBSD     | RT-DETR-L  |   0.5399 | 0.8699 |    0.8594 | 0.8404 |
| HRIPCB      | RT-DETR-L  |   0.5198 | 0.9613 |    0.9742 | 0.9507 |

## Main Findings

### RT-DETR-L is the strongest overall model

RT-DETR-L achieved the best merged-test result:

```text
mAP50     = 0.9315
mAP50-95  = 0.6595
Precision = 0.9215
Recall    = 0.8970
FPS       = 23.7442
```

It also achieved the best source-wise performance on Merged, DeepPCB, DsPCBSD, and HRIPCB test subsets.

### YOLOv11s is the fastest practical baseline

YOLOv11s achieved:

```text
FPS       = 90.7908
mAP50     = 0.8788
mAP50-95  = 0.5761
```

It is much faster than RT-DETR-L, but less accurate. It remains useful as a speed-oriented baseline.

### The custom YOLOv11s-P2 variant did not meaningfully improve accuracy

YOLOv11s-CBAMLite-BiFPNLite-P2 achieved:

```text
mAP50-95 = 0.5764
FPS      = 61.0634
```

This is only a negligible improvement over stock YOLOv11s:

```text
YOLOv11s mAP50-95    = 0.5761
Custom YOLO mAP50-95 = 0.5764
```

The added model complexity is not justified in its current combined form.

### Faster R-CNN improves recall but remains slow

Faster R-CNN achieved strong recall:

```text
Recall    = 0.9004
mAP50     = 0.8924
mAP50-95  = 0.5918
FPS       = 11.2345
```

It is useful as a two-stage detector reference, but it is less practical for real-time use.

### PCBNet-RTDETR-HybridOpt did not beat stock RT-DETR-L

PCBNet-RTDETR-HybridOpt remained competitive but did not outperform RT-DETR-L:

```text
mAP50     = 0.9177
mAP50-95  = 0.6112
FPS       = 20.9106
```

This suggests that optimization-level changes alone were not enough to outperform the stock RT-DETR-L baseline.

### Spur remains the hardest class

On the merged test set, `spur` remained the weakest class:

```text
Best spur mAP50-95 = 0.4847
Best model         = RT-DETR-L
```

This likely comes from small defect size, visual ambiguity, and source-domain variation.

## Additional Semi-DETR Research Summary

This release also includes a public-safe Semi-DETR research summary:

```text
docs/semidetr_private_research_summary.md
```

This document summarizes semi-supervised DETR-style research for PCB defect detection without exposing private data or confidential files.

## Included Artifacts

Main notebooks:

```text
notebooks/01_prepare_final_datapcb_clean_6cls_sourcewise.ipynb
notebooks/02_train_yolov11s_datapcb_clean_6cls_sourcewise.ipynb
notebooks/03_train_rtdetr_l_datapcb_clean_6cls_sourcewise.ipynb
notebooks/04_train_faster_rcnn_datapcb_clean_6cls_sourcewise.ipynb
notebooks/05_train_yolov11s_cbamlite_bifpnlite_p2_datapcb_clean_6cls_sourcewise.ipynb
notebooks/06_train_pcbnet_rtdetr_hybridopt_datapcb_clean_6cls_sourcewise.ipynb
notebooks/07_compare_sourcewise_benchmark_results.ipynb
```

Main reports:

```text
reports/tables/
reports/figures/
reports/benchmark_readme_summary.md
```

Documentation:

```text
docs/docs_data_sources.md
docs/experiment_log.md
docs/kaggle_links.md
docs/semidetr_private_research_summary.md
```

## Limitations

* The benchmark is limited to the cleaned 6-class PCB defect dataset.
* Performance may not generalize to unseen PCB manufacturing domains.
* Dataset source differences still affect model performance.
* Weak classes such as `spur` and `mouse_bite` remain difficult.
* Custom architecture changes were tested only in limited forms.
* Heavy model training was performed on Kaggle.
* Large datasets, model weights, and training runs are not committed to GitHub.

## Next Steps

1. Validate the benchmark on more PCB datasets.
2. Add stronger augmentation ablation studies.
3. Test small-object-specific detectors more systematically.
4. Add semi-supervised learning experiments with unlabeled PCB images.
5. Add model export and inference demo.
6. Add a lightweight Streamlit or Gradio inference visualization demo.
7. Compare deployment trade-offs between YOLOv11s and RT-DETR-L.

## CV-ready Summary

Recommended CV title:

```text
PCB Defect Detection Benchmark with YOLO, RT-DETR, and Faster R-CNN
```

Recommended CV bullets:

```text
- Built a source-wise PCB defect detection benchmark by cleaning and standardizing a merged 6-class dataset from DeepPCB, DsPCBSD, and HRIPCB into YOLO detection format.
- Benchmarked YOLOv11s, RT-DETR-L, Faster R-CNN ResNet50-FPN, and custom small-object-oriented detection variants across merged and source-specific test sets.
- Found RT-DETR-L achieved the best merged-test performance with 0.6595 mAP50-95 and 0.9315 mAP50, while YOLOv11s provided the fastest baseline at 90.79 FPS.
```
