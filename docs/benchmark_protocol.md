# Benchmark and Timing Protocol

## Scope

The public benchmark compares detection accuracy on the same cleaned 6-class dataset and reports results on the merged test set and on each source-specific test subset. The committed CSV files remain the source of record for the historical results.

The existing `inference_ms_per_image` and `FPS` columns are retained for provenance. They were collected by the original training/evaluation notebooks, but they were **not** produced by one controlled cross-framework latency harness. They must be read as recorded evaluation-time measurements, not as deployment latency claims.

## Accuracy Evaluation

For each model, the best available checkpoint was evaluated on:

```text
Merged test
DeepPCB test only
DsPCBSD test only
HRIPCB test only
```

The main ranking metric is `mAP50-95`. Precision, recall, F1, and `mAP50` provide supporting context. Source-wise evaluation uses the same model checkpoint and filters the merged test set by the source prefix added during dataset preparation.

The evaluation backends are not identical: the Ultralytics models use the Ultralytics validator, while Faster R-CNN uses TorchMetrics/pycocotools plus a fixed-threshold precision/recall calculation. Both report COCO-style AP, but strict cross-framework ranking would be stronger with one shared evaluator over exported predictions. In particular, precision, recall, and F1 should not be treated as threshold-equivalent across the two backends.

A future accuracy refresh should export predictions from every model to one documented COCO JSON schema and evaluate them with the same pinned pycocotools version and settings.

## Historical Timing Context

All five recorded runs used a Kaggle environment with two Tesla T4 GPUs visible. The model-specific settings and timing scopes were not identical:

| Model | Images | Input | Eval batch | Eval device | Timing source | Recorded ms/image | Recorded FPS |
|---|---:|---:|---:|---|---|---:|---:|
| YOLOv11s | 1,887 | 640 | 32 | CUDA 0,1 | Ultralytics `metrics.speed["inference"]` | 11.0143 | 90.7908 |
| RT-DETR-L | 1,887 | 640 | 32 | CUDA 0,1 | Ultralytics `metrics.speed["inference"]` | 42.1156 | 23.7442 |
| Faster R-CNN | 1,887 | 640 | 8 | CUDA 0 | Manual evaluation-loop wall time | 89.0116 | 11.2345 |
| YOLOv11s-CBAMLite-BiFPNLite-P2 | 1,887 | 640 | 16 | CUDA 0 | Ultralytics `metrics.speed["inference"]` | 16.3764 | 61.0634 |
| PCBNet-RTDETR-HybridOpt | 1,887 | 768 | 16 | CUDA 0,1 | Ultralytics `metrics.speed["inference"]` | 47.8226 | 20.9106 |

These are the merged-test rows; the dataset-preparation notebook records 1,887 merged test images. The machine-readable version is stored in `reports/tables/historical_timing_context.csv`.

For the Ultralytics models, the CSV stores the framework-reported inference component. For Faster R-CNN, the manual timer includes device transfer, model forward, conversion of predictions/targets to CPU, and `torchmetrics.update`; it excludes the later `metric.compute` call. Therefore, the Faster R-CNN value is not scope-equivalent to the Ultralytics inference field.

The original notebooks do not record a controlled warm-up count, repeated timing trials, percentile latency, explicit synchronization policy, or an evaluation precision mode shared by all five models. These fields are marked `not_recorded` rather than inferred after the fact.

## Valid and Invalid Uses

Valid uses of the recorded timing values:

- documenting what the original notebook runs reported;
- identifying candidates for a future controlled benchmark;
- discussing an approximate accuracy/compute trade-off with explicit qualification.

Invalid uses:

- claiming a hardware-independent deployment FPS;
- treating the five FPS values as a controlled cross-framework comparison;
- attributing the difference between two models only to architecture when batch size, device count, input size, or timed scope also changed.

## Standardized Latency Protocol for Future Runs

New latency results should be published separately from the historical columns and should satisfy all conditions below.

1. Use the same physical GPU model and exactly one GPU per model.
2. Record GPU, CUDA, driver, Python, PyTorch, Torchvision, Ultralytics, and model-export/runtime versions.
3. Use the same fixed image list and publish its manifest or hashes.
4. Report batch size 1 at 640×640 for the cross-model comparison. Any native-resolution run must be a separate table.
5. Report FP32 and FP16 as separate experiments; never mix precision modes in one ranking.
6. Run at least 50 warm-up iterations before timing.
7. Time at least 200 images and repeat the run at least 5 times.
8. Use `torch.inference_mode()` and explicit CUDA synchronization immediately before and after each timed region.
9. Use a monotonic high-resolution timer such as `time.perf_counter()`.
10. Report median and p90 latency in milliseconds, plus throughput derived from the same samples.
11. Report two scopes separately:
    - model-only: tensor input through raw model output;
    - end-to-end: decode/preprocess, model inference, and postprocess/NMS, excluding file discovery and unrelated metric computation.
12. Save the complete run configuration and raw timing samples with the generated summary.

## Current Status

The standardized latency protocol above has not yet been executed. Until it is, the README uses the label `Recorded FPS` and qualifies YOLOv11s as having the highest recorded throughput in the original notebook logs, not as a proven deployment-speed winner.
