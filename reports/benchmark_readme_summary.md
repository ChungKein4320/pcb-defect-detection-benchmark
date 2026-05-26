## Benchmark Results

### Overall comparison on merged test set

| model_display_name             |   precision |   recall |     f1 |   mAP50 |   mAP50_95 |     FPS |
|:-------------------------------|------------:|---------:|-------:|--------:|-----------:|--------:|
| YOLOv11s                       |      0.8529 |   0.8053 | 0.8285 |  0.8788 |     0.5761 | 90.7908 |
| RT-DETR-L                      |      0.9215 |   0.897  | 0.909  |  0.9315 |     0.6595 | 23.7442 |
| Faster R-CNN                   |      0.7823 |   0.9004 | 0.8372 |  0.8924 |     0.5918 | 11.2345 |
| YOLOv11s-CBAMLite-BiFPNLite-P2 |      0.8428 |   0.8146 | 0.8284 |  0.8818 |     0.5764 | 61.0634 |
| PCBNet-RTDETR-HybridOpt        |      0.8945 |   0.8767 | 0.8855 |  0.9177 |     0.6112 | 20.9106 |

### Best model per source

| test_source   | model_display_name   |   mAP50_95 |   mAP50 |   precision |   recall |
|:--------------|:---------------------|-----------:|--------:|------------:|---------:|
| Merged        | RT-DETR-L            |     0.6595 |  0.9315 |      0.9215 |   0.897  |
| DeepPCB       | RT-DETR-L            |     0.8345 |  0.9862 |      0.9875 |   0.9705 |
| DsPCBSD       | RT-DETR-L            |     0.5399 |  0.8699 |      0.8594 |   0.8404 |
| HRIPCB        | RT-DETR-L            |     0.5198 |  0.9613 |      0.9742 |   0.9507 |

### Best model per class on merged test

| class_name      | model_display_name   |   mAP50_95 |
|:----------------|:---------------------|-----------:|
| missing_hole    | RT-DETR-L            |     0.8618 |
| mouse_bite      | RT-DETR-L            |     0.5861 |
| open_circuit    | RT-DETR-L            |     0.6556 |
| short           | RT-DETR-L            |     0.6376 |
| spur            | RT-DETR-L            |     0.4847 |
| spurious_copper | RT-DETR-L            |     0.7314 |
