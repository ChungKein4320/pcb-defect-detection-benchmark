# Local benchmark data

Use the [six-class benchmark provenance guide](../docs/public_benchmark.md#dataset-and-provenance) to identify the source exports. The repository expects the following layout for local inspection and auditing.

```text
data/
├── raw/datapcb-project/
│   ├── DeepPCB/
│   ├── DsPCBSD/
│   └── HRIPCB/
└── processed/DataPCB_Final_Clean_6cls/
    ├── train/{images,labels}/
    ├── valid/{images,labels}/
    ├── test/{images,labels}/
    └── data.yaml
```

Each raw source also follows the YOLO split layout. Use the [preparation notebook](../notebooks/public_benchmark/01_prepare_dataset.ipynb) with the three raw exports or audit an authorized copy of the prepared dataset. Historical Kaggle paths and taxonomy are recorded in the [Kaggle config reference](../configs/data/kaggle_datapcb_final_clean_6cls.yaml). The [local config reference](../configs/data/local_datapcb_final_clean_6cls.example.yaml) documents the expected local location. See the [rerun guide](../docs/reproducibility.md#audit-and-rerun-the-six-class-benchmark) for commands.

Keep the coursework dataset outside this tree. Its access requirements and separate protocol are described in the [DETR research guide](../docs/semi_supervised_research.md).
