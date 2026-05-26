# Data Sources

The dataset files are not committed to this repository because they are large.

Public dataset archive folder:

```text
https://drive.google.com/drive/folders/1HgYeXju6ztRux0FNicaaQ8CKoi9qTl5g?usp=sharing
```

## Included Archives

The Google Drive folder contains four `.zip` files:

```text
DsPCBSD+.zip
HRIPCB.zip
DeepPCB.zip
DataPCB_Final_Clean_6cls.zip
```

## Archive Meaning

### Raw source datasets

```text
DsPCBSD+.zip
HRIPCB.zip
DeepPCB.zip
```

These are the original source datasets used to build the merged PCB defect benchmark.

### Final processed dataset

```text
DataPCB_Final_Clean_6cls.zip
```

This is the final cleaned 6-class dataset used for model training and evaluation.

## Expected Local Structure

After downloading and extracting the archives, the local data directory should follow this structure:

```text
data/
├── raw/
│   ├── archives/
│   │   ├── DsPCBSD+.zip
│   │   ├── HRIPCB.zip
│   │   └── DeepPCB.zip
│   └── datapcb-project/
│       ├── DeepPCB/
│       ├── DsPCBSD/
│       └── HRIPCB/
└── processed/
    ├── archives/
    │   └── DataPCB_Final_Clean_6cls.zip
    └── DataPCB_Final_Clean_6cls/
        ├── train/
        ├── valid/
        ├── test/
        └── data.yaml
```

## Notes

- `data/raw/` contains the original source datasets.
- `data/processed/DataPCB_Final_Clean_6cls/` contains the final cleaned 6-class dataset used for benchmark training.
- Dataset files are excluded from Git through `.gitignore`.
- Model weights, runs, and full training artifacts are also excluded from Git.

## Kaggle Training

Training was performed on Kaggle. The notebooks expect the processed dataset to be available on Kaggle under a path similar to:

```text
/kaggle/input/datasets/<owner>/pcb-merged/DataPCB_Final_Clean_6cls
```

During training, notebooks copy the dataset to:

```text
/kaggle/working/DataPCB_Final_Clean_6cls
```

to avoid read-only cache warnings from `/kaggle/input`.

## Reproducibility Note

There are two valid ways to reproduce the benchmark:

1. Use `DataPCB_Final_Clean_6cls.zip` directly as the final processed dataset.
2. Use the raw source archives and run the data preparation notebook:

```text
notebooks/01_prepare_final_datapcb_clean_6cls.ipynb
```

The first option is recommended if the goal is to reproduce the model benchmark quickly. The second option is recommended if the goal is to inspect the dataset cleaning and merging process.
