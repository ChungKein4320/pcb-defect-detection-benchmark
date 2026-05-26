# Data

This folder is used for local dataset storage only. Dataset files are not committed to GitHub.

## Download

Dataset archives are available from the Google Drive folder below:

```text
https://drive.google.com/drive/folders/1HgYeXju6ztRux0FNicaaQ8CKoi9qTl5g?usp=sharing
```

See the detailed data source guide:

```text
docs/data_sources.md
```

## Included Archives

The Google Drive folder contains four `.zip` files:

```text
DsPCBSD+.zip
HRIPCB.zip
DeepPCB.zip
DataPCB_Final_Clean_6cls.zip
```

## Expected Local Structure

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

## Folder Meaning

```text
data/raw/
```

Stores the original source datasets.

```text
data/processed/
```

Stores the final cleaned dataset used by the benchmark.

The main benchmark dataset is:

```text
data/processed/DataPCB_Final_Clean_6cls/
```

## Git Policy

The following folders are excluded from Git:

```text
data/raw/
data/processed/
```

Only this `README.md` file is committed to document the expected data layout.

## Training Environment

Training is performed on Kaggle. Local data is used only for backup, inspection, and project organization.

Full datasets, processed outputs, training runs, and weights are not committed to this repository.
