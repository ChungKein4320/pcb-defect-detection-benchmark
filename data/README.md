# Data

This folder is used only for local storage and is not committed to GitHub.

## Raw datasets

Expected local structure:

- data/raw/datapcb-project/DeepPCB/
- data/raw/datapcb-project/DsPCBSD/
- data/raw/datapcb-project/HRIPCB/

Original archives can be stored locally at:

- data/raw/archives/DeepPCB.zip
- data/raw/archives/DsPCBSD+.zip
- data/raw/archives/HRIPCB.zip

## Processed dataset

Final processed dataset:

- data/processed/DataPCB_Final_Clean_6cls/train/
- data/processed/DataPCB_Final_Clean_6cls/valid/
- data/processed/DataPCB_Final_Clean_6cls/test/
- data/processed/DataPCB_Final_Clean_6cls/data.yaml

Processed archive can be stored locally at:

- data/processed/archives/DataPCB_Final_Clean_6cls.zip

## Training environment

All training is performed on Kaggle.

Local data is kept only for backup, inspection, and project organization.

Kaggle raw dataset path:

- /kaggle/input/datasets/chungkein/datapcb-project

Final processed dataset path on Kaggle:

- /kaggle/working/datapcb_final_clean_6cls/DataPCB_Final_Clean_6cls

Dataset files are excluded from Git via `.gitignore`.