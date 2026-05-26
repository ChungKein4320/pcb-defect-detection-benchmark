# Semi-DETR Private Research Summary

This document summarizes a private academic experiment on semi-supervised object detection for PCB defect detection.

The original dataset, visual samples, prediction images, training outputs, checkpoints, and raw experiment artifacts are **not public** due to confidentiality constraints. The public repository only includes sanitized notebooks with cleared outputs and this high-level technical summary.

## Objective

The objective was to evaluate whether a Semi-DETR-style teacher-student pipeline can improve PCB defect detection under limited labeled data.

The experiments compare supervised and semi-supervised DETR-style detectors under two labeled-data settings:

```text
5% labeled data
10% labeled data
```

The main research question was:

```text
Can semi-supervised DETR-style training improve detection performance when only a small portion of PCB images are labeled?
```

## Public-safe notebook set

The sanitized notebooks should be placed under:

```text
notebooks/semidetr/
```

Recommended filenames:

```text
01_semidetr_data_audit_public_sanitized.ipynb
02_supervised_deformable_detr_ratio10_public_sanitized.ipynb
03_vanilla_detr_ssod_ratio10_public_sanitized.ipynb
04_detr_ssod_shm_ratio10_public_sanitized.ipynb
05_semidetr_shm_cqc_ablation_ratio10_public_sanitized.ipynb
06_semidetr_shm_cqc_cpm_full_ratio10_public_sanitized.ipynb
07_semidetr_summary_public_sanitized.ipynb
```

All notebook outputs should remain cleared before committing.

## Methods

### 1. Def-DETR Sup-only

A supervised Deformable DETR baseline trained only on the labeled subset.

Purpose:

```text
Measure the supervised lower-bound performance under limited labeled data.
```

### 2. Def-DETR SSOD

A vanilla DETR-style semi-supervised object detection baseline.

Purpose:

```text
Test whether teacher-student pseudo-labeling improves over supervised-only training.
```

### 3. DETR-SSOD + SHM

A semi-supervised baseline enhanced with SHM.

Purpose:

```text
Evaluate whether stage-wise hybrid matching improves pseudo-label usage and detection stability.
```

### 4. SHM + CQC

An ablation setting that combines SHM with CQC.

Purpose:

```text
Evaluate whether query consistency improves semi-supervised learning under limited labels.
```

### 5. Full SHM+CQC+CPM

The full Semi-DETR-style configuration using SHM, CQC, and CPM.

Purpose:

```text
Evaluate the combined effect of hybrid matching, query consistency, and pseudo-label control.
```

## Core ideas

### Teacher-student semi-supervised detection

A teacher detector generates pseudo-labels for unlabeled images. A student detector is trained on both labeled ground truth and filtered pseudo-labels.

### EMA teacher update

The teacher weights are updated from the student using exponential moving average, which stabilizes pseudo-label generation across training.

### Pseudo-label filtering

Pseudo-labels are filtered to reduce noisy supervision. This is important for PCB defects because small defects and visually similar classes can generate unstable pseudo-labels.

### SHM

SHM is used to improve the matching/training process in semi-supervised DETR-style detection.

### CQC

CQC is used to improve consistency at the query level, reducing unstable pseudo-label supervision.

### CPM

CPM is used to control or refine pseudo-label usage in the full semi-supervised setting.

## Summary results

The table below is copied from the sanitized NB07 summary. It reports validation and test AP-style metrics for both 5% and 10% labeled-data settings.

| Ratio   | Method            |   Best epoch |   Val AP |   Val AP50 |   Val AP75 |   Val APs |   Test AP |   Test AP50 |   Test AP75 |   Test APs |   Δ Test AP vs Sup-only |
|:--------|:------------------|-------------:|---------:|-----------:|-----------:|----------:|----------:|------------:|------------:|-----------:|------------------------:|
| 5%      | Def-DETR Sup-only |          150 |   0.5545 |     0.9237 |     0.619  |    0.4835 |    0.5422 |      0.9141 |      0.5974 |     0.4774 |                  0      |
| 5%      | Def-DETR SSOD     |          150 |   0.5475 |     0.9069 |     0.6209 |    0.469  |    0.5383 |      0.8985 |      0.6044 |     0.4801 |                 -0.0039 |
| 5%      | DETR-SSOD + SHM   |          150 |   0.5715 |     0.9325 |     0.6588 |    0.4936 |    0.5621 |      0.9283 |      0.6369 |     0.4911 |                  0.0199 |
| 5%      | SHM + CQC         |          145 |   0.562  |     0.9206 |     0.6478 |    0.4898 |    0.5545 |      0.9164 |      0.6275 |     0.4879 |                  0.0123 |
| 5%      | Full SHM+CQC+CPM  |          140 |   0.5577 |     0.9246 |     0.6324 |    0.4846 |    0.5439 |      0.915  |      0.6048 |     0.4784 |                  0.0018 |
| 10%     | Def-DETR Sup-only |          150 |   0.6074 |     0.9598 |     0.7303 |    0.5415 |    0.6007 |      0.9611 |      0.7029 |     0.5423 |                  0      |
| 10%     | Def-DETR SSOD     |          145 |   0.6201 |     0.9642 |     0.7467 |    0.5508 |    0.6116 |      0.9621 |      0.7218 |     0.556  |                  0.0109 |
| 10%     | DETR-SSOD + SHM   |          115 |   0.628  |     0.9639 |     0.7604 |    0.5587 |    0.6151 |      0.9636 |      0.7326 |     0.556  |                  0.0144 |
| 10%     | SHM + CQC         |          120 |   0.626  |     0.9643 |     0.7642 |    0.5607 |    0.6191 |      0.9642 |      0.7338 |     0.5584 |                  0.0185 |
| 10%     | Full SHM+CQC+CPM  |          120 |   0.6228 |     0.9621 |     0.7518 |    0.548  |    0.6211 |      0.9644 |      0.7425 |     0.5666 |                  0.0204 |

## Main observations

### 5% labeled data

The best test AP is achieved by:

```text
DETR-SSOD + SHM
Test AP = 0.5621
```

Compared with the supervised baseline:

```text
Def-DETR Sup-only Test AP = 0.5422
DETR-SSOD + SHM Test AP = 0.5621
Absolute gain = +0.0199
```

This suggests that SHM is the most useful component in the extremely low-label setting.

### 10% labeled data

The best test AP is achieved by:

```text
Full SHM+CQC+CPM
Test AP = 0.6211
```

Compared with the supervised baseline:

```text
Def-DETR Sup-only Test AP = 0.6007
Full SHM+CQC+CPM Test AP = 0.6211
Absolute gain = +0.0204
```

This suggests that the full Semi-DETR-style configuration becomes more beneficial when the labeled subset increases from 5% to 10%.

### General trend

The semi-supervised variants provide consistent gains over the supervised-only baseline in the 10% setting. In the 5% setting, SHM alone performs best, while the full SHM+CQC+CPM variant does not outperform SHM. This indicates that additional pseudo-label control modules may require enough labeled signal to become stable.

## Confidentiality policy

The following items are intentionally excluded from the public repository:

```text
private dataset
raw images
visual samples
prediction grids
confusion matrix images
model checkpoints
teacher/student weights
prediction JSON files
full training outputs
raw Kaggle outputs
```

Only sanitized code notebooks and this technical summary should be committed.

## Suggested README snippet

Add this section to the main `README.md`:

```md
## Additional Semi-DETR Research

I also studied a Semi-DETR-style semi-supervised object detection pipeline on a private academic PCB defect dataset.

The public-safe notebooks are available under:

```text
notebooks/semidetr/
```

The experiments include:

- supervised Deformable DETR baseline
- vanilla DETR-SSOD baseline
- DETR-SSOD + SHM
- Semi-DETR ablation with SHM + CQC
- full Semi-DETR-style SHM + CQC + CPM

The original dataset, visual samples, checkpoints, prediction files, and raw training outputs are not public due to confidentiality constraints.

A technical summary is available at:

```text
docs/semidetr_private_research_summary.md
```
```

## Commit checklist

Before committing the Semi-DETR material:

```powershell
Select-String -Path notebooks\semidetr\*.ipynb -Pattern '"image/png"', '"image/jpeg"', 'base64'
```

Expected result:

```text
No matches
```

Then check Git:

```powershell
git status
```

Do not commit:

```text
*.pt
*.pth
*.zip
runs/
weights/
data/raw/
data/processed/
prediction JSON files
private images
```
