# Semi-Supervised Deformable DETR Research

## Experimental design

The coursework team studied whether unlabeled PCB images help when annotation is limited. Supervised Deformable DETR establishes a reference at each label budget, using only the corresponding labeled subset. The 5% and 10% settings retain 126 and 252 labeled images from the 2,518-image training split, leaving 2,392 and 2,266 images for pseudo-labeling. The ten authoritative runs shuffle sorted training pairs with seed 42 and take a rounded fraction. An earlier standalone data-audit notebook used a different stratification routine and is outside the active run sequence.

All five recipes start from pretrained `SenseTime/deformable-detr` in PyTorch and Hugging Face Transformers, with a five-class head for IDs 0–4. Deformable attention samples a small set of locations over multi-scale features, which makes the detector a natural starting point for studying small defects. The validation and test splits contain 840 and 838 images. The runs train for 150 epochs, evaluate validation AP every five epochs, select the best validation checkpoint, and report final test bounding-box AP with pycocotools COCOeval. For vanilla SSOD, both teacher and student were evaluated and the student was the primary model. For the SHM-containing runs, final evaluation uses the student. [Deformable DETR (ICLR 2021)](https://arxiv.org/abs/2010.04159)

## Read the implementation

| Notebook | What it does |
| --- | --- |
| [01 · Supervised baseline](../notebooks/semi_supervised_research/01_supervised_deformable_detr.ipynb) | Fine-tunes on the labeled subset without consuming unlabeled images |
| [02 · Vanilla DETR-SSOD](../notebooks/semi_supervised_research/02_vanilla_detr_ssod.ipynb) | Runs supervised burn-in and then trains a student with teacher pseudo-labels |
| [03 · SHM approximation](../notebooks/semi_supervised_research/03_shm_approximation.ipynb) | Repeats pseudo-targets during epochs 81–120, then returns to one target per pseudo-box |
| [04 · SHM + CQC proxy](../notebooks/semi_supervised_research/04_shm_cqc_proxy.ipynb) | Adds agreement between matched weak-view teacher and strong-view student predictions |
| [05 · SHM + CQC + CPM](../notebooks/semi_supervised_research/05_shm_cqc_cpm.ipynb) | Uses assignment costs to filter which pseudo-boxes become consistency anchors |
| [06 · Historical results and analysis](../notebooks/semi_supervised_research/06_historical_results_and_analysis.ipynb) | Shows the saved test AP chart and links each recipe to its training code |

In the SSOD recipes, the teacher starts from the student and follows it by exponential moving average (momentum 0.995). After 80 epochs of supervised burn-in, the teacher predicts pseudo-boxes on weak views. The student learns from the matching strong views and from labeled images. The pseudo-label confidence threshold anneals from 0.50 to 0.25. The miners use box-size filtering, class-wise NMS where available, and a low-score fallback to recover candidates when the main threshold yields too few.

The [Semi-DETR paper](https://arxiv.org/abs/2307.08095) develops its own matching, cross-view query mechanism, and cost-based pseudo-label mining. The coursework makes the following adaptations.

- **SHM.** Duplicating each unsupervised target lets the unchanged Hugging Face one-to-one matcher assign multiple queries to copies of the same object. The implementation keeps the supervised targets unchanged.
- **CQC.** The coursework matches student predictions to teacher pseudo-boxes and penalizes cosine distance between class vectors and L1 box error. The paper builds cross-view query embeddings for feature-level consistency. The proxy operates on predictions and retains teacher query indices by forming candidates directly from logits and boxes. Vanilla and SHM-only instead start from the Hugging Face postprocessor.
- **CPM.** The coursework builds an assignment cost from class, L1, and IoU terms. A two-component Gaussian mixture selects the low-cost cluster, with a 0.60 quantile fallback. The filtered boxes feed the CQC branch. The pseudo-label detection loss is computed using the earlier pseudo-target set.

## Evidence and interpretation

The [historical results CSV](../reports/semi_supervised_research/historical_results.csv) records final validation and test AP, AP50, AP75, APs, selected epoch and stage, and a source locator and SHA-256 for each original run. The 10% vanilla run reports final validation AP of 0.6202. Its checkpoint metadata stores 0.6201504 before rounding. The 0.6201 value in the earlier repository summary differs from the saved final evaluation.

At 5%, the SHM recipe scored 0.0199 above supervised training and vanilla SSOD scored 0.0039 below it. Both consistency recipes scored below SHM alone. At 10%, the full recipe scored 0.0204 above supervised training, while SHM alone achieved the highest validation AP. The comparisons use one seed and do not isolate the added modules. CQC changes candidate generation, and CPM uses one unlabeled batch per labeled batch where the other SSOD recipes use two. The selected 10% SHM-containing checkpoints come from the duplication stage (epochs 115, 120, 120), even though those runs continued through the later one-to-one stage.

Five local 10% checkpoints match the saved selected epochs and ratios in their checkpoint metadata. Checkpoints were located for the 10% runs only. The [test AP chart](../reports/semi_supervised_research/figures/historical_test_ap.png) is a new CPU visualization of the ten historical `test_AP` values in the CSV, all previously reported in the public summary. Its points represent the test scores of validation-selected checkpoints. The notebook embeds the same PNG for viewing without execution. [Previously reported results](https://github.com/ChungKein4320/pcb-defect-detection-benchmark/blob/4f3eb23/docs/semidetr_private_research_summary.md#reported-internal-results)

A [local split audit](../reports/semi_supervised_research/split_audit.json) found 587 candidate filename families shared by train and test after stripping the export and variant suffixes. No images shared exact file bytes across those splits. The candidate count signals possible reuse of related images and does not establish which samples depict the same board. Any new grouped split would need its own run provenance and comparison.

## Access, attribution, and verification

The five-class study used DetectionProjects' `electronic_component_failure`, Roboflow version 1 dated February 19, 2025, in YOLOv8 format. The saved export metadata and the public listing both state CC BY 4.0. Credit DetectionProjects and the dataset, link the license, and indicate any changes when reusing the export. [Roboflow Universe listing](https://universe.roboflow.com/detectionprojects-3nwgx/electronic_component_failure) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The instructor restricted use of the coursework copy. The public release links to the upstream listing and includes the code and verified aggregate results. Coursework images, annotations, prediction grids, raw logs, checkpoints, the team report, and epoch-level validation artifacts stay local. See the [dataset setup](reproducibility.md#set-up-the-five-class-detr-dataset) for the five-class export.

The five training notebooks are sanitized 10% counterparts. Their archive sections show validation-selected epochs and aggregate scores from the original 5% and 10% runs. The original 5% counterparts use the same core training logic with different ratio settings, experiment labels and output paths. The active notebooks remove a credential assignment and use environment variables for data and output paths. The [reproducibility guide](reproducibility.md) explains how to verify the CSV against the original notebook outputs and inspect the saved validation histories locally.

This was a group project. The repository owner implemented the supervised Deformable DETR baseline, EMA teacher–student training and pseudo-labeling, and the coursework's Semi-DETR-inspired SHM, CQC, and CPM variants.
