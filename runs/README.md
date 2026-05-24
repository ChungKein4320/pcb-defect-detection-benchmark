# Runs

Training outputs are not committed to GitHub.

This folder is used locally for downloaded Kaggle experiment outputs, logs, metrics, and checkpoints when needed.

Example local structure:

- runs/yolov8n/results.csv
- runs/yolov8n/confusion_matrix.png
- runs/yolov8n/weights/best.pt
- runs/yolov8s/results.csv
- runs/yolov8s/weights/best.pt
- runs/faster_rcnn/metrics.json
- runs/faster_rcnn/checkpoints/

Git policy:

- Do not commit full training runs.
- Do not commit checkpoints or large artifacts.
- Only this README file should be committed.

Selected summary results should be copied to:

- reports/tables/
- reports/figures/