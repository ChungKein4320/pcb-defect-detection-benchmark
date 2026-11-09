"""Plot saved DETR scores on CPU. Write previews to the ignored .local directory."""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit_semidetr_evidence import EXPERIMENTS, RESULTS, ROOT


LOCAL = ROOT / ".local"
TEST_AP_FILENAME = "historical_test_ap.png"
VALIDATION_FILENAME = "historical_validation_ap.png"


def plot_test_ap(rows, output):
    scores = {
        (row["labeled_percent"], row["experiment"]): float(row["test_AP"]) for row in rows
    }
    fig, ax = plt.subplots(figsize=(9.2, 3.65))
    fig.subplots_adjust(left=0.29, right=0.97, top=0.71, bottom=0.19)
    fig.text(0.05, 0.92, "Historical test AP by label budget", fontsize=17, weight="bold")
    fig.text(0.05, 0.845, "Five-class PCB coursework  ·  One run per recipe and budget", color="#475569")

    for ratio, color, marker in (("5", "#007780", "o"), ("10", "#514299", "D")):
        values = [scores[ratio, key] for key, _, _ in EXPERIMENTS]
        ax.scatter(values, range(len(values)), label=f"{ratio}% labeled", color=color, marker=marker, s=46, zorder=3)
        for index, score in enumerate(values):
            ax.annotate(
                f"{score:.4f}", (score, index), xytext=(8, 0), textcoords="offset points",
                va="center", color=color, fontsize=10, weight="bold" if score == max(values) else "normal",
            )

    ax.set(
        xlim=(0.529, 0.638), ylim=(4.55, -0.55),
        xticks=[0.53, 0.55, 0.57, 0.59, 0.61, 0.63],
        xlabel="Test AP (IoU 0.50–0.95)",
    )
    ax.set_yticks(range(len(EXPERIMENTS)), labels=[method for _, method, _ in EXPERIMENTS])
    ax.grid(axis="x", color="#dbe3ea")
    ax.tick_params(axis="y", length=0, pad=10)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.03), frameon=False, ncol=2)
    fig.savefig(output, dpi=135, metadata={"Software": None})
    plt.close(fig)


def plot_validation_ap(history, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.73, bottom=0.18, wspace=0.09)
    fig.text(0.05, 0.93, "Historical validation AP across epochs", fontsize=17, weight="bold")
    fig.text(0.05, 0.87, "Local review only  ·  Saved observations every five epochs", color="#475569")

    colors = ("#525f70", "#007780", "#2456a6", "#8e4b9a", "#ac4c08")
    for ratio, ax in zip(("5", "10"), axes):
        ax.axvspan(80, 120, color="#e8edf5", zorder=0)
        for (key, method, _), color in zip(EXPERIMENTS, colors):
            points = [
                (int(row["epoch"]), float(row["val_AP"]))
                for row in history
                if row["labeled_percent"] == ratio and row["experiment"] == key and row["val_AP"]
            ]
            if [epoch for epoch, _ in points] != list(range(5, 151, 5)):
                raise ValueError(f"Missing, duplicate, or unordered validation observations for {ratio}% {key}.")
            epochs, values = zip(*points)
            ax.plot(epochs, values, label=method, color=color, marker="o", markersize=2.5, linewidth=1.35)
        ax.set(title=f"{ratio}% labeled", xlabel="Epoch", xticks=[0, 40, 80, 120, 150], ylim=(0, 0.7))
        ax.grid(axis="y", color="#dbe3ea")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation AP (IoU 0.50–0.95)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.845), ncol=5, fontsize=8, frameon=False)
    fig.text(0.09, 0.035, "Shading marks epochs 81–120, the duplication phase in the three SHM recipes.", color="#475569")
    fig.savefig(output, dpi=135, metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-history", action="store_true", help="Plot validation AP from the verified local history export.")
    args = parser.parse_args()
    LOCAL.mkdir(exist_ok=True)

    if args.local_history:
        history_path = LOCAL / "semidetr_epoch_metrics.csv"
        if not history_path.is_file():
            parser.error("Generate the local history with audit_semidetr_evidence.py --export-history first.")
        with history_path.open(encoding="utf-8", newline="") as handle:
            plot_validation_ap(list(csv.DictReader(handle)), LOCAL / VALIDATION_FILENAME)
        print(f"Local review only: .local/{VALIDATION_FILENAME}")
    else:
        with RESULTS.open(encoding="utf-8", newline="") as handle:
            plot_test_ap(list(csv.DictReader(handle)), LOCAL / TEST_AP_FILENAME)
        print(f"Comparison preview: .local/{TEST_AP_FILENAME}")


if __name__ == "__main__":
    main()
