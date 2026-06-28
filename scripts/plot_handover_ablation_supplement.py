#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = ["full", "no_prior", "single_token", "no_joint_id"]
LABELS = {
    "full": "Full",
    "no_prior": "w/o\nForcePrior",
    "single_token": "Single\nforce token",
    "no_joint_id": "w/o\nJoint ID",
}
COLORS = {
    "full": "#1F4E79",
    "no_prior": "#B85C38",
    "single_token": "#2F7F6F",
    "no_joint_id": "#6A5D9E",
}
FAILURE_TYPES = [
    ("no_close_intent", "no close\nintent"),
    ("weak_close_not_debounced", "weak close\nnot debounced"),
    ("no_effective_close", "ineffective\nclose"),
    ("no_release_after_close", "no release\nafter close"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis="y", linewidth=0.55, alpha=0.16)
    ax.set_axisbelow(True)


def _jitter(n: int, width: float = 0.12) -> list[float]:
    if n <= 1:
        return [0.0] * n
    return [width * (2 * i / (n - 1) - 1) for i in range(n)]


def plot(summary_path: Path, output_dir: Path) -> None:
    rows = _read_csv(summary_path)
    by_model = {model: [row for row in rows if row["model"] == model] for model in MODELS}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    xs = list(range(len(MODELS)))

    ax = axes[0]
    matrix = np.array(
        [
            [sum(1 for row in by_model[model] if row["failure_type"] == failure_type) for model in MODELS]
            for failure_type, _ in FAILURE_TYPES
        ],
        dtype=float,
    )
    image = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=max(float(matrix.max()), 1.0), aspect="auto")
    for yi in range(matrix.shape[0]):
        for xi in range(matrix.shape[1]):
            value = int(matrix[yi, xi])
            ax.text(
                xi,
                yi,
                str(value),
                ha="center",
                va="center",
                color="white" if value >= 3 else "black",
                fontsize=8.0,
                fontweight="bold" if value else "normal",
            )
    ax.set_xticks(xs, [LABELS[m] for m in MODELS])
    ax.set_yticks(range(len(FAILURE_TYPES)), [label for _, label in FAILURE_TYPES])
    ax.set_title("a  Failure modes", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("failed trials", fontsize=7.6)
    cbar.ax.tick_params(labelsize=7.2, length=2)

    ax = axes[1]
    toggle_trials = [sum(1 for row in by_model[model] if int(row["early_toggle_count"]) > 0) for model in MODELS]
    toggle_totals = [sum(int(row["early_toggle_count"]) for row in by_model[model]) for model in MODELS]
    ax.bar(xs, toggle_trials, color=[COLORS[m] for m in MODELS], edgecolor="black", linewidth=0.45, width=0.62)
    for x, trials, total in zip(xs, toggle_trials, toggle_totals):
        ax.text(x, trials + 0.45, f"{trials}/30\n{total} toggles", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(xs, [LABELS[m] for m in MODELS])
    ax.set_ylim(0, max(toggle_trials + [1]) + 3.0)
    ax.set_ylabel("trials with early toggles")
    ax.set_title("b  Grasp-phase instability", loc="left", fontweight="bold")
    _style(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "handover_ablation_supplement.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / "handover_ablation_supplement.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "handover_ablation_supplement.svg", bbox_inches="tight")
    print(f"Wrote {output_dir / 'handover_ablation_supplement.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("logs/handover_gripper/analysis_4models_30trials/handover_30trial_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/handover_gripper/analysis_4models_30trials"),
    )
    args = parser.parse_args()
    plot(args.summary, args.output_dir)


if __name__ == "__main__":
    main()
