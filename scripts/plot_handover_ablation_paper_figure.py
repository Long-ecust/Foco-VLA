#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


MODEL_ORDER = ["full", "no_prior", "single_token", "no_joint_id"]
MODEL_LABELS = {
    "full": "Full",
    "no_prior": "w/o ForcePrior",
    "single_token": "Single force token",
    "no_joint_id": "w/o joint ID",
}
COLORS = {
    "full": "#1F4E79",
    "no_prior": "#B85C38",
    "single_token": "#2F7F6F",
    "no_joint_id": "#6A5D9E",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(value: str | None) -> float:
    try:
        return float(value) if value not in {None, ""} else math.nan
    except ValueError:
        return math.nan


def _valid(values: list[float]) -> list[float]:
    return [v for v in values if v == v]


def _mean(values: list[float]) -> float:
    values = _valid(values)
    return sum(values) / len(values) if values else math.nan


def _sem(values: list[float]) -> float:
    values = _valid(values)
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _rows_by_model(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {model: [row for row in rows if row["model"] == model] for model in MODEL_ORDER}


def _successful(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("success") == "True"]


def _bar_labels(ax, xs: list[int], values: list[float], suffix: str = "") -> None:
    top = ax.get_ylim()[1]
    for x, value in zip(xs, values):
        ax.text(x, value + top * 0.025, f"{value:g}{suffix}", ha="center", va="bottom", fontsize=9)


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis="y", alpha=0.16, linewidth=0.6)
    ax.set_axisbelow(True)


def _jitter(n: int, width: float = 0.12) -> list[float]:
    if n <= 1:
        return [0.0] * n
    return [width * (2 * i / (n - 1) - 1) for i in range(n)]


def _mean_sem_points(
    ax,
    by_model: dict[str, list[dict[str, str]]],
    metric: str,
    ylabel: str,
    title: str,
    *,
    success_only: bool = True,
) -> None:
    xs = list(range(len(MODEL_ORDER)))
    for x, model in zip(xs, MODEL_ORDER):
        rows = _successful(by_model[model]) if success_only else by_model[model]
        vals = _valid([_float(row[metric]) for row in rows])
        for dx, val in zip(_jitter(len(vals)), vals):
            ax.scatter(
                x + dx,
                val,
                s=22,
                color=COLORS[model],
                alpha=0.72,
                linewidth=0,
                zorder=2,
            )
        if vals:
            mean = _mean(vals)
            sem = _sem(vals)
            ax.errorbar(
                x,
                mean,
                yerr=sem,
                fmt="o",
                color="black",
                markersize=4.5,
                capsize=3,
                elinewidth=1.0,
                markeredgewidth=0,
                zorder=4,
            )
            ax.hlines(mean, x - 0.18, x + 0.18, color="black", linewidth=1.1, zorder=3)
    ax.set_xticks(xs, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
    _style_axis(ax)


def plot_paper_figure(summary_path: Path, output_dir: Path) -> None:
    rows = _read_csv(summary_path)
    by_model = _rows_by_model(rows)
    xs = list(range(len(MODEL_ORDER)))
    labels = [MODEL_LABELS[model] for model in MODEL_ORDER]
    colors = [COLORS[model] for model in MODEL_ORDER]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.7), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # (a) Success rate with failed trials highlighted as missing completion.
    ax = axes[0, 0]
    success_counts = [sum(1 for row in by_model[model] if row.get("success") == "True") for model in MODEL_ORDER]
    failure_counts = [10 - count for count in success_counts]
    ax.bar(xs, [10] * len(xs), color="#F1F1F1", edgecolor="#BDBDBD", linewidth=0.6, width=0.58)
    ax.bar(xs, success_counts, color=colors, edgecolor="black", linewidth=0.5, width=0.58)
    for x, succ, fail in zip(xs, success_counts, failure_counts):
        ax.text(x, succ + 0.32, f"{succ}/10", ha="center", va="bottom", fontsize=8.8)
        if fail:
            ax.text(x, succ - 0.7, f"{fail} fail", ha="center", va="top", fontsize=8.0, color="white")
    ax.set_ylim(0, 10.9)
    ax.set_xticks(xs, labels, rotation=18, ha="right")
    ax.set_ylabel("task success")
    ax.set_title("a  Completion rate", loc="left", fontweight="bold", fontsize=10.5)
    _style_axis(ax)

    _mean_sem_points(
        axes[0, 1],
        by_model,
        "main_close_from_first_s",
        "time to main close (s)",
        "b  Grasp initiation",
    )

    _mean_sem_points(
        axes[1, 0],
        by_model,
        "early_toggle_count",
        "early toggles per trial",
        "c  Grasp command stability",
        success_only=False,
    )

    # (d) Compact failure mode matrix.
    ax = axes[1, 1]
    grasp_failures = [sum(1 for row in by_model[model] if row.get("success") != "True") for model in MODEL_ORDER]
    early_totals = [sum(int(row["early_toggle_count"]) for row in by_model[model]) for model in MODEL_ORDER]
    release_anomalies = [
        sum(
            1
            for row in by_model[model]
            if "delayed_release" in row.get("note", "") or "direction_specific_release" in row.get("note", "")
        )
        for model in MODEL_ORDER
    ]
    matrix = [early_totals, grasp_failures, release_anomalies]
    row_labels = ["early\ntoggles", "grasp\nfailures", "release\nanomalies"]
    image = ax.imshow(matrix, cmap="Greys", vmin=0, vmax=max(max(row) for row in matrix), aspect="auto")
    for yi, row in enumerate(matrix):
        for xi, value in enumerate(row):
            ax.text(
                xi,
                yi,
                str(value),
                ha="center",
                va="center",
                fontsize=9.2,
                color="white" if value >= 5 else "black",
                fontweight="bold" if value else "normal",
            )
    ax.set_xticks(xs, labels, rotation=18, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title("d  Failure-mode profile", loc="left", fontweight="bold", fontsize=10.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8, length=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "handover_ablation_paper_figure.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / "handover_ablation_paper_figure.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "handover_ablation_paper_figure.svg", bbox_inches="tight")
    print(f"Wrote {output_dir / 'handover_ablation_paper_figure.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("logs/handover_gripper/paper_analysis/handover_ablation_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/handover_gripper/paper_analysis"),
    )
    args = parser.parse_args()
    plot_paper_figure(args.summary, args.output_dir)


if __name__ == "__main__":
    main()
