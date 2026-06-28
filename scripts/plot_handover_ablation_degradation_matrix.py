#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = ["full", "no_prior", "single_token", "no_joint_id"]
MODEL_LABELS = {
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(value: str | None) -> float:
    try:
        return float(value) if value not in {None, ""} else math.nan
    except ValueError:
        return math.nan


def _mean(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else math.nan


def _model_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        success_rows = [row for row in model_rows if row["success"] == "True"]
        out[model] = {
            "success_rate": len(success_rows) / len(model_rows) if model_rows else math.nan,
            "time_to_grasp": _mean([_float(row["main_close_from_first_s"]) for row in success_rows]),
            "early_toggles": sum(int(row["early_toggle_count"]) for row in model_rows),
            "grasp_failures": sum(1 for row in model_rows if row["success"] != "True"),
            "release_anomalies": sum(
                1
                for row in model_rows
                if "delayed_release" in row.get("note", "")
                or "direction_specific_release" in row.get("note", "")
            ),
        }
    return out


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis="y", linewidth=0.55, alpha=0.16)
    ax.set_axisbelow(True)


def plot(summary_path: Path, output_dir: Path) -> None:
    rows = _read_csv(summary_path)
    stats = _model_stats(rows)
    ablations = ["no_prior", "single_token", "no_joint_id"]

    full = stats["full"]
    metrics = [
        ("success loss", [100 * (full["success_rate"] - stats[m]["success_rate"]) for m in ablations], "pp"),
        ("grasp delay", [stats[m]["time_to_grasp"] - full["time_to_grasp"] for m in ablations], "s"),
        ("extra early toggles", [stats[m]["early_toggles"] - full["early_toggles"] for m in ablations], "count"),
        ("grasp failures", [stats[m]["grasp_failures"] for m in ablations], "count"),
        ("release anomalies", [stats[m]["release_anomalies"] for m in ablations], "count"),
    ]
    raw = np.array([values for _, values, _ in metrics], dtype=float)
    # Positive values are degradation. Negative values are not treated as evidence of degradation.
    clipped = np.maximum(raw, 0.0)
    row_max = np.maximum(clipped.max(axis=1), 1e-9)
    normalized = clipped / row_max[:, None]

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

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), constrained_layout=True)

    ax = axes[0]
    xs = np.arange(len(MODELS))
    success = [100 * stats[m]["success_rate"] for m in MODELS]
    ax.bar(xs, success, color=[COLORS[m] for m in MODELS], edgecolor="black", linewidth=0.45, width=0.58)
    for x, model, value in zip(xs, MODELS, success):
        n = sum(1 for row in rows if row["model"] == model)
        k = round(value / 100 * n)
        ax.text(x, min(104, value + 3), f"{k}/{n}", ha="center", fontsize=7.6)
    ax.set_xticks(xs, [MODEL_LABELS[m] for m in MODELS])
    ax.set_ylim(0, 108)
    ax.set_ylabel("success rate (%)")
    ax.set_title("a  Task completion", loc="left", fontweight="bold")
    _style(ax)

    ax = axes[1]
    image = ax.imshow(normalized, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ablations)), [MODEL_LABELS[m] for m in ablations])
    ax.set_yticks(range(len(metrics)), [name for name, _, _ in metrics])
    ax.set_title("b  Degradation after removing each module", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for yi, (name, values, unit) in enumerate(metrics):
        for xi, value in enumerate(values):
            if abs(value) < 0.05:
                text = "0"
            elif unit == "pp":
                text = f"{value:+.0f} pp"
            elif unit == "s":
                text = f"{value:+.1f} s"
            else:
                text = f"{value:+.0f}"
            ax.text(
                xi,
                yi,
                text,
                ha="center",
                va="center",
                color="white" if normalized[yi, xi] > 0.55 else "black",
                fontsize=7.2,
                fontweight="bold" if normalized[yi, xi] > 0.55 else "normal",
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("relative degradation", fontsize=7.8)
    cbar.ax.tick_params(labelsize=7.2, length=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "handover_ablation_degradation_matrix.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / "handover_ablation_degradation_matrix.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "handover_ablation_degradation_matrix.svg", bbox_inches="tight")
    print(f"Wrote {output_dir / 'handover_ablation_degradation_matrix.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("logs/handover_gripper/paper_analysis/handover_ablation_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/handover_gripper/paper_analysis"))
    args = parser.parse_args()
    plot(args.summary, args.output_dir)


if __name__ == "__main__":
    main()
