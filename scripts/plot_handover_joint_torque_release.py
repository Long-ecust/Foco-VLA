#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PROMPT_DIR = "hand_the_bottle_of_tea_to_me"
MODELS = [
    ("full", "Full", "longvla_handover_full"),
    ("no_prior", "w/o ForcePrior", "longvla_handover_no_prior"),
    ("single_token", "Single force token", "longvla_handover_single_token"),
    ("no_joint_id", "w/o Joint ID", "longvla_handover_no_joint_id"),
]
COLORS = {
    "full": "#1F4E79",
    "no_prior": "#B85C38",
    "single_token": "#2F7F6F",
    "no_joint_id": "#6A5D9E",
}
JOINT_KEYS = [f"r_tau_j{i}" for i in range(1, 8)]
JOINT_LABELS = [f"R-J{i}" for i in range(1, 8)]
ARM_JOINT_IDXS = list(range(6))
ARM_JOINT_LABELS = [JOINT_LABELS[i] for i in ARM_JOINT_IDXS]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(value: str | None) -> float:
    try:
        return float(value) if value not in {None, ""} else math.nan
    except ValueError:
        return math.nan


def _valid(values: list[float]) -> list[float]:
    return [value for value in values if value == value]


def _mean(values: list[float]) -> float:
    values = _valid(values)
    return sum(values) / len(values) if values else math.nan


def _sem(values: list[float]) -> float:
    values = _valid(values)
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _latest_open_time(events: list[dict[str, str]]) -> float:
    opens = [event for event in events if event.get("event") == "R_OPEN"]
    return _float(opens[-1].get("elapsed_s")) if opens else math.nan


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    if not xs or x < xs[0] or x > xs[-1]:
        return math.nan
    for i in range(1, len(xs)):
        if xs[i] >= x:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return y1
            alpha = (x - x0) / (x1 - x0)
            return y0 + alpha * (y1 - y0)
    return ys[-1]


def _trial_paths(log_dir: Path, model_dir: str) -> list[Path]:
    return sorted((log_dir / model_dir / PROMPT_DIR).glob("trial_*"))


def _baseline_delta(actions: list[dict[str, str]], open_t: float) -> tuple[list[float], list[list[float]]]:
    elapsed = [_float(row.get("elapsed_s")) for row in actions]
    joints = [[_float(row.get(key)) for row in actions] for key in JOINT_KEYS]
    baseline_idx = [i for i, t in enumerate(elapsed) if open_t - 5.0 <= t <= open_t - 3.0]
    if not baseline_idx:
        baseline_idx = [i for i, t in enumerate(elapsed) if t < open_t]
    baselines = [_mean([joint[i] for i in baseline_idx]) for joint in joints]
    deltas = [[value - base for value in joint] for joint, base in zip(joints, baselines)]
    return elapsed, deltas


def _release_metrics(trial_dir: Path) -> dict[str, object] | None:
    action_path = trial_dir / "gripper_actions.csv"
    event_path = trial_dir / "gripper_events.csv"
    if not action_path.exists() or not event_path.exists():
        return None
    actions = _read_csv(action_path)
    events = _read_csv(event_path)
    open_t = _latest_open_time(events)
    if open_t != open_t:
        return None
    elapsed, deltas = _baseline_delta(actions, open_t)
    window_idx = [i for i, t in enumerate(elapsed) if open_t - 2.0 <= t <= open_t + 0.2]
    if not window_idx:
        return None
    peak_abs = [max(abs(delta[i]) for i in window_idx if delta[i] == delta[i]) for delta in deltas]
    signed_at_peak = []
    for delta in deltas:
        peak_i = max(window_idx, key=lambda i: abs(delta[i]) if delta[i] == delta[i] else -1.0)
        signed_at_peak.append(delta[peak_i])
    dominant_idx = max(ARM_JOINT_IDXS, key=lambda i: peak_abs[i])
    return {
        "trial": trial_dir.name,
        "open_t": open_t,
        "peak_abs": peak_abs,
        "signed_at_peak": signed_at_peak,
        "dominant_joint": dominant_idx,
    }


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis="y", linewidth=0.55, alpha=0.16)
    ax.set_axisbelow(True)


def _plot_example(log_dir: Path, output_dir: Path) -> None:
    trial_dir = log_dir / "longvla_handover_full" / PROMPT_DIR / "trial_009"
    actions = _read_csv(trial_dir / "gripper_actions.csv")
    events = _read_csv(trial_dir / "gripper_events.csv")
    open_t = _latest_open_time(events)
    elapsed, deltas = _baseline_delta(actions, open_t)
    t = [x - open_t for x in elapsed]
    raw = [_float(row.get("raw_r_gripper")) for row in actions]
    pub = [_float(row.get("pub_r_gripper")) for row in actions]
    tau_sum = [
        sum(abs(_float(row.get(JOINT_KEYS[j]))) for j in ARM_JOINT_IDXS)
        for row in actions
    ]
    idx = [i for i, x in enumerate(t) if -5.0 <= x <= 2.0]
    tt = [t[i] for i in idx]
    heat = np.array([[delta[i] for i in idx] for delta in deltas], dtype=float)
    release_window_idx = [i for i, x in enumerate(t) if -2.0 <= x <= 0.2]
    pre_baseline_idx = [i for i, x in enumerate(t) if -5.0 <= x <= -3.0]
    post_open_idx = [i for i, x in enumerate(t) if 0.3 <= x <= 1.3]
    peak_abs = []
    signed_peak = []
    peak_times = []
    for delta in deltas:
        peak_i = max(release_window_idx, key=lambda i: abs(delta[i]) if delta[i] == delta[i] else -1.0)
        peak_abs.append(abs(delta[peak_i]))
        signed_peak.append(delta[peak_i])
        peak_times.append(t[peak_i])
    dominant_idx = max(ARM_JOINT_IDXS, key=lambda i: peak_abs[i])
    total_peak_i = max([i for i in idx if t[i] < 0], key=lambda i: tau_sum[i])
    total_peak_t = t[total_peak_i]
    total_baseline = _mean([tau_sum[i] for i in pre_baseline_idx])
    total_post_open = _mean([tau_sum[i] for i in post_open_idx])
    total_peak_value = tau_sum[total_peak_i]
    raw_at_open = _interp(elapsed, raw, open_t)
    pub_at_open = _interp(elapsed, pub, open_t)

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
    fig = plt.figure(figsize=(6.25, 5.35), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.35])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axes[0].plot(tt, [raw[i] for i in idx], color="#333333", linewidth=1.2, label="raw gripper")
    axes[0].plot(tt, [pub[i] for i in idx], color="#1F4E79", linewidth=1.1, label="published gripper")
    axes[0].axvline(0, color="#C62828", linestyle="--", linewidth=0.9)
    axes[0].axhline(0.092, color="#777777", linestyle=":", linewidth=0.8)
    axes[0].fill_betweenx(
        axes[0].get_ylim(),
        total_peak_t,
        0,
        color="#C62828",
        alpha=0.06,
        linewidth=0,
    )
    axes[0].scatter([0], [pub_at_open], s=18, color="#C62828", zorder=4)
    axes[0].scatter([0], [raw_at_open], s=18, facecolor="white", edgecolor="#333333", linewidth=0.8, zorder=4)
    axes[0].set_ylabel("gripper pos. (m)")
    axes[0].set_title("a  Gripper opening aligned with joint-torque change", loc="left", fontweight="bold")
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.86,
        fontsize=6.7,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=2,
        handlelength=1.6,
        columnspacing=0.8,
        borderpad=0.25,
    )
    _style(axes[0])

    axes[1].plot(tt, [tau_sum[i] for i in idx], color="#1F4E79", linewidth=1.3)
    axes[1].axvline(0, color="#C62828", linestyle="--", linewidth=0.9)
    axes[1].axvline(total_peak_t, color="#1F4E79", linestyle=":", linewidth=0.9)
    axes[1].fill_betweenx(axes[1].get_ylim(), total_peak_t, 0, color="#C62828", alpha=0.06, linewidth=0)
    axes[1].axhline(total_baseline, color="#777777", linestyle=":", linewidth=0.75)
    axes[1].axhline(total_post_open, color="#777777", linestyle="--", linewidth=0.75, alpha=0.85)
    axes[1].scatter([total_peak_t], [total_peak_value], s=18, color="#1F4E79", zorder=4)
    axes[1].scatter([0], [_interp(elapsed, tau_sum, open_t)], s=18, color="#C62828", zorder=4)
    axes[1].annotate(
        f"peak-to-open\n{abs(total_peak_t):.2f} s",
        xy=(total_peak_t, tau_sum[total_peak_i]),
        xytext=(total_peak_t - 1.15, tau_sum[total_peak_i] * 0.82),
        arrowprops={"arrowstyle": "->", "linewidth": 0.75, "color": "#333333"},
        fontsize=7.0,
        ha="right",
    )
    axes[1].set_ylabel("summed arm effort (a.u.)")
    axes[1].set_title("b  Summed right-arm joint effort", loc="left", fontweight="bold")
    _style(axes[1])

    vmax = np.nanmax(np.abs(heat))
    image = axes[2].imshow(
        heat,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[tt[0], tt[-1], len(JOINT_LABELS) - 0.5, -0.5],
    )
    axes[2].axvline(0, color="black", linestyle="--", linewidth=0.85)
    axes[2].axvline(peak_times[dominant_idx], color="#C62828", linestyle=":", linewidth=0.85)
    axes[2].axhline(5.5, color="#555555", linestyle="--", linewidth=0.65, alpha=0.75)
    axes[2].add_patch(
        Rectangle(
            (tt[0], dominant_idx - 0.5),
            tt[-1] - tt[0],
            1.0,
            fill=False,
            edgecolor="#C62828",
            linewidth=1.1,
        )
    )
    axes[2].set_yticks(range(len(JOINT_LABELS)), JOINT_LABELS)
    axes[2].set_title("c  Per-joint effort change from pre-release baseline", loc="left", fontweight="bold")
    cbar = fig.colorbar(image, ax=axes[2], fraction=0.03, pad=0.02)
    cbar.set_label("effort delta (a.u.)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7.0, length=2)

    inset = axes[2].inset_axes([1.08, 0.05, 0.18, 0.82])
    y = np.arange(len(JOINT_LABELS))
    bar_colors = ["#BDBDBD"] * len(JOINT_LABELS)
    bar_colors[dominant_idx] = "#C62828"
    inset.barh(y, signed_peak, color=bar_colors, edgecolor="black", linewidth=0.25)
    inset.invert_yaxis()
    inset.axvline(0, color="black", linewidth=0.6)
    inset.set_yticks([])
    inset.set_xlabel("peak\nDelta", fontsize=6.8)
    inset.tick_params(axis="x", labelsize=6.5, length=2)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_visible(False)
    inset.spines["bottom"].set_linewidth(0.6)
    axes[2].text(
        tt[0] + 0.05,
        dominant_idx - 0.34,
        f"dominant arm joint: {JOINT_LABELS[dominant_idx]}",
        color="#C62828",
        fontsize=7.2,
        va="top",
        ha="left",
        fontweight="bold",
    )
    axes[2].annotate(
        f"{JOINT_LABELS[dominant_idx]} peak\n{signed_peak[dominant_idx]:+.1f}",
        xy=(peak_times[dominant_idx], dominant_idx),
        xytext=(peak_times[dominant_idx] - 1.0, dominant_idx + 1.05),
        arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "#C62828"},
        fontsize=6.9,
        ha="right",
        color="#C62828",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.5},
    )
    axes[2].text(
        tt[0] + 0.08,
        6.05,
        "R-J7: gripper effort, excluded",
        fontsize=6.2,
        color="#555555",
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
    )
    for ax in axes[:3]:
        ax.set_xlim(tt[0], tt[-1])
    axes[2].set_xlabel("time to R_OPEN (s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "handover_joint_torque_release_example.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / "handover_joint_torque_release_example.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "handover_joint_torque_release_example.svg", bbox_inches="tight")


def _plot_summary(log_dir: Path, output_dir: Path) -> None:
    metrics_by_model: dict[str, list[dict[str, object]]] = {}
    for key, label, model_dir in MODELS:
        metrics = []
        for trial_dir in _trial_paths(log_dir, model_dir):
            item = _release_metrics(trial_dir)
            if item is not None:
                metrics.append(item)
        metrics_by_model[key] = metrics

    mean_peak = []
    dominant_counts = []
    for key, _, _ in MODELS:
        metrics = metrics_by_model[key]
        mean_peak.append([_mean([m["peak_abs"][j] for m in metrics]) for j in ARM_JOINT_IDXS])
        dominant_counts.append([
            sum(1 for m in metrics if int(m["dominant_joint"]) == j) for j in ARM_JOINT_IDXS
        ])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
    image = axes[0].imshow(np.array(mean_peak), cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(len(ARM_JOINT_LABELS)), ARM_JOINT_LABELS)
    axes[0].set_yticks(range(len(MODELS)), [label for _, label, _ in MODELS])
    axes[0].set_title("a  Mean per-joint effort change before release", loc="left", fontweight="bold")
    for yi, row in enumerate(mean_peak):
        for xi, value in enumerate(row):
            axes[0].text(xi, yi, f"{value:.1f}", ha="center", va="center", fontsize=6.8, color="white" if value > np.nanmax(mean_peak) * 0.55 else "black")
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.02)
    cbar.set_label("mean |effort delta| (a.u.)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7.0, length=2)

    bottom = np.zeros(len(MODELS))
    xs = np.arange(len(MODELS))
    palette = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#9C755F", "#76B7B2"]
    for ji, joint_label in enumerate(ARM_JOINT_LABELS):
        values = [counts[ji] for counts in dominant_counts]
        axes[1].bar(xs, values, bottom=bottom, color=palette[ji], edgecolor="black", linewidth=0.25, label=joint_label)
        for x, base, value in zip(xs, bottom, values):
            if value > 0:
                axes[1].text(
                    x,
                    base + value / 2,
                    joint_label,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if value > 4 else "black",
                    fontweight="bold",
                )
        bottom += np.array(values)
    for x, count in zip(xs, bottom):
        axes[1].text(x, count + 0.35, f"n={int(count)}", ha="center", va="bottom", fontsize=7.0)
    axes[1].set_xticks(xs, [label for _, label, _ in MODELS], rotation=12, ha="right")
    axes[1].set_ylabel("released trials")
    axes[1].set_title("b  Dominant arm joint near release", loc="left", fontweight="bold")
    axes[1].set_ylim(0, max(bottom) + 3.0)
    _style(axes[1])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "handover_joint_torque_release_summary.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / "handover_joint_torque_release_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "handover_joint_torque_release_summary.svg", bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs/handover_gripper"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/handover_gripper/analysis_4models_30trials"))
    args = parser.parse_args()
    _plot_example(args.log_dir, args.output_dir)
    _plot_summary(args.log_dir, args.output_dir)
    print(f"Wrote joint torque release figures to {args.output_dir}")


if __name__ == "__main__":
    main()
