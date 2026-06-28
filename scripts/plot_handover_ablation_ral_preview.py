#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


PROMPT_DIR = "hand_the_bottle_of_tea_to_me"
MODELS = [
    ("full", "Full", "longvla_handover_full"),
    ("no_prior", "-Prior", "longvla_handover_no_prior"),
    ("single_token", "Single", "longvla_handover_single_token"),
    ("no_joint_id", "-ID", "longvla_handover_no_joint_id"),
]
COLORS = {
    "full": "#1F4E79",
    "no_prior": "#B85C38",
    "single_token": "#2F7F6F",
    "no_joint_id": "#6A5D9E",
}
FORCE_THRESHOLD = 14.0
RESPONSE_WINDOW_S = 1.5


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


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return math.nan, math.nan, math.nan
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return phat, max(0.0, center - margin), min(1.0, center + margin)


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis="y", linewidth=0.55, alpha=0.16)
    ax.set_axisbelow(True)


def _jitter(n: int, width: float = 0.10) -> list[float]:
    if n <= 1:
        return [0.0] * n
    return [width * (2 * i / (n - 1) - 1) for i in range(n)]


def _force_peaks(
    elapsed: list[float],
    tau: list[float],
    start_t: float,
    end_t: float,
    threshold: float = FORCE_THRESHOLD,
    min_separation_s: float = 1.0,
) -> list[tuple[float, float]]:
    idxs = [i for i, t in enumerate(elapsed) if start_t <= t <= end_t and tau[i] == tau[i]]
    peaks: list[tuple[float, float]] = []
    for pos in range(1, len(idxs) - 1):
        i = idxs[pos]
        if tau[i] < threshold:
            continue
        if tau[i] < tau[idxs[pos - 1]] or tau[i] < tau[idxs[pos + 1]]:
            continue
        if peaks and elapsed[i] - peaks[-1][0] < min_separation_s:
            if tau[i] > peaks[-1][1]:
                peaks[-1] = (elapsed[i], tau[i])
        else:
            peaks.append((elapsed[i], tau[i]))
    return peaks


def _release_response_stats(log_dir: Path, model_dir: str) -> tuple[int, int]:
    root = log_dir / model_dir / PROMPT_DIR
    total = 0
    missed = 0
    for trial_dir in sorted(root.glob("trial_*")):
        action_path = trial_dir / "gripper_actions.csv"
        event_path = trial_dir / "gripper_events.csv"
        if not action_path.exists() or not event_path.exists():
            continue
        actions = _read_csv(action_path)
        events = _read_csv(event_path)
        closes = [event for event in events if event.get("event") == "R_CLOSE"]
        opens = [event for event in events if event.get("event") == "R_OPEN"]
        if not closes or not opens:
            continue
        start_t = _float(closes[0].get("elapsed_s"))
        end_t = _float(opens[-1].get("elapsed_s"))
        elapsed = [_float(row.get("elapsed_s")) for row in actions]
        tau = [_float(row.get("r_tau_abs_sum")) for row in actions]
        peaks = _force_peaks(elapsed, tau, start_t, end_t)
        for peak_t, _ in peaks:
            total += 1
            responded = any(
                event.get("event") == "R_OPEN"
                and peak_t < _float(event.get("elapsed_s")) <= peak_t + RESPONSE_WINDOW_S
                for event in events
            )
            if not responded:
                missed += 1
    return missed, total


def plot(summary_path: Path, log_dir: Path, output_dir: Path) -> None:
    rows = _read_csv(summary_path)
    by_model = {key: [row for row in rows if row["model"] == key] for key, _, _ in MODELS}

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

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.75), constrained_layout=True)
    xs = list(range(len(MODELS)))
    labels = [label for _, label, _ in MODELS]

    ax = axes[0, 0]
    for x, (key, label, _) in zip(xs, MODELS):
        n = len(by_model[key])
        k = sum(1 for row in by_model[key] if row.get("success") == "True")
        rate, lo, hi = _wilson(k, n)
        ax.errorbar(
            x,
            rate * 100,
            yerr=[[100 * (rate - lo)], [100 * (hi - rate)]],
            fmt="o",
            markersize=5.2,
            capsize=3,
            color=COLORS[key],
            ecolor=COLORS[key],
            elinewidth=1.0,
        )
        ax.text(x, min(104, rate * 100 + 5.5), f"{k}/{n}", ha="center", fontsize=7.5)
    ax.set_xticks(xs, labels)
    ax.set_ylim(60, 108)
    ax.set_ylabel("success rate (%)")
    ax.set_title("a  Completion", loc="left", fontweight="bold")
    _style(ax)

    ax = axes[0, 1]
    for x, (key, _, _) in zip(xs, MODELS):
        vals = _valid([
            _float(row.get("main_close_from_first_s"))
            for row in by_model[key]
            if row.get("success") == "True"
        ])
        for dx, val in zip(_jitter(len(vals)), vals):
            ax.scatter(x + dx, val, s=16, color=COLORS[key], alpha=0.66, linewidth=0)
        if vals:
            mean = _mean(vals)
            sem = _sem(vals)
            ax.errorbar(x, mean, yerr=sem, fmt="o", color="black", markersize=4.0, capsize=3, elinewidth=1.0)
            ax.hlines(mean, x - 0.17, x + 0.17, color="black", linewidth=1.0)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("time to grasp (s)")
    ax.set_title("b  Grasp initiation", loc="left", fontweight="bold")
    _style(ax)

    ax = axes[1, 0]
    toggle_totals = [sum(int(row["early_toggle_count"]) for row in by_model[key]) for key, _, _ in MODELS]
    ax.bar(xs, toggle_totals, color=[COLORS[key] for key, _, _ in MODELS], edgecolor="black", linewidth=0.45, width=0.62)
    for x, value in zip(xs, toggle_totals):
        ax.text(x, value + 0.28, str(value), ha="center", fontsize=7.8)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, max(toggle_totals) + 1.6)
    ax.set_ylabel("early toggles / 30 trials")
    ax.set_title("c  Grasp command stability", loc="left", fontweight="bold")
    _style(ax)

    ax = axes[1, 1]
    for x, (key, _, model_dir) in zip(xs, MODELS):
        missed, total = _release_response_stats(log_dir, model_dir)
        rate, lo, hi = _wilson(missed, total)
        ax.errorbar(
            x,
            rate * 100,
            yerr=[[100 * (rate - lo)], [100 * (hi - rate)]],
            fmt="o",
            markersize=5.2,
            capsize=3,
            color=COLORS[key],
            ecolor=COLORS[key],
            elinewidth=1.0,
        )
        ax.text(x, min(104, rate * 100 + 6), f"{missed}/{total}", ha="center", fontsize=7.5)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 70)
    ax.set_ylabel("non-release rate (%)")
    ax.set_title("d  Contact-conditioned release", loc="left", fontweight="bold")
    _style(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "handover_ablation_ral_preview.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / "handover_ablation_ral_preview.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "handover_ablation_ral_preview.svg", bbox_inches="tight")
    print(f"Wrote {output_dir / 'handover_ablation_ral_preview.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("logs/handover_gripper/paper_analysis/handover_ablation_summary.csv"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/handover_gripper"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/handover_gripper/paper_analysis"))
    args = parser.parse_args()
    plot(args.summary, args.log_dir, args.output_dir)


if __name__ == "__main__":
    main()
