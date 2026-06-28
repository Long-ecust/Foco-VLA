#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


PROMPT_DIR = "hand_the_bottle_of_tea_to_me"
MODEL_DIRS = {
    "Full": "longvla_handover_full",
    "w/o ForcePrior": "longvla_handover_no_prior",
    "Single force token": "longvla_handover_single_token",
    "w/o joint ID": "longvla_handover_no_joint_id",
}
MODEL_ORDER = ["Full", "w/o ForcePrior", "Single force token", "w/o joint ID"]
COLORS = {
    "Full": "#1F4E79",
    "w/o ForcePrior": "#B85C38",
    "Single force token": "#2F7F6F",
    "w/o joint ID": "#6A5D9E",
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


def _latest_open_time(events: list[dict[str, str]]) -> float:
    opens = [event for event in events if event.get("event") == "R_OPEN"]
    return _float(opens[-1].get("elapsed_s")) if opens else math.nan


def _force_peak_delay(actions: list[dict[str, str]], open_t: float) -> float:
    rows = [
        row
        for row in actions
        if open_t == open_t and open_t - 5.0 <= _float(row.get("elapsed_s")) < open_t
    ]
    if not rows:
        return math.nan
    peak = max(rows, key=lambda row: _float(row.get("r_tau_abs_sum")))
    return open_t - _float(peak.get("elapsed_s"))


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


def _close_open_bounds(events: list[dict[str, str]]) -> tuple[float, float]:
    closes = [event for event in events if event.get("event") == "R_CLOSE"]
    opens = [event for event in events if event.get("event") == "R_OPEN"]
    if not closes or not opens:
        return math.nan, math.nan
    return _float(closes[0].get("elapsed_s")), _float(opens[-1].get("elapsed_s"))


def _has_release_response(events: list[dict[str, str]], peak_t: float) -> bool:
    for event in events:
        if event.get("event") != "R_OPEN":
            continue
        event_t = _float(event.get("elapsed_s"))
        if peak_t < event_t <= peak_t + RESPONSE_WINDOW_S:
            return True
    return False


def _release_response_stats(log_dir: Path, model_label: str) -> tuple[int, int, list[int]]:
    root = log_dir / MODEL_DIRS[model_label] / PROMPT_DIR
    total_peaks = 0
    unreleased_peaks = 0
    unreleased_by_trial: list[int] = []
    for trial_dir in sorted(root.glob("trial_*")):
        action_path = trial_dir / "gripper_actions.csv"
        event_path = trial_dir / "gripper_events.csv"
        if not action_path.exists() or not event_path.exists():
            continue
        actions = _read_csv(action_path)
        events = _read_csv(event_path)
        first_close_t, final_open_t = _close_open_bounds(events)
        if first_close_t != first_close_t or final_open_t != final_open_t:
            unreleased_by_trial.append(0)
            continue
        elapsed = [_float(row.get("elapsed_s")) for row in actions]
        tau = [_float(row.get("r_tau_abs_sum")) for row in actions]
        peaks = _force_peaks(elapsed, tau, first_close_t, final_open_t)
        unreleased = [peak for peak in peaks if not _has_release_response(events, peak[0])]
        total_peaks += len(peaks)
        unreleased_peaks += len(unreleased)
        unreleased_by_trial.append(len(unreleased))
    return total_peaks, unreleased_peaks, unreleased_by_trial


def _collect_model(log_dir: Path, model_label: str) -> dict[str, object]:
    root = log_dir / MODEL_DIRS[model_label] / PROMPT_DIR
    grid = [round(-5.0 + i * 0.1, 3) for i in range(81)]
    tau_traces: list[list[float]] = []
    pub_traces: list[list[float]] = []
    raw_traces: list[list[float]] = []
    delays: list[float] = []

    for trial_dir in sorted(root.glob("trial_*")):
        action_path = trial_dir / "gripper_actions.csv"
        event_path = trial_dir / "gripper_events.csv"
        if not action_path.exists() or not event_path.exists():
            continue
        actions = _read_csv(action_path)
        events = _read_csv(event_path)
        open_t = _latest_open_time(events)
        if open_t != open_t:
            continue
        ts = [_float(row.get("elapsed_s")) - open_t for row in actions]
        tau = [_float(row.get("r_tau_abs_sum")) for row in actions]
        pub = [_float(row.get("pub_r_gripper")) for row in actions]
        raw = [_float(row.get("raw_r_gripper")) for row in actions]
        tau_traces.append([_interp(ts, tau, x) for x in grid])
        pub_traces.append([_interp(ts, pub, x) for x in grid])
        raw_traces.append([_interp(ts, raw, x) for x in grid])
        delays.append(_force_peak_delay(actions, open_t))

    return {
        "grid": grid,
        "tau": tau_traces,
        "pub": pub_traces,
        "raw": raw_traces,
        "delays": delays,
    }


def _mean_sem_trace(traces: list[list[float]]) -> tuple[list[float], list[float]]:
    if not traces:
        return [], []
    n = len(traces[0])
    means = []
    sems = []
    for i in range(n):
        vals = [trace[i] for trace in traces if trace[i] == trace[i]]
        means.append(_mean(vals))
        sems.append(_sem(vals))
    return means, sems


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis="y", alpha=0.16, linewidth=0.6)
    ax.set_axisbelow(True)


def _plot_representative_timeline(ax, trial_dir: Path, title: str, color: str) -> None:
    actions = _read_csv(trial_dir / "gripper_actions.csv")
    events = _read_csv(trial_dir / "gripper_events.csv")
    first_close_t, final_open_t = _close_open_bounds(events)
    if first_close_t != first_close_t or final_open_t != final_open_t:
        return
    elapsed = [_float(row.get("elapsed_s")) for row in actions]
    tau = [_float(row.get("r_tau_abs_sum")) for row in actions]
    raw = [_float(row.get("raw_r_gripper")) for row in actions]

    start_t = first_close_t
    end_t = min(elapsed[-1], final_open_t + 2.5)
    idxs = [i for i, t in enumerate(elapsed) if start_t <= t <= end_t]
    x = [elapsed[i] - first_close_t for i in idxs]
    tau_w = [tau[i] for i in idxs]
    raw_w = [raw[i] for i in idxs]

    ax.plot(x, tau_w, color=color, linewidth=1.45, label="effort")
    ax.axhline(FORCE_THRESHOLD, color="#666666", linestyle=":", linewidth=0.85)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("time from first R_CLOSE (s)")
    ax.set_ylabel("right-arm effort sum")
    _style_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(x, raw_w, color="#333333", linewidth=0.95, alpha=0.62, label="raw gripper")
    ax2.axhline(0.092, color="#222222", linestyle=":", linewidth=0.75, alpha=0.6)
    ax2.set_ylabel("raw gripper")
    ax2.set_ylim(0.055, 0.105)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.8)

    peaks = _force_peaks(elapsed, tau, first_close_t, final_open_t)
    unreleased = [(pt, pv) for pt, pv in peaks if not _has_release_response(events, pt)]
    if unreleased:
        ax.scatter(
            [pt - first_close_t for pt, _ in unreleased],
            [pv for _, pv in unreleased],
            marker="x",
            s=48,
            color="#C62828",
            linewidths=1.8,
            label="missed release",
            zorder=5,
        )

    for event in events:
        event_t = _float(event.get("elapsed_s"))
        if event_t != event_t or not (start_t <= event_t <= end_t):
            continue
        event_name = event.get("event", "")
        event_color = "#2F7F6F" if event_name == "R_CLOSE" else "#C62828"
        ax.axvline(event_t - first_close_t, color=event_color, linestyle="--", linewidth=0.9)
        if event_name == "R_OPEN":
            ax.text(
                event_t - first_close_t,
                ax.get_ylim()[1],
                "OPEN",
                color=event_color,
                rotation=90,
                ha="right",
                va="top",
                fontsize=7.5,
            )

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=6.8, loc="upper left", handlelength=1.4)


def plot_figure(log_dir: Path, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.6,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.25), constrained_layout=True)
    fig.patch.set_facecolor("white")

    _plot_representative_timeline(
        axes[0],
        log_dir / "longvla_handover_full" / PROMPT_DIR / "trial_001",
        "a  Full: rapid force-triggered release",
        COLORS["Full"],
    )
    _plot_representative_timeline(
        axes[1],
        log_dir / "longvla_handover_no_prior" / PROMPT_DIR / "trial_004",
        "b  w/o ForcePrior: delayed release",
        COLORS["w/o ForcePrior"],
    )

    ax = axes[2]
    xs = list(range(len(MODEL_ORDER)))
    unreleased_counts = []
    ratios = []
    totals = []
    for model in MODEL_ORDER:
        total, unreleased, _ = _release_response_stats(log_dir, model)
        totals.append(total)
        unreleased_counts.append(unreleased)
        ratios.append(unreleased / total if total else 0.0)
    ax.bar(xs, ratios, color=[COLORS[m] for m in MODEL_ORDER], edgecolor="black", linewidth=0.45, width=0.62)
    for x, ratio, unreleased, total in zip(xs, ratios, unreleased_counts, totals):
        ax.text(x, min(ratio + 0.045, 0.96), f"{unreleased}/{total}", ha="center", va="bottom", fontsize=8.2)
    ax.set_xticks(xs, MODEL_ORDER, rotation=22, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("c  High-force non-release rate", loc="left", fontweight="bold")
    ax.set_ylabel("unreleased pulls / high-force pulls")
    _style_axis(ax)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "handover_ablation_force_release_figure.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / "handover_ablation_force_release_figure.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "handover_ablation_force_release_figure.svg", bbox_inches="tight")
    print(f"Wrote {output_dir / 'handover_ablation_force_release_figure.png'}")

    fig_v, axes_v = plt.subplots(3, 1, figsize=(4.1, 7.2), constrained_layout=True)
    fig_v.patch.set_facecolor("white")
    _plot_representative_timeline(
        axes_v[0],
        log_dir / "longvla_handover_full" / PROMPT_DIR / "trial_001",
        "a  Full: force-triggered release",
        COLORS["Full"],
    )
    _plot_representative_timeline(
        axes_v[1],
        log_dir / "longvla_handover_no_prior" / PROMPT_DIR / "trial_004",
        "b  w/o ForcePrior: delayed release",
        COLORS["w/o ForcePrior"],
    )

    ax_v = axes_v[2]
    ax_v.bar(xs, ratios, color=[COLORS[m] for m in MODEL_ORDER], edgecolor="black", linewidth=0.45, width=0.62)
    for x, ratio, unreleased, total in zip(xs, ratios, unreleased_counts, totals):
        ax_v.text(x, min(ratio + 0.045, 0.96), f"{unreleased}/{total}", ha="center", va="bottom", fontsize=8.0)
    ax_v.set_xticks(xs, ["Full", "-Prior", "Single", "-ID"], rotation=0)
    ax_v.set_ylim(0, 1.0)
    ax_v.set_title("c  High-force non-release rate", loc="left", fontweight="bold")
    ax_v.set_ylabel("unreleased / high-force pulls")
    _style_axis(ax_v)
    fig_v.savefig(output_dir / "handover_ablation_force_release_ral_vertical.png", dpi=450, bbox_inches="tight")
    fig_v.savefig(output_dir / "handover_ablation_force_release_ral_vertical.pdf", bbox_inches="tight")
    fig_v.savefig(output_dir / "handover_ablation_force_release_ral_vertical.svg", bbox_inches="tight")
    print(f"Wrote {output_dir / 'handover_ablation_force_release_ral_vertical.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs/handover_gripper"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/handover_gripper/paper_analysis"))
    args = parser.parse_args()
    plot_figure(args.log_dir, args.output_dir)


if __name__ == "__main__":
    main()
