#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_handover_comparison_logs import _read_csv, _summarize_trial  # noqa: E402


PROMPT_DIR = "hand_the_bottle_of_tea_to_me"
TAVLA_MODEL_DIR = "tavla_efforthis_handover"
FULL_MODEL_DIR = "longvla_handover_full"
DELAYED_RELEASE_THRESHOLD_S = 2.5


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else math.nan
    except (TypeError, ValueError):
        return math.nan


def _valid(values: list[float]) -> list[float]:
    return [value for value in values if value == value]


def _mean(values: list[float]) -> float:
    values = _valid(values)
    return sum(values) / len(values) if values else math.nan


def _sd(values: list[float]) -> float:
    values = _valid(values)
    return statistics.stdev(values) if len(values) > 1 else math.nan


def _sem(values: list[float]) -> float:
    values = _valid(values)
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.nan


def _trial_rows(log_dir: Path, model_label: str, model_dir: str) -> list[dict[str, object]]:
    root = log_dir / model_dir / PROMPT_DIR
    rows: list[dict[str, object]] = []
    for trial_dir in sorted(root.glob("trial_*")):
        action_path = trial_dir / "gripper_actions.csv"
        event_path = trial_dir / "gripper_events.csv"
        if not action_path.exists() or not event_path.exists():
            continue
        row = _summarize_trial(model_label, trial_dir)
        row["delayed_release"] = _float(row["force_to_open_delay_s"]) > DELAYED_RELEASE_THRESHOLD_S
        row["review_flag"] = bool(
            int(row["early_toggle_count"]) > 0
            or int(row["high_force_without_release_count"]) > 0
            or row["delayed_release"]
        )
        rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for model in sorted({str(row["model"]) for row in rows}):
        mr = [row for row in rows if row["model"] == model]
        delays = [_float(row["force_to_open_delay_s"]) for row in mr]
        close_times = [_float(row["main_close_from_first_s"]) for row in mr]
        high_force = [int(row["high_force_without_release_count"]) for row in mr]
        out.append(
            {
                "model": model,
                "event_complete_count": sum(1 for row in mr if row["observed_close_open"]),
                "n_trials": len(mr),
                "review_trial_count": sum(1 for row in mr if row["review_flag"]),
                "delayed_release_trials": sum(1 for row in mr if row["delayed_release"]),
                "early_toggle_trials": sum(1 for row in mr if int(row["early_toggle_count"]) > 0),
                "early_toggle_total": sum(int(row["early_toggle_count"]) for row in mr),
                "high_force_without_release_total": sum(high_force),
                "high_force_without_release_trials": sum(1 for value in high_force if value > 0),
                "force_to_open_delay_s_mean": _mean(delays),
                "force_to_open_delay_s_sd": _sd(delays),
                "force_to_open_delay_s_sem": _sem(delays),
                "main_close_from_first_s_mean": _mean(close_times),
                "main_close_from_first_s_sd": _sd(close_times),
                "main_close_from_first_s_sem": _sem(close_times),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.55, alpha=0.18)
    ax.set_axisbelow(True)


def _plot_tavla_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    tavla = [row for row in rows if row["model"] == "TA-VLA"]
    labels = [
        "event\ncomplete",
        "review\nflag",
        "delayed\nrelease",
        "early\ntoggle",
        "unreleased\npull trial",
    ]
    values = [
        sum(1 for row in tavla if row["observed_close_open"]),
        sum(1 for row in tavla if row["review_flag"]),
        sum(1 for row in tavla if row["delayed_release"]),
        sum(1 for row in tavla if int(row["early_toggle_count"]) > 0),
        sum(1 for row in tavla if int(row["high_force_without_release_count"]) > 0),
    ]
    colors = ["#3B7A57", "#C47A2C", "#B85C38", "#7E6AAD", "#B85C38"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)

    axes[0].bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(range(len(labels)), labels)
    axes[0].set_ylim(0, 32)
    axes[0].set_ylabel("trials / 30")
    axes[0].set_title("a  TA-VLA handover stage outcomes", loc="left", fontweight="bold")
    for x, value in enumerate(values):
        axes[0].text(x, value + 0.6, str(value), ha="center", va="bottom", fontsize=8.5)
    _style(axes[0])

    xs = np.arange(1, len(tavla) + 1)
    delays = [_float(row["force_to_open_delay_s"]) for row in tavla]
    unreleased = [int(row["high_force_without_release_count"]) for row in tavla]
    delayed = [bool(row["delayed_release"]) for row in tavla]
    axes[1].scatter(xs, delays, s=30, color="#1F4E79", label="force-to-open delay")
    axes[1].scatter(
        [x for x, flag in zip(xs, delayed) if flag],
        [d for d, flag in zip(delays, delayed) if flag],
        s=52,
        facecolor="#B85C38",
        edgecolor="black",
        linewidth=0.4,
        label="delayed",
        zorder=3,
    )
    for x, count, delay in zip(xs, unreleased, delays):
        if count > 0:
            axes[1].text(x, delay + 0.18, str(count), ha="center", va="bottom", fontsize=6.8, color="#B85C38")
    axes[1].axhline(DELAYED_RELEASE_THRESHOLD_S, color="#777777", linestyle=":", linewidth=0.9)
    axes[1].set_xlabel("trial")
    axes[1].set_ylabel("force-to-open delay (s)")
    axes[1].set_title("b  Release response latency", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7.0, loc="upper left")
    _style(axes[1])

    fig.savefig(output_dir / "tavla_handover_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "tavla_handover_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "tavla_handover_summary.svg", bbox_inches="tight")


def _plot_comparison(rows: list[dict[str, object]], output_dir: Path) -> None:
    models = ["LongVLA full", "TA-VLA"]
    colors = {"LongVLA full": "#1F4E79", "TA-VLA": "#B85C38"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.45), constrained_layout=True)
    for ax in axes:
        _style(ax)

    totals = [
        sum(int(row["high_force_without_release_count"]) for row in rows if row["model"] == model)
        for model in models
    ]
    axes[0].bar(models, totals, color=[colors[m] for m in models], edgecolor="black", linewidth=0.4)
    axes[0].set_title("a  High-force pulls\nwithout release", loc="left", fontweight="bold")
    axes[0].set_ylabel("count / 30 trials")
    for x, value in enumerate(totals):
        axes[0].text(x, value + 0.7, str(value), ha="center", fontsize=8.0)

    for i, model in enumerate(models):
        vals = [int(row["high_force_without_release_count"]) for row in rows if row["model"] == model]
        jitter = np.linspace(-0.08, 0.08, len(vals))
        axes[1].scatter(np.full(len(vals), i) + jitter, vals, s=18, color=colors[model], alpha=0.75)
        axes[1].plot([i - 0.16, i + 0.16], [_mean(vals), _mean(vals)], color="black", linewidth=1.4)
    axes[1].set_xticks(range(len(models)), models, rotation=12, ha="right")
    axes[1].set_ylabel("count per trial")
    axes[1].set_title("b  Per-trial\nunreleased pulls", loc="left", fontweight="bold")

    for i, model in enumerate(models):
        vals = [_float(row["force_to_open_delay_s"]) for row in rows if row["model"] == model]
        vals = _valid(vals)
        jitter = np.linspace(-0.08, 0.08, len(vals))
        axes[2].scatter(np.full(len(vals), i) + jitter, vals, s=18, color=colors[model], alpha=0.75)
        axes[2].errorbar(i, _mean(vals), yerr=_sem(vals), color="black", marker="_", markersize=16, capsize=3)
    axes[2].axhline(DELAYED_RELEASE_THRESHOLD_S, color="#777777", linestyle=":", linewidth=0.9)
    axes[2].set_xticks(range(len(models)), models, rotation=12, ha="right")
    axes[2].set_ylabel("delay (s)")
    axes[2].set_title("c  Force-to-open\ndelay", loc="left", fontweight="bold")

    fig.savefig(output_dir / "tavla_vs_longvla_release_resistance.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "tavla_vs_longvla_release_resistance.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "tavla_vs_longvla_release_resistance.svg", bbox_inches="tight")


def _plot_case(log_dir: Path, output_dir: Path, trial_name: str = "trial_026") -> None:
    trial_dir = log_dir / TAVLA_MODEL_DIR / PROMPT_DIR / trial_name
    actions = _read_csv(trial_dir / "gripper_actions.csv")
    events = _read_csv(trial_dir / "gripper_events.csv")
    elapsed = [_float(row.get("elapsed_s")) for row in actions]
    tau = [_float(row.get("r_tau_abs_sum")) for row in actions]
    raw = [_float(row.get("raw_r_gripper")) for row in actions]
    pub = [_float(row.get("pub_r_gripper")) for row in actions]
    closes = [event for event in events if event.get("event") == "R_CLOSE"]
    opens = [event for event in events if event.get("event") == "R_OPEN"]
    close_t = _float(closes[-1].get("elapsed_s")) if closes else elapsed[0]
    open_t = _float(opens[-1].get("elapsed_s")) if opens else elapsed[-1]
    t = [value - close_t for value in elapsed]
    idx = [i for i, value in enumerate(elapsed) if close_t - 1.0 <= value <= open_t + 2.0]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.1,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.45), sharex=True, constrained_layout=True)
    axes[0].plot([t[i] for i in idx], [tau[i] for i in idx], color="#B85C38", linewidth=1.2)
    axes[0].axvline(0, color="#2F7F6F", linestyle="--", linewidth=0.9, label="R_CLOSE")
    axes[0].axvline(open_t - close_t, color="#C62828", linestyle="--", linewidth=0.9, label="R_OPEN")
    axes[0].axhline(14.0, color="#777777", linestyle=":", linewidth=0.9, label="high-force threshold")
    axes[0].set_ylabel("right-arm effort sum")
    axes[0].set_title(f"a  TA-VLA delayed release example ({trial_name})", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7.0, loc="upper left", ncol=3)
    _style(axes[0])

    axes[1].plot([t[i] for i in idx], [raw[i] for i in idx], color="#333333", linewidth=1.0, label="raw gripper")
    axes[1].plot([t[i] for i in idx], [pub[i] for i in idx], color="#1F4E79", linewidth=1.1, label="published gripper")
    axes[1].axvline(0, color="#2F7F6F", linestyle="--", linewidth=0.9)
    axes[1].axvline(open_t - close_t, color="#C62828", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("time since final R_CLOSE (s)")
    axes[1].set_ylabel("gripper pos. (m)")
    axes[1].set_title("b  Gripper remains closed during repeated force peaks", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7.0, loc="upper left", ncol=2)
    _style(axes[1])

    fig.savefig(output_dir / "tavla_delayed_release_case.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "tavla_delayed_release_case.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "tavla_delayed_release_case.svg", bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs/handover_gripper"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/handover_gripper/tavla_analysis"))
    args = parser.parse_args()

    tavla_rows = _trial_rows(args.log_dir, "TA-VLA", TAVLA_MODEL_DIR)
    full_rows = _trial_rows(args.log_dir, "LongVLA full", FULL_MODEL_DIR)
    if not tavla_rows:
        raise SystemExit("No TA-VLA logs found")

    rows = full_rows + tavla_rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "tavla_trial_summary.csv", tavla_rows)
    _write_csv(args.output_dir / "tavla_vs_longvla_trial_summary.csv", rows)
    _write_csv(args.output_dir / "tavla_vs_longvla_aggregate.csv", _aggregate(rows))
    _plot_tavla_summary(rows, args.output_dir)
    if full_rows:
        _plot_comparison(rows, args.output_dir)
    _plot_case(args.log_dir, args.output_dir)
    print(f"Wrote TA-VLA handover analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
