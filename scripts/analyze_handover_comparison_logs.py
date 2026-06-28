#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


MODEL_DIRS = {
    "LongVLA full": ("longvla_handover_full", "hand_the_bottle_of_tea_to_me"),
    "pi0.5 vision": ("pi05_vision", "put_tea_bottle_to_hand"),
}
PULL_FORCE_THRESHOLD = 14.0
RELEASE_RESPONSE_WINDOW_S = 1.5


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(value: str | None, default: float = math.nan) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _col(rows: list[dict[str, str]], key: str) -> list[float]:
    return [_float(row.get(key)) for row in rows]


def _valid(xs: list[float]) -> list[float]:
    return [x for x in xs if x == x]


def _mean(xs: list[float]) -> float:
    xs = _valid(xs)
    return sum(xs) / len(xs) if xs else math.nan


def _sd(xs: list[float]) -> float:
    xs = _valid(xs)
    return statistics.stdev(xs) if len(xs) > 1 else math.nan


def _sem(xs: list[float]) -> float:
    xs = _valid(xs)
    return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else math.nan


def _min(xs: list[float]) -> float:
    xs = _valid(xs)
    return min(xs) if xs else math.nan


def _max(xs: list[float]) -> float:
    xs = _valid(xs)
    return max(xs) if xs else math.nan


def _event_pairs(events: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str], float]]:
    pairs = []
    current_close = None
    for event in events:
        name = event.get("event")
        if name == "R_CLOSE":
            current_close = event
        elif name == "R_OPEN" and current_close is not None:
            dt = _float(event.get("elapsed_s")) - _float(current_close.get("elapsed_s"))
            pairs.append((current_close, event, dt))
            current_close = None
    return pairs


def _force_peaks(
    elapsed: list[float],
    tau: list[float],
    start_t: float,
    end_t: float,
    threshold: float = PULL_FORCE_THRESHOLD,
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


def _summarize_trial(model: str, trial_dir: Path) -> dict[str, object]:
    actions = _read_csv(trial_dir / "gripper_actions.csv")
    events = _read_csv(trial_dir / "gripper_events.csv")

    elapsed = _col(actions, "elapsed_s")
    raw = _col(actions, "raw_r_gripper")
    pub = _col(actions, "pub_r_gripper")
    tau = _col(actions, "r_tau_abs_sum")
    tau_delta = _col(actions, "r_tau_abs_delta")

    closes = [e for e in events if e.get("event") == "R_CLOSE"]
    opens = [e for e in events if e.get("event") == "R_OPEN"]
    pairs = _event_pairs(events)

    main_open = opens[-1] if opens else None
    main_open_t = _float(main_open.get("elapsed_s")) if main_open else math.nan
    main_close = None
    if main_open is not None:
        prev_closes = [e for e in closes if _float(e.get("elapsed_s")) < main_open_t]
        main_close = prev_closes[-1] if prev_closes else None
    main_close_t = _float(main_close.get("elapsed_s")) if main_close else math.nan

    first_t = elapsed[0] if elapsed else math.nan
    success_observed = bool(main_close and main_open)
    first_close_t = _float(closes[0].get("elapsed_s")) if closes else math.nan

    early_pairs = [p for p in pairs[:-1] if p[2] < 2.0]
    post_release_reclose = 0
    if main_open is not None:
        first_open_t = _float(opens[0].get("elapsed_s"))
        post_release_reclose = sum(1 for e in closes if _float(e.get("elapsed_s")) > first_open_t)

    pre5_idx = [i for i, t in enumerate(elapsed) if main_open_t == main_open_t and main_open_t - 5.0 <= t < main_open_t]
    pre2_idx = [i for i, t in enumerate(elapsed) if main_open_t == main_open_t and main_open_t - 2.0 <= t < main_open_t]
    carry_idx = [
        i
        for i, t in enumerate(elapsed)
        if main_close_t == main_close_t and main_open_t == main_open_t and main_close_t <= t <= main_open_t
    ]

    force_peak = math.nan
    force_to_open_delay = math.nan
    if pre5_idx:
        peak_i = max(pre5_idx, key=lambda i: tau[i])
        force_peak = tau[peak_i]
        force_to_open_delay = main_open_t - elapsed[peak_i]

    pull_peaks = (
        _force_peaks(elapsed, tau, first_close_t, main_open_t)
        if first_close_t == first_close_t and main_open_t == main_open_t
        else []
    )
    unreleased_pull_peaks = [
        (peak_t, peak_tau)
        for peak_t, peak_tau in pull_peaks
        if main_open_t - peak_t > RELEASE_RESPONSE_WINDOW_S
    ]

    return {
        "model": model,
        "trial": trial_dir.name,
        "observed_close_open": success_observed,
        "num_events": len(events),
        "num_close_events": len(closes),
        "num_open_events": len(opens),
        "early_toggle_count": len(early_pairs),
        "post_first_open_reclose_count": post_release_reclose,
        "high_force_pull_count": len(pull_peaks),
        "high_force_without_release_count": len(unreleased_pull_peaks),
        "max_unreleased_pull_tau": _max([peak_tau for _, peak_tau in unreleased_pull_peaks]),
        "main_close_from_first_s": main_close_t - first_t if main_close_t == main_close_t and first_t == first_t else math.nan,
        "force_peak_pre_open_5s": force_peak,
        "force_to_open_delay_s": force_to_open_delay,
        "pre_open_2s_tau_mean": _mean([tau[i] for i in pre2_idx]),
        "pre_open_2s_tau_max": _max([tau[i] for i in pre2_idx]),
        "max_tau_delta_pre_open_5s": _max([tau_delta[i] for i in pre5_idx]),
        "raw_min_during_carry": _min([raw[i] for i in carry_idx]),
        "pub_min_during_carry": _min([pub[i] for i in carry_idx]),
        "raw_at_open": _float(main_open.get("raw_gripper")) if main_open else math.nan,
        "pub_at_open": _float(main_open.get("pub_gripper")) if main_open else math.nan,
    }


def _write_csvs(summary: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "handover_comparison_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    metrics = [
        "main_close_from_first_s",
        "force_peak_pre_open_5s",
        "force_to_open_delay_s",
        "pre_open_2s_tau_mean",
        "early_toggle_count",
        "post_first_open_reclose_count",
        "high_force_pull_count",
        "high_force_without_release_count",
        "max_unreleased_pull_tau",
        "raw_min_during_carry",
    ]
    aggregate_path = output_dir / "handover_comparison_aggregate.csv"
    with aggregate_path.open("w", newline="") as f:
        fieldnames = [
            "model",
            "observed_close_open_count",
            "n_trials",
            "early_toggle_total",
            "post_release_reclose_total",
            "high_force_without_release_total",
        ]
        for metric in metrics:
            fieldnames += [f"{metric}_mean", f"{metric}_sd", f"{metric}_sem"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in MODEL_DIRS:
            rows = [r for r in summary if r["model"] == model]
            observed = [r for r in rows if r["observed_close_open"]]
            out = {
                "model": model,
                "observed_close_open_count": len(observed),
                "n_trials": len(rows),
                "early_toggle_total": sum(int(r["early_toggle_count"]) for r in rows),
                "post_release_reclose_total": sum(int(r["post_first_open_reclose_count"]) for r in rows),
                "high_force_without_release_total": sum(int(r["high_force_without_release_count"]) for r in rows),
            }
            for metric in metrics:
                vals = [float(r[metric]) for r in observed if float(r[metric]) == float(r[metric])]
                out[f"{metric}_mean"] = _mean(vals)
                out[f"{metric}_sd"] = _sd(vals)
                out[f"{metric}_sem"] = _sem(vals)
            writer.writerow(out)


def _scatter_metric(ax, summary: list[dict[str, object]], metric: str, ylabel: str, colors: dict[str, str]) -> None:
    models = list(MODEL_DIRS.keys())
    for xi, model in enumerate(models):
        rows = [r for r in summary if r["model"] == model]
        vals = [
            float(r[metric])
            for r in rows
            if float(r[metric]) == float(r[metric]) and (r["observed_close_open"] or metric in {"early_toggle_count", "post_first_open_reclose_count"})
        ]
        ax.scatter([xi] * len(vals), vals, color=colors[model], s=44, alpha=0.82)
        if vals:
            sd = _sd(vals)
            ax.errorbar([xi], [_mean(vals)], yerr=[[0], [sd if sd == sd else 0.0]], fmt="o", color="black", capsize=4)
    ax.set_xticks(range(len(models)), models, rotation=10)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def _plot_summary(summary: list[dict[str, object]], output_dir: Path) -> None:
    colors = {
        "LongVLA full": "#1f77b4",
        "pi0.5 vision": "#d62728",
    }
    metrics = [
        ("main_close_from_first_s", "time to main R_CLOSE (s)"),
        ("early_toggle_count", "early close/open toggles"),
        ("post_first_open_reclose_count", "re-close after first R_OPEN"),
        ("force_peak_pre_open_5s", "force peak before R_OPEN"),
        ("force_to_open_delay_s", "force peak to R_OPEN delay (s)"),
        ("raw_min_during_carry", "min raw gripper during carry"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        _scatter_metric(ax, summary, metric, ylabel, colors)
    fig.suptitle("Handover comparison: force-aware LongVLA full vs pure-vision pi0.5")
    fig.savefig(output_dir / "handover_comparison_metrics.png", dpi=180)

    models = list(MODEL_DIRS.keys())
    observed_counts = [sum(1 for r in summary if r["model"] == m and r["observed_close_open"]) for m in models]
    reclose_totals = [sum(int(r["post_first_open_reclose_count"]) for r in summary if r["model"] == m) for m in models]
    early_totals = [sum(int(r["early_toggle_count"]) for r in summary if r["model"] == m) for m in models]

    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes2[0].bar(models, observed_counts, color=[colors[m] for m in models])
    axes2[0].set_ylim(0, 11)
    axes2[0].set_ylabel("observed close-open / 10")
    axes2[0].set_title("Observed task completion")
    for i, c in enumerate(observed_counts):
        axes2[0].text(i, c + 0.25, f"{c}/10", ha="center")

    axes2[1].bar(models, early_totals, color=[colors[m] for m in models])
    axes2[1].set_ylim(0, max(early_totals + [1]) + 1)
    axes2[1].set_title("Grasp-phase toggles")
    axes2[1].set_ylabel("total count / 10 trials")
    for i, c in enumerate(early_totals):
        axes2[1].text(i, c + 0.2, str(c), ha="center")

    axes2[2].bar(models, reclose_totals, color=[colors[m] for m in models])
    axes2[2].set_ylim(0, max(reclose_totals + [1]) + 1)
    axes2[2].set_title("Re-close after release")
    axes2[2].set_ylabel("total count / 10 trials")
    for i, c in enumerate(reclose_totals):
        axes2[2].text(i, c + 0.2, str(c), ha="center")

    for ax in axes2:
        ax.grid(True, axis="y", alpha=0.25)
    fig2.savefig(output_dir / "handover_comparison_counts.png", dpi=180)

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    models = list(MODEL_DIRS.keys())
    unreleased_totals = [sum(int(r["high_force_without_release_count"]) for r in summary if r["model"] == m) for m in models]
    pull_totals = [sum(int(r["high_force_pull_count"]) for r in summary if r["model"] == m) for m in models]
    axes3[0].bar(models, unreleased_totals, color=[colors[m] for m in models])
    axes3[0].set_title("High-force pulls without release")
    axes3[0].set_ylabel(f"total count / 10 trials\n(tau >= {PULL_FORCE_THRESHOLD:g}, no R_OPEN within {RELEASE_RESPONSE_WINDOW_S:g}s)")
    axes3[0].set_ylim(0, max(unreleased_totals + [1]) + 2)
    for i, c in enumerate(unreleased_totals):
        axes3[0].text(i, c + 0.3, str(c), ha="center")

    for xi, model in enumerate(models):
        vals = [int(r["high_force_without_release_count"]) for r in summary if r["model"] == model]
        axes3[1].scatter([xi] * len(vals), vals, color=colors[model], s=46, alpha=0.82)
        axes3[1].errorbar([xi], [_mean(vals)], yerr=[[0], [_sd(vals) if _sd(vals) == _sd(vals) else 0.0]], fmt="o", color="black", capsize=4)
    axes3[1].set_xticks(range(len(models)), models, rotation=10)
    axes3[1].set_title("Per-trial unreleased pull attempts")
    axes3[1].set_ylabel("count per trial")

    for xi, model in enumerate(models):
        vals = [float(r["max_unreleased_pull_tau"]) for r in summary if r["model"] == model and float(r["max_unreleased_pull_tau"]) == float(r["max_unreleased_pull_tau"])]
        axes3[2].scatter([xi] * len(vals), vals, color=colors[model], s=46, alpha=0.82)
        if vals:
            axes3[2].errorbar([xi], [_mean(vals)], yerr=[[0], [_sd(vals) if _sd(vals) == _sd(vals) else 0.0]], fmt="o", color="black", capsize=4)
    axes3[2].set_xticks(range(len(models)), models, rotation=10)
    axes3[2].set_title("Peak force during unreleased pulls")
    axes3[2].set_ylabel("max tau_abs_sum")

    for ax in axes3:
        ax.grid(True, axis="y", alpha=0.25)
    fig3.suptitle("Resistance analysis: interaction force that did not trigger release")
    fig3.savefig(output_dir / "handover_comparison_unreleased_pulls.png", dpi=180)

    fig4, axes4 = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    total_pulls = [sum(int(r["high_force_pull_count"]) for r in summary if r["model"] == m) for m in models]
    unreleased_rates = [
        unreleased_totals[i] / total_pulls[i] if total_pulls[i] else math.nan for i in range(len(models))
    ]
    axes4[0].bar(models, unreleased_totals, color=[colors[m] for m in models])
    axes4[0].set_title("Unreleased high-force pulls")
    axes4[0].set_ylabel("total count / 10 trials")
    axes4[0].set_ylim(0, max(unreleased_totals + [1]) + 2)
    for i, c in enumerate(unreleased_totals):
        axes4[0].text(i, c + 0.3, str(c), ha="center")

    axes4[1].bar(models, unreleased_rates, color=[colors[m] for m in models])
    axes4[1].set_title("Non-release ratio under high force")
    axes4[1].set_ylabel("unreleased pulls / high-force pulls")
    axes4[1].set_ylim(0, 1.0)
    for i, rate in enumerate(unreleased_rates):
        label = f"{unreleased_totals[i]}/{total_pulls[i]}"
        axes4[1].text(i, min(rate + 0.04, 0.96), label, ha="center")

    for ax in axes4:
        ax.grid(True, axis="y", alpha=0.25)
    fig4.suptitle(
        f"Release resistance statistics (tau >= {PULL_FORCE_THRESHOLD:g}, response window {RELEASE_RESPONSE_WINDOW_S:g}s)"
    )
    fig4.savefig(output_dir / "handover_comparison_release_resistance_stats.png", dpi=180)


def _plot_case_trial(
    ax,
    trial_dir: Path,
    model_label: str,
    color: str,
) -> None:
    actions = _read_csv(trial_dir / "gripper_actions.csv")
    events = _read_csv(trial_dir / "gripper_events.csv")
    elapsed = _col(actions, "elapsed_s")
    tau = _col(actions, "r_tau_abs_sum")
    pub = _col(actions, "pub_r_gripper")
    raw = _col(actions, "raw_r_gripper")
    closes = [e for e in events if e.get("event") == "R_CLOSE"]
    opens = [e for e in events if e.get("event") == "R_OPEN"]
    if not elapsed or not closes or not opens:
        return

    first_close_t = _float(closes[0].get("elapsed_s"))
    final_open_t = _float(opens[-1].get("elapsed_s"))
    window_start = max(elapsed[0], first_close_t - 2.0)
    window_end = min(elapsed[-1], final_open_t + 3.0)
    idxs = [i for i, t in enumerate(elapsed) if window_start <= t <= window_end]
    t = [elapsed[i] - first_close_t for i in idxs]
    tau_w = [tau[i] for i in idxs]
    pub_w = [pub[i] for i in idxs]
    raw_w = [raw[i] for i in idxs]

    ax.plot(t, tau_w, color=color, linewidth=1.6, label="right-arm effort sum")
    ax.axhline(PULL_FORCE_THRESHOLD, color="#555555", linestyle=":", linewidth=1.0, label="high-force threshold")
    ax.set_ylabel("r_tau_abs_sum")
    ax.set_title(f"{model_label} ({trial_dir.name})")
    ax.grid(True, alpha=0.25)

    ax_gripper = ax.twinx()
    ax_gripper.plot(t, pub_w, color="#222222", linewidth=1.2, label="published gripper")
    ax_gripper.plot(t, raw_w, color="#777777", linewidth=0.9, linestyle="--", alpha=0.75, label="raw gripper")
    ax_gripper.set_ylabel("gripper position")
    ax_gripper.set_ylim(0.0, 0.105)

    pull_peaks = _force_peaks(elapsed, tau, first_close_t, final_open_t)
    unreleased_peaks = [(pt, pv) for pt, pv in pull_peaks if final_open_t - pt > RELEASE_RESPONSE_WINDOW_S]
    if unreleased_peaks:
        ax.scatter(
            [pt - first_close_t for pt, _ in unreleased_peaks],
            [pv for _, pv in unreleased_peaks],
            color="#d62728",
            marker="x",
            s=70,
            linewidths=2.0,
            label="high force, no release",
            zorder=5,
        )

    for event in events:
        event_t = _float(event.get("elapsed_s"))
        if event_t != event_t or not (window_start <= event_t <= window_end):
            continue
        name = event.get("event", "")
        event_color = "#2ca02c" if name == "R_CLOSE" else "#d62728"
        ax.axvline(event_t - first_close_t, color=event_color, linestyle="--", linewidth=1.1, alpha=0.85)
        ax.text(
            event_t - first_close_t,
            ax.get_ylim()[1],
            name,
            color=event_color,
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
        )

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_gripper.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", ncol=2, fontsize=8)


def _plot_case_study(log_dir: Path, output_dir: Path) -> None:
    full_trial = log_dir / "longvla_handover_full" / "hand_the_bottle_of_tea_to_me" / "trial_001"
    pi05_trial = log_dir / "pi05_vision" / "put_tea_bottle_to_hand" / "trial_006"
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=False, constrained_layout=True)
    _plot_case_trial(axes[0], full_trial, "LongVLA full: force peak followed by release", "#1f77b4")
    _plot_case_trial(axes[1], pi05_trial, "pi0.5 vision: repeated pulls without release", "#d62728")
    axes[1].set_xlabel("time relative to first R_CLOSE (s)")
    fig.suptitle("Representative force-release timelines")
    fig.savefig(output_dir / "handover_comparison_force_release_case.png", dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs/handover_gripper"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/handover_gripper/comparison_analysis"))
    args = parser.parse_args()

    summary: list[dict[str, object]] = []
    for model, (model_dir, prompt_dir) in MODEL_DIRS.items():
        root = args.log_dir / model_dir / prompt_dir
        for trial_dir in sorted(root.glob("trial_*")):
            if (trial_dir / "gripper_actions.csv").exists() and (trial_dir / "gripper_events.csv").exists():
                summary.append(_summarize_trial(model, trial_dir))

    if not summary:
        raise SystemExit("No comparison logs found.")
    _write_csvs(summary, args.output_dir)
    _plot_summary(summary, args.output_dir)
    _plot_case_study(args.log_dir, args.output_dir)
    print(f"Wrote comparison analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
