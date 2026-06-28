#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


PROMPT_DIR = "hand_the_bottle_of_tea_to_me"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(value: str | None, default: float = math.nan) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _col(rows: list[dict[str, str]], key: str) -> list[float]:
    return [_float(r.get(key)) for r in rows]


def _valid(xs: list[float]) -> list[float]:
    return [x for x in xs if x == x]


def _mean(xs: list[float]) -> float:
    xs = _valid(xs)
    return sum(xs) / len(xs) if xs else math.nan


def _sd(xs: list[float]) -> float:
    xs = _valid(xs)
    return statistics.stdev(xs) if len(xs) > 1 else math.nan


def _min(xs: list[float]) -> float:
    xs = _valid(xs)
    return min(xs) if xs else math.nan


def _max(xs: list[float]) -> float:
    xs = _valid(xs)
    return max(xs) if xs else math.nan


def _sem(xs: list[float]) -> float:
    xs = _valid(xs)
    if len(xs) <= 1:
        return math.nan
    return statistics.stdev(xs) / math.sqrt(len(xs))


def _summarize_trial(model: str, trial_dir: Path) -> dict[str, object]:
    actions = _read_csv(trial_dir / "gripper_actions.csv")
    events = _read_csv(trial_dir / "gripper_events.csv")

    elapsed = _col(actions, "elapsed_s")
    raw = _col(actions, "raw_r_gripper")
    pub = _col(actions, "pub_r_gripper")
    tau = _col(actions, "r_tau_abs_sum")

    first_elapsed = elapsed[0] if elapsed else math.nan
    closes = [e for e in events if e.get("event") == "R_CLOSE"]
    opens = [e for e in events if e.get("event") == "R_OPEN"]

    main_open = opens[-1] if opens else None
    main_close = None
    if main_open is not None:
        open_t = _float(main_open["elapsed_s"])
        prev_closes = [e for e in closes if _float(e["elapsed_s"]) < open_t]
        main_close = prev_closes[-1] if prev_closes else None

    main_open_t = _float(main_open["elapsed_s"]) if main_open else math.nan
    main_close_t = _float(main_close["elapsed_s"]) if main_close else math.nan

    event_pairs = []
    current_close = None
    for event in events:
        if event.get("event") == "R_CLOSE":
            current_close = event
        elif event.get("event") == "R_OPEN" and current_close is not None:
            dt = _float(event["elapsed_s"]) - _float(current_close["elapsed_s"])
            event_pairs.append((current_close, event, dt))
            current_close = None

    early_toggles = [p for p in event_pairs[:-1] if p[2] < 2.0]
    success = bool(main_open and main_close)

    # Manual annotations from the physical trials.
    note = ""
    if model == "no_prior" and trial_dir.name in {"trial_007", "trial_009"}:
        success = False
        note = "grasp_failure"
    if model == "no_prior" and trial_dir.name == "trial_004":
        note = "delayed_release_multiple_pulls"
    if model == "no_joint_id" and trial_dir.name == "trial_009":
        note = "direction_specific_release_downward_pull_only"
    if early_toggles:
        note = (note + ";" if note else "") + f"early_toggle_x{len(early_toggles)}"

    pre_release_window_s = 5.0
    pre_idx = [
        i
        for i, t in enumerate(elapsed)
        if main_open_t == main_open_t and main_open_t - pre_release_window_s <= t < main_open_t
    ]
    pre2_idx = [
        i for i, t in enumerate(elapsed) if main_open_t == main_open_t and main_open_t - 2.0 <= t < main_open_t
    ]
    carry_idx = [
        i
        for i, t in enumerate(elapsed)
        if main_close_t == main_close_t and main_open_t == main_open_t and main_close_t <= t <= main_open_t
    ]

    force_peak = math.nan
    force_peak_t = math.nan
    force_to_open_delay = math.nan
    if pre_idx:
        peak_i = max(pre_idx, key=lambda i: tau[i])
        force_peak = tau[peak_i]
        force_peak_t = elapsed[peak_i]
        force_to_open_delay = main_open_t - force_peak_t

    return {
        "model": model,
        "trial": trial_dir.name,
        "success": success,
        "note": note,
        "num_events": len(events),
        "early_toggle_count": len(early_toggles),
        "main_close_from_first_s": main_close_t - first_elapsed if main_close_t == main_close_t else math.nan,
        "force_peak_pre_open_5s": force_peak,
        "force_to_open_delay_s": force_to_open_delay,
        "pre_open_2s_tau_mean": _mean([tau[i] for i in pre2_idx]),
        "pre_open_2s_tau_max": _max([tau[i] for i in pre2_idx]),
        "raw_min_during_carry": _min([raw[i] for i in carry_idx]),
        "raw_at_open": _float(main_open["raw_gripper"]) if main_open else math.nan,
        "pub_at_open": _float(main_open["pub_gripper"]) if main_open else math.nan,
    }


def _write_summary(summary: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "handover_ablation_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    aggregate_path = output_dir / "handover_ablation_aggregate.csv"
    models = ["full", "no_prior", "single_token", "no_joint_id"]
    metrics = [
        "main_close_from_first_s",
        "force_peak_pre_open_5s",
        "force_to_open_delay_s",
        "pre_open_2s_tau_mean",
        "early_toggle_count",
    ]
    with aggregate_path.open("w", newline="") as f:
        fieldnames = ["model", "success_count", "n_trials", "early_toggle_total", "early_toggle_trials"]
        for metric in metrics:
            fieldnames += [f"{metric}_mean", f"{metric}_sd", f"{metric}_sem"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            rows = [r for r in summary if r["model"] == model]
            succ = [r for r in rows if r["success"]]
            out = {
                "model": model,
                "success_count": len(succ),
                "n_trials": len(rows),
                "early_toggle_total": sum(int(r["early_toggle_count"]) for r in rows),
                "early_toggle_trials": sum(1 for r in rows if int(r["early_toggle_count"]) > 0),
            }
            for metric in metrics:
                vals = [float(r[metric]) for r in succ if float(r[metric]) == float(r[metric])]
                out[f"{metric}_mean"] = _mean(vals)
                out[f"{metric}_sd"] = _sd(vals)
                out[f"{metric}_sem"] = _sem(vals)
            writer.writerow(out)


def _plot_summary(summary: list[dict[str, object]], output_dir: Path) -> None:
    models = ["full", "no_prior", "single_token", "no_joint_id"]
    colors = {
        "full": "#1f77b4",
        "no_prior": "#d62728",
        "single_token": "#2ca02c",
        "no_joint_id": "#9467bd",
    }
    metrics = [
        ("main_close_from_first_s", "time to main R_CLOSE from first action (s)"),
        ("early_toggle_count", "grasp-phase early close/open toggles"),
        ("force_peak_pre_open_5s", "force peak before R_OPEN (5s window)"),
        ("force_to_open_delay_s", "force peak to R_OPEN delay (s)"),
        ("pre_open_2s_tau_mean", "mean effort before R_OPEN (2s window)"),
        ("raw_min_during_carry", "min raw gripper during carry"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for xi, model in enumerate(models):
            rows = [r for r in summary if r["model"] == model]
            vals = [
                float(r[metric])
                for r in rows
                if float(r[metric]) == float(r[metric]) and (r["success"] or metric == "early_toggle_count")
            ]
            ax.scatter([xi] * len(vals), vals, color=colors[model], s=42, alpha=0.8)
            if vals:
                sd = _sd(vals)
                ax.errorbar(
                    [xi],
                    [_mean(vals)],
                    yerr=[[0], [sd if sd == sd else 0.0]],
                    fmt="o",
                    color="black",
                    capsize=4,
                )
        ax.set_xticks(range(len(models)), models, rotation=18)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Handover ablation: grasp stability and force-conditioned release")
    fig.savefig(output_dir / "handover_ablation_metrics.png", dpi=180)

    fig2, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    counts = [sum(1 for r in summary if r["model"] == m and r["success"]) for m in models]
    ax.bar(models, counts, color=[colors[m] for m in models])
    ax.set_ylim(0, 10)
    ax.set_ylabel("success count / 10")
    ax.set_title("Observed task success count")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.2, f"{c}/10", ha="center")
    ax.grid(True, axis="y", alpha=0.25)
    fig2.savefig(output_dir / "handover_ablation_success_count.png", dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs/handover_gripper"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/handover_gripper/paper_analysis"))
    args = parser.parse_args()

    model_dirs = {
        "full": args.log_dir / "longvla_handover_full" / PROMPT_DIR,
        "no_prior": args.log_dir / "longvla_handover_no_prior" / PROMPT_DIR,
        "single_token": args.log_dir / "longvla_handover_single_token" / PROMPT_DIR,
        "no_joint_id": args.log_dir / "longvla_handover_no_joint_id" / PROMPT_DIR,
    }

    summary: list[dict[str, object]] = []
    for model, model_dir in model_dirs.items():
        for trial_dir in sorted(model_dir.glob("trial_*")):
            if (trial_dir / "gripper_actions.csv").exists() and (trial_dir / "gripper_events.csv").exists():
                summary.append(_summarize_trial(model, trial_dir))

    if not summary:
        raise SystemExit("No handover gripper logs found.")

    _write_summary(summary, args.output_dir)
    _plot_summary(summary, args.output_dir)
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
