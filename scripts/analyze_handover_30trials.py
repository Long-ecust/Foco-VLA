#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics


PROMPT_DIR = "hand_the_bottle_of_tea_to_me"
MODELS = {
    "full": "longvla_handover_full",
    "no_prior": "longvla_handover_no_prior",
    "single_token": "longvla_handover_single_token",
    "no_joint_id": "longvla_handover_no_joint_id",
}
RIGHT_CLOSE_RAW_THRESHOLD = 0.075
RIGHT_OPEN_RAW_THRESHOLD = 0.092


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


def _sd(values: list[float]) -> float:
    values = _valid(values)
    return statistics.stdev(values) if len(values) > 1 else math.nan


def _sem(values: list[float]) -> float:
    values = _valid(values)
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.nan


def _event_pairs(events: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str], float]]:
    pairs = []
    current_close = None
    for event in events:
        if event.get("event") == "R_CLOSE":
            current_close = event
        elif event.get("event") == "R_OPEN" and current_close is not None:
            dt = _float(event.get("elapsed_s")) - _float(current_close.get("elapsed_s"))
            pairs.append((current_close, event, dt))
            current_close = None
    return pairs


def _summarize_trial(model: str, trial_dir: Path) -> dict[str, object]:
    actions = _read_csv(trial_dir / "gripper_actions.csv")
    events = _read_csv(trial_dir / "gripper_events.csv")
    elapsed = [_float(row.get("elapsed_s")) for row in actions]
    raw = [_float(row.get("raw_r_gripper")) for row in actions]
    pub = [_float(row.get("pub_r_gripper")) for row in actions]
    fb = [_float(row.get("fb_r_gripper")) for row in actions]
    tau = [_float(row.get("r_tau_abs_sum")) for row in actions]

    closes = [event for event in events if event.get("event") == "R_CLOSE"]
    opens = [event for event in events if event.get("event") == "R_OPEN"]
    pairs = _event_pairs(events)
    early_toggles = [pair for pair in pairs[:-1] if pair[2] < 2.0]

    main_open = opens[-1] if opens else None
    main_open_t = _float(main_open.get("elapsed_s")) if main_open else math.nan
    main_close = None
    if main_open_t == main_open_t:
        prev_closes = [event for event in closes if _float(event.get("elapsed_s")) < main_open_t]
        main_close = prev_closes[-1] if prev_closes else None
    main_close_t = _float(main_close.get("elapsed_s")) if main_close else math.nan

    raw_min = min(_valid(raw)) if _valid(raw) else math.nan
    pub_min = min(_valid(pub)) if _valid(pub) else math.nan
    fb_min = min(_valid(fb)) if _valid(fb) else math.nan
    tau_max = max(_valid(tau)) if _valid(tau) else math.nan
    first_elapsed = elapsed[0] if elapsed else math.nan

    failure_type = ""
    if not closes and not opens:
        if raw_min == raw_min and raw_min <= RIGHT_CLOSE_RAW_THRESHOLD and (pub_min != pub_min or pub_min > 0.08):
            failure_type = "weak_close_not_debounced"
        elif raw_min == raw_min and raw_min > RIGHT_OPEN_RAW_THRESHOLD:
            failure_type = "no_close_intent"
        else:
            failure_type = "no_effective_close"
    elif closes and not opens:
        failure_type = "no_release_after_close"
    elif opens and not closes:
        failure_type = "open_without_close"

    success = bool(main_close and main_open and not failure_type)
    notes = []
    if failure_type:
        notes.append(failure_type)
    if early_toggles:
        notes.append(f"early_toggle_x{len(early_toggles)}")
    if len(events) > 4:
        notes.append(f"many_events_x{len(events)}")

    pre5_idx = [
        i for i, t in enumerate(elapsed) if main_open_t == main_open_t and main_open_t - 5.0 <= t < main_open_t
    ]
    force_peak = math.nan
    force_to_open_delay = math.nan
    if pre5_idx:
        peak_i = max(pre5_idx, key=lambda i: tau[i])
        force_peak = tau[peak_i]
        force_to_open_delay = main_open_t - elapsed[peak_i]

    return {
        "model": model,
        "trial": trial_dir.name,
        "success": success,
        "failure_type": failure_type,
        "note": ";".join(notes),
        "num_events": len(events),
        "num_close": len(closes),
        "num_open": len(opens),
        "early_toggle_count": len(early_toggles),
        "main_close_from_first_s": main_close_t - first_elapsed if main_close_t == main_close_t else math.nan,
        "force_peak_pre_open_5s": force_peak,
        "force_to_open_delay_s": force_to_open_delay,
        "raw_min": raw_min,
        "pub_min": pub_min,
        "fb_min": fb_min,
        "tau_abs_max": tau_max,
    }


def _write_csv(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "handover_30trial_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_path = output_dir / "handover_30trial_aggregate.csv"
    with aggregate_path.open("w", newline="") as f:
        fieldnames = [
            "model",
            "success_count",
            "n_trials",
            "failure_count",
            "no_close_intent",
            "weak_close_not_debounced",
            "no_effective_close",
            "no_release_after_close",
            "early_toggle_total",
            "early_toggle_trials",
            "many_event_trials",
            "main_close_from_first_s_mean",
            "main_close_from_first_s_sd",
            "main_close_from_first_s_sem",
            "force_to_open_delay_s_mean",
            "force_to_open_delay_s_sd",
            "force_to_open_delay_s_sem",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in MODELS:
            mr = [row for row in rows if row["model"] == model]
            succ = [row for row in mr if row["success"]]
            close_times = [float(row["main_close_from_first_s"]) for row in succ]
            delays = [float(row["force_to_open_delay_s"]) for row in succ]
            out = {
                "model": model,
                "success_count": len(succ),
                "n_trials": len(mr),
                "failure_count": sum(1 for row in mr if not row["success"]),
                "no_close_intent": sum(1 for row in mr if row["failure_type"] == "no_close_intent"),
                "weak_close_not_debounced": sum(1 for row in mr if row["failure_type"] == "weak_close_not_debounced"),
                "no_effective_close": sum(1 for row in mr if row["failure_type"] == "no_effective_close"),
                "no_release_after_close": sum(1 for row in mr if row["failure_type"] == "no_release_after_close"),
                "early_toggle_total": sum(int(row["early_toggle_count"]) for row in mr),
                "early_toggle_trials": sum(1 for row in mr if int(row["early_toggle_count"]) > 0),
                "many_event_trials": sum(1 for row in mr if int(row["num_events"]) > 4),
                "main_close_from_first_s_mean": _mean(close_times),
                "main_close_from_first_s_sd": _sd(close_times),
                "main_close_from_first_s_sem": _sem(close_times),
                "force_to_open_delay_s_mean": _mean(delays),
                "force_to_open_delay_s_sd": _sd(delays),
                "force_to_open_delay_s_sem": _sem(delays),
            }
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs/handover_gripper"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/handover_gripper/analysis_4models_30trials"))
    args = parser.parse_args()

    rows = []
    for model, model_dir in MODELS.items():
        root = args.log_dir / model_dir / PROMPT_DIR
        for trial_dir in sorted(root.glob("trial_*")):
            if (trial_dir / "gripper_actions.csv").exists() and (trial_dir / "gripper_events.csv").exists():
                rows.append(_summarize_trial(model, trial_dir))
    if not rows:
        raise SystemExit("No logs found")
    _write_csv(rows, args.output_dir)
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
