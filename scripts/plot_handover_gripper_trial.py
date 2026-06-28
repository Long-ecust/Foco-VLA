#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(value: str | None, default: float = float("nan")) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _col(rows: list[dict[str, str]], key: str) -> list[float]:
    return [_float(row.get(key)) for row in rows]


def plot_trial(trial_dir: Path) -> None:
    actions_path = trial_dir / "gripper_actions.csv"
    events_path = trial_dir / "gripper_events.csv"
    if not actions_path.exists():
        raise FileNotFoundError(actions_path)
    if not events_path.exists():
        raise FileNotFoundError(events_path)

    actions = _read_csv(actions_path)
    events = _read_csv(events_path)

    elapsed = _col(actions, "elapsed_s")
    if not elapsed:
        raise ValueError(f"No action rows in {actions_path}")
    t0 = elapsed[0]
    t = [x - t0 for x in elapsed]

    raw = _col(actions, "raw_r_gripper")
    pub = _col(actions, "pub_r_gripper")
    fb = _col(actions, "fb_r_gripper")
    tau_abs = _col(actions, "r_tau_abs_sum")
    tau_delta = _col(actions, "r_tau_abs_delta")
    gripper_tau = _col(actions, "r_gripper_tau")

    event_rows = [(ev.get("event", ""), _float(ev.get("elapsed_s")) - t0, ev) for ev in events]

    model = actions[0].get("model_name", trial_dir.parents[2].name if len(trial_dir.parents) > 2 else "")
    trial = trial_dir.name

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t, raw, label="raw_r_gripper", color="#1f77b4", linewidth=1.4)
    axes[0].plot(t, pub, label="pub_r_gripper", color="#d62728", linewidth=1.2)
    axes[0].plot(t, fb, label="fb_r_gripper", color="#2ca02c", linewidth=1.0, alpha=0.85)
    axes[0].set_ylabel("gripper pos")
    axes[0].set_title(f"{model} {trial}: right gripper and effort")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", ncol=3)

    axes[1].plot(t, tau_abs, label="r_tau_abs_sum", color="#9467bd", linewidth=1.2)
    axes[1].plot(t, gripper_tau, label="r_gripper_tau", color="#8c564b", linewidth=1.0, alpha=0.85)
    axes[1].set_ylabel("effort")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", ncol=2)

    axes[2].plot(t, tau_delta, label="r_tau_abs_delta", color="#ff7f0e", linewidth=0.9)
    axes[2].axhline(0, color="black", linewidth=0.6, alpha=0.4)
    axes[2].set_ylabel("effort delta")
    axes[2].set_xlabel("time since first logged action (s)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    for event_name, event_t, _ in event_rows:
        if event_t != event_t:
            continue
        color = "#d62728" if event_name == "R_OPEN" else "#2ca02c"
        for ax in axes:
            ax.axvline(event_t, color=color, linestyle="--", linewidth=1.2, alpha=0.85)
        axes[0].text(event_t, axes[0].get_ylim()[1], event_name, color=color, rotation=90, va="top", ha="right", fontsize=9)

    overview_path = trial_dir / f"{model}_{trial}_gripper_effort_overview.png"
    fig.savefig(overview_path, dpi=180)

    open_events = [(name, ev_t, ev) for name, ev_t, ev in event_rows if name == "R_OPEN" and ev_t == ev_t]
    if open_events:
        _, rel_t, _ = open_events[-1]
        idxs = [i for i, x in enumerate(t) if rel_t - 5 <= x <= rel_t + 5]
        zr = [t[i] - rel_t for i in idxs]
        fig2, axes2 = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
        axes2[0].plot(zr, [raw[i] for i in idxs], label="raw", color="#1f77b4")
        axes2[0].plot(zr, [pub[i] for i in idxs], label="published", color="#d62728")
        axes2[0].plot(zr, [fb[i] for i in idxs], label="feedback", color="#2ca02c")
        axes2[0].axvline(0, color="#d62728", linestyle="--", label="main R_OPEN")
        axes2[0].set_ylabel("gripper pos")
        axes2[0].set_title(f"{model} {trial}: release-aligned view")
        axes2[0].grid(True, alpha=0.25)
        axes2[0].legend(ncol=4)

        axes2[1].plot(zr, [tau_abs[i] for i in idxs], label="r_tau_abs_sum", color="#9467bd")
        axes2[1].plot(zr, [tau_delta[i] for i in idxs], label="r_tau_abs_delta", color="#ff7f0e")
        axes2[1].plot(zr, [gripper_tau[i] for i in idxs], label="r_gripper_tau", color="#8c564b")
        axes2[1].axvline(0, color="#d62728", linestyle="--")
        axes2[1].axhline(0, color="black", linewidth=0.6, alpha=0.4)
        axes2[1].set_ylabel("effort")
        axes2[1].set_xlabel("time relative to main R_OPEN (s)")
        axes2[1].grid(True, alpha=0.25)
        axes2[1].legend(ncol=3)
        fig2.savefig(trial_dir / f"{model}_{trial}_release_aligned.png", dpi=180)

    print(f"Wrote {overview_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_dir", type=Path)
    args = parser.parse_args()
    plot_trial(args.trial_dir)


if __name__ == "__main__":
    main()
