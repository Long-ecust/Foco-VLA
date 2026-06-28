"""Structural + signal sanity inspection for longvla_raw_v2 HDF5 episodes.

Usage:
    uv run python examples/piper_bimanual/inspect_longvla_dataset.py \
        examples/datasets [--max-episodes N] [--quiet]

Reports per-episode and cross-episode:
  - schema/attrs, field shapes & dtypes, NaN/Inf scan
  - qpos/qvel/effort/filtered_effort/end_pose ranges and stats
  - static-segment noise on filtered_effort (joints stationary)
  - timestamp dt, drift between image / joint / end_pose streams
  - image brightness, black/frozen-frame detection (subsampled)
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np


ARM_DIM = 7  # 6 joints + 1 gripper per arm
END_DIM = 7  # x y z qx qy qz qw per arm
NUM_ARMS = 2
JOINT_LABELS = [f"L_{n}" for n in ("j1", "j2", "j3", "j4", "j5", "j6", "grip")] + [
    f"R_{n}" for n in ("j1", "j2", "j3", "j4", "j5", "j6", "grip")
]
GRIPPER_IDX = (ARM_DIM - 1, 2 * ARM_DIM - 1)  # 6, 13


@dataclass
class EpReport:
    path: str
    attrs: dict[str, Any]
    shapes: dict[str, tuple]
    nan_inf: dict[str, tuple[int, int]] = field(default_factory=dict)
    qpos_stats: dict | None = None
    qvel_stats: dict | None = None
    effort_stats: dict | None = None
    fil_effort_stats: dict | None = None
    end_pose_stats: dict | None = None
    static_noise: dict | None = None
    timestamp_stats: dict | None = None
    image_stats: dict | None = None
    issues: list[str] = field(default_factory=list)


# ---------- helpers ----------

def _stats(x: np.ndarray, axis: int = 0) -> dict:
    return {
        "min": np.nanmin(x, axis=axis),
        "max": np.nanmax(x, axis=axis),
        "mean": np.nanmean(x, axis=axis),
        "std": np.nanstd(x, axis=axis),
        "ptp": np.nanmax(x, axis=axis) - np.nanmin(x, axis=axis),
    }


def _nan_inf_count(x: np.ndarray) -> tuple[int, int]:
    if not np.issubdtype(x.dtype, np.floating):
        return 0, 0
    return int(np.isnan(x).sum()), int(np.isinf(x).sum())


def _fmt_vec(v: np.ndarray, fmt: str = "{:+.3f}") -> str:
    return "[" + ", ".join(fmt.format(float(x)) for x in v) + "]"


# ---------- analyses ----------

def analyze_static_noise(qvel: np.ndarray, fil_effort: np.ndarray, fps: float) -> dict:
    """Identify frames where every joint is approximately stationary, then
    measure filtered_effort std on those frames. Reveals sensor floor noise."""
    # per-frame "speed" — use max abs qvel across joints; gripper excluded since it
    # may report jitter even when arm is still
    arm_mask = [i for i in range(qvel.shape[1]) if i not in GRIPPER_IDX]
    speed = np.max(np.abs(qvel[:, arm_mask]), axis=1)
    thresh = max(1e-3, np.percentile(speed, 5) * 1.5)  # adaptive low-speed threshold
    static = speed < thresh
    n_static = int(static.sum())
    if n_static < int(fps):  # need at least ~1s of static data
        return {"n_static_frames": n_static, "threshold": float(thresh), "noise_std": None}
    noise_std = fil_effort[static].std(axis=0)
    noise_p2p = np.ptp(fil_effort[static], axis=0)
    return {
        "n_static_frames": n_static,
        "threshold": float(thresh),
        "static_fraction": float(n_static / len(static)),
        "noise_std": noise_std,
        "noise_p2p": noise_p2p,
    }


def analyze_timestamps(f: h5py.File, fps: float) -> dict:
    frame_t = f["timestamps/frame"][...]
    dt = np.diff(frame_t)
    expected = 1.0 / fps
    drops = int(((dt > expected * 1.5)).sum())

    # cross-stream drift: compare arrival times of first-of-each-frame events
    streams = {}
    for name in ("image", "joint_feedback", "joint_feedback_filtered", "end_pose"):
        key = f"timestamps/{name}"
        if key in f:
            ts = f[key][...]
            # take the earliest column (some streams have multiple cameras / sources)
            if ts.ndim == 2:
                ts0 = np.nanmin(ts, axis=1)
            else:
                ts0 = ts
            streams[name] = ts0

    drift = {}
    if "image" in streams:
        for k, v in streams.items():
            if k == "image":
                continue
            d = v - streams["image"]
            drift[f"{k}_minus_image_ms"] = (
                float(np.nanmean(d) * 1000),
                float(np.nanstd(d) * 1000),
                float(np.nanmax(np.abs(d)) * 1000),
            )

    return {
        "dt_mean_ms": float(dt.mean() * 1000),
        "dt_std_ms": float(dt.std() * 1000),
        "dt_min_ms": float(dt.min() * 1000),
        "dt_max_ms": float(dt.max() * 1000),
        "expected_dt_ms": expected * 1000,
        "n_dropped_frames": drops,
        "duration_s": float(frame_t[-1] - frame_t[0]),
        "drift_mean_std_max_ms": drift,
    }


def analyze_images(f: h5py.File, n_samples: int = 32) -> dict:
    out = {}
    cams = list(f["observations/images"].keys())
    for cam in cams:
        ds = f[f"observations/images/{cam}"]
        T = ds.shape[0]
        idx = np.linspace(0, T - 1, min(n_samples, T)).astype(int)
        # read sampled frames; HDF5 indexing requires sorted unique
        idx = np.unique(idx)
        frames = ds[idx, ...]  # (k, H, W, 3) uint8
        means = frames.reshape(len(idx), -1).mean(axis=1)
        stds = frames.reshape(len(idx), -1).std(axis=1)
        # frozen-frame detection: consecutive sampled frames with near-identical mean+std
        frozen = 0
        for i in range(1, len(idx)):
            if abs(means[i] - means[i - 1]) < 0.05 and abs(stds[i] - stds[i - 1]) < 0.05:
                frozen += 1
        out[cam] = {
            "mean_brightness": float(means.mean()),
            "std_brightness": float(stds.mean()),
            "min_mean": float(means.min()),
            "max_mean": float(means.max()),
            "frozen_pairs_in_sample": frozen,
            "n_sampled": len(idx),
            "n_total": T,
        }
    return out


def inspect_episode(path: str) -> EpReport:
    rep = EpReport(path=path, attrs={}, shapes={})
    with h5py.File(path, "r") as f:
        rep.attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}

        # --- structural map + nan/inf scan ---
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                rep.shapes[name] = obj.shape
                if np.issubdtype(obj.dtype, np.floating) and obj.size < 5_000_000:
                    n_nan, n_inf = _nan_inf_count(obj[...])
                    if n_nan or n_inf:
                        rep.nan_inf[name] = (n_nan, n_inf)
                        rep.issues.append(f"{name}: NaN={n_nan} Inf={n_inf}")

        f.visititems(visit)

        # --- numeric signals ---
        qpos = f["observations/qpos"][...]
        qvel = f["observations/qvel"][...]
        effort = f["observations/effort"][...]
        fil_effort = f["observations/filtered_effort"][...]
        end_pose = f["observations/end_pose"][...]

        # also scan large image arrays for NaN by sampling a few frames
        # (uint8 → no NaN possible; just check dtype is uint8)
        for cam in f["observations/images"]:
            ds = f[f"observations/images/{cam}"]
            if ds.dtype != np.uint8:
                rep.issues.append(f"images/{cam}: unexpected dtype {ds.dtype}")

        rep.qpos_stats = _stats(qpos)
        rep.qvel_stats = _stats(qvel)
        rep.effort_stats = _stats(effort)
        rep.fil_effort_stats = _stats(fil_effort)
        rep.end_pose_stats = _stats(end_pose)

        # quaternion norm sanity (cols 3..6 per arm)
        for arm_name, base in (("L", 0), ("R", END_DIM)):
            quat = end_pose[:, base + 3 : base + 7]
            qnorm = np.linalg.norm(quat, axis=1)
            err = float(np.nanmax(np.abs(qnorm - 1.0)))
            if err > 1e-3:
                rep.issues.append(f"end_pose {arm_name} quaternion norm err={err:.4f}")

        # --- static-segment noise ---
        rep.static_noise = analyze_static_noise(qvel, fil_effort, float(rep.attrs.get("fps", 30.0)))

        # --- timestamps ---
        rep.timestamp_stats = analyze_timestamps(f, float(rep.attrs.get("fps", 30.0)))
        # flag big drift
        for k, (m, s, mx) in rep.timestamp_stats["drift_mean_std_max_ms"].items():
            if mx > 50.0:
                rep.issues.append(f"timestamp drift {k}: max |Δ|={mx:.1f}ms")
        if rep.timestamp_stats["n_dropped_frames"] > 0:
            rep.issues.append(f"{rep.timestamp_stats['n_dropped_frames']} dropped frames (dt>1.5x expected)")

        # --- images ---
        rep.image_stats = analyze_images(f)
        for cam, s in rep.image_stats.items():
            if s["min_mean"] < 5:
                rep.issues.append(f"images/{cam}: black frame detected (min mean={s['min_mean']:.1f})")
            if s["frozen_pairs_in_sample"] >= 2:
                rep.issues.append(
                    f"images/{cam}: {s['frozen_pairs_in_sample']} frozen pairs in sampled frames"
                )

    return rep


# ---------- printing ----------

def print_episode(rep: EpReport) -> None:
    name = os.path.basename(rep.path)
    a = rep.attrs
    ts = rep.timestamp_stats
    print(f"\n=== {name} ===")
    print(
        f"  task: {a.get('task')!r}  fps={a.get('fps')}  frames={a.get('num_frames')}  "
        f"duration={ts['duration_s']:.2f}s  schema={a.get('schema_version')}"
    )
    print(
        f"  dt: mean={ts['dt_mean_ms']:.2f}ms expected={ts['expected_dt_ms']:.2f}ms "
        f"min={ts['dt_min_ms']:.2f} max={ts['dt_max_ms']:.2f} drops={ts['n_dropped_frames']}"
    )
    for k, (m, s, mx) in ts["drift_mean_std_max_ms"].items():
        print(f"  drift {k}: mean={m:+.2f}ms std={s:.2f}ms max|Δ|={mx:.2f}ms")

    qs = rep.qpos_stats
    print(
        f"  qpos ptp (rad): "
        + _fmt_vec(qs["ptp"], "{:.2f}")
    )
    print(
        f"  qvel max|.|:    "
        + _fmt_vec(np.maximum(np.abs(qs["min"] * 0), np.maximum(np.abs(rep.qvel_stats["min"]), np.abs(rep.qvel_stats["max"]))), "{:.2f}")
    )

    es = rep.effort_stats
    fes = rep.fil_effort_stats
    eff_std = es["std"]
    fil_std = fes["std"]
    ratio = np.where(fil_std > 1e-9, eff_std / fil_std, np.nan)
    print(f"  effort std:     " + _fmt_vec(eff_std, "{:6.2f}"))
    print(f"  fil_effort std: " + _fmt_vec(fil_std, "{:6.2f}"))
    print(f"  std ratio (raw/filt): " + _fmt_vec(ratio, "{:5.2f}"))

    sn = rep.static_noise
    if sn["noise_std"] is not None:
        print(
            f"  static-frame noise (filt_effort std on {sn['n_static_frames']}/"
            f"{a.get('num_frames')} frames, frac={sn['static_fraction']:.2f}): "
            + _fmt_vec(sn["noise_std"], "{:5.2f}")
        )
    else:
        print(f"  static-frame noise: too few static frames ({sn['n_static_frames']})")

    eps = rep.end_pose_stats
    print(
        f"  end_pose pos ptp L={_fmt_vec(eps['ptp'][:3], '{:.3f}')}  "
        f"R={_fmt_vec(eps['ptp'][7:10], '{:.3f}')}  (m)"
    )

    for cam, s in rep.image_stats.items():
        print(
            f"  img {cam}: bright μ={s['mean_brightness']:6.2f} σ={s['std_brightness']:5.2f}  "
            f"min_mean={s['min_mean']:6.2f}  frozen_pairs={s['frozen_pairs_in_sample']}/{s['n_sampled']}"
        )

    if rep.nan_inf:
        for k, (n, i) in rep.nan_inf.items():
            print(f"  !! {k}: NaN={n} Inf={i}")
    if rep.issues:
        print("  Issues:")
        for it in rep.issues:
            print(f"    - {it}")
    else:
        print("  Issues: none")


def print_summary(reps: list[EpReport]) -> None:
    print("\n=== CROSS-EPISODE SUMMARY ===")
    print(f"  episodes: {len(reps)}")
    durations = [r.timestamp_stats["duration_s"] for r in reps]
    print(
        f"  duration s: min={min(durations):.2f} max={max(durations):.2f} "
        f"mean={np.mean(durations):.2f}"
    )
    prompts = {r.attrs.get("task") for r in reps}
    print(f"  unique tasks: {prompts}")

    # field-shape consistency (ignore time dim)
    for key in ("observations/qpos", "observations/effort", "observations/filtered_effort",
                "observations/end_pose", "observations/images/camera_f"):
        shapes = {r.shapes.get(key, ("missing",))[1:] for r in reps}
        print(f"  {key} non-time shape set: {shapes}")

    # aggregate static-frame filtered_effort std
    std_stack = [r.static_noise["noise_std"] for r in reps if r.static_noise.get("noise_std") is not None]
    if std_stack:
        med = np.median(np.stack(std_stack), axis=0)
        print("  median static filtered_effort std across episodes:")
        for i, lab in enumerate(JOINT_LABELS):
            print(f"    {lab:8s} {med[i]:7.3f}")

    # per-joint qpos ptp medians
    ptps = np.stack([r.qpos_stats["ptp"] for r in reps])
    med_ptp = np.median(ptps, axis=0)
    print("  median per-joint qpos ptp (rad except gripper):")
    for i, lab in enumerate(JOINT_LABELS):
        print(f"    {lab:8s} {med_ptp[i]:6.3f}")

    issues_total = sum(len(r.issues) for r in reps)
    print(f"  total issues across episodes: {issues_total}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="directory containing *.hdf5 episodes")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--quiet", action="store_true", help="only print summary")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dataset_dir, "*.hdf5")))
    if args.max_episodes:
        paths = paths[: args.max_episodes]
    if not paths:
        raise SystemExit(f"no .hdf5 files in {args.dataset_dir}")

    reports = []
    for p in paths:
        try:
            rep = inspect_episode(p)
        except Exception as e:
            print(f"!! failed: {p}: {e}")
            continue
        reports.append(rep)
        if not args.quiet:
            print_episode(rep)

    print_summary(reports)


if __name__ == "__main__":
    main()
