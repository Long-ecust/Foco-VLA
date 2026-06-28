#!/usr/bin/env python3
"""Canonical, confound-aware TA-VLA vs LongVLA contact-conditioned release comparison.

复用 analyze_handover_comparison_logs.py 的已审指标函数(同一 τ>=14 / 1.5s 口径),
对全部模型统一计算:非释放*率*(带真实分母 + Wilson 95% CI)、trial 级发生率
(对单次 pull 力度不敏感、更稳健)、以及 pull 力度分布(把 full/TA-VLA 力度区间
不重叠这一混淆如实摊开)。输出到 tavla_analysis_v2,不覆盖既有图。
"""
from __future__ import annotations
import importlib.util, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

spec = importlib.util.spec_from_file_location("ahc", "scripts/analyze_handover_comparison_logs.py")
ahc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ahc)

LOG = Path("logs/handover_gripper")
OUT = LOG / "tavla_analysis_v2"; OUT.mkdir(parents=True, exist_ok=True)
MODELS = [
    ("LongVLA full",   "longvla_handover_full",        "hand_the_bottle_of_tea_to_me", "#1F4E79"),
    ("w/o Joint ID",   "longvla_handover_no_joint_id", "hand_the_bottle_of_tea_to_me", "#6A5D9E"),
    ("w/o ForcePrior", "longvla_handover_no_prior",    "hand_the_bottle_of_tea_to_me", "#B07AA1"),
    ("Single token",   "longvla_handover_single_token","hand_the_bottle_of_tea_to_me", "#2F7F6F"),
    ("TA-VLA",         "tavla_efforthis_handover",     "hand_the_bottle_of_tea_to_me", "#B85C38"),
    ("pi0.5 vision",   "pi05_vision",                  "put_tea_bottle_to_hand",       "#C62828"),
]

def wilson(k, n, z=1.96):
    if n == 0: return (math.nan, math.nan, math.nan)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0, c-h), min(1, c+h)

# ---- gather ----
rows_by = {}
pull_mags = {}
for name, md, pd, _ in MODELS:
    rows, mags = [], []
    for td in sorted((LOG/md/pd).glob("trial_*")):
        if not (td/"gripper_actions.csv").exists(): continue
        rows.append(ahc._summarize_trial(name, td))
        a = ahc._read_csv(td/"gripper_actions.csv"); ev = ahc._read_csv(td/"gripper_events.csv")
        el = ahc._col(a,"elapsed_s"); tau = ahc._col(a,"r_tau_abs_sum")
        cl=[e for e in ev if e.get("event")=="R_CLOSE"]; op=[e for e in ev if e.get("event")=="R_OPEN"]
        if cl and op:
            for _, pv in ahc._force_peaks(el, tau, ahc._float(cl[0]["elapsed_s"]), ahc._float(op[-1]["elapsed_s"])):
                mags.append(pv)
    rows_by[name] = rows; pull_mags[name] = mags

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.4,"axes.titlesize":9.2,
                     "axes.labelsize":8.6,"xtick.labelsize":7.8,"ytick.labelsize":7.8,
                     "pdf.fonttype":42,"ps.fonttype":42})
names  = [n for n,_,_,_ in MODELS]
colors = [c for *_ ,c in MODELS]
x = np.arange(len(names))

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)

# (a) non-release RATE with Wilson CI
ax = axes[0]
for i,(name,*_ ) in enumerate(MODELS):
    u = sum(int(r["high_force_without_release_count"]) for r in rows_by[name])
    p = sum(int(r["high_force_pull_count"]) for r in rows_by[name])
    rate, lo, hi = wilson(u, p)
    ax.bar(i, 100*rate, color=colors[i], edgecolor="black", linewidth=0.4)
    ax.errorbar(i, 100*rate, yerr=[[100*(rate-lo)],[100*(hi-rate)]], fmt="none", ecolor="black", capsize=3, lw=0.9)
    ax.text(i, 100*hi+2.5, f"{u}/{p}", ha="center", va="bottom", fontsize=7.2)
ax.set_ylim(0,100); ax.set_ylabel("non-release rate under high force (%)")
ax.set_title("a  Per-pull non-release rate", loc="left", fontweight="bold")
ax.set_xticks(x, names, rotation=18, ha="right")
ax.grid(True, axis="y", alpha=0.18)

# (b) trial-level incidence
ax = axes[1]
for i,(name,*_ ) in enumerate(MODELS):
    N = len(rows_by[name]); k = sum(1 for r in rows_by[name] if int(r["high_force_without_release_count"])>0)
    rate, lo, hi = wilson(k, N)
    ax.bar(i, 100*rate, color=colors[i], edgecolor="black", linewidth=0.4)
    ax.errorbar(i, 100*rate, yerr=[[100*(rate-lo)],[100*(hi-rate)]], fmt="none", ecolor="black", capsize=3, lw=0.9)
    ax.text(i, 100*hi+2.5, f"{k}/{N}", ha="center", va="bottom", fontsize=7.2)
ax.set_ylim(0,100); ax.set_ylabel("trials with ≥1 unreleased pull (%)")
ax.set_title("b  Trial-level non-release incidence", loc="left", fontweight="bold")
ax.set_xticks(x, names, rotation=18, ha="right")
ax.grid(True, axis="y", alpha=0.18)

# (c) pull magnitude distribution (transparency on the confound)
ax = axes[2]
for i,(name,*_ ) in enumerate(MODELS):
    mags = pull_mags[name]
    if mags:
        jit = (np.random.default_rng(i).random(len(mags))-0.5)*0.5
        ax.scatter(np.full(len(mags),i)+jit, mags, s=12, color=colors[i], alpha=0.55, edgecolor="none")
        ax.plot([i-0.28,i+0.28],[np.median(mags)]*2, color="black", lw=1.6)
ax.axhline(14, color="#777", ls=":", lw=0.9)
ax.text(len(names)-0.5, 14.4, "τ≥14 pull threshold", ha="right", va="bottom", fontsize=6.6, color="#555")
ax.set_ylabel("pull peak effort  (a.u., uncalibrated)")
ax.set_title("c  Pull-force regime (stimulus)", loc="left", fontweight="bold")
ax.set_xticks(x, names, rotation=18, ha="right")
ax.grid(True, axis="y", alpha=0.18)

fig.suptitle("Contact-conditioned release: force-structured (LongVLA full) vs single-token / vision baselines",
             fontsize=9.6, fontweight="bold")
for ext in ("png","pdf","svg"):
    fig.savefig(OUT/f"contact_release_canonical.{ext}", dpi=400, bbox_inches="tight")
print("wrote", OUT/"contact_release_canonical.png")
