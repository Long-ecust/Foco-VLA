#!/usr/bin/env python3
"""Paper-style visualization for LongVLA training metrics.

读 csv_logs/ 下的 metrics.csv (单 run) 或 csv_logs/<run_name>/metrics.csv (多 run),
画三张论文级别的图:

  fig_total_loss.pdf     : flow vs anomaly_weighted 分解 + aux 比例
  fig_force_dynamics.pdf : tau_hat 追赶 tau_t + mismatch 残差
  fig_summary.pdf        : 4 panel 综合 (适合 main paper / supplementary)

单 run mode 会在同一张图上画 tau_t 参考线; 多 run mode 自动用子目录名做 legend.

Usage:
    python draw_picture.py
    python draw_picture.py --smooth 7
    python draw_picture.py --csv-dir csv_logs --out-dir figures
"""
from __future__ import annotations

import argparse
import csv
import pathlib
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np


# Okabe-Ito 配色 (色弱友好, Nature 推荐). 论文里常用, 比 matplotlib 默认更适合 print.
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

SEMANTIC_COLORS = {
    "flow": OKABE_ITO["blue"],
    "anomaly_mse": OKABE_ITO["vermillion"],
    "anomaly_weighted": OKABE_ITO["vermillion"],
    "tau_t": OKABE_ITO["green"],
    "tau_hat": OKABE_ITO["purple"],
    "mismatch": OKABE_ITO["orange"],
    "total": OKABE_ITO["black"],
}


def set_paper_style() -> None:
    """ICRA/CoRL/Nature 系常用 matplotlib 风格: serif, 简洁 spine, 浅虚线 grid."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # Type 42 (TrueType) 让 PDF 在 Illustrator / Acrobat 里字体可编辑, IEEE/ACM 要求.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


class Run(NamedTuple):
    label: str
    # 列名 → 1D float array. 用 dict 而不是 pandas DataFrame 减少依赖 (系统 python3
    # 通常有 numpy 没 pandas, .venv 反过来——dict 在两边都能跑).
    cols: dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:  # type: ignore[override]
        # 兼容旧 run.df["col"] 风格的访问
        return self.cols[key]

    @property
    def n(self) -> int:
        return len(next(iter(self.cols.values())))


def _read_metrics_csv(path: pathlib.Path) -> dict[str, np.ndarray]:
    """读 metrics.csv → 列字典. 比 numpy.genfromtxt 健壮 (后者对空文件/缺失列处理不友好)."""
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    fieldnames = reader.fieldnames or []
    return {name: np.array([float(r[name]) for r in rows], dtype=np.float64) for name in fieldnames}


def load_runs(csv_dir: pathlib.Path) -> list[Run]:
    """加载 runs.

    Mode 1: csv_dir/metrics.csv 直接存在 → 单 run, label = csv_dir 名
    Mode 2: csv_dir 下有多个子目录, 每个子目录里有 metrics.csv → 多 run 对比
    Mode 3: csv_dir 下有多个 metrics_<tag>.csv (扁平命名的消融) → 多 run 对比,
            label = <tag>. 这种适合手动重命名好几个 run 丢同一个目录里对比.
    """
    direct = csv_dir / "metrics.csv"
    if direct.exists():
        return [Run(label=csv_dir.name, cols=_read_metrics_csv(direct))]

    runs: list[Run] = []
    for child in sorted(csv_dir.iterdir()):
        if child.is_dir() and (child / "metrics.csv").exists():
            runs.append(Run(label=child.name, cols=_read_metrics_csv(child / "metrics.csv")))
    if runs:
        return runs

    # Mode 3: 扁平的 metrics_<tag>.csv
    for path in sorted(csv_dir.glob("metrics_*.csv")):
        tag = path.stem[len("metrics_"):] or path.stem
        runs.append(Run(label=tag, cols=_read_metrics_csv(path)))
    return runs


def ema(values: np.ndarray, window: int) -> np.ndarray:
    """EMA 平滑. window=1 时是 no-op. 比 rolling mean 好——没有 boundary NaN, 早期点没消失."""
    if window <= 1:
        return values.astype(np.float64)
    alpha = 2.0 / (window + 1)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _format_step_axis(ax: plt.Axes) -> None:
    """x 轴用 k 格式 (10000 → 10k), 论文里更紧凑."""
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}"
    ))


# Ablation 里某些 run 与另一个 run 在某些指标上**架构性等价**——画图时给它们一个
# 视觉提示 (虚线 + 灰色 + 半透明), 避免审稿人看到两条 bit-identical 曲线产生疑虑.
# 当前已知的等价对: aux0 ≡ no_prior 在所有 force/* 指标 (zero-init fc_out +
# stop_gradient + aux_weight=0 → ForcePrior 永远输出 0). 详见 longvla.py.
REDUNDANT_LABELS = {"aux0", "longvla_handover_aux0"}


def _run_style(label: str, default_color=None) -> dict:
    """给定 run label, 返回 matplotlib plot kwargs.

    REDUNDANT_LABELS 里的 run 用灰色虚线+半透明; 其他 run 走 matplotlib 默认配色循环.
    """
    if label in REDUNDANT_LABELS:
        return {"color": "0.55", "linestyle": "--", "alpha": 0.7, "linewidth": 1.2}
    if default_color is not None:
        return {"color": default_color}
    return {}


def _save_fig(fig: plt.Figure, out_path: pathlib.Path) -> None:
    """同名 .pdf (paper) + .png (slack 预览) 各存一份."""
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))


def plot_total_loss(runs: list[Run], out_path: pathlib.Path, smooth_window: int) -> None:
    """Fig 1: 总 loss 分解.
    (a) flow vs anomaly_weighted 同图, log y, 看二者量级差
    (b) anomaly_weighted / total 比例, linear, 看 aux 贡献占比
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    # (a) 分项 loss
    ax = axes[0]
    for run in runs:
        steps = run["step"]
        flow_raw = run["loss/flow"]
        anom_raw = run["loss/anomaly_weighted"]
        flow = ema(flow_raw, smooth_window)
        anom = ema(anom_raw, smooth_window)
        total = flow + anom

        if len(runs) == 1:
            # 单 run: 显示三条 + 半透明 raw
            ax.plot(steps, flow_raw, color=SEMANTIC_COLORS["flow"], alpha=0.18, linewidth=0.7)
            ax.plot(steps, anom_raw, color=SEMANTIC_COLORS["anomaly_weighted"], alpha=0.18, linewidth=0.7)
            ax.plot(steps, flow, label=r"$\mathcal{L}_{\mathrm{flow}}$",
                    color=SEMANTIC_COLORS["flow"])
            ax.plot(steps, anom, label=r"$\lambda \,\mathcal{L}_{\mathrm{anomaly}}$",
                    color=SEMANTIC_COLORS["anomaly_weighted"], linestyle="--")
            ax.plot(steps, total, label="Total",
                    color=SEMANTIC_COLORS["total"], linewidth=1.0, alpha=0.55)
        else:
            ax.plot(steps, total, label=run.label, **_run_style(run.label))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("(a) Loss decomposition" if len(runs) == 1 else "(a) Total loss across runs")
    ax.legend(loc="upper right")
    _format_step_axis(ax)

    # (b) aux 占比
    ax = axes[1]
    for run in runs:
        steps = run["step"]
        flow = run["loss/flow"]
        anom = run["loss/anomaly_weighted"]
        total = flow + anom
        ratio = anom / np.maximum(total, 1e-12)
        ratio_smooth = ema(ratio, smooth_window)
        if len(runs) == 1:
            ax.plot(steps, ratio, color=SEMANTIC_COLORS["anomaly_weighted"], alpha=0.2, linewidth=0.7)
            ax.plot(steps, ratio_smooth, color=SEMANTIC_COLORS["anomaly_weighted"])
        else:
            ax.plot(steps, ratio_smooth, label=run.label, **_run_style(run.label))
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\lambda\,\mathcal{L}_{\mathrm{anomaly}} / \mathcal{L}_{\mathrm{total}}$")
    ax.set_title("(b) Auxiliary loss share")
    ax.set_ylim(bottom=0)
    if len(runs) > 1:
        ax.legend(loc="upper right")
    _format_step_axis(ax)

    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_force_dynamics(runs: list[Run], out_path: pathlib.Path, smooth_window: int) -> None:
    """Fig 2: ForcePrior 学习动态.
    (a) tau_t (measured) vs tau_hat (predicted) 同图, 直接看 prior 追赶速度
    (b) mismatch = |tau_t - tau_hat|, 看残差衰减
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    # (a) tau magnitudes
    ax = axes[0]
    if len(runs) == 1:
        run = runs[0]
        steps = run["step"]
        tau_t = ema(run["force/tau_t_abs_mean"], smooth_window)
        tau_hat = ema(run["force/tau_hat_abs_mean"], smooth_window)
        ax.plot(steps, tau_t, label=r"$\overline{|\tau_t|}$ measured",
                color=SEMANTIC_COLORS["tau_t"])
        ax.plot(steps, tau_hat, label=r"$\overline{|\hat{\tau}|}$ predicted",
                color=SEMANTIC_COLORS["tau_hat"], linestyle="--")
    else:
        # 多 run 对比时只画 tau_hat (tau_t 是数据固有的, 不随 run 变, 比也没意义).
        for run in runs:
            steps = run["step"]
            tau_hat = ema(run["force/tau_hat_abs_mean"], smooth_window)
            ax.plot(steps, tau_hat, label=run.label, **_run_style(run.label))
        # 用第一个 run 的 tau_t 画灰色参考线
        run0 = runs[0]
        tau_t_ref = ema(run0["force/tau_t_abs_mean"], smooth_window)
        ax.plot(run0["step"], tau_t_ref, color="gray",
                linestyle=":", linewidth=1.0, label=r"$\overline{|\tau_t|}$ ref", alpha=0.7)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Mean $|\tau|$ (N$\cdot$m)")
    ax.set_title("(a) Force magnitude tracking")
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0)
    _format_step_axis(ax)

    # (b) mismatch
    ax = axes[1]
    for run in runs:
        steps = run["step"]
        mis_raw = run["force/mismatch_abs_mean"]
        mis = ema(mis_raw, smooth_window)
        if len(runs) == 1:
            ax.plot(steps, mis_raw, color=SEMANTIC_COLORS["mismatch"], alpha=0.2, linewidth=0.7)
            ax.plot(steps, mis, color=SEMANTIC_COLORS["mismatch"])
        else:
            ax.plot(steps, mis, label=run.label, **_run_style(run.label))
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\overline{|\tau_t - \hat{\tau}|}$ (N$\cdot$m)")
    ax.set_title("(b) Force prediction residual")
    ax.set_ylim(bottom=0)
    if len(runs) > 1:
        ax.legend(loc="upper right")
    _format_step_axis(ax)

    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_summary_4panel(runs: list[Run], out_path: pathlib.Path, smooth_window: int) -> None:
    """Fig 3: 4 panel 综合视图, 适合放 main paper 的 training analysis section."""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6))

    panels = [
        ("loss/flow", "(a) Flow matching loss",
         r"$\mathcal{L}_{\mathrm{flow}}$", True, SEMANTIC_COLORS["flow"]),
        ("loss/anomaly_mse", "(b) Anomaly MSE",
         r"$\mathcal{L}_{\mathrm{anomaly}}$", True, SEMANTIC_COLORS["anomaly_mse"]),
        ("force/tau_hat_abs_mean", "(c) Predicted force magnitude",
         r"$\overline{|\hat{\tau}|}$ (N$\cdot$m)", False, SEMANTIC_COLORS["tau_hat"]),
        ("force/mismatch_abs_mean", "(d) Force residual",
         r"$\overline{|\tau_t - \hat{\tau}|}$ (N$\cdot$m)", False, SEMANTIC_COLORS["mismatch"]),
    ]

    for ax, (col, title, ylabel, log_y, single_color) in zip(axes.flat, panels):
        for run in runs:
            steps = run["step"]
            vals_raw = run[col]
            vals = ema(vals_raw, smooth_window)
            if len(runs) == 1:
                ax.plot(steps, vals_raw, color=single_color, alpha=0.18, linewidth=0.7)
                ax.plot(steps, vals, color=single_color)
            else:
                ax.plot(steps, vals, label=run.label, **_run_style(run.label))

        # (c) 上叠 tau_t 参考线 (single run 时)
        if col == "force/tau_hat_abs_mean" and len(runs) == 1:
            tau_t = ema(runs[0]["force/tau_t_abs_mean"], smooth_window)
            ax.plot(runs[0]["step"], tau_t,
                    label=r"$\overline{|\tau_t|}$ target",
                    color=SEMANTIC_COLORS["tau_t"], linestyle=":", linewidth=1.2)
            ax.legend(loc="lower right")

        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        if len(runs) > 1:
            ax.legend(loc="best")
        _format_step_axis(ax)

    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def print_final_stats(runs: list[Run]) -> None:
    """打印每个 run 最后 5% steps 的均值——论文 'final' 数字一般取尾段平均, 不取最后一点."""
    print("\nFinal-window statistics (last 5% of training):")
    print(f"  {'run':<35} {'flow':>10} {'anom_mse':>12} {'|tau_hat|':>11} {'mismatch':>10}")
    print("  " + "-" * 80)
    for run in runs:
        n = run.n
        tail_start = max(0, n - n // 20)
        print(f"  {run.label:<35} "
              f"{run['loss/flow'][tail_start:].mean():>10.4f} "
              f"{run['loss/anomaly_mse'][tail_start:].mean():>12.4f} "
              f"{run['force/tau_hat_abs_mean'][tail_start:].mean():>11.4f} "
              f"{run['force/mismatch_abs_mean'][tail_start:].mean():>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-dir", type=pathlib.Path,
                        default=pathlib.Path(__file__).parent / "csv_logs",
                        help="Dir with metrics.csv (single) or subdirs each with metrics.csv (multi-run)")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=pathlib.Path(__file__).parent / "figures",
                        help="Output dir for .pdf / .png")
    parser.add_argument("--smooth", type=int, default=5,
                        help="EMA smoothing window (1 = off). With log_interval=100 and "
                             "smooth=5, effective smoothing horizon ≈ 500 steps.")
    args = parser.parse_args()

    set_paper_style()

    runs = load_runs(args.csv_dir)
    if not runs:
        raise SystemExit(f"No metrics.csv found under {args.csv_dir}")

    print(f"Loaded {len(runs)} run(s) from {args.csv_dir}:")
    for r in runs:
        steps = r["step"]
        print(f"  {r.label}: {r.n} rows, "
              f"steps {int(steps.min())}-{int(steps.max())}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def emit(group: list[Run], out_dir: pathlib.Path, tag: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_total_loss(group, out_dir / "fig_total_loss", args.smooth)
        plot_force_dynamics(group, out_dir / "fig_force_dynamics", args.smooth)
        plot_summary_4panel(group, out_dir / "fig_summary", args.smooth)
        print(f"  [{tag}] -> {out_dir}/fig_*.pdf")

    # 每个 run 单独出一套图 (single-run mode: 带 raw + 分解).
    for run in runs:
        emit([run], args.out_dir / run.label, run.label)

    # 多 run 时再出一套对比图.
    if len(runs) > 1:
        emit(runs, args.out_dir / "comparison", "comparison")

    print_final_stats(runs)


if __name__ == "__main__":
    main()
