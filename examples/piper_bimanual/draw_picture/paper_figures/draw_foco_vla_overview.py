#!/usr/bin/env python3
"""Draw the main RA-L overview figure for FoCo-VLA.

The figure is intentionally compact for a two-column robotics paper:
inputs on the left, the Force-as-Context module in the center, and the
VLA action policy on the right.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
from matplotlib import patches


OUT_DIR = pathlib.Path(__file__).resolve().parent


COLORS = {
    "blue": "#3B82F6",
    "blue_light": "#EFF6FF",
    "green": "#22C55E",
    "green_light": "#F0FDF4",
    "purple": "#8B5CF6",
    "purple_light": "#F5F3FF",
    "orange": "#F59E0B",
    "orange_light": "#FFFBEB",
    "red": "#EF4444",
    "red_light": "#FEF2F2",
    "gray": "#374151",
    "gray_light": "#F9FAFB",
    "line": "#6B7280",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def rounded_box(ax, x, y, w, h, label, *, fc, ec, lw=1.2, fontsize=8.2, weight="bold"):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#111827",
        linespacing=1.15,
    )
    return box


def arrow(ax, x1, y1, x2, y2, *, color=None, lw=1.25, style="-|>"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color or COLORS["line"],
            lw=lw,
            shrinkA=2,
            shrinkB=2,
            mutation_scale=10,
        ),
    )


def token_row(ax, x, y, n, color, *, dx=0.037, size=0.022, alpha=1.0):
    for i in range(n):
        ax.add_patch(
            patches.Rectangle(
                (x + i * dx, y),
                size,
                size,
                linewidth=0.55,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
            )
        )
    ax.text(x + n * dx + 0.01, y + size / 2, "...", va="center", ha="left", fontsize=8, color=COLORS["gray"])


def small_signal(ax, x, y, color, label):
    ax.plot([x, x + 0.035, x + 0.07, x + 0.105], [y, y + 0.02, y - 0.006, y + 0.014], color=color, lw=1.5)
    ax.text(x + 0.12, y + 0.006, label, va="center", ha="left", fontsize=7.5, color=COLORS["gray"])


def draw() -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.965,
        "FoCo-VLA: Force-as-Context Vision-Language-Action Policies",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#111827",
    )

    # Input column.
    rounded_box(ax, 0.035, 0.72, 0.18, 0.15, "Multi-view RGB", fc=COLORS["blue_light"], ec=COLORS["blue"], fontsize=8.3)
    token_row(ax, 0.065, 0.675, 5, COLORS["blue"], dx=0.028, size=0.018)
    ax.text(0.055, 0.647, "SigLIP vision tokens", fontsize=7.0, color=COLORS["gray"])

    rounded_box(ax, 0.035, 0.455, 0.18, 0.14, "Instruction", fc=COLORS["green_light"], ec=COLORS["green"], fontsize=8.3)
    token_row(ax, 0.065, 0.412, 5, COLORS["green"], dx=0.028, size=0.018)
    ax.text(0.055, 0.384, "prompt + state tokens", fontsize=7.0, color=COLORS["gray"])

    rounded_box(ax, 0.035, 0.18, 0.18, 0.15, "Force history", fc=COLORS["purple_light"], ec=COLORS["purple"], fontsize=8.3)
    small_signal(ax, 0.058, 0.13, COLORS["purple"], r"$q, \dot{q}, \tau$ over $W$ frames")

    # FoCo center module.
    rounded_box(
        ax,
        0.31,
        0.565,
        0.23,
        0.235,
        "",
        fc=COLORS["orange_light"],
        ec=COLORS["orange"],
        fontsize=8.2,
    )
    ax.text(0.425, 0.75, "Force prior", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#111827")
    ax.text(0.335, 0.71, "task-conditioned expected force", fontsize=7.0, color=COLORS["gray"])
    token_row(ax, 0.348, 0.625, 6, COLORS["orange"], dx=0.026, size=0.018)
    ax.text(0.348, 0.598, r"$\hat{\tau}_t$ per joint", fontsize=7.0, color=COLORS["gray"])

    rounded_box(
        ax,
        0.31,
        0.18,
        0.23,
        0.245,
        "",
        fc=COLORS["purple_light"],
        ec=COLORS["purple"],
        fontsize=8.2,
    )
    ax.text(0.425, 0.375, "Force tokenizer", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#111827")
    ax.text(0.333, 0.342, "per-joint history tokens", fontsize=7.0, color=COLORS["gray"])
    token_row(ax, 0.348, 0.265, 7, COLORS["purple"], dx=0.025, size=0.018)
    ax.text(0.348, 0.238, "joint-level context tokens", fontsize=7.0, color=COLORS["gray"])

    rounded_box(
        ax,
        0.58,
        0.33,
        0.18,
        0.28,
        "",
        fc=COLORS["gray_light"],
        ec="#9CA3AF",
        fontsize=8.3,
    )
    ax.text(0.67, 0.575, "Context tokens", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#111827")
    token_row(ax, 0.605, 0.515, 6, COLORS["blue"], dx=0.022, size=0.016)
    token_row(ax, 0.605, 0.455, 6, COLORS["green"], dx=0.022, size=0.016)
    token_row(ax, 0.605, 0.395, 7, COLORS["purple"], dx=0.022, size=0.016)
    ax.text(0.715, 0.52, "vision", fontsize=6.5, color=COLORS["gray"])
    ax.text(0.715, 0.46, "language", fontsize=6.5, color=COLORS["gray"])
    ax.text(0.715, 0.40, "force", fontsize=6.5, color=COLORS["gray"])

    # VLA policy column.
    rounded_box(
        ax,
        0.80,
        0.54,
        0.16,
        0.145,
        "PaliGemma\nVLA backbone",
        fc=COLORS["blue_light"],
        ec=COLORS["blue"],
        fontsize=8.0,
    )
    ax.text(0.826, 0.512, "prefix attention", fontsize=6.7, color=COLORS["gray"])

    rounded_box(
        ax,
        0.80,
        0.285,
        0.16,
        0.13,
        "Action expert",
        fc=COLORS["orange_light"],
        ec=COLORS["orange"],
        fontsize=8.2,
    )
    ax.text(0.817, 0.257, "flow matching", fontsize=6.7, color=COLORS["gray"])
    token_row(ax, 0.815, 0.205, 6, COLORS["purple"], dx=0.023, size=0.017)
    ax.text(0.815, 0.172, r"$T$-step dual-arm actions", fontsize=7.0, color=COLORS["gray"])

    # Arrows.
    arrow(ax, 0.215, 0.745, 0.575, 0.525, color=COLORS["blue"])
    arrow(ax, 0.215, 0.49, 0.575, 0.465, color=COLORS["green"])
    arrow(ax, 0.215, 0.245, 0.307, 0.285, color=COLORS["purple"])
    arrow(ax, 0.423, 0.56, 0.423, 0.428, color=COLORS["orange"])
    arrow(ax, 0.54, 0.305, 0.58, 0.405, color=COLORS["purple"])
    arrow(ax, 0.76, 0.47, 0.80, 0.595)
    arrow(ax, 0.88, 0.54, 0.88, 0.415)
    arrow(ax, 0.88, 0.285, 0.88, 0.23, color=COLORS["orange"])

    # Compact legend.
    legend_y = 0.055
    legend = [
        (COLORS["blue"], "vision token"),
        (COLORS["green"], "language token"),
        (COLORS["purple"], "force/action token"),
        (COLORS["orange"], "predicted force"),
        (COLORS["red"], "residual contact signal"),
    ]
    x = 0.075
    for color, text in legend:
        ax.add_patch(patches.Rectangle((x, legend_y), 0.018, 0.018, facecolor=color, edgecolor=color, lw=0.5))
        ax.text(x + 0.024, legend_y + 0.009, text, va="center", ha="left", fontsize=7.1, color=COLORS["gray"])
        x += 0.17 if text != "residual contact signal" else 0.0

    fig.tight_layout(pad=0.15)
    return fig


def main() -> None:
    fig = draw()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"fig1_foco_vla_overview.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FoCo-VLA overview figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
