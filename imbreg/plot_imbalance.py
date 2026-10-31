#!/usr/bin/env python3
"""Target-distribution plots for the three tabular imbalanced-regression datasets.

Shows *why* these count as imbalanced regression: each target bin is coloured by
how many training points fall in it, using this repo's own shot-region thresholds
(many >= 100, medium 20-99, few < 20 -- metrics.MANY_SHOT_MIN / FEW_SHOT_MAX).
The few-shot bins are the ones every method in this study is trying to fix.

Nothing is re-run; this reads the dataset loaders directly.

    python3 plot_imbalance.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import data as data_mod
from metrics import MANY_SHOT_MIN, FEW_SHOT_MAX

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")

# Ordinal blue ramp (dataviz reference palette). Darker = more training data.
# On a light surface the ordinal rule forbids anything lighter than step 250.
C_MANY, C_MEDIUM, C_FEW = "#0d366b", "#2a78d6", "#86b6ef"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

DATASETS = [("abalone", "Abalone", "Rings (age proxy)"),
            ("communities", "Communities & Crime", "Violent crimes per capita"),
            ("wine_white", "Wine Quality (white)", "Quality score")]


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    fig.patch.set_facecolor(SURFACE)

    for ax, (key, title, xlabel) in zip(axes, DATASETS):
        ds = data_mod.LOADERS[key]()
        y = np.asarray(ds.y, float)
        bw = ds.bin_width

        lo = np.floor(y.min() / bw) * bw
        hi = np.ceil(y.max() / bw) * bw
        edges = np.arange(lo, hi + bw, bw)
        counts, _ = np.histogram(y, bins=edges)

        colors = [C_MANY if c >= MANY_SHOT_MIN else
                  (C_MEDIUM if c >= FEW_SHOT_MAX else C_FEW) for c in counts]
        ax.bar(edges[:-1], counts, width=bw * 0.9, align="edge",
               color=colors, linewidth=0)

        n = len(y)
        pct = {}
        for name, m in (("many", counts >= MANY_SHOT_MIN),
                        ("medium", (counts >= FEW_SHOT_MAX) & (counts < MANY_SHOT_MIN)),
                        ("few", counts < FEW_SHOT_MAX)):
            pct[name] = 100.0 * counts[m].sum() / n

        # Headroom so the annotation never lands on a bar (wine's mode bar is tall).
        ax.set_ylim(0, counts.max() * 1.34)

        ax.set_facecolor(SURFACE)
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=10)
        ax.set_xlabel(xlabel, fontsize=10, color=INK_2)
        ax.set_ylabel("Training samples in bin", fontsize=10, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)

        n_few = int((counts < FEW_SHOT_MAX).sum())
        ax.text(0.97, 0.95,
                f"n = {n:,}   {len(counts)} bins\n"
                f"{n_few} few-shot bins hold {pct['few']:.1f}% of data\n"
                f"many {pct['many']:.1f}%  ·  medium {pct['medium']:.1f}%  ·  few {pct['few']:.1f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=INK_2, linespacing=1.5)

    handles = [Patch(facecolor=C_MANY, label=f"many-shot  (>= {MANY_SHOT_MIN} per bin)"),
               Patch(facecolor=C_MEDIUM, label=f"medium-shot  ({FEW_SHOT_MAX}-{MANY_SHOT_MIN - 1})"),
               Patch(facecolor=C_FEW, label=f"few-shot  (< {FEW_SHOT_MAX})")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.062, 0.975),
               ncol=3, frameon=False, fontsize=10, labelcolor=INK_2)

    fig.suptitle("Target imbalance in the three tabular regression datasets",
                 fontsize=15, fontweight="bold", color=INK, x=0.062, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(FIGS, "01_dataset_imbalance.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
