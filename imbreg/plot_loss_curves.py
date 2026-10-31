#!/usr/bin/env python3
"""Figure: did Gaussian-context fine-tuning reduce TabPFN's training error?

Two rows, because the two quantities answer different questions and only one of them
is comparable between the arms:

  row 1  TRAINING loss per epoch. Answers "did each arm learn?" -- read each line
         against its own starting point. The two lines are NOT comparable with each
         other: the Gaussian arm computes its loss on episodes whose context is
         deliberately skewed, so it is solving a harder problem on every step and sits
         higher for reasons unrelated to how well it is learning.

  row 2  VALIDATION MSE per epoch, computed on the same held-out set for both arms.
         This one IS directly comparable, and it is what early stopping acts on.

Colours are slots 1 and 2 of the reference categorical palette, in fixed order
(blue = uniform, orange = Gaussian). The palette validator ships as a Node script and
no Node runtime is installed here, so the documented slots are used unchanged rather
than new colours being invented.
"""
from __future__ import annotations
import collections, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = [("abalone", "Abalone"), ("communities", "Communities"),
            ("wine_white", "Wine Quality")]
ARMS = [("ft_uniform", "Uniform context", "#2a78d6"),
        ("ft_gauss", "Gaussian context", "#eb6834")]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#5c5c58", "#e4e4e0"


def load():
    rows = [json.loads(l) for p in glob.glob(os.path.join(HERE, "results", "curves_*_losscurve.jsonl"))
            for l in open(p) if l.strip()]
    runs = collections.defaultdict(list)
    for r in rows:
        if r["event"] == "epoch":
            runs[(r["dataset"], r["arm"], r["seed"])].append(r)
    out = collections.defaultdict(list)
    for (ds, arm, seed), v in runs.items():
        v = sorted(v, key=lambda x: x["train/epoch"])
        out[(ds, arm)].append({
            "train": [x["train/mean_loss"] for x in v if "train/mean_loss" in x],
            "val": [x["val/MSE"] for x in v if "val/MSE" in x]})
    return out


def band(ax, series, color, label):
    """Mean across seeds with a min-max band, truncated to where >= 2 seeds remain."""
    if not series:
        return
    n = max(len(s) for s in series)
    xs, mean, lo, hi = [], [], [], []
    for i in range(n):
        vals = [s[i] for s in series if len(s) > i]
        if len(vals) < 2:
            break
        xs.append(i + 1); mean.append(np.mean(vals))
        lo.append(np.min(vals)); hi.append(np.max(vals))
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(xs, mean, color=color, linewidth=2, label=label, solid_capstyle="round")
    best = int(np.argmin(mean))
    ax.plot([xs[best]], [mean[best]], marker="o", markersize=8, color=color,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)


def main():
    data = load()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.33, wspace=0.26, top=0.84, bottom=0.11, left=0.07, right=0.98)

    for col, (ds, nice) in enumerate(DATASETS):
        for row, key in enumerate(("train", "val")):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            for arm, label, color in ARMS:
                band(ax, [r[key] for r in data.get((ds, arm), []) if r[key]], color, label)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=9, length=0)
            if row == 0:
                ax.set_title(nice, color=INK, fontsize=12, fontweight="bold", pad=8)
            if row == 1:
                ax.set_xlabel("Epoch", color=MUTED, fontsize=10)
            if col == 0:
                ax.set_ylabel("Training loss\n(bar-distribution NLL)" if row == 0
                              else "Validation MSE", color=INK, fontsize=10)

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.07, 0.95), ncol=2,
               frameon=False, fontsize=11, labelcolor=INK)
    fig.text(0.07, 0.965, "Does Gaussian-context fine-tuning reduce TabPFN-2.5's error?",
             fontsize=14, fontweight="bold", color=INK, ha="left")
    fig.text(0.98, 0.955,
             "Mean of 3 seeds; band = min–max. Dot = best epoch.\n"
             "Top row: compare each line to its OWN start, not to the other line.",
             fontsize=8.5, color=MUTED, ha="right", va="top")

    out = os.path.join(HERE, "results", "loss_curves.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
