#!/usr/bin/env python3
"""True vs predicted age scatter for every entry in Table 1.

One panel per (TabPFN variant, encoder) cell, in Table 1's order. Each panel pools
the 5 seeds at that cell's validation-selected learning rate -- the same selection
rule the table itself uses -- so a panel shows the same model the table scored.

The diagonal is y = x. The orange line is the mean prediction in each 5-year bin of
true age: where it is flatter than the diagonal, the model is regressing toward the
training mean, which is exactly the failure imbalanced regression is about.

Nothing is re-run; this reads results/preds/*.csv.

    python3 plot_scatter_table1.py
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "results", "preds")
FIGS = os.path.join(HERE, "figures")

# dataviz reference palette, categorical slots 1 and 2 (all-pairs safe at <= 3 series).
C_POINTS, C_TREND = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID, DIAG = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#9a9892"

# (row label, encoder label, dataset key, arm) in Table 1 order.
CELLS = [
    ("Frozen", "E1", "ptbxl_l1_enc", "zeroshot"),
    ("Frozen", "E2", "ptbxl_tabpfn_enc", "zeroshot"),
    ("Uniform fine-tuning", "E1", "ptbxl_l1_enc", "ft_uniform"),
    ("Uniform fine-tuning", "E2", "ptbxl_tabpfn_enc", "ft_uniform"),
    ("Gaussian fine-tuning", "E1", "ptbxl_l1_enc", "ft_gauss"),
    ("Gaussian fine-tuning", "E2", "ptbxl_tabpfn_enc", "ft_gauss"),
    ("Gaussian fine-tuning", "E2.1", "ptbxl_enc_e21", "ft_gauss"),
    ("Gaussian + Uniform", "E1", "ptbxl_l1_enc", "ft_mixed"),
    ("Gaussian + Uniform", "E2", "ptbxl_tabpfn_enc", "ft_mixed"),
    ("Gaussian + Uniform", "E2.2", "ptbxl_enc_e22", "ft_mixed"),
]
BEST = ("Gaussian fine-tuning", "E2")
PROTOCOL = "dir"


def val_selected_tags(dataset: str, arm: str) -> dict[int, str]:
    """Per seed, the learning-rate tag with the lowest validation RMSE."""
    rows = []
    for p in glob.glob(os.path.join(HERE, "results", f"{dataset}_*.jsonl")):
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    df = df[(df.protocol == PROTOCOL) & (df.arm == arm) &
            (df.tag.str.startswith("dir_") == (PROTOCOL == "dir"))]
    if df.empty or "val_rmse" not in df:
        return {}
    best = df.sort_values("val_rmse").groupby("seed", as_index=False).first()
    return dict(zip(best.seed, best.tag))


def load_cell(dataset: str, arm: str) -> tuple[np.ndarray, np.ndarray, int]:
    tags = val_selected_tags(dataset, arm)
    if not tags:
        tags = {s: f"{'dir_' if PROTOCOL == 'dir' else ''}baseline" for s in range(5)}
    yt, yp = [], []
    for seed, tag in sorted(tags.items()):
        f = os.path.join(PRED, f"{dataset}_{tag}_{PROTOCOL}_{arm}_s{seed}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        yt.append(d.y_true.to_numpy())
        yp.append(d.y_pred.to_numpy())
    if not yt:
        return np.array([]), np.array([]), 0
    return np.concatenate(yt), np.concatenate(yp), len(yt)


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(21.5, 9.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    lo, hi = 10, 95
    for ax, (variant, enc, ds, arm) in zip(axes.ravel(), CELLS):
        yt, yp, nseed = load_cell(ds, arm)
        ax.set_facecolor(SURFACE)
        if yt.size == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color=INK_2)
            continue

        ax.plot([lo, hi], [lo, hi], color=DIAG, linewidth=1.4,
                linestyle=(0, (5, 4)), zorder=1)
        ax.scatter(yt, yp, s=3.5, color=C_POINTS, alpha=0.055,
                   linewidths=0, zorder=2, rasterized=True)

        # Conditional mean of the prediction in each 5-year bin of true age.
        edges = np.arange(15, 91, 5.0)
        idx = np.digitize(yt, edges) - 1
        cx, cy = [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() >= 25:
                cx.append(0.5 * (edges[b] + edges[b + 1]))
                cy.append(float(yp[m].mean()))
        ax.plot(cx, cy, color=C_TREND, linewidth=2.0, zorder=3,
                solid_capstyle="round")

        mae = float(np.abs(yp - yt).mean())
        is_best = (variant, enc) == BEST
        ax.set_title(f"{variant}\n{enc}" + ("   ★ best" if is_best else ""),
                     fontsize=10.5, color=INK,
                     fontweight="bold" if is_best else "normal", pad=8)
        ax.text(0.04, 0.955, f"MAE {mae:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, color=INK,
                fontweight="bold" if is_best else "normal")
        ax.text(0.04, 0.885, f"{nseed} seeds pooled", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=INK_2)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
        if is_best:
            for s in ("left", "bottom"):
                ax.spines[s].set_color(INK)
                ax.spines[s].set_linewidth(1.6)

    for ax in axes[1, :]:
        ax.set_xlabel("True age (years)", fontsize=10, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted age (years)", fontsize=10, color=INK_2)

    handles = [
        Line2D([], [], color=C_POINTS, marker="o", linestyle="none",
               markersize=7, label="Test prediction"),
        Line2D([], [], color=C_TREND, linewidth=2.4,
               label="Mean prediction per 5-year bin"),
        Line2D([], [], color=DIAG, linewidth=1.6, linestyle=(0, (5, 4)),
               label="Perfect prediction (y = x)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.962),
               ncol=3, frameon=False, fontsize=10.5, labelcolor=INK_2)
    fig.suptitle("True vs predicted age, every Table 1 cell  (PTB-XL, balanced test set)",
                 fontsize=16, fontweight="bold", color=INK, x=0.045, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    # aspect="equal" shrinks each panel inside its slot, so the default row gap
    # leaves the lower titles sitting on the upper row's frame.
    fig.subplots_adjust(hspace=0.42)
    out = os.path.join(FIGS, "02_true_vs_pred_table1.png")
    fig.savefig(out, dpi=165, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
