#!/usr/bin/env python3
"""Is the best model's predictive uncertainty calibrated?

Reads results/calibration/*.csv (written by calibration_experiment.py) for the best
Table 1 cell -- Gaussian fine-tuning on the E2 encoder -- and asks four questions:

  1. Does a larger sigma actually mean a larger error?  (the per-point view)
  2. Is sigma on the right SCALE, or only the right order?  (the calibration curve:
     within a bin of similar sigma, the realised RMSE should equal that sigma)
  3. Do the nominal intervals cover what they claim?  (+-1 sigma should hold 68.3%)
  4. Does the model know WHERE it is uncertain -- does sigma rise in the
     under-represented age tails that this whole study is about?

    python3 plot_calibration.py
"""
from __future__ import annotations
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "calibration")
FIGS = os.path.join(HERE, "figures")

C_1, C_2, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, GRID, DIAG = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#9a9892"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, pad=9)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_2)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)


def main() -> None:
    files = sorted(glob.glob(os.path.join(SRC, "*.csv")))
    if not files:
        raise SystemExit(f"no calibration CSVs in {SRC} -- run calibration_experiment.py first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    y, p, s = df.y_true.to_numpy(), df.y_pred.to_numpy(), df.sigma.to_numpy()
    err = np.abs(p - y)
    n_seeds = len(files)

    from scipy.stats import spearmanr
    rho = float(spearmanr(s, err).statistic)

    os.makedirs(FIGS, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(23.5, 5.4))
    fig.patch.set_facecolor(SURFACE)
    a1, a2, a3, a4 = axes

    # 1 -- per point
    a1.scatter(s, err, s=3.5, color=C_1, alpha=0.07, linewidths=0, rasterized=True)
    qs = np.quantile(s, np.linspace(0, 1, 13))
    cx, cy = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum() >= 30:
            cx.append(float(s[m].mean()))
            cy.append(float(err[m].mean()))
    a1.plot(cx, cy, color=C_2, linewidth=2.2, zorder=3, solid_capstyle="round")
    style(a1, "1 · Does a bigger σ mean a bigger error?",
          "Predicted σ (years)", "Absolute error (years)")
    a1.text(0.04, 0.95, f"Spearman ρ = {rho:.3f}\n{len(y):,} test points, {n_seeds} seeds",
            transform=a1.transAxes, ha="left", va="top", fontsize=9,
            color=INK, linespacing=1.6)

    # 2 -- calibration curve: realised RMSE within each sigma bin vs that bin's sigma
    bx, by = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum() >= 30:
            bx.append(float(s[m].mean()))
            by.append(float(np.sqrt(np.mean((p[m] - y[m]) ** 2))))
    lim = [0, max(max(bx), max(by)) * 1.08]
    a2.plot(lim, lim, color=DIAG, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
    a2.plot(bx, by, color=C_1, linewidth=2.2, marker="o", markersize=7,
            zorder=3, solid_capstyle="round")
    a2.set_xlim(lim)
    a2.set_ylim(lim)
    style(a2, "2 · Is σ on the right scale?",
          "Mean predicted σ in bin (years)", "Realised RMSE in bin (years)")
    ratio = float(np.mean(np.array(by) / np.array(bx)))
    a2.text(0.04, 0.95,
            f"Points above the dashed line are\nover-confident (σ too small).\n\n"
            f"mean RMSE/σ = {ratio:.2f}×",
            transform=a2.transAxes, ha="left", va="top", fontsize=9,
            color=INK_2, linespacing=1.6)

    # 3 -- interval coverage vs nominal
    levels = [(1.0, 68.27), (1.645, 90.0), (1.96, 95.0), (2.576, 99.0)]
    obs = [100.0 * float((err <= k * s).mean()) for k, _ in levels]
    nom = [v for _, v in levels]
    xpos = np.arange(len(levels))
    a3.bar(xpos - 0.2, nom, width=0.38, color=DIAG, linewidth=0, label="Nominal")
    a3.bar(xpos + 0.2, obs, width=0.38, color=C_1, linewidth=0, label="Observed")
    for i, (o, nv) in enumerate(zip(obs, nom)):
        a3.text(i + 0.2, o + 1.4, f"{o:.1f}", ha="center", fontsize=8.5, color=INK)
        a3.text(i - 0.2, nv + 1.4, f"{nv:.1f}", ha="center", fontsize=8.5, color=INK_2)
    a3.set_xticks(xpos)
    a3.set_xticklabels([f"±{k:g}σ" for k, _ in levels])
    # Headroom so neither the legend nor the note lands on a bar.
    a3.set_ylim(0, 140)
    a3.set_yticks([0, 20, 40, 60, 80, 100])
    style(a3, "3 · Do the intervals cover what they claim?", "Interval", "Coverage (%)")
    a3.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=2,
              loc="upper left", bbox_to_anchor=(0.0, 1.0))
    a3.text(0.0, 0.84, "Observed sits below nominal at every level: over-confident overall.",
            transform=a3.transAxes, ha="left", va="top",
            fontsize=8.5, color=INK_2, linespacing=1.5)

    # 4 -- sigma across the age range, against where the error actually is
    edges = np.arange(15, 91, 5.0)
    idx = np.digitize(y, edges) - 1
    ax_, s_m, e_m = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 30:
            ax_.append(0.5 * (edges[b] + edges[b + 1]))
            s_m.append(float(s[m].mean()))
            e_m.append(float(err[m].mean()))
    a4.plot(ax_, e_m, color=C_2, linewidth=2.2, marker="o", markersize=6,
            label="Actual mean |error|", solid_capstyle="round")
    a4.plot(ax_, s_m, color=C_1, linewidth=2.2, marker="o", markersize=6,
            label="Predicted σ", solid_capstyle="round")
    style(a4, "4 · Does it know WHERE it is uncertain?",
          "True age (years)", "Years")
    a4.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")

    # The headline failure: past 80 the error explodes while sigma does not move.
    m80 = y >= 80
    if m80.sum() > 50:
        a4.axvspan(80, 92, color=C_2, alpha=0.07, zorder=0)
        a4.text(0.97, 0.52,
                f"age > 80  (n={int(m80.sum())})\n"
                f"mean |error| {err[m80].mean():.1f} yr\n"
                f"mean σ        {s[m80].mean():.1f} yr\n"
                f"±1σ covers {100 * float((err[m80] <= s[m80]).mean()):.0f}%, not 68%",
                transform=a4.transAxes, ha="right", va="top",
                fontsize=8.5, color=INK, linespacing=1.55)

    fig.suptitle("Predictive uncertainty of the best model  ·  "
                 "Gaussian fine-tuning on encoder E2  (PTB-XL, balanced test set)",
                 fontsize=15, fontweight="bold", color=INK, x=0.028, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGS, "04_uncertainty_calibration.png")
    fig.savefig(out, dpi=165, facecolor=SURFACE)
    print("wrote", out)

    print(f"\nSpearman(sigma, |error|) = {rho:.4f}")
    print(f"mean RMSE/sigma over bins = {ratio:.3f}")
    for (k, nv), o in zip(levels, obs):
        print(f"  +-{k:g} sigma: observed {o:5.1f}%  nominal {nv:5.1f}%")


if __name__ == "__main__":
    main()
