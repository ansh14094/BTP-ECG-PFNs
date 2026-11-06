#!/usr/bin/env python3
"""Figures 1 and 2 for the paper, as single-column vector PDFs.

Fig 1  method schematic. Left: the target histogram with three Gaussian context
       weightings drawn from the actual prior used by the sampler. Right: the query
       distribution, which stays uniform. Makes no claim about results.
Fig 2  prediction against truth, binned, for the two headline comparisons. Shows
       whether the Gaussian arm pulls tail predictions less hard toward the mean.

Style follows paper_completion_plan.md section 3: 3.3 in wide, 8 pt, vector PDF.
The palette is #2a78d6 / #eb6834, checked with eval/check_palette.py -- that pair
separates by dE 27-36 under normal vision and all three dichromacies, the widest
margin available in the repo's palette. Line style differs as well, so the figures
survive greyscale printing.

    python3 eval/make_figures.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRED = os.path.join(ROOT, "imbreg", "results", "preds")
FIGS = os.path.join(ROOT, "paper", "figs")

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a1a", "#5a5a5a", "#d8d8d8"
W = 3.3

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed real glyphs, not Type-3
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.8)
    ax.set_axisbelow(True)


def ptbxl_ages() -> np.ndarray:
    d = pd.read_csv(os.path.join(ROOT, "data", "1000_timesteps.csv"))
    return d.age.to_numpy(float)


# --------------------------------------------------------------------- figure 1
def fig1() -> None:
    y = ptbxl_ages()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.5))

    # Left: what the context sees.
    a1.hist(y, bins=40, color="#c9d6e4", edgecolor="white", linewidth=0.3, density=True)
    xs = np.linspace(y.min(), y.max(), 400)
    std = y.std()
    # Three draws from the sampler's own prior: mu = quantile(y, U(.05,.95)),
    # sigma = std(y) * exp(U(log .15, log 1.5)).
    draws = [(np.quantile(y, 0.12), std * 0.30),
             (np.quantile(y, 0.50), std * 0.55),
             (np.quantile(y, 0.88), std * 0.22)]
    for k, ((mu, sg), ls) in enumerate(zip(draws, ["-", "--", ":"])):
        w = np.exp(-0.5 * ((xs - mu) / sg) ** 2)
        a1.plot(xs, w * a1.get_ylim()[1] * 0.92, color=BLUE, linewidth=1.0,
                linestyle=ls, label="context weight" if k == 0 else None)
    a1.set_title("Context: Gaussian", pad=3)
    a1.set_xlabel("target (age, years)", labelpad=1.5)
    a1.set_ylabel("density", labelpad=1.5)
    a1.set_yticks([])
    a1.text(0.03, 0.97, "3 draws", transform=a1.transAxes, ha="left", va="top",
            fontsize=6.5, color=MUTED)
    style(a1)

    # Right: what the query sees -- unchanged.
    a2.hist(y, bins=40, color="#c9d6e4", edgecolor="white", linewidth=0.3, density=True)
    lo, hi = a2.get_xlim()
    a2.plot([lo, hi], [a2.get_ylim()[1] * 0.55] * 2, color=ORANGE, linewidth=1.0)
    a2.set_xlim(lo, hi)
    a2.set_title("Query: uniform", pad=3)
    a2.set_xlabel("target (age, years)", labelpad=1.5)
    a2.set_yticks([])
    style(a2)

    fig.tight_layout(pad=0.3)
    out = os.path.join(FIGS, "fig1_method.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------- figure 2
def _pool(dataset: str, tag: str, protocol: str, arm: str):
    """Seed-pooled (y_true, y_pred), or None when the cell has not been run."""
    fs = sorted(glob.glob(os.path.join(
        PRED, f"{dataset}_{tag}_{protocol}_{arm}_s*.csv")))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d.y_true.to_numpy(float), d.y_pred.to_numpy(float)


def _binned(ax, y, p, color, ls, label, nbins=12):
    qs = np.quantile(y, np.linspace(0, 1, nbins + 1))
    qs = np.unique(qs)
    cx, cm, cs = [], [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (y >= lo) & (y < hi)
        if m.sum() < 20:
            continue
        cx.append(y[m].mean()); cm.append(p[m].mean()); cs.append(p[m].std())
    cx, cm, cs = map(np.asarray, (cx, cm, cs))
    ax.fill_between(cx, cm - cs, cm + cs, color=color, alpha=0.13, linewidth=0)
    ax.plot(cx, cm, color=color, linewidth=1.1, linestyle=ls, label=label)


def fig2() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.75))
    notes = []

    # Panel A -- abalone, prior-level stock vs Gaussian.
    for tag, arm, lab, c, ls in [("ticl_base", "ticl_zeroshot", "zero-shot", BLUE, "-"),
                                 ("ticl_lr1e-4", "ticl_ft_gauss", "Gaussian", ORANGE, "--")]:
        got = _pool("abalone", tag, "dir", arm)
        if got:
            _binned(a1, *got, c, ls, lab)
    lim = [1, 25]
    a1.plot(lim, lim, color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_title("Abalone (fine-tuned)", pad=3)
    a1.set_xlabel("true", labelpad=1.5); a1.set_ylabel("predicted", labelpad=1.5)
    style(a1); a1.legend(frameon=False, loc="upper left", handlelength=1.4,
                         borderpad=0.1, labelspacing=0.25)

    # Panel B -- PTB-XL, E3 under the stock and Gaussian bodies. Prefer the clean
    # official split; fall back to the DIR runs and say so in the caption.
    tag, proto = "clean_matrix", "fixed"
    if not glob.glob(os.path.join(PRED, f"ptbxl_enc_e3_{tag}_{proto}_*.csv")):
        tag, proto = "dir_matrix", "dir"
        notes.append("PTB-XL panel uses the DIR split; clean-split runs not yet done")
    for arm, lab, c, ls in [("ticl_zeroshot", "stock", BLUE, "-"),
                            ("ticl_prior_gauss", "Gaussian", ORANGE, "--")]:
        got = _pool("ptbxl_enc_e3", tag, proto, arm)
        if got:
            _binned(a2, *got, c, ls, lab)
    lim2 = [15, 90]
    a2.plot(lim2, lim2, color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)
    a2.set_xlim(lim2); a2.set_ylim(lim2)
    a2.set_title("PTB-XL, E3 (prior only)", pad=3)
    a2.set_xlabel("true", labelpad=1.5)
    style(a2); a2.legend(frameon=False, loc="upper left", handlelength=1.4,
                         borderpad=0.1, labelspacing=0.25)

    fig.tight_layout(pad=0.3)
    out = os.path.join(FIGS, "fig2_pred_vs_true.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out, *(f"\n  NOTE: {n}" for n in notes))


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig1()
    fig2()
