#!/usr/bin/env python3
"""How the Gaussian context sampler works, drawn step by step.

Every panel is produced by calling the real sampler in biased_split.py -- the
(mu, sigma) values shown are the ones it actually drew, captured through its
`_TRACE` hook, not a reimplementation.

The mechanism, for one fine-tuning episode:

  1. The episode gets a pool of (X, y).
  2. The QUERY points are drawn first and uniformly, so the thing the model is
     scored on keeps the pool's natural target distribution. This is what makes
     the comparison against uniform context fair.
  3. A centre mu and a width sigma are drawn fresh:
         mu    = quantile(y_remaining, U(0.05, 0.95))
         sigma = std(y_remaining) * exp(U(log 0.15, log 1.5))
     so sigma is log-uniform over 0.15-1.5 times the episode's own target spread.
  4. The CONTEXT is drawn from what is left with weight
         w_i  proportional to  exp(-0.5 * ((y_i - mu) / sigma)^2)
     using Gumbel-top-k (weighted sampling without replacement, done in log space
     because the Gaussian weights underflow to exactly zero several sigma out).

Resampling (mu, sigma) every episode is the point: the model is not taught one
particular skew, it is taught to cope with a whole family of them.

    python3 plot_gaussian_sampling.py
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import biased_split as bs
import data as data_mod

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")

C_CTX, C_QRY, C_POOL = "#2a78d6", "#eb6834", "#c9c7c0"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

QUERY_RATIO, MIN_QUERY, POOL_N = 0.2, 10, 1200


def style(ax, title, xlabel, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, pad=9)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    ds = data_mod.LOADERS["abalone"]()
    rng = np.random.default_rng(7)
    pool = rng.choice(len(ds.y), size=POOL_N, replace=False)
    y_pool = np.asarray(ds.y, float)[pool]
    X_pool = ds.X.iloc[pool].reset_index(drop=True)

    edges = np.arange(np.floor(y_pool.min()), np.ceil(y_pool.max()) + 1, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.6))
    fig.patch.set_facecolor(SURFACE)
    a1, a2, a3, a4, a5, a6 = axes.ravel()

    # ---- one traced episode, used by panels 1-4 ------------------------------
    bs._TRACE = []
    Xc, Xq, yc, yq = bs.imbalanced_ctx_query_split(
        X_pool, y_pool, query_ratio=QUERY_RATIO, min_query_size=MIN_QUERY,
        random_state=3, mode="gauss")
    tr = bs._TRACE[-1]
    mu, sigma, y_rem = tr["mu"], tr["sigma"], tr["y_rem"]
    bs._TRACE = None

    # 1 -- the pool
    a1.hist(y_pool, bins=edges, color=C_POOL, linewidth=0)
    style(a1, "1 · The episode pool", "Target  (abalone rings)", "Points")
    a1.text(0.97, 0.94, f"n = {len(y_pool)}", transform=a1.transAxes,
            ha="right", va="top", fontsize=9, color=INK_2)

    # 2 -- query drawn first, uniformly
    a2.hist(y_pool, bins=edges, color=C_POOL, linewidth=0, label="Pool")
    a2.hist(yq, bins=edges, color=C_QRY, linewidth=0, label="Query")
    style(a2, "2 · Query drawn FIRST, uniformly", "Target", "Points")
    a2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    a2.text(0.97, 0.76, f"{len(yq)} query points\nkeeps the pool's shape",
            transform=a2.transAxes, ha="right", va="top",
            fontsize=8.5, color=INK_2, linespacing=1.5)

    # 3 -- the drawn Gaussian weight curve over what remains
    a3.hist(y_rem, bins=edges, color=C_POOL, linewidth=0)
    a3.set_ylabel("Points", fontsize=9.5, color=INK_2)
    tw = a3.twinx()
    grid_y = np.linspace(edges[0], edges[-1], 400)
    tw.plot(grid_y, np.exp(-0.5 * ((grid_y - mu) / sigma) ** 2),
            color=C_CTX, linewidth=2.4)
    tw.set_ylim(0, 1.35)
    tw.set_ylabel("Selection weight  w", fontsize=9.5, color=C_CTX)
    tw.tick_params(colors=C_CTX, labelsize=8.5)
    for s in ("top", "left"):
        tw.spines[s].set_visible(False)
    tw.spines["right"].set_color(C_CTX)
    tw.spines["bottom"].set_color(GRID)
    tw.axvline(mu, color=C_CTX, linewidth=1.2, linestyle=(0, (4, 3)))
    tw.annotate("", xy=(mu - sigma, 0.607), xytext=(mu + sigma, 0.607),
                arrowprops=dict(arrowstyle="<->", color=C_CTX, linewidth=1.2))
    tw.text(mu + 1.15 * sigma, 0.607, f"2σ = {2 * sigma:.2f}", ha="left",
            va="center", fontsize=8.5, color=C_CTX)
    tw.text(mu, 1.06, f"μ = {mu:.2f}", ha="center", fontsize=9,
            color=C_CTX, fontweight="bold")
    style(a3, "3 · Draw μ and σ for this episode", "Target")
    a3.text(0.03, 0.94, f"σ = {sigma:.2f}\n  = {sigma / tr['std']:.2f} × pool std",
            transform=a3.transAxes, ha="left", va="top",
            fontsize=8.5, color=INK_2, linespacing=1.5)

    # 4 -- the context actually selected
    a4.hist(y_rem, bins=edges, color=C_POOL, linewidth=0, label="Available")
    a4.hist(yc, bins=edges, color=C_CTX, linewidth=0, label="Context chosen")
    style(a4, "4 · Context = Gumbel-top-k by weight", "Target", "Points")
    a4.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    a4.text(0.97, 0.74,
            f"{len(yc)} context points\nstd {np.std(yc):.2f} vs pool {np.std(y_pool):.2f}",
            transform=a4.transAxes, ha="right", va="top",
            fontsize=8.5, color=INK_2, linespacing=1.5)

    # 5 -- a different (mu, sigma) every episode
    bs._TRACE = []
    for s in range(9):
        bs.imbalanced_ctx_query_split(
            X_pool, y_pool, query_ratio=QUERY_RATIO, min_query_size=MIN_QUERY,
            random_state=100 + s, mode="gauss")
    draws = list(bs._TRACE)
    bs._TRACE = None
    a5.hist(y_pool, bins=edges, color=C_POOL, linewidth=0)
    tw5 = a5.twinx()
    for t in draws:
        tw5.plot(grid_y, np.exp(-0.5 * ((grid_y - t["mu"]) / t["sigma"]) ** 2),
                 color=C_CTX, linewidth=1.5, alpha=0.62)
    tw5.set_ylim(0, 1.35)
    tw5.set_ylabel("Selection weight  w", fontsize=9.5, color=C_CTX)
    tw5.tick_params(colors=C_CTX, labelsize=8.5)
    for s_ in ("top", "left"):
        tw5.spines[s_].set_visible(False)
    tw5.spines["right"].set_color(C_CTX)
    tw5.spines["bottom"].set_color(GRID)
    style(a5, "5 · Every episode redraws μ and σ", "Target", "Points")
    a5.text(0.03, 0.94,
            "9 episodes.\nμ ~ quantile U(0.05, 0.95)\nσ ~ log-U(0.15, 1.5) × std",
            transform=a5.transAxes, ha="left", va="top",
            fontsize=8.5, color=INK_2, linespacing=1.6)

    # 6 -- aggregate over many episodes
    n_ep = 600
    ctx_all, qry_all = [], []
    for s in range(n_ep):
        _, _, yc_s, yq_s = bs.imbalanced_ctx_query_split(
            X_pool, y_pool, query_ratio=QUERY_RATIO, min_query_size=MIN_QUERY,
            random_state=1000 + s, mode="gauss")
        ctx_all.append(np.asarray(yc_s, float))
        qry_all.append(np.asarray(yq_s, float))
    ctx_all = np.concatenate(ctx_all)
    qry_all = np.concatenate(qry_all)

    h_pool, _ = np.histogram(y_pool, bins=edges, density=True)
    a6.fill_between(centers, h_pool, color=C_POOL, zorder=1, label="Pool")
    for arr, color, label in ((qry_all, C_QRY, "Query, pooled"),
                              (ctx_all, C_CTX, "Context, pooled")):
        h, _ = np.histogram(arr, bins=edges, density=True)
        a6.plot(centers, h, color=color, linewidth=2.3, zorder=2,
                label=label, solid_capstyle="round")
    style(a6, "6 · Pooled over 600 episodes", "Target", "Density")
    a6.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")

    lo_q, hi_q = np.quantile(y_pool, 0.10), np.quantile(y_pool, 0.90)
    tail = lambda a: float(((a < lo_q) | (a > hi_q)).mean())
    a6.text(0.97, 0.60,
            f"Query matches the pool exactly\n"
            f"(std {qry_all.std():.2f} vs {y_pool.std():.2f}) — the control.\n\n"
            f"Context stays concentrated even\n"
            f"pooled: std {ctx_all.std():.2f}, and only\n"
            f"{tail(ctx_all) / tail(y_pool):.2f}× the pool's tail mass.",
            transform=a6.transAxes, ha="right", va="top",
            fontsize=8.5, color=INK_2, linespacing=1.5)

    fig.suptitle("How Gaussian context sampling builds one fine-tuning episode",
                 fontsize=16, fontweight="bold", color=INK, x=0.035, ha="left", y=0.985)
    fig.text(0.035, 0.945,
             "Drawn by calling the real sampler in biased_split.py; "
             "μ and σ shown are the values it drew.",
             fontsize=10, color=INK_2, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.subplots_adjust(hspace=0.40, wspace=0.34)
    out = os.path.join(FIGS, "03_gaussian_sampling_mechanism.png")
    fig.savefig(out, dpi=165, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
