#!/usr/bin/env python3
"""The two priors the Gaussian context sampler draws from, measured not asserted.

    mu    = quantile(y_remaining, U(0.05, 0.95))
    sigma = std(y_remaining) * exp(U(log 0.15, log 1.5))

Every number plotted here was drawn by the real sampler in `biased_split.py`,
captured through its `_TRACE` hook over N_EPISODES episodes. Nothing is a
reimplementation of the formulas.

Two things the formulas do not make obvious, and that the figure is for:

  * mu is uniform over QUANTILES, not over the value range. Its distribution is
    therefore the dataset's own target distribution, clipped at the 5th and 95th
    percentiles -- centres land where the data is, so a skewed target gets skewed
    centres. Drawing mu uniformly over [min, max] instead would put most centres
    in sparse regions and is a different method.
  * sigma is log-uniform, so the TYPICAL draw is the geometric mean
    sqrt(0.15 * 1.5) = 0.474 of the episode's target spread, not the arithmetic
    midpoint 0.825. Half of all episodes are narrower than 0.474 sigma.

Caveat recorded on the figure: the trace captures the sigma as first drawn. When a
context comes out degenerate the sampler doubles sigma and redraws, so the sigma
actually used is occasionally larger than the one shown here.

    python3 plot_mu_sigma.py
"""
from __future__ import annotations
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import biased_split as bs
import data as data_mod

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")

C_MU, C_SIGMA, C_POOL = "#2a78d6", "#eb6834", "#c9c7c0"
# Ordinal ramp for the sigma bands in panel 4: narrow -> wide.
RAMP = ("#0d366b", "#2a78d6", "#86b6ef")
SURFACE, INK, INK_2, GRID, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#9a9892"

DATASET = "ptbxl_tabpfn_enc"
POOL_N, N_EPISODES = 1500, 4000
QUERY_RATIO, MIN_QUERY = 0.2, 10
SIG_LO, SIG_HI = 0.15, 1.5


def style(ax, title, xlabel, ylabel=None, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, pad=9)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=INK_2)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    ds = data_mod.LOADERS[DATASET]()
    rng = np.random.default_rng(7)
    pool = rng.choice(len(ds.y), size=POOL_N, replace=False)
    y_pool = np.asarray(ds.y, float)[pool]
    X_pool = ds.X.iloc[pool].reset_index(drop=True)

    # Drive the real sampler and collect what it drew.
    bs._TRACE = []
    for ep in range(N_EPISODES):
        bs.imbalanced_ctx_query_split(
            X_pool, y_pool, query_ratio=QUERY_RATIO, min_query_size=MIN_QUERY,
            random_state=ep, mode="gauss")
    trace = bs._TRACE
    bs._TRACE = None

    mu = np.array([t["mu"] for t in trace])
    sig_rel = np.array([t["sigma"] / t["std"] for t in trace])
    print(f"{len(trace)} episodes traced")
    print(f"  mu       min {mu.min():.3f}  median {np.median(mu):.3f}  max {mu.max():.3f}")
    print(f"  sigma/std min {sig_rel.min():.4f}  median {np.median(sig_rel):.4f}  "
          f"max {sig_rel.max():.4f}")
    print(f"  geometric mean of sigma/std: {np.exp(np.mean(np.log(sig_rel))):.4f} "
          f"(predicted {np.sqrt(SIG_LO * SIG_HI):.4f})")

    fig, axes = plt.subplots(1, 4, figsize=(23.5, 5.4))
    fig.patch.set_facecolor(SURFACE)
    a1, a2, a3, a4 = axes

    # 1 -- where mu lands, against the target distribution it is drawn from
    lo, hi = y_pool.min(), y_pool.max()
    edges = np.linspace(lo, hi, 36)
    a1.hist(y_pool, bins=edges, density=True, color=C_POOL, linewidth=0,
            label="Episode targets")
    a1.hist(mu, bins=edges, density=True, histtype="step", color=C_MU,
            linewidth=2.0, label="Drawn μ")
    q05, q95 = np.quantile(y_pool, [0.05, 0.95])
    for q in (q05, q95):
        a1.axvline(q, color=RULE, linewidth=1.3, linestyle=(0, (5, 4)), zorder=1)
    style(a1, "1 · Where the centre μ lands", f"Target  (PTB-XL age, years)", "Density")
    # Headroom so neither the legend nor the note lands on a bar.
    a1.set_ylim(0, a1.get_ylim()[1] * 1.42)
    a1.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    a1.text(0.03, 0.96,
            f"μ tracks the data, not the axis:\nit is uniform over QUANTILES.\n"
            f"Dashed rules are the 5th and\n95th percentiles ({q05:.0f} and {q95:.0f} yr),\n"
            f"outside which μ never falls.",
            transform=a1.transAxes, ha="left", va="top",
            fontsize=8.5, color=INK_2, linespacing=1.55)

    # 2 -- how wide sigma is, on the log axis it was drawn on
    bins = np.geomspace(SIG_LO, SIG_HI, 31)
    a2.hist(sig_rel, bins=bins, color=C_SIGMA, linewidth=0)
    a2.set_xscale("log")
    gm = float(np.sqrt(SIG_LO * SIG_HI))
    a2.axvline(gm, color=INK, linewidth=1.6, zorder=3)
    a2.annotate(f"median {gm:.2f}", xy=(gm, 0), xytext=(gm * 1.07, 0.60),
                textcoords=("data", "axes fraction"), fontsize=8.5, color=INK)
    style(a2, "2 · How wide the Gaussian σ is",
          "σ  (in units of the episode's target std, log scale)", "Episodes")
    a2.set_xticks([0.15, 0.3, 0.5, 1.0, 1.5])
    a2.set_xticklabels(["0.15", "0.3", "0.5", "1.0", "1.5"])
    # Headroom so the note never lands on a bar.
    a2.set_ylim(0, max(np.histogram(sig_rel, bins=bins)[0]) * 1.62)
    a2.text(0.03, 0.96,
            f"Flat on a log axis = log-uniform.\n"
            f"Typical draw is the geometric mean\n"
            f"√(0.15×1.5) = {gm:.3f}, not the midpoint 0.825.\n"
            f"Half of all episodes are narrower.",
            transform=a2.transAxes, ha="left", va="top",
            fontsize=8.5, color=INK_2, linespacing=1.55)

    # 3 -- the two draws are independent
    a3.scatter(mu, sig_rel, s=7, color=C_MU, alpha=0.13, linewidths=0, rasterized=True)
    a3.set_yscale("log")
    a3.set_yticks([0.15, 0.3, 0.5, 1.0, 1.5])
    a3.set_yticklabels(["0.15", "0.3", "0.5", "1.0", "1.5"])
    style(a3, "3 · μ and σ are drawn independently", "Drawn μ  (target units)",
          "Drawn σ  (× target std)", grid_axis="both")
    from scipy.stats import spearmanr
    rho = float(spearmanr(mu, sig_rel).statistic)
    a3.text(0.97, 0.95, f"Spearman ρ = {rho:+.3f}\n{len(mu):,} episodes",
            transform=a3.transAxes, ha="right", va="top", fontsize=9,
            color=INK, linespacing=1.6)

    # 4 -- what those two draws mean for the selection weight
    grid = np.linspace(lo, hi, 400)
    band = np.digitize(sig_rel, [0.3, 0.7])          # narrow / medium / wide
    labels = ("σ < 0.3 (severe skew)", "0.3 ≤ σ < 0.7", "σ ≥ 0.7 (near-uniform)")
    sub = np.random.default_rng(1).choice(len(mu), size=130, replace=False)
    for i in sub:
        a4.plot(grid, np.exp(-0.5 * ((grid - mu[i]) / (sig_rel[i] * y_pool.std())) ** 2),
                color=RAMP[band[i]], linewidth=1.0, alpha=0.38, zorder=2)
    style(a4, "4 · The family of weight curves this produces",
          "Target  (PTB-XL age, years)", "Selection weight  w(y)")
    a4.set_ylim(0, 1.3)
    a4.legend(handles=[Line2D([], [], color=RAMP[k], linewidth=2.4, label=labels[k])
                       for k in range(3)],
              frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper right")
    a4.text(0.03, 0.95,
            "Redrawn every episode, so the model is\n"
            "never taught one skew — it is taught to\n"
            "cope with the whole family.",
            transform=a4.transAxes, ha="left", va="top",
            fontsize=8.5, color=INK_2, linespacing=1.55)

    fig.suptitle("The two priors of the Gaussian context sampler  ·  "
                 f"μ and σ as actually drawn over {N_EPISODES:,} episodes (PTB-XL age)",
                 fontsize=15, fontweight="bold", color=INK, x=0.028, ha="left", y=0.985)
    fig.text(0.028, 0.012,
             "σ shown is the value as first drawn; where a context comes out degenerate "
             "the sampler doubles σ and redraws, so the σ actually used is occasionally larger.",
             fontsize=8.5, color=INK_2, ha="left")
    fig.tight_layout(rect=[0, 0.035, 1, 0.92])
    out = os.path.join(FIGS, "05_mu_sigma_priors.png")
    fig.savefig(out, dpi=165, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
