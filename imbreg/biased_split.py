"""Imbalance-aware context/query sampling for TabPFN fine-tuning.

Background
----------
TabPFN is pre-trained on synthetic datasets drawn from a prior, but Prior Labs did
not open-source that prior generator -- the released `tabpfn` package contains
inference and fine-tuning only. What it *does* expose is the fine-tuning episode
sampler, which has the same structure as a pre-training step: each episode draws a
context set and a query set from the data, and the model is scored on how well it
predicts the query given the context.

Stock behaviour (`tabpfn.finetuning.finetuned_base._ratio_ctx_query_split`) draws
that split uniformly at random, so the context's target distribution matches the
query's. The model therefore never has to predict a target region that its context
under-represents -- exactly the situation imbalanced regression puts it in at test
time.

What this module changes
------------------------
Context points are drawn with a Gaussian weight in target space,

    w_i  proportional to  exp( -0.5 * ((y_i - mu) / sigma)^2 ),

so the context concentrates around a centre `mu` and the tails become rare, while
the query points are drawn *independently and without that weighting*, so they still
cover the full target range. Each episode resamples (mu, sigma), which makes the
fine-tuning objective an expectation over a family of selection mechanisms rather
than one fixed skew.

Why the query set is drawn first
--------------------------------
Selecting context points on the basis of y is non-ignorable selection: the context's
y-marginal is deliberately not the population's. Drawing the query set *before*
applying the context weighting keeps the query marginal untouched, so the learning
signal is "predict the full target range from a skewed context" rather than
"predict a skewed subset from a skewed context", which would be the same problem the
model already solves.

The confound this code is careful to avoid
------------------------------------------
Gaussian weighting can only bias the context if the context is a strict *subset* of
the available points -- sampling all of them, weighted or not, is just a permutation.
So the context is subsampled to `ctx_keep_frac` of what is available. That shrinks
the context, which would on its own change performance. The uniform baseline
therefore applies the identical subsampling with uniform weights: both arms see the
same number of context points and the same number of query points, and the *only*
difference between them is the sampling weight. Without this, any measured gain would
be confounded with context size.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np

# Episode counter. The fine-tuning loop passes the same `random_state` for every
# chunk within an epoch, so without extra entropy every episode in an epoch would
# draw the same (mu, sigma) and the "family of selection mechanisms" would collapse
# to a single one. The counter is deterministic given call order, so runs stay
# reproducible. Safe because the fine-tuning DataLoader runs with num_workers=0
# (single process) -- see finetuned_base.py DataLoader construction.
_EPISODE_COUNTER = itertools.count()

# Opt-in instrumentation for the explanatory figure (plot_gaussian_sampling.py).
# Left as None in every production path, so this costs nothing and changes nothing;
# set it to a list to have each episode append its drawn (mu, sigma) and selections.
_TRACE: list | None = None

# A context whose target standard deviation falls below this fraction of the
# episode's own is treated as degenerate; see the guard in the sampler.
DEGENERATE_CONTEXT_FRAC = 0.05
_MAX_WIDEN_RETRIES = 6


def reset_episode_counter() -> None:
    """Reset episode entropy; call once before each fine-tuning run."""
    global _EPISODE_COUNTER
    _EPISODE_COUNTER = itertools.count()


def _take(obj: Any, idx: np.ndarray) -> Any:
    """Index a DataFrame/Series by position, or a plain array normally."""
    return obj.iloc[idx] if hasattr(obj, "iloc") else obj[idx]


def _gumbel_top_k(log_w: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted sampling without replacement via the Gumbel-top-k trick.

    Drawing `argtop_k(log w_i + g_i)` with `g_i ~ Gumbel(0,1)` is equivalent to
    successive weighted sampling without replacement (Plackett-Luce ordering). Used
    instead of `rng.choice(..., replace=False, p=w)` because it works directly in log
    space -- the Gaussian weights underflow to exactly zero several sigma out, and
    `rng.choice` raises when fewer than k entries have non-zero probability, which is
    precisely the strong-imbalance regime we want to be able to sample.
    """
    g = rng.gumbel(size=log_w.shape)
    return np.argpartition(-(log_w + g), k - 1)[:k]


def _inverse_density_log_weights(y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """log weights proportional to 1 / (bin frequency), i.e. a near-balanced sampler."""
    edges = np.linspace(y.min(), y.max() + 1e-12, n_bins + 1)
    idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    counts[counts == 0] = 1.0
    return -np.log(counts[idx])


def imbalanced_ctx_query_split(
    X: Any,
    y: Any,
    *,
    query_ratio: float,
    min_query_size: int,
    random_state: int,
    stratify: Any = None,  # noqa: ARG001 - regression never stratifies; kept for API parity
    mode: str = "gauss",
    ctx_keep_frac: float = 0.5,
    sigma_range: tuple[float, float] = (0.15, 1.5),
    center_mode: str = "random_quantile",
    query_mode: str = "uniform",
) -> list[Any]:
    """Drop-in replacement for `_ratio_ctx_query_split` with a biased context.

    Args:
        X, y: the episode's features and targets.
        query_ratio, min_query_size, random_state: supplied by the fine-tuning loop,
            same meaning as in the stock splitter.
        mode: "uniform" reproduces the stock sampler *at the reduced context size*
            (the matched baseline); "gauss" applies the Gaussian target weighting;
            "mixed" draws half the context by Gaussian weighting and half uniformly,
            so the tails stay represented instead of being stripped out. "mixed" is a
            middle ground: the model still sees a context skewed toward one region,
            but never one that has lost the tails entirely, which is closer to how a
            naturally imbalanced dataset actually looks.
        ctx_keep_frac: fraction of non-query points kept as context. Must be < 1 for
            the weighting to have any effect.
        sigma_range: Gaussian width is drawn log-uniformly from this range, in units
            of the episode's target standard deviation. Small sigma = severe
            imbalance, large sigma = nearly uniform.
        center_mode: "random_quantile" draws mu at a random quantile of y each
            episode (a prior over where the over-represented region sits);
            "mode" always centres on the densest target region, matching the
            structure of a naturally right-skewed dataset more literally.
        query_mode: "uniform" leaves the query marginal at the episode's natural
            distribution; "balanced" draws queries with inverse-density weights so
            the tails are over-represented in the loss.

    Returns:
        [X_context, X_query, y_context, y_query] -- the order sklearn's
        `train_test_split` returns, which is what the caller unpacks.
    """
    yv = np.asarray(y, dtype=float).ravel()
    n = len(yv)

    n_query = max(int(n * query_ratio), min_query_size)
    n_query = int(np.clip(n_query, 1, n - 2))  # leave >= 2 points for a context

    rng = np.random.default_rng([int(random_state), next(_EPISODE_COUNTER)])

    # --- 1. Query set, drawn independently of the context bias -------------------
    all_idx = np.arange(n)
    if query_mode == "balanced":
        q_pos = _gumbel_top_k(_inverse_density_log_weights(yv), n_query, rng)
    elif query_mode == "uniform":
        q_pos = rng.choice(n, size=n_query, replace=False)
    else:
        raise ValueError(f"unknown query_mode {query_mode!r}")
    query_idx = all_idx[q_pos]

    remaining = np.setdiff1d(all_idx, query_idx, assume_unique=False)

    # --- 2. Context set, subsampled from what is left ----------------------------
    n_ctx = max(2, int(round(len(remaining) * ctx_keep_frac)))
    n_ctx = min(n_ctx, len(remaining))

    if mode == "uniform":
        c_sel = rng.choice(len(remaining), size=n_ctx, replace=False)
    elif mode == "gauss":
        y_rem = yv[remaining]
        std = float(np.std(y_rem))
        if std < 1e-12:  # degenerate target in this episode; nothing to bias
            c_sel = rng.choice(len(remaining), size=n_ctx, replace=False)
        else:
            if center_mode == "random_quantile":
                mu = float(np.quantile(y_rem, rng.uniform(0.05, 0.95)))
            elif center_mode == "mode":
                hist, edges = np.histogram(y_rem, bins=min(20, max(3, len(y_rem) // 20)))
                b = int(np.argmax(hist))
                mu = float(0.5 * (edges[b] + edges[b + 1]))
            else:
                raise ValueError(f"unknown center_mode {center_mode!r}")
            sigma = std * float(np.exp(rng.uniform(np.log(sigma_range[0]),
                                                   np.log(sigma_range[1]))))
            log_w = -0.5 * ((y_rem - mu) / sigma) ** 2
            c_sel = _gumbel_top_k(log_w, n_ctx, rng)

            # Guard against a degenerate context. On a low-cardinality target a
            # narrow Gaussian can select a context whose entries are ALL the same
            # value, giving it zero variance. TabPFN standardises the targets by the
            # context's own standard deviation, so a zero there divides by ~0 and the
            # bar-distribution NLL explodes -- measured at 7e15 on wine quality, which
            # has only 7 distinct values, on 1.4% of episodes. Widening sigma and
            # redrawing keeps the context biased while guaranteeing it still spans
            # more than one target value.
            min_std = DEGENERATE_CONTEXT_FRAC * std
            if _TRACE is not None:
                _TRACE.append({"mu": mu, "sigma": sigma, "std": std,
                               "y_rem": y_rem.copy()})
            for _ in range(_MAX_WIDEN_RETRIES):
                if float(np.std(y_rem[c_sel])) >= min_std:
                    break
                sigma *= 2.0
                c_sel = _gumbel_top_k(-0.5 * ((y_rem - mu) / sigma) ** 2, n_ctx, rng)
            else:
                # Still degenerate after widening: fall back to an unbiased draw
                # rather than emit an episode that would produce a garbage gradient.
                c_sel = rng.choice(len(remaining), size=n_ctx, replace=False)
    elif mode == "mixed":
        # Half the context Gaussian-weighted, half uniform from what the Gaussian
        # half did not take. Sizes are the same as the other modes, so the three
        # arms remain comparable on everything except the sampling weight.
        n_g = n_ctx // 2
        y_rem = yv[remaining]
        std = float(np.std(y_rem))
        if std < 1e-12:
            c_sel = rng.choice(len(remaining), size=n_ctx, replace=False)
        else:
            if center_mode == "random_quantile":
                mu = float(np.quantile(y_rem, rng.uniform(0.05, 0.95)))
            else:
                hist, edges = np.histogram(y_rem, bins=min(20, max(3, len(y_rem) // 20)))
                b = int(np.argmax(hist))
                mu = float(0.5 * (edges[b] + edges[b + 1]))
            sigma = std * float(np.exp(rng.uniform(np.log(sigma_range[0]),
                                                   np.log(sigma_range[1]))))
            g_sel = _gumbel_top_k(-0.5 * ((y_rem - mu) / sigma) ** 2, n_g, rng)
            rest = np.setdiff1d(np.arange(len(remaining)), g_sel)
            n_u = min(n_ctx - n_g, len(rest))
            u_sel = rng.choice(rest, size=n_u, replace=False)
            c_sel = np.concatenate([g_sel, u_sel])
            # The uniform half guarantees tail coverage, so the degenerate-context
            # failure mode that affects pure "gauss" cannot arise here.
    else:
        raise ValueError(f"unknown mode {mode!r}")

    context_idx = remaining[c_sel]

    return [_take(X, context_idx), _take(X, query_idx),
            _take(y, context_idx), _take(y, query_idx)]
