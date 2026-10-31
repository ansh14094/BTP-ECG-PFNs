"""Imbalance-aware context sampling for TabICL fine-tuning.

TabICL's fine-tuning episode sampler has the same shape as TabPFN's: each chunk is
split into an in-context set and a query set, and the loss is computed on the query.
The split happens in `tabicl._finetune.data._split_ctx_query`, which takes the chunk's
targets and returns index arrays -- a cleaner hook than TabPFN's, which operates on the
feature matrix too.

The intervention is identical to `biased_split.py` so the two models are tested on the
same idea rather than on two subtly different ones:

  * the query set is drawn FIRST and without weighting, so its target marginal is the
    natural one and the model must predict the full range;
  * the context is then drawn from what remains with weight proportional to
    exp(-0.5 * ((y - mu) / sigma)^2), concentrating it and making the tails rare;
  * (mu, sigma) is resampled every episode, so the objective is an expectation over a
    family of selection mechanisms rather than one fixed skew;
  * the context is subsampled to `ctx_keep_frac` of what is available, because
    weighting a sample that takes *every* remaining point is just a permutation --
    and the uniform baseline applies the same subsampling, so context size is matched
    and the sampling weight is the only difference between the arms.
"""
from __future__ import annotations

import itertools
import numpy as np

_EPISODE_COUNTER = itertools.count()


def reset_episode_counter() -> None:
    global _EPISODE_COUNTER
    _EPISODE_COUNTER = itertools.count()


def _gumbel_top_k(log_w: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted sampling without replacement in log space (Gumbel-top-k).

    Needed because the Gaussian weights underflow to exactly zero several sigma out,
    which `rng.choice(..., p=...)` cannot handle -- precisely the strong-imbalance
    regime this exists for.
    """
    g = rng.gumbel(size=log_w.shape)
    return np.argpartition(-(log_w + g), k - 1)[:k]


def make_split_fn(mode: str = "gauss", ctx_keep_frac: float = 0.5,
                  sigma_range: tuple[float, float] = (0.15, 1.5),
                  center_mode: str = "random_quantile"):
    """Build a drop-in replacement for `tabicl._finetune.data._split_ctx_query`."""

    def split_ctx_query(y_chunk: np.ndarray, *, query_size: int, seed: int,
                        stratify: bool) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y_chunk, dtype=float).ravel()
        n = len(y)
        query_size = max(1, min(query_size, n - 2))
        # Extra entropy: the loop reuses one seed per epoch across chunks, so without
        # this every episode in an epoch would share the same (mu, sigma).
        rng = np.random.default_rng([int(seed), next(_EPISODE_COUNTER)])

        q_idx = rng.choice(n, size=query_size, replace=False)
        remaining = np.setdiff1d(np.arange(n), q_idx)
        n_ctx = min(len(remaining), max(2, int(round(len(remaining) * ctx_keep_frac))))

        if mode == "uniform":
            sel = rng.choice(len(remaining), size=n_ctx, replace=False)
        elif mode == "gauss":
            y_rem = y[remaining]
            std = float(np.std(y_rem))
            if std < 1e-12:
                sel = rng.choice(len(remaining), size=n_ctx, replace=False)
            else:
                if center_mode == "random_quantile":
                    mu = float(np.quantile(y_rem, rng.uniform(0.05, 0.95)))
                else:
                    hist, edges = np.histogram(y_rem, bins=min(20, max(3, len(y_rem) // 20)))
                    b = int(np.argmax(hist))
                    mu = float(0.5 * (edges[b] + edges[b + 1]))
                sigma = std * float(np.exp(rng.uniform(np.log(sigma_range[0]),
                                                       np.log(sigma_range[1]))))
                sel = _gumbel_top_k(-0.5 * ((y_rem - mu) / sigma) ** 2, n_ctx, rng)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        return remaining[sel], q_idx

    return split_ctx_query
