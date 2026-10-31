#!/usr/bin/env python3
"""Gaussian context sampling at the PRIOR level, for TabICL continued pre-training.

TabICL splits each synthetic dataset into context and query *positionally*: the
dataset carries a `train_size`, and rows ``[:train_size]`` are the context the model
conditions on. The split point is drawn uniformly in
``[min_train_size, max_train_size]``; which rows land on which side is just whatever
order the SCM emitted them in.

So we do not need to touch the sampler, the model, or the training loop. We reorder
the rows. Rows whose target sits near a drawn centre mu are moved to the front, so
they become the context; rows in the tails fall past the cut into the query. That is
the same intervention as our TabPFN fine-tuning work, expressed positionally.

mu and sigma use the same priors as `imbreg/biased_split.py`:

    mu    = quantile(y, U(0.05, 0.95))
    sigma = std(y) * exp(U(log 0.15, log 1.5))
    w_i   proportional to exp(-0.5 * ((y_i - mu) / sigma)^2)

and selection is Gumbel-top-k, i.e. weighted sampling without replacement done in
log space, because the Gaussian weights underflow to exactly zero several sigma out.

Both arms of the experiment use this module. `mode="uniform"` applies a random
permutation of the same shape and cost, so the control differs from the treatment in
the ordering rule alone -- not in whether a permutation happened at all.

The class is module-level and picklable on purpose: the trainer's DataLoader runs
`n_jobs` worker processes and pickles the dataset into each one.
"""
from __future__ import annotations

import numpy as np
import torch

from tabicl.prior import PriorDataset

SIGMA_RANGE = (0.15, 1.5)
QUANTILE_RANGE = (0.05, 0.95)
# A context whose targets are all but identical makes the model's own target
# standardisation divide by ~0. Mirrors DEGENERATE_CONTEXT_FRAC in biased_split.py.
DEGENERATE_CONTEXT_FRAC = 0.05
MAX_WIDEN_RETRIES = 6


def _gumbel_top_k(log_w: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """Indices of k entries sampled without replacement with weights exp(log_w).

    The clamps keep u strictly inside (0, 1): log(0) and log(1)=0 would both send
    the outer log to +-inf. Note the parentheses around each negation -- `.clamp_()`
    binds tighter than unary minus, so `-torch.log(u).clamp_min(e)` silently
    computes `-(log(u).clamp_min(e))`, which is -e for every u < 1 and makes the
    outer log NaN. All-NaN scores make topk degenerate to "the last k indices",
    which looks like a working sampler and is not one.
    """
    u = torch.rand(log_w.shape, generator=gen).clamp_(1e-20, 1.0 - 1e-7)
    gumbel = -torch.log(-torch.log(u))
    assert torch.isfinite(gumbel).all(), "non-finite Gumbel noise"
    return torch.topk(log_w + gumbel, k).indices


def gaussian_row_order(y: torch.Tensor, train_size: int, gen: torch.Generator) -> torch.Tensor:
    """Permutation putting a Gaussian-weighted subset of rows into [:train_size].

    Args:
        y: 1-D targets for ONE dataset (already trimmed to its true sequence length).
        train_size: number of leading rows that will be used as context.
        gen: torch RNG.

    Returns:
        A permutation of ``arange(len(y))``.
    """
    n = y.numel()
    if train_size <= 0 or train_size >= n:
        return torch.randperm(n, generator=gen)

    yf = y.float()
    std = float(yf.std())
    if not np.isfinite(std) or std < 1e-12:
        return torch.randperm(n, generator=gen)

    q = float(torch.empty(1).uniform_(*QUANTILE_RANGE, generator=gen))
    mu = float(torch.quantile(yf, q))
    sigma = std * float(np.exp(np.random.RandomState(
        int(torch.randint(0, 2**31 - 1, (1,), generator=gen))).uniform(
            np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1]))))

    ctx = _gumbel_top_k(-0.5 * ((yf - mu) / sigma) ** 2, train_size, gen)

    # Widen rather than emit a context the model cannot standardise.
    min_std = DEGENERATE_CONTEXT_FRAC * std
    for _ in range(MAX_WIDEN_RETRIES):
        if float(yf[ctx].std()) >= min_std:
            break
        sigma *= 2.0
        ctx = _gumbel_top_k(-0.5 * ((yf - mu) / sigma) ** 2, train_size, gen)
    else:
        return torch.randperm(n, generator=gen)

    mask = torch.ones(n, dtype=torch.bool)
    mask[ctx] = False
    rest = torch.nonzero(mask, as_tuple=True)[0]
    rest = rest[torch.randperm(rest.numel(), generator=gen)]
    return torch.cat([ctx, rest])


def mixed_row_order(y: torch.Tensor, train_size: int, gen: torch.Generator) -> torch.Tensor:
    """Half the context Gaussian-weighted, half drawn uniformly from the remainder.

    Mirrors `mode="mixed"` in imbreg/biased_split.py. The context is the same SIZE as
    in the other two modes, so the three arms stay comparable on everything except
    the weighting rule.

    The uniform half guarantees the context still spans the target range, so the
    degenerate-context failure that pure `gauss` has to guard against cannot arise
    here and no widening loop is needed.
    """
    n = y.numel()
    if train_size <= 0 or train_size >= n:
        return torch.randperm(n, generator=gen)

    yf = y.float()
    std = float(yf.std())
    if not np.isfinite(std) or std < 1e-12:
        return torch.randperm(n, generator=gen)

    n_g = train_size // 2
    q = float(torch.empty(1).uniform_(*QUANTILE_RANGE, generator=gen))
    mu = float(torch.quantile(yf, q))
    sigma = std * float(np.exp(np.random.RandomState(
        int(torch.randint(0, 2**31 - 1, (1,), generator=gen))).uniform(
            np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1]))))

    g_sel = _gumbel_top_k(-0.5 * ((yf - mu) / sigma) ** 2, n_g, gen)
    mask = torch.ones(n, dtype=torch.bool)
    mask[g_sel] = False
    rest = torch.nonzero(mask, as_tuple=True)[0]
    rest = rest[torch.randperm(rest.numel(), generator=gen)]
    n_u = min(train_size - n_g, rest.numel())
    return torch.cat([g_sel, rest[:n_u], rest[n_u:]])


def _row_perm(y_i: torch.Tensor, train_size: int, mode: str,
              gen: torch.Generator) -> torch.Tensor:
    if mode == "gauss":
        return gaussian_row_order(y_i, train_size, gen)
    if mode == "mixed":
        return mixed_row_order(y_i, train_size, gen)
    if mode == "uniform":
        # Same operation, unbiased: the control for "a permutation happened".
        return torch.randperm(y_i.numel(), generator=gen)
    raise ValueError(f"unknown mode {mode!r}")


def reorder_batch(X: torch.Tensor, y: torch.Tensor, seq_lens: torch.Tensor,
                  train_sizes: torch.Tensor, mode: str, gen: torch.Generator
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder rows within each dataset of a prior batch. Shapes are preserved.

    With `seq_len_per_gp=True` the generator emits NESTED tensors, because datasets
    in one batch have different lengths; `.shape[0]` is not even defined on those.
    They are rebuilt with `torch.nested.nested_tensor`, the same call the generator
    itself uses, so the trainer's later `to_padded_tensor` is unaffected.
    """
    if X.is_nested:
        xs, ys = list(X.unbind()), list(y.unbind())
        out_x, out_y = [], []
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            n = yi.numel()
            if n < 4:
                out_x.append(xi)
                out_y.append(yi)
                continue
            perm = _row_perm(yi, int(train_sizes[i]), mode, gen)
            out_x.append(xi[perm])
            out_y.append(yi[perm])
        return (torch.nested.nested_tensor(out_x, device=X.device),
                torch.nested.nested_tensor(out_y, device=y.device))

    X, y = X.clone(), y.clone()
    for i in range(X.shape[0]):
        n = int(seq_lens[i]) if seq_lens is not None else X.shape[1]
        n = min(n, X.shape[1])
        if n < 4:
            continue
        perm = _row_perm(y[i, :n], int(train_sizes[i]), mode, gen)
        X[i, :n] = X[i, :n][perm]
        y[i, :n] = y[i, :n][perm]
    return X, y


class GaussPriorDataset(PriorDataset):
    """PriorDataset whose rows are reordered so the context is target-concentrated.

    Everything about generation is inherited unchanged -- same SCM, same
    hyperparameter sampling, same sequence lengths, same train_size draws. Only the
    order of rows within each dataset differs, and `mode="uniform"` makes even that
    an unbiased shuffle so the two arms are matched in every operation performed.
    """

    reorder_mode: str = "gauss"
    reorder_seed: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Copy the class-level settings onto the INSTANCE. The trainer's DataLoader
        # uses spawn, so each worker re-imports this module and gets the class
        # defaults back -- a class attribute set by install() does not travel. Only
        # the pickled instance __dict__ does. Without this, every arm silently runs
        # with the default mode and the control is not a control.
        self.reorder_mode = type(self).reorder_mode
        self.reorder_seed = type(self).reorder_seed

    # PriorDataset is its own iterator: its __iter__ returns self and __next__ does
    # the work. Overriding __iter__ and calling super().__iter__() therefore returns
    # *this* object and recurses forever, so the hook belongs in __next__.
    def __iter__(self):
        it = super().__iter__()
        gen = torch.Generator()
        # Offset by worker id so parallel DataLoader workers do not draw identical
        # (mu, sigma) sequences, which would collapse the family of skews.
        info = torch.utils.data.get_worker_info()
        gen.manual_seed(int(self.reorder_seed) + 100003 * (0 if info is None else int(info.id)))
        self._reorder_gen = gen
        return it

    def __next__(self):
        batch = super().__next__()
        gen = getattr(self, "_reorder_gen", None)
        if gen is None:
            gen = torch.Generator()
            gen.manual_seed(int(self.reorder_seed))
            self._reorder_gen = gen
        X, y, d, seq_lens, train_sizes = batch[0], batch[1], batch[2], batch[3], batch[4]
        X, y = reorder_batch(X, y, seq_lens, train_sizes, self.reorder_mode, gen)
        return (X, y, d, seq_lens, train_sizes, *batch[5:])


def install(mode: str = "gauss", seed: int = 0) -> None:
    """Point the trainer at the reordering dataset. Call before TabICL builds it."""
    import tabicl.train._run as run_mod

    GaussPriorDataset.reorder_mode = mode
    GaussPriorDataset.reorder_seed = seed
    run_mod.PriorDataset = GaussPriorDataset
    print(f"[gauss_prior] installed: mode={mode} seed={seed}", flush=True)
