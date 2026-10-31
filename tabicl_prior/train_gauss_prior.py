#!/usr/bin/env python3
"""Continued pre-training of TabICL on its own prior, with Gaussian context ordering.

This is `python -m tabicl.train` with one thing changed: the dataset class the
trainer instantiates is swapped for `GaussPriorDataset`, which reorders the rows of
each synthetic dataset so the positional context/query cut lands on a
target-concentrated context. Every TabICL flag is passed through untouched.

Run it exactly as you would the stock trainer, plus --reorder_mode:

    PYTHONPATH=/mnt/storage/Aryan\\ LBP/BTP/tabicl_prior \\
    python3 train_gauss_prior.py --reorder_mode gauss  [all the usual tabicl.train flags]

--reorder_mode uniform is the control arm: it performs an unbiased permutation of
the same shape and cost, so the two arms differ in the ordering rule alone.

The trainer spawns DataLoader workers, which re-import this package rather than
inheriting the parent's patched module object, so `gauss_prior` must be importable
in the children -- hence the PYTHONPATH above. The patch is applied in-process too,
because the Trainer builds the dataset in the parent before handing it to workers.
"""
from __future__ import annotations

import sys

from torch.multiprocessing import set_start_method

import gauss_prior


def main() -> None:
    # Pull our own flags out before TabICL's parser sees argv.
    argv = sys.argv[1:]
    mode, seed, rest = "gauss", 0, []
    i = 0
    while i < len(argv):
        if argv[i] == "--reorder_mode":
            mode = argv[i + 1]
            i += 2
        elif argv[i] == "--reorder_seed":
            seed = int(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    if mode not in ("gauss", "uniform", "mixed"):
        raise SystemExit(f"--reorder_mode must be gauss, uniform or mixed, got {mode!r}")

    gauss_prior.install(mode=mode, seed=seed)

    from tabicl.train._train_config import build_parser
    from tabicl.train._run import Trainer

    sys.argv = [sys.argv[0], *rest]
    config = build_parser().parse_args()

    if config.prior_dir is not None:
        raise SystemExit(
            "--prior_dir loads pre-generated data through LoadPriorDataset, which this "
            "hook does not wrap. Use on-the-fly generation (omit --prior_dir), as the "
            "released stage scripts do."
        )

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    Trainer(config).train()


if __name__ == "__main__":
    main()
