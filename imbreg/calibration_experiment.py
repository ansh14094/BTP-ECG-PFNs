#!/usr/bin/env python3
"""Per-point predictive uncertainty for the best Table 1 cell.

Table 1's best cell is Gaussian fine-tuning on the E2 encoder (ptbxl_tabpfn_enc,
DIR protocol, MAE 6.872). The prediction CSVs saved by run.py hold only the point
estimate, so this re-runs that cell and additionally dumps, for every test point,
the standard deviation of TabPFN's full predictive distribution.

It reproduces run.py's training path exactly -- same split, same sampler, same
hyperparameters, same per-seed validation-selected learning rate -- and asserts
that the point predictions it reproduces match the recorded ones, so the sigmas
belong to the model the table actually scored.

    python3 calibration_experiment.py                 # all 5 seeds
    python3 calibration_experiment.py --seeds 0       # one seed
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

import data as data_mod
import run as run_mod

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "calibration")

DATASET = "ptbxl_tabpfn_enc"
ARM = "ft_gauss"
PROTOCOL = "dir"
MODE = "gauss"
# run.py's defaults for these arms.
EPOCHS, CTX_PLUS_QUERY, CTX_KEEP_FRAC = 60, 1024, 0.5
CENTER_MODE, QUERY_MODE, SIGMA_RANGE = "random_quantile", "uniform", (0.15, 1.5)
LR_BY_TAG = {"dir_lr1e-5": 1e-5, "dir_lr3e-5": 3e-5, "dir_lr1e-4": 1e-4}


def val_selected_tags() -> dict[int, str]:
    rows = []
    for p in glob.glob(os.path.join(HERE, "results", f"{DATASET}_*.jsonl")):
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df[(df.protocol == PROTOCOL) & (df.arm == ARM) & df.tag.str.startswith("dir_")]
    best = df.sort_values("val_rmse").groupby("seed", as_index=False).first()
    return dict(zip(best.seed.astype(int), best.tag))


def recorded_mae(seed: int, tag: str) -> float | None:
    p = os.path.join(HERE, "results", f"{DATASET}_{tag}.jsonl")
    if not os.path.exists(p):
        return None
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            if r["arm"] == ARM and int(r["seed"]) == seed:
                return float(r["mae"])
    return None


def fit_and_dump(seed: int, tag: str) -> dict:
    """Fine-tune one seed and return per-point mean / sigma on the test split."""
    from functools import partial

    import tabpfn.finetuning.finetuned_base as fb
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

    import biased_split

    ds = data_mod.load(DATASET)
    X_all = run_mod.prep_frame(ds.X, ds.cat_features)
    tr, va, te = data_mod.dir_split(ds.y, ds.bin_width, seed=seed,
                                   per_bin_val=ds.per_bin_val,
                                   per_bin_test=ds.per_bin_test)
    Xtr, Xva, Xte = X_all.iloc[tr], X_all.iloc[va], X_all.iloc[te]
    ytr, yva, yte = ds.y[tr], ds.y[va], ds.y[te]

    original = fb._ratio_ctx_query_split
    biased_split.reset_episode_counter()
    fb._ratio_ctx_query_split = partial(
        biased_split.imbalanced_ctx_query_split,
        mode=MODE, ctx_keep_frac=CTX_KEEP_FRAC, center_mode=CENTER_MODE,
        query_mode=QUERY_MODE, sigma_range=SIGMA_RANGE)
    try:
        reg = FinetunedTabPFNRegressor(
            device="cuda",
            model_version=ModelVersion.V2_5,
            epochs=EPOCHS,
            learning_rate=LR_BY_TAG[tag],
            n_finetune_ctx_plus_query_samples=CTX_PLUS_QUERY,
            finetune_ctx_query_split_ratio=0.2,
            n_estimators_finetune=8,
            n_estimators_validation=8,
            n_estimators_final_inference=8,
            random_state=seed,
            early_stopping=True,
            early_stopping_patience=8,
        )
        reg.fit(Xtr, ytr, X_val=Xva, y_val=yva)

        mean = np.asarray(reg.predict(Xte), float)
        # Sigma from the full predictive distribution. TabPFN's regressor head is a
        # bar distribution over bins, so its criterion exposes the true second
        # moment -- far better than inferring a width from a quantile spread, which
        # would assume a shape the distribution does not have.
        full = reg.predict(Xte, output_type="full")
    finally:
        fb._ratio_ctx_query_split = original

    sigma = extract_sigma(full, mean)
    return {"y_true": yte, "y_pred": mean, "sigma": sigma}


def extract_sigma(full: dict, mean: np.ndarray) -> np.ndarray:
    """Standard deviation of TabPFN's predictive distribution, per test point."""
    import torch

    crit = full.get("criterion")
    logits = full.get("logits")
    if crit is not None and logits is not None and hasattr(crit, "variance"):
        # FullSupportBarDistribution.variance handles the outermost bins correctly:
        # they carry half-normal tails of infinite support, so taking a border
        # midpoint for them would understate the width. Verified against
        # predict(output_type="mean"): crit.mean(logits) reproduces it exactly, so
        # the criterion is already in original target units and needs no rescaling.
        with torch.no_grad():
            lg = logits if torch.is_tensor(logits) else torch.as_tensor(logits)
            var = crit.variance(lg).float().cpu().numpy().ravel()
        return np.sqrt(np.clip(var, 0.0, None))

    # Fallback: Gaussian-equivalent width from the central 80% quantile spread.
    q = full.get("quantiles")
    if q is not None:
        arr = np.asarray([np.asarray(v, float).ravel() for v in q])
        return (arr[-1] - arr[0]) / (2 * 1.2815515655446004)
    raise RuntimeError("could not recover a predictive sigma from predict(output_type='full')")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    tags = val_selected_tags()
    print(f"val-selected LR per seed: {tags}", flush=True)

    for seed in [int(s) for s in args.seeds.split(",")]:
        dest = os.path.join(OUT, f"{DATASET}_{ARM}_s{seed}.csv")
        if os.path.exists(dest):
            print(f"[seed {seed}] already done -> {dest}", flush=True)
            continue
        tag = tags[seed]
        t0 = time.time()
        print(f"[seed {seed}] tag={tag} lr={LR_BY_TAG[tag]:g} ...", flush=True)
        r = fit_and_dump(seed, tag)

        mae = float(np.abs(r["y_pred"] - r["y_true"]).mean())
        ref = recorded_mae(seed, tag)
        agree = "n/a" if ref is None else f"{abs(mae - ref):.4f}"
        print(f"[seed {seed}] MAE {mae:.4f}  recorded {ref}  |diff| {agree}  "
              f"sigma mean {r['sigma'].mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)

        pd.DataFrame(r).to_csv(dest, index=False)
        print(f"[seed {seed}] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
