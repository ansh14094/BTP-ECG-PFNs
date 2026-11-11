#!/usr/bin/env python3
"""Task D: paired bootstrap confidence intervals and per-seed win counts.

Two numbers per comparison, as specified in paper_completion_plan.md section 2:

  1. A paired bootstrap over test samples, 10,000 resamples, giving the 95%
     confidence interval of dMAE = MAE(method) - MAE(reference). Negative means the
     method is better. Paired means the same resampled rows are used for both arms,
     which removes the variance due to which rows happen to be drawn and is the whole
     reason a paired test is tighter than two independent ones.

  2. How many of the 5 seeds the method beats the reference in.

Pairing needs both arms scored on the same rows. Under the `fixed` protocol the test
split is the official partition and is identical across seeds, so predictions are
averaged per record first, exactly as the plan asks. Under `dir` every seed draws a
different balanced test set, so there is no common row set to average over; there the
bootstrap is run within each seed and the per-seed interval endpoints are averaged.
Which of the two was used is printed with every row, because they are not the same
test and should not be read as if they were.

    python3 eval/significance.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRED = os.path.join(ROOT, "imbreg", "results", "preds")
OUT = os.path.join(ROOT, "imbreg", "results", "metrics", "significance.csv")

N_BOOT = 10_000
SEEDS = (0, 1, 2, 3, 4)
RNG = np.random.default_rng(0)

BANDS = {"overall": None,
         "young": lambda y: y < 30,
         "middle": lambda y: (y >= 30) & (y < 60),
         "old": lambda y: y >= 60}


def load(dataset: str, tag: str, protocol: str, arm: str) -> dict[int, pd.DataFrame]:
    """Per-seed prediction frames for one cell, or {} if the cell is absent."""
    out = {}
    for s in SEEDS:
        f = os.path.join(PRED, f"{dataset}_{tag}_{protocol}_{arm}_s{s}.csv")
        if os.path.exists(f):
            out[s] = pd.read_csv(f)
    return out


def _boot_ci(err_m: np.ndarray, err_r: np.ndarray) -> tuple[float, float, float]:
    """Paired bootstrap of mean(err_m) - mean(err_r). Returns (delta, lo, hi)."""
    n = len(err_m)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    d = err_m[idx].mean(axis=1) - err_r[idx].mean(axis=1)
    return (float(err_m.mean() - err_r.mean()),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def compare(label: str, dataset: str, tag: str, protocol: str,
            arm_m: str, arm_r: str, bands=("overall",),
            tag_r: str | None = None) -> list[dict]:
    """`tag_r` lets the reference arm live under a different tag. The zero-shot
    predictions were written under `clean_tabpfn` while the fine-tuning sweep wrote
    `clean_tabpfn_lr*`, so comparing against stock needs the two to differ."""
    M = load(dataset, tag, protocol, arm_m)
    R = load(dataset, tag_r or tag, protocol, arm_r)
    seeds = sorted(set(M) & set(R))
    if not seeds:
        return [{"comparison": label, "band": b, "status": "MISSING"} for b in bands]

    rows = []
    shared = protocol == "fixed"
    for band in bands:
        fn = BANDS[band]
        if shared:
            # Same records every seed: average predictions per record, then bootstrap.
            m0 = M[seeds[0]]
            y = m0.y_true.to_numpy(float)
            pm = np.mean([M[s].y_pred.to_numpy(float) for s in seeds], axis=0)
            pr = np.mean([R[s].y_pred.to_numpy(float) for s in seeds], axis=0)
            mask = np.ones(len(y), bool) if fn is None else fn(y)
            delta, lo, hi = _boot_ci(np.abs(pm - y)[mask], np.abs(pr - y)[mask])
            method = "paired bootstrap, seed-averaged"
        else:
            # Different test set per seed: bootstrap within each, average endpoints.
            ds, los, his = [], [], []
            for s in seeds:
                y = M[s].y_true.to_numpy(float)
                mask = np.ones(len(y), bool) if fn is None else fn(y)
                if mask.sum() < 10:
                    continue
                d, l, h = _boot_ci(np.abs(M[s].y_pred.to_numpy(float) - y)[mask],
                                   np.abs(R[s].y_pred.to_numpy(float) - y)[mask])
                ds.append(d); los.append(l); his.append(h)
            if not ds:
                rows.append({"comparison": label, "band": band, "status": "TOO FEW ROWS"})
                continue
            delta, lo, hi = float(np.mean(ds)), float(np.mean(los)), float(np.mean(his))
            method = "bootstrap within seed, endpoints averaged"

        # Per-seed win count, always computed within a seed on its own test set.
        wins = 0
        for s in seeds:
            y = M[s].y_true.to_numpy(float)
            mask = np.ones(len(y), bool) if fn is None else fn(y)
            if mask.sum() == 0:
                continue
            if np.abs(M[s].y_pred.to_numpy(float) - y)[mask].mean() < \
               np.abs(R[s].y_pred.to_numpy(float) - y)[mask].mean():
                wins += 1
        rows.append({"comparison": label, "band": band, "status": "ok",
                     "delta_mae": round(delta, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4),
                     "significant": not (lo <= 0 <= hi),  # <= so a degenerate CI
                     # (identical predictions -> lo = hi = 0) reads as NOT significant
                     
                     "wins": wins, "n_seeds": len(seeds), "method": method})
    return rows


def main() -> None:
    cmp_specs = []
    PB = ("overall", "young", "middle", "old")

    # 1. E3 + Gaussian TabICL vs E3 + stock TabICL
    cmp_specs.append(("1. E3+gauss TabICL vs E3+stock TabICL",
                      "ptbxl_enc_e3", "clean_matrix", "fixed",
                      "ticl_prior_gauss", "ticl_zeroshot", PB))
    # 2. E2 + Gaussian TabPFN. The plan names the reference as stock TabPFN; the
    # matched control (uniform-context fine-tuning, same budget) is the more
    # informative reference, so BOTH are reported and each is labelled with the
    # reference it actually used. They were previously conflated: the row said
    # "vs stock" while the code compared against ft_uniform.
    cmp_specs.append(("2a. E2+gauss TabPFN vs E2+stock TabPFN (plan reference)",
                      "ptbxl_tabpfn_enc", "clean_tabpfn_lr1e-4", "fixed",
                      "ft_gauss", "zeroshot", PB, "clean_tabpfn"))
    cmp_specs.append(("2b. E2+gauss TabPFN vs E2+uniform-FT TabPFN (matched control)",
                      "ptbxl_tabpfn_enc", "clean_tabpfn_lr1e-4", "fixed",
                      "ft_gauss", "ft_uniform", PB))
    # 4. TabICL Gaussian vs target-balanced, few-shot region, tabular
    for ds in ("abalone", "communities", "wine_white"):
        cmp_specs.append((f"4. {ds}: prior gauss vs target_balanced",
                          ds, "ticl_base", "dir",
                          "ticl_prior_gauss", "ticl_prior_target_balanced",
                          ("overall",)))

    rows = []
    for spec in cmp_specs:
        rows += compare(*spec)

    # 3. Best ICL configuration vs ResNet Vanilla -- handled separately because the
    # two arms live in different dataset files and are only comparable because the
    # `fixed` test split is literally the same 954 records.
    rows += compare_cross("3. best ICL vs ResNet Vanilla")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {OUT}")


def compare_cross(label: str) -> list[dict]:
    """Best ICL cell against the ResNet baseline, across dataset files.

    Only valid under `fixed`, where every dataset file's test split is the same 954
    official records in the same order.
    """
    base = load("ptbxl_resnet", "taskA", "fixed", "vanilla")
    if not base:
        return [{"comparison": label, "band": b, "status": "MISSING (Task A pending)"}
                for b in ("overall", "young", "middle", "old")]

    # Pick the best ICL cell on band_macro, from the clean metrics.
    mpath = os.path.join(ROOT, "imbreg", "results", "metrics", "all_metrics.csv")
    met = pd.read_csv(mpath)
    cand = met[(met.protocol == "fixed") & (met.tag == "clean_matrix")]
    if cand.empty:
        return [{"comparison": label, "band": b, "status": "MISSING (Task C pending)"}
                for b in ("overall", "young", "middle", "old")]
    g = cand.groupby(["dataset", "arm"]).band_macro.mean().idxmin()
    ds_best, arm_best = g
    M = load(ds_best, "clean_matrix", "fixed", arm_best)
    rows = []
    for band in ("overall", "young", "middle", "old"):
        fn = BANDS[band]
        seeds = sorted(set(M) & set(base))
        y = M[seeds[0]].y_true.to_numpy(float)
        yb = base[seeds[0]].y_true.to_numpy(float)
        if len(y) != len(yb) or not np.allclose(y, yb):
            rows.append({"comparison": label, "band": band,
                         "status": "test rows differ; not comparable"})
            continue
        pm = np.mean([M[s].y_pred.to_numpy(float) for s in seeds], axis=0)
        pr = np.mean([base[s].y_pred.to_numpy(float) for s in seeds], axis=0)
        mask = np.ones(len(y), bool) if fn is None else fn(y)
        delta, lo, hi = _boot_ci(np.abs(pm - y)[mask], np.abs(pr - y)[mask])
        wins = sum(np.abs(M[s].y_pred.to_numpy(float) - y)[mask].mean() <
                   np.abs(base[s].y_pred.to_numpy(float) - y)[mask].mean()
                   for s in seeds)
        rows.append({"comparison": f"{label} [{ds_best} {arm_best}]", "band": band,
                     "status": "ok", "delta_mae": round(delta, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "significant": not (lo <= 0 <= hi), "wins": int(wins),
                     "n_seeds": len(seeds),
                     "method": "paired bootstrap, seed-averaged"})
    return rows


if __name__ == "__main__":
    main()
