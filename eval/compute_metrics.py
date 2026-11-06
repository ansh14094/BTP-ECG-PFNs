#!/usr/bin/env python3
"""The one place metrics are computed. Nothing is measured inside a training loop.

Reads the per-prediction CSVs under `imbreg/results/preds/` and writes tidy
per-(dataset, arm, protocol, seed) rows to `imbreg/results/metrics/`.

Protocol, as fixed by paper_completion_plan.md section 1:

PTB-XL age bands, non-overlapping, in years:
    young   age < 30
    middle  30 <= age < 60
    old     age >= 60

PTB-XL metrics, all in years:
    mae_young / mae_middle / mae_old
    natural       MAE over the test split as it occurs -- requires the `fixed`
                  protocol, whose test set is the official PTB-XL partition and
                  therefore carries the natural age distribution.
    band_macro    plain mean of the three band MAEs; an explicit, reproducible
                  balancing that does not depend on how a test set was resampled.
    balanced      MAE under the `dir` protocol, whose test set is resampled to be
                  near-uniform over the target (see imbreg/data.py:dir_split, which
                  caps each 1-year age bin at `per_bin_test` rows). This is what the
                  existing code has always called balanced evaluation. It is
                  reported as-is and NOT recomputed differently.
    gm            exp(mean(log(|err| + 1e-8)))

Tabular benchmarks use the repo's existing shot regions (imbreg/metrics.py): bin the
target, count TRAIN rows per bin, and label a test row many- (>100), few- (<20) or
medium-shot by its bin's training frequency, following Yang et al. (2021).

    python3 eval/compute_metrics.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRED = os.path.join(ROOT, "imbreg", "results", "preds")
OUT = os.path.join(ROOT, "imbreg", "results", "metrics")

EPS = 1e-8
BANDS = (("young", lambda y: y < 30),
         ("middle", lambda y: (y >= 30) & (y < 60)),
         ("old", lambda y: y >= 60))

# <dataset>_<tag>_<protocol>_<arm>_s<seed>.csv. Dataset names are matched against
# the registry rather than by a wildcard: nearly every one of them contains an
# underscore ("ptbxl_enc_e3", "wine_white"), so a non-greedy group silently
# truncates them to the first segment and collapses distinct datasets into one.
# Longest-first so "ptbxl_enc_e3" is tried before any prefix of it.
def _dataset_names() -> list[str]:
    sys.path.insert(0, os.path.join(ROOT, "imbreg"))
    import data as data_mod
    # ptbxl_resnet is not a harness dataset -- it is the direct ResNet baseline
    # from eval/predict_resnet.py -- but its predictions are scored the same way.
    return sorted(set(data_mod.LOADERS) | {"ptbxl_resnet"}, key=len, reverse=True)


DATASETS = _dataset_names()
FNAME = re.compile(r"^(?P<dataset>" + "|".join(re.escape(d) for d in DATASETS) +
                   r")_(?P<tag>.+)_(?P<protocol>dir|fixed|natural)_"
                   r"(?P<arm>.+)_s(?P<seed>\d+)\.csv$")


def gm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Geometric mean of the absolute errors, per the plan's definition."""
    return float(np.exp(np.mean(np.log(np.abs(y_pred - y_true) + EPS))))


_DS_CACHE: dict = {}


def shot_metrics(dataset: str, protocol: str, seed: int,
                 y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Many-/medium-/few-shot MAE, following the repo's existing region definition.

    Shot regions are a property of the TRAINING distribution, so the split has to be
    reconstructed for this (dataset, protocol, seed) rather than read off a stored
    number. `dir_split` is deterministic in the seed, so this reproduces exactly the
    split the run used.
    """
    sys.path.insert(0, os.path.join(ROOT, "imbreg"))
    import data as data_mod
    import metrics as metrics_mod

    if dataset not in _DS_CACHE:
        if dataset not in data_mod.LOADERS:
            return {}
        _DS_CACHE[dataset] = data_mod.LOADERS[dataset]()
    ds = _DS_CACHE[dataset]

    if protocol == "dir":
        tr, _, _ = data_mod.dir_split(ds.y, ds.bin_width, seed=seed,
                                      per_bin_val=ds.per_bin_val,
                                      per_bin_test=ds.per_bin_test)
    elif protocol == "fixed" and ds.fixed_split is not None:
        tr = ds.fixed_split[0]
    else:
        return {}

    masks, _ = metrics_mod.shot_regions(ds.y[tr], y_true, bin_width=ds.bin_width)
    err = np.abs(y_pred - y_true)
    out = {}
    for region, m in masks.items():
        out[f"n_{region}"] = int(m.sum())
        out[f"mae_{region}"] = float(err[m].mean()) if m.sum() else float("nan")
    return out


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray, is_ptbxl: bool) -> dict:
    err = np.abs(y_pred - y_true)
    out = {"n": int(len(y_true)), "mae": float(err.mean()), "gm": gm(y_true, y_pred),
           "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2)))}
    if not is_ptbxl:
        return out
    band_means = []
    for name, fn in BANDS:
        m = fn(y_true)
        out[f"n_{name}"] = int(m.sum())
        v = float(err[m].mean()) if m.sum() else float("nan")
        out[f"mae_{name}"] = v
        band_means.append(v)
    out["band_macro"] = float(np.mean(band_means))
    return out


def main() -> None:
    files = sorted(glob.glob(os.path.join(PRED, "*.csv")))
    if not files:
        sys.exit(f"no prediction files in {PRED}")

    rows, skipped = [], []
    for f in files:
        m = FNAME.match(os.path.basename(f))
        if not m:
            skipped.append(os.path.basename(f))
            continue
        d = pd.read_csv(f)
        if not {"y_true", "y_pred"} <= set(d.columns):
            skipped.append(os.path.basename(f))
            continue
        g = m.groupdict()
        is_ptbxl = g["dataset"].startswith("ptbxl")
        rec = {k: g[k] for k in ("dataset", "tag", "protocol", "arm")}
        rec["seed"] = int(g["seed"])
        # `natural` and `balanced` are the same number measured on different test
        # sets, so the protocol column -- not a separate metric -- distinguishes them.
        rec["eval_set"] = {"fixed": "natural", "dir": "balanced"}.get(g["protocol"],
                                                                     g["protocol"])
        yt, yp = d.y_true.to_numpy(float), d.y_pred.to_numpy(float)
        rec.update(metrics_for(yt, yp, is_ptbxl))
        # Shot regions for the tabular benchmarks; the PTB-XL tables use age bands.
        if not is_ptbxl:
            rec.update(shot_metrics(g["dataset"], g["protocol"], rec["seed"], yt, yp))
        rec["has_id"] = "id" in d.columns
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values(
        ["dataset", "protocol", "arm", "tag", "seed"]).reset_index(drop=True)
    os.makedirs(OUT, exist_ok=True)
    allp = os.path.join(OUT, "all_metrics.csv")
    df.to_csv(allp, index=False)
    print(f"wrote {allp}: {len(df)} rows from {len(files)} prediction files")
    if skipped:
        print(f"skipped {len(skipped)} unparseable: {skipped[:4]}")

    for ds, sub in df.groupby("dataset"):
        p = os.path.join(OUT, f"{ds}.csv")
        sub.to_csv(p, index=False)
    print(f"wrote {df.dataset.nunique()} per-dataset files to {OUT}")
    print(f"\nprotocols present: {sorted(df.protocol.unique())}")
    print(f"datasets: {sorted(df.dataset.unique())}")


if __name__ == "__main__":
    main()
