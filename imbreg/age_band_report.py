#!/usr/bin/env python3
"""Age-band breakdown for the PTB-XL arms (age <= 30, 30-60, >= 60).

The main reports bin test points by how often their age appears in TRAINING
(many/medium/few-shot). That is the imbalanced-regression convention, but it is not
the clinically readable one, and it is not what this repo's earlier ECG work used.
This recomputes the same predictions by absolute age band instead -- matching the
`many (age<=30) / median (30<age<60) / low (age>=60)` split in
tabpfn_feature_experiment.py -- so the two are directly comparable.

Nothing is re-run: it reads the saved per-prediction CSVs.
"""
from __future__ import annotations
import glob, json, os, re, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "results", "preds")
BANDS = [("age <= 30", lambda y: y <= 30),
         ("30 < age < 60", lambda y: (y > 30) & (y < 60)),
         ("age >= 60", lambda y: y >= 60)]
ARMS = ["zeroshot", "zeroshot_recal_iso", "ft_uniform", "ft_gauss", "ft_mixed", "ft_gauss_recal"]


def val_selected_lr(dataset, protocol, arm):
    """Mirror the reports: pick each seed's learning rate by validation RMSE."""
    rows = []
    for p in glob.glob(os.path.join(HERE, "results", f"{dataset}_*.jsonl")):
        for l in open(p):
            if l.strip():
                rows.append(json.loads(l))
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    df = df[(df.protocol == protocol) & (df.arm == arm) &
            (df.tag.str.startswith("dir_") == (protocol == "dir"))]
    if df.empty or "val_rmse" not in df:
        return {}
    best = df.sort_values("val_rmse").groupby("seed", as_index=False).first()
    return dict(zip(best.seed, best.tag))


def collect(dataset, protocol):
    pre = "dir_" if protocol == "dir" else ""
    out = {}
    for arm in ARMS:
        tags = val_selected_lr(dataset, protocol, arm)
        if not tags:
            tags = {s: f"{pre}baseline" for s in range(5)}
        per_band = collections.defaultdict(list)
        overall = []
        for seed, tag in tags.items():
            f = os.path.join(PRED, f"{dataset}_{tag}_{protocol}_{arm}_s{seed}.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            err = np.abs(d.y_pred - d.y_true)
            overall.append(err.mean())
            for name, fn in BANDS:
                m = fn(d.y_true.to_numpy())
                if m.sum():
                    per_band[name].append(float(err[m].mean()))
        if overall:
            out[arm] = {"overall": float(np.mean(overall)),
                        **{b: float(np.mean(v)) for b, v in per_band.items()},
                        "n_seeds": len(overall)}
    return out


def counts(dataset, protocol):
    tags = {0: ("dir_baseline" if protocol == "dir" else "baseline")}
    f = os.path.join(PRED, f"{dataset}_{tags[0]}_{protocol}_zeroshot_s0.csv")
    if not os.path.exists(f):
        return {}
    y = pd.read_csv(f).y_true.to_numpy()
    return {name: int(fn(y).sum()) for name, fn in BANDS}


def main():
    for protocol, desc in [("fixed", "official PTB-XL split"),
                           ("dir", "age-rebalanced split")]:
        for ds in ("ptbxl_l1_enc", "ptbxl_tabpfn_enc", "ptbxl_enc_e22"):
            res = collect(ds, protocol)
            if not res:
                continue
            n = counts(ds, protocol)
            print(f"\n=== {ds}  [{protocol} — {desc}]   test points per band: "
                  + ", ".join(f"{k}={v}" for k, v in n.items()))
            print(f"{'arm':22s}{'Overall':>9s}" + "".join(f"{b:>16s}" for b, _ in BANDS))
            for arm in ARMS:
                if arm not in res:
                    continue
                r = res[arm]
                print(f"{arm:22s}{r['overall']:9.3f}" +
                      "".join(f"{r.get(b, float('nan')):16.3f}" for b, _ in BANDS))


if __name__ == "__main__":
    main()
