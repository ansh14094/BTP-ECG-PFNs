#!/usr/bin/env python3
"""Per-dataset verdict: arm table plus the paired comparisons that carry the claims.

Learning rate is selected per (arm, seed) on VALIDATION RMSE, for every fine-tuned
arm including the baselines, so no arm is reported at a rate another arm was denied.
"""
import glob, json, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

M = ["mae", "rmse", "sera", "mae_many", "mae_medium", "mae_few"]
ORDER = ["lgbm", "zeroshot", "ft_uniform", "ft_gauss",
         "zeroshot_recal", "zeroshot_recal_iso", "ft_gauss_recal"]
PAIRS = [("ft_uniform", "ft_gauss"),               # does biased sampling beat uniform?
         ("zeroshot_recal_iso", "ft_gauss"),       # does it beat free recalibration?
         ("zeroshot_recal_iso", "ft_gauss_recal")] # does the combination?


def load(ds):
    rows = []
    for p in glob.glob(f"results/{ds}_*.jsonl"):
        for l in open(p):
            if l.strip():
                rows.append(json.loads(l))
    df = pd.DataFrame(rows)
    df = df[df.tag != "main"]
    ft = df[df.arm.str.startswith("ft_")]
    rest = df[~df.arm.str.startswith("ft_")]
    best = ft.sort_values("val_rmse").groupby(["arm", "seed"], as_index=False).first()
    return pd.concat([rest, best])


def main(ds):
    sel = load(ds)
    if sel.empty:
        print(f"{ds}: no results"); return
    g = sel[sel.arm.isin(ORDER)].groupby("arm")
    t = g[M].mean().reindex([a for a in ORDER if a in g.groups])
    t.insert(0, "n", g.size().reindex(t.index))
    print(f"=== {ds.upper()} (LR selected on validation) ===")
    print(t.round(3).to_string())

    for a, b in PAIRS:
        da, db = sel[sel.arm == a].set_index("seed"), sel[sel.arm == b].set_index("seed")
        s = sorted(set(da.index) & set(db.index))
        if not s:
            continue
        print(f"\n  {b} vs {a}  ({len(s)} seeds, negative = {b} better)")
        for m in M:
            va, vb = da.loc[s, m].to_numpy(float), db.loc[s, m].to_numpy(float)
            d = vb - va
            try:
                p = wilcoxon(va, vb).pvalue
            except ValueError:
                p = float("nan")
            w = int((d < 0).sum())
            tag = "WIN " if w >= 4 else ("LOSS" if w <= 1 else "tie ")
            print(f"    {m:11s} {d.mean():+7.3f} ({100*d.mean()/va.mean():+6.1f}%)  "
                  f"{w}/{len(s)}  p={p:.4f}  {tag}")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["abalone", "communities", "wine_white"]):
        main(ds); print()
