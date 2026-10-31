#!/usr/bin/env python3
"""Aggregate benchmark results: per-arm means across seeds, plus paired comparisons.

Comparisons are paired by seed because every arm sees the identical DIR split for a
given seed, so the seed-to-seed split variance cancels. With only a handful of seeds
the Wilcoxon signed-rank test cannot go below p = 2^-(n-1) (p = 0.0625 at n = 5), so
the per-seed win count is reported alongside it rather than leaning on a p-value.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

METRICS = ["mae", "rmse", "gm", "sera", "mae_many", "mae_medium", "mae_few"]
LOWER_IS_BETTER = set(METRICS)  # every metric here is an error measure


def load(tags: list[str] | None, dataset: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(HERE, "results", f"{dataset}_*.jsonl")):
        for line in open(path):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if tags:
        df = df[df["tag"].isin(tags)]
    # Keep the last record for a given (arm, seed, tag) so re-runs supersede.
    return df.drop_duplicates(subset=["arm", "seed", "tag"], keep="last")


def select_lr_on_val(df: pd.DataFrame, prefix: str = "lr") -> pd.DataFrame:
    """Collapse the learning-rate sweep by picking, per (arm, seed), the LR with the
    lowest VALIDATION RMSE.

    Selection uses validation only -- the test set never participates. Doing this for
    both fine-tuning arms (not just the proposed one) is what makes the comparison
    fair: each arm is reported at its own best learning rate.
    """
    sweep = df[df["tag"].astype(str).str.startswith(prefix)].copy()
    if sweep.empty or "val_rmse" not in sweep:
        return pd.DataFrame()
    best = (sweep.sort_values("val_rmse")
                 .groupby(["arm", "seed"], as_index=False)
                 .first())
    best["tag"] = "lr_selected"
    return best


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["tag", "arm"])
    out = g[METRICS].agg(["mean", "std"])
    out[("n_seeds", "")] = g.size()
    return out


def paired(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Paired per-seed comparison of arm `b` against arm `a` (the reference)."""
    from scipy.stats import wilcoxon

    da = df[df["arm"] == a].set_index("seed")
    db = df[df["arm"] == b].set_index("seed")
    seeds = sorted(set(da.index) & set(db.index))
    if not seeds:
        return pd.DataFrame()

    rows = []
    for m in METRICS:
        va, vb = da.loc[seeds, m].to_numpy(float), db.loc[seeds, m].to_numpy(float)
        if np.isnan(va).any() or np.isnan(vb).any():
            continue
        diff = vb - va                      # negative => b better (lower error)
        wins = int((diff < 0).sum())        # seeds where b beat a
        try:
            p = float(wilcoxon(va, vb).pvalue) if len(seeds) >= 3 else float("nan")
        except ValueError:                  # all differences identically zero
            p = float("nan")
        rows.append({
            "metric": m,
            f"{a}": va.mean(),
            f"{b}": vb.mean(),
            "abs_delta": diff.mean(),
            "rel_delta_%": 100.0 * diff.mean() / va.mean(),
            "wins/n": f"{wins}/{len(seeds)}",
            "wilcoxon_p": p,
        })
    return pd.DataFrame(rows).set_index("metric")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="abalone")
    ap.add_argument("--tags", default=None, help="comma-separated; default = all")
    ap.add_argument("--select-lr", action="store_true",
                    help="collapse tags starting with 'lr' by best validation RMSE")
    ap.add_argument("--compare", default=None,
                    help="'ref:alt' pairs separated by commas, e.g. "
                         "'zeroshot:ft_gauss,ft_uniform:ft_gauss'")
    args = ap.parse_args()

    tags = args.tags.split(",") if args.tags else None
    df = load(tags, args.dataset)
    if df.empty:
        print("no results yet")
        return

    if args.select_lr:
        picked = select_lr_on_val(df)
        if not picked.empty:
            chosen = (picked.groupby(["arm", "config"], dropna=False).size()
                      if "config" in picked else None)
            print("=== learning rate chosen on VALIDATION rmse (per arm, per seed) ===")
            print(picked.assign(lr=picked["config"].apply(
                lambda c: c.get("lr") if isinstance(c, dict) else None))
                  .pivot_table(index="arm", columns="lr", values="seed",
                               aggfunc="count", fill_value=0).to_string())
            print()
            df = pd.concat([df[~df["tag"].astype(str).str.startswith("lr")], picked])

    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(f"=== {args.dataset}: per-arm summary (mean over seeds) ===")
    print(summary(df).round(4).to_string())

    if args.compare:
        for pair in args.compare.split(","):
            a, b = pair.split(":")
            print(f"\n=== paired: {b} vs {a}  (negative delta = {b} better) ===")
            res = paired(df, a, b)
            print("no overlapping seeds" if res.empty else res.round(4).to_string())


if __name__ == "__main__":
    main()
