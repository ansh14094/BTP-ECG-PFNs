#!/usr/bin/env python3
"""Mean absolute error by absolute age band for the PTB-XL TabICL arms.

The headline tables bin test points by how often their age appears in TRAINING
(many/medium/few-shot), which is the imbalanced-regression convention. This bins by
absolute age instead -- under 30, 30 to 60, 60 and over -- which is the clinically
readable split and the one the earlier ECG work in this repo used.

Nothing is re-run: it reads the per-prediction CSVs the harness already saved. Each
fine-tuned arm is taken at its own validation-selected learning rate, exactly as the
main tables do, so these numbers decompose those.

    python3 ticl_age_bands.py
"""
from __future__ import annotations
import glob, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "results", "preds")

BANDS = [("age < 30",      lambda y: y < 30),
         ("30 <= age < 60", lambda y: (y >= 30) & (y < 60)),
         ("age >= 60",     lambda y: y >= 60)]
DATASETS = {"PTB-XL encoder E1": "ptbxl_l1_enc", "PTB-XL encoder E2": "ptbxl_tabpfn_enc"}
ARMS = [("ticl_zeroshot", "TabICL zero-shot"),
        ("ticl_ft_uniform", "TabICL ft uniform"),
        ("ticl_ft_gauss", "TabICL ft Gaussian")]
SEEDS = [0, 1, 2, 3, 4]


def best_tag_per_seed(dataset: str, arm: str) -> dict[int, str]:
    """The learning-rate tag each seed selected on validation RMSE."""
    rows = []
    for p in glob.glob(os.path.join(HERE, "results", f"{dataset}_dir_ticl_*.jsonl")):
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df[df.arm == arm]
    if df.empty:
        return {}
    if arm == "ticl_zeroshot":
        return {int(s): t for s, t in zip(df.seed, df.tag)}
    best = df.sort_values("val_rmse").groupby("seed", as_index=False).first()
    return {int(s): t for s, t in zip(best.seed, best.tag)}


def main() -> None:
    for label, dsname in DATASETS.items():
        print(f"\n{'='*86}\n{label}   (mean absolute error in years, 5 seeds)")
        table = {}
        counts = {}
        for arm, pretty in ARMS:
            tags = best_tag_per_seed(dsname, arm)
            per_seed = {b: [] for b, _ in BANDS}
            n_seen = {b: [] for b, _ in BANDS}
            for s in SEEDS:
                tag = tags.get(s)
                if tag is None:
                    continue
                f = os.path.join(PRED, f"{dsname}_{tag}_dir_{arm}_s{s}.csv")
                if not os.path.exists(f):
                    print(f"  MISSING {os.path.basename(f)}")
                    continue
                d = pd.read_csv(f)
                y, p = d.y_true.to_numpy(float), d.y_pred.to_numpy(float)
                for bname, fn in BANDS:
                    m = fn(y)
                    if m.sum():
                        per_seed[bname].append(float(np.mean(np.abs(p[m] - y[m]))))
                        n_seen[bname].append(int(m.sum()))
            table[pretty] = per_seed
            counts = n_seen
        hdr = f"  {'arm':22s}" + "".join(f"{b:>18s}" for b, _ in BANDS)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for _, pretty in ARMS:
            row = table.get(pretty, {})
            cells = []
            for b, _ in BANDS:
                v = row.get(b, [])
                cells.append(f"{np.mean(v):10.3f} ±{np.std(v, ddof=1):5.3f}" if len(v) > 1
                             else (f"{np.mean(v):10.3f}       " if v else " " * 18))
            print(f"  {pretty:22s}" + "".join(cells))
        print(f"  {'test points / seed':22s}" +
              "".join(f"{int(np.mean(counts[b])):>18d}" for b, _ in BANDS))
        # paired per-seed comparison, Gaussian against uniform
        u, g = table.get("TabICL ft uniform", {}), table.get("TabICL ft Gaussian", {})
        print(f"  {'gauss - uniform':22s}", end="")
        for b, _ in BANDS:
            du, dg = np.array(u.get(b, [])), np.array(g.get(b, []))
            if len(du) == len(dg) and len(du):
                d = dg - du
                print(f"{d.mean():+10.3f} ({int((d<0).sum())}/{len(d)})", end="")
            else:
                print(" " * 18, end="")
        print()


if __name__ == "__main__":
    main()
