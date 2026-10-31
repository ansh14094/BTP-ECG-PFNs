#!/usr/bin/env python3
"""Generate AGEDB_EXPERIMENT_LOG.md. Every figure computed from imbreg/results."""
from __future__ import annotations
import datetime, glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "imbreg", "results")
ENC = {"agedb_l1_enc": "ResNet-18 trained on L1 age loss (control)",
       "agedb_tabpfn_enc": "ResNet-18 trained through frozen TabPFN-2.5"}
M = ["mae", "rmse", "gm", "sera", "mae_many", "mae_medium", "mae_few"]
ARMS = ["lgbm", "zeroshot", "zeroshot_recal", "zeroshot_recal_iso",
        "ft_uniform", "ft_gauss", "ft_gauss_recal"]
LBL = {"lgbm": "LightGBM on embeddings", "zeroshot": "TabPFN-2.5 zero-shot",
       "zeroshot_recal": "zero-shot + linear recalibration",
       "zeroshot_recal_iso": "zero-shot + isotonic recalibration",
       "ft_uniform": "TabPFN FT, uniform context", "ft_gauss": "TabPFN FT, Gaussian context",
       "ft_gauss_recal": "Gaussian-context FT + recalibration"}
# Published AgeDB-DIR numbers from this repo's README (Yang et al. protocol, MAE).
PRIOR = [("LDS", 7.67), ("FDS", 7.69), ("LDS + FDS", 7.47)]


def load():
    rows = []
    for ds in ENC:
        for p in glob.glob(os.path.join(RES, f"{ds}_*.jsonl")):
            for l in open(p):
                if l.strip():
                    rows.append(json.loads(l))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["dataset", "arm", "seed", "tag"], keep="last")
    ft, rest = df[df.arm.str.startswith("ft_")], df[~df.arm.str.startswith("ft_")]
    if ft.empty or "val_rmse" not in ft:
        return rest
    best = ft.sort_values("val_rmse").groupby(["dataset", "arm", "seed"], as_index=False).first()
    return pd.concat([rest, best], ignore_index=True)


def md(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def arm_table(df, ds):
    sub = df[df.dataset == ds]
    if sub.empty:
        return "_no results_"
    rows = []
    for a in ARMS:
        x = sub[sub.arm == a]
        if x.empty:
            continue
        rows.append([LBL[a], len(x)] + [f"{x[m].mean():.3f} ± {x[m].std():.3f}" for m in M])
    return md(rows, ["arm", "seeds", "MAE", "RMSE", "GM", "SERA",
                     "MAE many", "MAE medium", "MAE few"])


def paired(df, ds_a, arm_a, ds_b, arm_b):
    a = df[(df.dataset == ds_a) & (df.arm == arm_a)].set_index("seed")
    b = df[(df.dataset == ds_b) & (df.arm == arm_b)].set_index("seed")
    s = sorted(set(a.index) & set(b.index))
    if not s:
        return None
    rows = []
    for m in M:
        va, vb = a.loc[s, m].to_numpy(float), b.loc[s, m].to_numpy(float)
        if np.isnan(va).any() or np.isnan(vb).any():
            continue
        d = vb - va
        try:
            p = f"{wilcoxon(va, vb).pvalue:.4f}"
        except ValueError:
            p = "n/a"
        w = int((d < 0).sum())
        rows.append([m, f"{va.mean():.3f}", f"{vb.mean():.3f}",
                     f"{100*d.mean()/va.mean():+.1f}%", f"{w}/{len(s)}", p,
                     "WIN" if w >= 4 else ("LOSS" if w <= 1 else "tie")])
    return md(rows, ["metric", arm_a, arm_b, "rel delta", "seeds won", "p", ""])


def noop(df):
    rows = []
    for ds in ENC:
        z = df[(df.dataset == ds) & (df.arm == "zeroshot")].set_index("seed")["mae"]
        for a in ("ft_uniform", "ft_gauss", "ft_gauss_recal"):
            f = df[(df.dataset == ds) & (df.arm == a)].set_index("seed")["mae"]
            s = sorted(set(z.index) & set(f.index))
            if not s:
                continue
            same = int(np.sum(np.isclose(z.loc[s], f.loc[s], atol=1e-6)))
            rows.append([f"`{ds}`", f"`{a}`", f"{same}/{len(s)}",
                         "**VACUOUS — reverted to zero-shot**" if same >= len(s) - 1
                         else ("partly reverted" if same else "trained")])
    return md(rows, ["encoder", "arm", "seeds identical to zero-shot", "status"])


def main():
    df = load()
    if df.empty:
        print("no AgeDB results yet"); return
    doc = f"""# AgeDB-DIR age prediction with TabPFN-2.5 — experiment log

Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by `make_agedb_report.py`.
All figures computed from `imbreg/results/*.jsonl`; none typed by hand.

## 1. Why this dataset

AgeDB-DIR tests whether the two ideas validated on PTB-XL ECG transfer to a different
modality — face images instead of 12-lead signals — with the same imbalanced-regression
structure. 16,488 images, official split 12,208 / 2,140 / 2,140, ages 1–101, training
bin-count ratio **353:1**. Its val and test sets are already balanced over age bins by
construction (Yang et al., ICML 2021), so `--protocol fixed` is correct here and no
rebalancing is needed — unlike PTB-XL, whose official split had only six few-shot test
points.

### Published numbers for this benchmark (this repo's README)

{md([[n, f"{v:.2f}"] for n, v in PRIOR], ["method", "MAE"])}

## 2. A cleaner Stage-1 comparison than PTB-XL had

For PTB-XL the control encoder was a pre-existing checkpoint trained earlier on its own
schedule, so "encoder trained through TabPFN" vs "encoder trained for a linear head"
was not perfectly matched. Here both encoders start from the **same ImageNet ResNet-18
weights** and get the same optimiser, schedule, step budget and episode sampling. Only
the loss differs:

- `agedb_l1_enc` — ResNet-18 + linear head, L1 on age (control)
- `agedb_tabpfn_enc` — ResNet-18 features into a frozen TabPFN-2.5, bar-distribution
  NLL on query images (treatment)

Both are scored the same way at every evaluation — features into TabPFN — so the
comparison is on the quantity that matters rather than on each objective's own loss.

Stage 2 then fine-tunes TabPFN on the frozen embeddings with Gaussian context sampling,
identical to the PTB-XL and tabular runs.

## 3. Did the fine-tuning actually change the weights?

{noop(df)}

Early stopping restores best-validation weights, so a run whose validation score never
improves silently reverts to the original checkpoint and the arm is just zero-shot
relabelled. Any comparison involving such an arm is vacuous.

## 4. Results

"""
    for ds, desc in ENC.items():
        doc += f"### {ds} — {desc}\n\n{arm_table(df, ds)}\n\n"
        for a, b in [("ft_uniform", "ft_gauss"), ("zeroshot_recal_iso", "ft_gauss"),
                     ("zeroshot_recal_iso", "ft_gauss_recal")]:
            t = paired(df, ds, a, ds, b)
            if t:
                doc += f"**`{b}` vs `{a}`** (negative = `{b}` better)\n\n{t}\n\n"

    doc += "### Stage-1 effect: the two encoders under the same arm\n\n"
    for arm in ("zeroshot", "ft_gauss"):
        t = paired(df, "agedb_l1_enc", arm, "agedb_tabpfn_enc", arm)
        if t:
            doc += f"**arm = `{arm}`** (negative = TabPFN-trained encoder better)\n\n{t}\n\n"

    doc += """## 5. Caveats

- Five seeds; Wilcoxon cannot report below p = 0.0625 at n = 5, so seed-win counts and
  effect sizes carry the information, not the p-values.
- The encoders are ResNet-18 at 112x112, not the deeper backbones at 224x224 used by
  the published AgeDB-DIR numbers, so absolute MAEs are not directly comparable with
  them; the within-this-table comparisons are the meaningful ones.
- Gaussian context sampling is applied to TabPFN's fine-tuning episode sampler, not to
  its pre-training prior, which Prior Labs never released.
"""
    out = os.path.join(HERE, "AGEDB_EXPERIMENT_LOG.md")
    open(out, "w").write(doc)
    print(f"wrote {out} ({len(doc)} chars)")


if __name__ == "__main__":
    main()
