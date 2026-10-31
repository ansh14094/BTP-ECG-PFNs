#!/usr/bin/env python3
"""Generate TABICL_TRANSFER_LOG.md — does the Gaussian-context idea transfer from
TabPFN-2.5 to a different in-context model?

All figures computed from results/*.jsonl; none typed by hand."""
from __future__ import annotations
import datetime, glob, json, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATASETS = ["abalone", "communities", "wine_white", "ptbxl_tabpfn_enc", "agedb_tabpfn_enc"]
M = ["mae", "rmse", "sera", "mae_many", "mae_medium", "mae_few"]
# (TabPFN arm, TabICL arm) pairs that answer the same question for each model.
PAIRS = {"zeroshot": "ticl_zeroshot", "ft_uniform": "ticl_ft_uniform",
         "ft_gauss": "ticl_ft_gauss"}


def load():
    rows = []
    for p in glob.glob(os.path.join(RES, "*.jsonl")):
        for l in open(p):
            if l.strip():
                rows.append(json.loads(l))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df.dataset.isin(DATASETS)]
    df = df.drop_duplicates(subset=["dataset", "arm", "seed", "tag"], keep="last")
    ft = df[df.arm.str.contains("ft_")]
    rest = df[~df.arm.str.contains("ft_")]
    if ft.empty or "val_rmse" not in ft:
        return rest
    best = ft.sort_values("val_rmse").groupby(["dataset", "arm", "seed"], as_index=False).first()
    return pd.concat([rest, best], ignore_index=True)


def md(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def paired(df, ds, ref, alt, metric="mae"):
    a = df[(df.dataset == ds) & (df.arm == ref)].set_index("seed")
    b = df[(df.dataset == ds) & (df.arm == alt)].set_index("seed")
    s = sorted(set(a.index) & set(b.index))
    if not s:
        return None
    va, vb = a.loc[s, metric].to_numpy(float), b.loc[s, metric].to_numpy(float)
    if np.isnan(va).any() or np.isnan(vb).any():
        return None
    d = vb - va
    w = int((d < 0).sum())
    return dict(ref=va.mean(), alt=vb.mean(), rel=100 * d.mean() / va.mean(),
                wins=f"{w}/{len(s)}", verdict="yes" if w >= 4 else ("no" if w <= 1 else "tie"))


def main():
    df = load()
    if df.empty:
        print("no results"); return

    head = """# Does the Gaussian-context idea transfer to another in-context model?

Generated {date} by `make_tabicl_report.py`.
All figures computed from `imbreg/results/*.jsonl`; none typed by hand.

## The question

Every result so far concerns TabPFN-2.5. If Gaussian-weighted context sampling only
works there, it is a quirk of one model. If it works on **TabICL v2** as well — a
different tabular foundation model, different architecture, different pre-training —
then the effect belongs to in-context learning itself, which is the claim worth making.

The intervention is deliberately identical. TabICL's fine-tuning splits each chunk via
`tabicl._finetune.data._split_ctx_query`, structurally the same hook as TabPFN's
`_ratio_ctx_query_split`. The replacement draws the query set first and unweighted,
then draws the context with Gaussian weight in target space, resampling (mu, sigma)
every episode, and subsamples the context to the same fraction in both arms so only
the sampling weight differs. Both arms get the same learning-rate sweep and are
reported at their own validation-selected rate.

## Headline: does Gaussian beat uniform, per model?

""".format(date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))

    rows = []
    for ds in DATASETS:
        r_pfn = paired(df, ds, "ft_uniform", "ft_gauss")
        r_icl = paired(df, ds, "ticl_ft_uniform", "ticl_ft_gauss")
        if not r_pfn and not r_icl:
            continue
        rows.append([
            f"`{ds}`",
            f"{r_pfn['rel']:+.1f}% ({r_pfn['wins']}) {r_pfn['verdict']}" if r_pfn else "—",
            f"{r_icl['rel']:+.1f}% ({r_icl['wins']}) {r_icl['verdict']}" if r_icl else "—",
        ])
    body = md(rows, ["dataset", "TabPFN-2.5: ft_gauss vs ft_uniform",
                     "TabICL v2: ft_gauss vs ft_uniform"])
    body += ("\n\nNegative percentages are improvements; the fraction is the number of "
             "seeds on which Gaussian sampling won. Seed-win counts rather than "
             "p-values, since Wilcoxon cannot go below p = 0.0625 at five seeds.\n")

    body += "\n## Absolute numbers, both models side by side\n\n"
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        rows = []
        for pfn_arm, icl_arm in PAIRS.items():
            for label, arm in ((f"TabPFN {pfn_arm}", pfn_arm), (f"TabICL {icl_arm.replace('ticl_','')}", icl_arm)):
                x = sub[sub.arm == arm]
                if x.empty:
                    continue
                rows.append([label, len(x)] + [f"{x[m].mean():.3f}" for m in M])
        if rows:
            body += f"### {ds}\n\n" + md(rows, ["arm", "seeds"] + M) + "\n\n"

    tail = """## Reading this

The comparison that carries the claim is `ft_gauss` vs `ft_uniform` **within** each
model: same model, same data, same budget, same episode sizes, same learning-rate
sweep, with the sampling weight as the only difference. Comparing TabPFN against
TabICL on absolute error is a separate and less interesting question — they are
different models with different pre-training, so whichever wins says little about the
intervention.

### Caveats

- Five seeds per cell.
- TabICL fine-tuning is far more memory-hungry than TabPFN's (roughly 30 GB on the
  PTB-XL embeddings against about 3 GB), so these runs were executed sequentially;
  that is a resource note, not a result.
- Where a dataset shows "—" the corresponding run has not completed.
"""
    out = os.path.join(HERE, "..", "TABICL_TRANSFER_LOG.md")
    out = os.path.normpath(out)
    open(out, "w").write(head + body + tail)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
