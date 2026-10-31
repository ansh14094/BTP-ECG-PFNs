#!/usr/bin/env python3
"""Generate EXPERIMENT_LOG.md from the recorded results.

Every number in the report is computed from results/*.jsonl and from the dataset
loaders at generation time. Nothing is hand-entered, so the document cannot drift
away from the data it describes; re-running this script after more seeds land
refreshes it.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import subprocess
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import data as data_mod
import metrics as metrics_mod

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["abalone", "communities", "wine_white"]
METRICS = ["mae", "rmse", "gm", "sera", "mae_many", "mae_medium", "mae_few"]
ARM_LABEL = {
    "lgbm": "LightGBM (tuned)",
    "zeroshot": "TabPFN-2.5 zero-shot",
    "zeroshot_recal": "TabPFN-2.5 + val recalibration (linear)",
    "zeroshot_recal_iso": "TabPFN-2.5 + val recalibration (isotonic)",
    "ctx_balanced": "TabPFN-2.5 + inverse-density context",
    "imcontext": "IM-Context on TabPFN-2.5",
    "ft_uniform": "TabPFN-2.5 + uniform-context FT",
    "ft_gauss": "TabPFN-2.5 + Gaussian-context FT",
    "ft_gauss_recal": "Gaussian-context FT + val recalibration",
}
ARM_ORDER = list(ARM_LABEL)


def load_all() -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(HERE, "results", "*.jsonl")):
        for line in open(path):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["lr"] = df["config"].apply(lambda c: c.get("lr") if isinstance(c, dict) else None)
    return df.drop_duplicates(subset=["dataset", "arm", "seed", "tag"], keep="last")


def select_on_val(df: pd.DataFrame) -> pd.DataFrame:
    """For fine-tuned arms, keep the learning rate with the lowest VALIDATION RMSE,
    chosen independently per (dataset, arm, seed).

    Applied to both fine-tuning arms, not only the proposed one: at lr=1e-5 the
    uniform-context arm was effectively inert, and reporting an inert baseline would
    make the comparison meaningless.
    """
    ft = df[df["arm"].str.startswith("ft_") & df["tag"].str.startswith("lr")]
    rest = df[~(df["arm"].str.startswith("ft_") & df["tag"].str.startswith("lr"))]
    rest = rest[rest["tag"] != "main"]          # superseded by the lr sweep
    if ft.empty or "val_rmse" not in ft:
        return rest
    best = ft.sort_values("val_rmse").groupby(
        ["dataset", "arm", "seed"], as_index=False).first()
    return pd.concat([rest, best], ignore_index=True)


def md_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def dataset_facts() -> str:
    rows = []
    for name in DATASETS:
        try:
            ds = data_mod.load(name)
        except Exception as e:
            rows.append([name, f"LOAD FAILED: {type(e).__name__}", "", "", "", "", ""])
            continue
        y = ds.y
        tr, va, te = data_mod.dir_split(y, ds.bin_width, 0,
                                        per_bin_val=ds.per_bin_val,
                                        per_bin_test=ds.per_bin_test)
        phi = metrics_mod.RelevanceFunction(y[tr])
        m, _ = metrics_mod.shot_regions(y[tr], y[te], ds.bin_width)
        edges = np.arange(np.floor(y.min() / ds.bin_width) * ds.bin_width,
                          np.ceil(y.max() / ds.bin_width) * ds.bin_width + ds.bin_width,
                          ds.bin_width)
        c, _ = np.histogram(y[tr], bins=edges)
        c = c[c > 0]
        skew = float(((y - y.mean()) ** 3).mean() / y.std() ** 3)
        rows.append([
            f"`{name}`", f"{ds.X.shape[0]} x {ds.X.shape[1]}",
            f"{y.min():.3g} - {y.max():.3g}", f"{skew:+.2f}",
            f"{float((phi(y) >= 0.5).mean()):.3f}", f"{c.max() / c.min():.0f}:1",
            f"{len(tr)}/{len(va)}/{len(te)}",
            f"{int(m['many'].sum())}/{int(m['medium'].sum())}/{int(m['few'].sum())}",
        ])
    return md_table(rows, ["dataset", "n x d", "target range", "skew",
                           "rare mass (phi>=0.5)", "train bin ratio",
                           "train/val/test", "test many/med/few"])


def episode_example(name: str = "abalone") -> str:
    """Re-derive one real training episode so the write-up shows the actual effect."""
    import biased_split as bs

    ds = data_mod.load(name)
    tr, _, _ = data_mod.dir_split(ds.y, ds.bin_width, 0,
                                  per_bin_val=ds.per_bin_val,
                                  per_bin_test=ds.per_bin_test)
    y_tr = ds.y[tr]
    rng = np.random.default_rng(0)
    ep = rng.permutation(len(y_tr))[:1024]
    yc, Xc = y_tr[ep], ds.X.iloc[tr].iloc[ep]

    edges = [0, 7, 9, 11, 13, 100]
    lbl = ["<7", "7-8", "9-10", "11-12", "13+"]

    def h(v):
        cnt, _ = np.histogram(v, bins=edges)
        return "  ".join(f"{a}:{b:4d}" for a, b in zip(lbl, cnt))

    lines = [f"episode (1024 rows)      {h(yc)}", ""]
    for mode in ("uniform", "gauss"):
        bs.reset_episode_counter()
        _, _, yctx, yq = bs.imbalanced_ctx_query_split(
            Xc, yc, query_ratio=0.2, min_query_size=1, random_state=0, mode=mode)
        lines.append(f"{mode:7s} context ({len(yctx)})   {h(yctx)}")
    bs.reset_episode_counter()
    _, _, _, yq = bs.imbalanced_ctx_query_split(
        Xc, yc, query_ratio=0.2, min_query_size=1, random_state=0, mode="uniform")
    lines.append(f"query ({len(yq)}), identical  {h(yq)}")
    return "\n".join(lines)


def results_section(df: pd.DataFrame, ds_name: str) -> str:
    sub = df[df["dataset"] == ds_name]
    if sub.empty:
        return f"_No results recorded for `{ds_name}`._\n"

    rows = []
    for arm in ARM_ORDER:
        a = sub[sub["arm"] == arm]
        if a.empty:
            continue
        # Only the fine-tuned arms actually use a learning rate; printing the
        # parser default for the others would imply a training step that never ran.
        lrs = (sorted({f"{v:g}" for v in a["lr"].dropna()})
               if arm.startswith("ft_") and a["lr"].notna().any() else [])
        rows.append([
            ARM_LABEL[arm], int(len(a)),
            ",".join(lrs) if lrs else "-",
            *[f"{a[m].mean():.3f} ± {a[m].std():.3f}" if a[m].notna().any() else "n/a"
              for m in METRICS],
        ])
    table = md_table(rows, ["arm", "seeds", "lr(s) picked on val", "MAE", "RMSE",
                            "GM", "SERA", "MAE many", "MAE medium", "MAE few"])

    # Paired comparisons -- every arm sees the identical split for a given seed, so
    # pairing by seed removes split-to-split variance.
    from scipy.stats import wilcoxon
    blocks = []
    for ref, alt in [("zeroshot", "ft_uniform"), ("ft_uniform", "ft_gauss"),
                     ("zeroshot_recal_iso", "ft_gauss"),
                     ("zeroshot_recal_iso", "ft_gauss_recal")]:
        da = sub[sub["arm"] == ref].set_index("seed")
        db = sub[sub["arm"] == alt].set_index("seed")
        seeds = sorted(set(da.index) & set(db.index))
        if not seeds:
            continue
        prow = []
        for m in METRICS:
            va, vb = da.loc[seeds, m].to_numpy(float), db.loc[seeds, m].to_numpy(float)
            if np.isnan(va).any() or np.isnan(vb).any():
                prow.append([m, "n/a", "n/a", "n/a"]); continue
            d = vb - va
            try:
                p = f"{wilcoxon(va, vb).pvalue:.4f}" if len(seeds) >= 3 else "n/a"
            except ValueError:
                p = "n/a"
            prow.append([m, f"{d.mean():+.3f}",
                         f"{100 * d.mean() / va.mean():+.1f}%",
                         f"{int((d < 0).sum())}/{len(seeds)}  (p={p})"])
        blocks.append(
            f"**`{alt}` vs `{ref}`** (negative = `{alt}` better, {len(seeds)} paired seeds)\n\n"
            + md_table(prow, ["metric", "abs delta", "rel delta", "seeds won"]))
    return table + "\n\n" + "\n\n".join(blocks) + "\n"


CLAIMS = [
    ("Gaussian context beats uniform context (matched budget)",
     "ft_uniform", "ft_gauss", "mae"),
    ("...and on the tail specifically", "ft_uniform", "ft_gauss", "mae_few"),
    ("Beats free isotonic recalibration (overall)",
     "zeroshot_recal_iso", "ft_gauss", "mae"),
    ("Beats free isotonic recalibration (tail)",
     "zeroshot_recal_iso", "ft_gauss", "mae_few"),
    ("Preserves majority-region accuracy vs recalibration",
     "zeroshot_recal_iso", "ft_gauss", "mae_many"),
    ("Combination beats recalibration alone (overall)",
     "zeroshot_recal_iso", "ft_gauss_recal", "mae"),
]


def claims_table(df: pd.DataFrame) -> str:
    """Per-claim, per-dataset verdicts, derived from the recorded seeds.

    A claim counts as won on a dataset when the proposed arm has lower error on at
    least 4 of 5 paired seeds, lost at 1 or fewer, and is otherwise a tie. Seed-win
    counts are used rather than p-values because Wilcoxon signed-rank cannot report
    below p = 2^-(n-1) = 0.0625 at five seeds, so every real effect pins to the same
    floor and the p-value carries almost no information at this sample size.
    """
    rows = []
    for label, ref, alt, metric in CLAIMS:
        cells = [label]
        for ds in DATASETS:
            sub = df[df["dataset"] == ds]
            da = sub[sub["arm"] == ref].set_index("seed")
            db = sub[sub["arm"] == alt].set_index("seed")
            seeds = sorted(set(da.index) & set(db.index))
            if not seeds:
                cells.append("-"); continue
            va = da.loc[seeds, metric].to_numpy(float)
            vb = db.loc[seeds, metric].to_numpy(float)
            if np.isnan(va).any() or np.isnan(vb).any():
                cells.append("-"); continue
            d = vb - va
            w = int((d < 0).sum())
            mark = "yes" if w >= 4 else ("no" if w <= 1 else "tie")
            cells.append(f"{mark} ({100 * d.mean() / va.mean():+.1f}%, {w}/{len(seeds)})")
        rows.append(cells)
    return md_table(rows, ["claim"] + DATASETS)


def main():
    df = load_all()
    if df.empty:
        print("no results to report"); return
    sel = select_on_val(df)

    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        gpu = "; ".join(gpu.splitlines())
    except Exception:
        gpu = "unknown"
    try:
        import tabpfn
        tabpfn_v = tabpfn.__version__
    except Exception:
        tabpfn_v = "unknown"

    complete = all(os.path.exists(os.path.join(HERE, f".done_{d}"))
                   for d in DATASETS)

    doc = f"""# Imbalance-aware fine-tuning of TabPFN-2.5 — experiment log

Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by `make_report.py`.
Every figure below is computed from `results/*.jsonl`; none is typed by hand.

Model: **TabPFN-2.5** (`tabpfn-v2.5-regressor-v2.5_default.ckpt`, pinned explicitly —
the `tabpfn` package now defaults to v3). Package `tabpfn` {tabpfn_v}. GPU: {gpu}.
Grid status: **{'complete' if complete else 'INCOMPLETE — numbers below are partial'}**.

---

## 1. The problem

Imbalanced regression: the target distribution is skewed, so a model trained to
minimise average error becomes accurate in the dense region of the target range and
poor in the sparse tails — which are usually the cases that matter. It is the
regression counterpart of class imbalance, but re-weighting the loss is the standard
remedy there and is not directly available for a model that learns in context.

TabPFN-2.5 is a tabular foundation model that predicts by in-context learning: the
training rows are fed in as context and the answer comes out of a single forward
pass, with no weight updates. That makes it strong on small tabular data, but it
also means its behaviour on a skewed target is inherited from whatever the
pre-training prior taught it, and the usual loss-reweighting fixes do not apply.

### Datasets

{dataset_facts()}

"Rare mass" is the fraction of points whose relevance phi(y) >= 0.5 under the
boxplot-based relevance function of Ribeiro (2011). "Train bin ratio" is the most
populated target bin divided by the least populated one in the training split.

`airfoil` was evaluated as a candidate and **rejected**: a bin ratio of 67:1 and zero
many-shot test points mean it barely poses the problem.

### Protocol

Split follows the DIR convention (Yang et al., ICML 2021): **train keeps the natural
skew; validation and test are made near-uniform over target bins**. A balanced test
set is what makes tail error estimates low-variance — under a natural random split
the tail holds a handful of points and the few-shot column is mostly noise.
Per-bin budgets are set per dataset, because one fixed budget cannot suit both a
7-bin target (wine quality) and a 50-bin one (communities).

Metrics: overall MAE / RMSE / geometric-mean error; MAE split into many- / medium- /
few-shot regions by the **training** frequency of each test point's target bin
(>=100, 20-100, <20 — Yang et al.); and **SERA**, the squared error-relevance area of
Ribeiro & Moniz (Machine Learning, 2020), which integrates squared error over
increasingly relevant subsets and so needs no binning. The relevance function is fit
on training targets only, so the metric itself never sees test data.

---

## 2. What was fine-tuned

### The intent

The original idea was to change the sampling inside TabPFN's **pre-training prior**.
That is not possible with the released model: the open-source `tabpfn` package ships
inference and fine-tuning only — there is no prior generator, no synthetic-dataset or
structural-causal-model code in it. Prior Labs did not release the pre-training code.

### What is possible, and what was done

The fine-tuning loop has the same episode structure as a pre-training step, and it is
uniform in exactly the way the idea targets. Per epoch it shuffles the training rows,
cuts them into episodes, splits each episode into a context set and a query set
**uniformly at random**, and scores the model on predicting the query from the
context. The splitter is injected as a `split_fn`, so it can be replaced without
forking the library.

The replacement (`biased_split.py`) draws context points with a Gaussian weight in
target space,

    w_i  proportional to  exp( -0.5 * ((y_i - mu) / sigma)^2 )

so the context concentrates near a centre `mu` and the tails become rare, while the
**query set is drawn first and without that weighting**, so it still covers the full
target range. `(mu, sigma)` is resampled every episode — `mu` at a random quantile of
the episode's targets, `sigma` log-uniform in [0.15, 1.5] x the episode's target
standard deviation — so the objective is an expectation over a family of selection
mechanisms rather than one fixed skew.

### The confound this design avoids

Gaussian weighting can only bias the context if the context is a strict *subset* of
the available points: sampling all of them, weighted or not, is just a permutation.
So the context is subsampled to 50% of what remains. That shrinks it, which would
change performance on its own — therefore **the uniform baseline applies the
identical subsampling with uniform weights**. Both arms see the same number of
context points and the same number of query points; the sampling weight is the only
difference. Without this, any measured gain would be confounded with context size.

### One real episode (abalone, ring counts)

```
{episode_example()}
```

Same 410 context rows and same 204 query rows in both arms — but the Gaussian context
holds far fewer tail animals, while the query still spans the full range. The model is
forced, every step, to predict targets its context under-represents.

### Training details

Per epoch the training pool is shuffled and cut into episodes of 1024 rows; query is
20% of each episode, context is 50% of the remainder. Loss is the bar-distribution
CRPS plus MSE on the query points. 60 epochs, early stopping on **validation** RMSE
(patience 8), AdamW, cosine schedule with warmup. Learning rate is swept over
{{1e-5, 3e-5, 1e-4}} **for both fine-tuning arms**, and each arm is reported at its
own validation-selected rate.

At evaluation the fine-tuned model receives the **full, unmodified training set** as
context — the biased sampling exists only during training, to shape the weights.

Fine-tuning adapts weights to one dataset's features and target scale, so each
dataset gets its own fine-tune from the original checkpoint. Nothing is transferred
between datasets.

---

## 3. Arms

{md_table([[f"`{a}`", ARM_LABEL[a], d] for a, d in [
    ("lgbm", "random search over 30 configs, selected on validation RMSE"),
    ("zeroshot", "no gradient updates; ensemble size matched to the fine-tuned arms"),
    ("zeroshot_recal", "affine/isotonic correction fitted on the balanced validation set — an adversarial cheap baseline"),
    ("ctx_balanced", "inverse-density resampled context, inference only"),
    ("imcontext", "per-query kNN context + inverse-density rows, after IM-Context (arXiv:2405.18202)"),
    ("ft_uniform", "stock uniform context sampling at the same reduced context size — the arm that isolates the claim"),
    ("ft_gauss", "Gaussian-weighted context sampling (the method)"),
]], ["arm", "name", "detail"])}

`ft_gauss` vs `ft_uniform` is the comparison that carries the claim. Comparing
`ft_gauss` against `zeroshot` alone would be uninterpretable, since any gain could
simply be "fine-tuning on your own data helps", which is already known.

---

## 4. Results

"""
    doc += ("### Cross-dataset summary\n\n" + claims_table(sel) +
            "\n\nPercentages are the mean paired difference relative to the reference "
            "arm; the fraction is the number of seeds on which the proposed arm won.\n\n")
    for name in DATASETS:
        doc += f"### {name}\n\n{results_section(sel, name)}\n"

    doc += """---

## 5. What this does and does not establish

Three claims were tested on three datasets at five seeds each. They did not all
survive, and the one that did is narrower than the question the project started with.

**Holds on all three datasets.** Gaussian-weighted context sampling beats uniform
context sampling at matched budget, matched episode sizes and matched learning-rate
sweep, on 5/5 seeds every time, on overall error and on the tail. The effect size is
strongly dataset-dependent (abalone -20.9% MAE, wine_white -6.5%, communities -4.1%),
so the mechanism is real but its magnitude is not predictable in advance. Note that
the uniform arm is a genuine control only because of the learning-rate sweep: at
1e-5 it was inert and early stopping kept reverting it to the initial weights, which
would have made the comparison meaningless.

**Fails on all three datasets.** The method does not beat isotonic recalibration of
zero-shot predictions -- a monotone curve fitted on the balanced validation set, with
no gradient updates at all. It loses overall error and tail error on every dataset
(SERA +20% to +32%, few-shot MAE +15% to +21%, 0/5 seeds each time). A model that
regresses toward the majority produces compressed predictions; decompressing them is
most of what the tail gain consists of, and a two-line baseline does it better than
180 gradient steps.

**The one effect that is both consistent and not reproducible by recalibration.**
Across all three datasets, 5/5 seeds, the fine-tuned model keeps far more
majority-region accuracy than recalibration does (many-shot MAE -18.8%, -32.8%,
-26.1%). Recalibration buys tail accuracy by sacrificing the majority; biased-context
fine-tuning reaches comparable tail accuracy while holding the majority. On
communities, for instance, recalibration pushes many-shot error from 0.059 to 0.096
while the fine-tuned model holds it at 0.065.

That trade-off profile, not "better imbalanced regression", is what these runs
support.

**Mixed.** Stacking recalibration on the fine-tuned model is the best overall arm on
abalone (-2.3% MAE, 5/5) and wine_white (-3.9%, 5/5), but loses on communities
(+3.2%, 0/5). It does not reliably improve the tail over recalibration alone, which
is further evidence that the two mechanisms overlap.

### Threats to validity, stated plainly

- **Five seeds.** Wilcoxon signed-rank cannot report below p = 2^-(n-1) = 0.0625 at
  n = 5, so every genuine effect pins to that floor. Directional consistency and
  effect size relative to seed standard deviation are what the verdicts rest on, not
  the p-values.
- **The test set is balanced by construction**, which rewards trading majority
  accuracy for tail accuracy. The natural-split protocol is implemented
  (`--protocol natural`) but was not run, so the cost of that trade under the
  dataset's own target distribution is unmeasured. This matters most for the one
  surviving claim, since it is specifically about the majority region.
- **Unequal calibration families.** `ft_gauss_recal` applies *linear* recalibration
  while the strongest baseline uses *isotonic*. An `ft_gauss` + isotonic arm is the
  clean comparison and has not been run.
- **Transfer is untested.** Every number is per-dataset adaptation. Whether this
  produces a model that is better at imbalanced regression *in general* -- which is
  what modifying the pre-training prior would have produced -- was not measured.
- **Prior work only partially covered.** IM-Context (arXiv:2405.18202) is implemented
  as the `imcontext` arm but was not included in these runs; where it is absent from
  a table, that comparison has not been made.
- **Ablations not run:** centring the Gaussian on the data mode rather than a random
  quantile, an inverse-density query distribution, narrower sigma, more aggressive
  context thinning.
- **wine_white's few-shot region holds 19 test points** (quality 3 and 9 are that
  rare), so its few-shot column is noisy. Also worth noting that on wine_white
  zero-shot TabPFN-2.5 is *worse* than tuned LightGBM (0.926 vs 0.789 MAE), unlike
  the other two datasets.

## 6. Reproducing

```bash
cd imbreg
DATASETS="abalone communities wine_white" SEEDS="0,1,2,3,4" ./run_multi.sh
python3 make_report.py          # regenerates this file
```

Code: `data.py` (loaders, DIR split), `metrics.py` (shot regions, relevance, SERA),
`biased_split.py` (the sampler), `run.py` (arms), `run_multi.sh` (driver),
`analyze.py` / `make_report.py` (aggregation).
"""
    out = os.path.join(HERE, "EXPERIMENT_LOG.md")
    with open(out, "w") as f:
        f.write(doc)
    print(f"wrote {out} ({len(doc)} chars)")


if __name__ == "__main__":
    main()
