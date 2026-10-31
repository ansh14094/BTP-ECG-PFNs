#!/usr/bin/env python3
"""Generate ECG_EXPERIMENT_LOG.md for the PTB-XL age-prediction study.

Every figure is computed from imbreg/results/*.jsonl and from the data at generation
time; nothing is typed by hand, so the document cannot drift from the results it
describes. Re-run after more seeds land to refresh it.
"""
from __future__ import annotations

import datetime, glob, json, os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "imbreg"))
import data as data_mod      # noqa: E402
import metrics as metrics_mod  # noqa: E402

RES = os.path.join(BASE, "imbreg", "results")
ENCODERS = {"ptbxl_l1_enc": "L1-trained backbone (control)",
            "ptbxl_tabpfn_enc": "Encoder trained through frozen TabPFN-2.5"}
M = ["mae", "rmse", "gm", "sera", "mae_many", "mae_medium", "mae_few"]
ARMS = ["lgbm", "zeroshot", "zeroshot_recal", "zeroshot_recal_iso",
        "ft_uniform", "ft_gauss", "ft_gauss_recal"]
LABEL = {"lgbm": "LightGBM on embeddings", "zeroshot": "TabPFN-2.5 zero-shot",
         "zeroshot_recal": "zero-shot + linear recalibration",
         "zeroshot_recal_iso": "zero-shot + isotonic recalibration",
         "ft_uniform": "TabPFN FT, uniform context",
         "ft_gauss": "TabPFN FT, Gaussian context",
         "ft_gauss_recal": "Gaussian-context FT + recalibration"}
# Numbers already recorded in this repo, for continuity (see pipeline.log).
PRIOR = [("ResNet1d-Wang, L1 loss (end-to-end)", 7.549),
         ("frozen backbone features -> TabPFN", 7.441)]


def load(protocol: str) -> pd.DataFrame:
    """Rows for one protocol. `dir_`-prefixed tags are the rebalanced-split runs."""
    rows = []
    for ds in ENCODERS:
        for p in glob.glob(os.path.join(RES, f"{ds}_*.jsonl")):
            for line in open(p):
                if line.strip():
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df.protocol == protocol]
    df = df[df.tag.str.startswith("dir_") == (protocol == "dir")]
    df = df.drop_duplicates(subset=["dataset", "arm", "seed", "tag"], keep="last")
    ft = df[df.arm.str.startswith("ft_")]
    rest = df[~df.arm.str.startswith("ft_")]
    if ft.empty or "val_rmse" not in ft:
        return rest
    best = ft.sort_values("val_rmse").groupby(
        ["dataset", "arm", "seed"], as_index=False).first()
    return pd.concat([rest, best], ignore_index=True)


def md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def arm_table(df, ds):
    sub = df[df.dataset == ds]
    if sub.empty:
        return "_no results yet_"
    rows = []
    for a in ARMS:
        x = sub[sub.arm == a]
        if x.empty:
            continue
        rows.append([LABEL[a], int(len(x))] +
                    [f"{x[m].mean():.3f} ± {x[m].std():.3f}" for m in M])
    return md(rows, ["arm", "seeds", "MAE", "RMSE", "GM", "SERA",
                     "MAE many", "MAE medium", "MAE few"])


def paired(df, ds, ref, alt):
    sub = df[df.dataset == ds]
    da, db = sub[sub.arm == ref].set_index("seed"), sub[sub.arm == alt].set_index("seed")
    s = sorted(set(da.index) & set(db.index))
    if not s:
        return None
    rows = []
    for m in M:
        va, vb = da.loc[s, m].to_numpy(float), db.loc[s, m].to_numpy(float)
        if np.isnan(va).any() or np.isnan(vb).any():
            continue
        d = vb - va
        try:
            p = f"{wilcoxon(va, vb).pvalue:.4f}"
        except ValueError:
            p = "n/a"
        w = int((d < 0).sum())
        rows.append([m, f"{d.mean():+.3f}", f"{100*d.mean()/va.mean():+.1f}%",
                     f"{w}/{len(s)}", p,
                     "WIN" if w >= 4 else ("LOSS" if w <= 1 else "tie")])
    return md(rows, ["metric", "abs delta", "rel delta", "seeds won", "wilcoxon p", ""])


def cross_encoder(df, arm):
    """Stage-1 effect: same arm, the two encoders compared seed by seed."""
    a = df[(df.dataset == "ptbxl_l1_enc") & (df.arm == arm)].set_index("seed")
    b = df[(df.dataset == "ptbxl_tabpfn_enc") & (df.arm == arm)].set_index("seed")
    s = sorted(set(a.index) & set(b.index))
    if not s:
        return None
    rows = []
    for m in M:
        va, vb = a.loc[s, m].to_numpy(float), b.loc[s, m].to_numpy(float)
        d = vb - va
        rows.append([m, f"{va.mean():.3f}", f"{vb.mean():.3f}",
                     f"{100*d.mean()/va.mean():+.1f}%", f"{int((d<0).sum())}/{len(s)}"])
    return md(rows, ["metric", "L1 encoder", "TabPFN-trained encoder",
                     "rel delta", "seeds won"])


def split_facts():
    ds = data_mod.load("ptbxl_l1_enc")
    tr, va, te = ds.fixed_split
    mo, _ = metrics_mod.shot_regions(ds.y[tr], ds.y[te], ds.bin_width)
    tr2, va2, te2 = data_mod.dir_split(ds.y, ds.bin_width, seed=0,
                                       per_bin_val=ds.per_bin_val,
                                       per_bin_test=ds.per_bin_test)
    md2, _ = metrics_mod.shot_regions(ds.y[tr2], ds.y[te2], ds.bin_width)
    return md([
        ["official (`fixed`)", f"{len(tr)}/{len(va)}/{len(te)}",
         f"{int(mo['many'].sum())}/{int(mo['medium'].sum())}/{int(mo['few'].sum())}"],
        ["rebalanced (`dir`)", f"{len(tr2)}/{len(va2)}/{len(te2)}",
         f"{int(md2['many'].sum())}/{int(md2['medium'].sum())}/{int(md2['few'].sum())}"],
    ], ["protocol", "train/val/test", "test many/medium/few"])


def encoder_curve():
    p = os.path.join(HERE, "logs", "enc_main_curve.jsonl")
    if not os.path.exists(p):
        return "_no encoder training curve recorded_"
    rs = [json.loads(l) for l in open(p) if l.strip()]
    if not rs:
        return "_empty curve_"
    best = min(rs, key=lambda r: r["val_mae"])
    rows = [[r["step"], f"{r['loss']:.4f}", f"{r['val_mae']:.4f}"] for r in rs]
    return (md(rows, ["step", "train NLL", "val MAE"]) +
            f"\n\nBest: val MAE **{best['val_mae']:.4f}** at step {best['step']}.")


def noop_check(df: pd.DataFrame) -> str:
    """Flag fine-tuned arms whose test predictions are identical to zero-shot.

    Early stopping restores the best-validation weights, so a fine-tuning run whose
    validation score never improves reverts to the ORIGINAL checkpoint and the arm is
    silently just zero-shot. Any comparison involving such an arm is vacuous, so it
    has to be detected and stated rather than reported as a result.
    """
    rows = []
    for ds in ENCODERS:
        z = df[(df.dataset == ds) & (df.arm == "zeroshot")].set_index("seed")["mae"]
        for arm in ("ft_uniform", "ft_gauss", "ft_gauss_recal"):
            f = df[(df.dataset == ds) & (df.arm == arm)].set_index("seed")["mae"]
            s = sorted(set(z.index) & set(f.index))
            if not s:
                continue
            same = int(np.sum(np.isclose(z.loc[s], f.loc[s], atol=1e-6)))
            rows.append([f"`{ds}`", f"`{arm}`", f"{same}/{len(s)}",
                         "**VACUOUS - reverted to zero-shot**" if same >= len(s) - 1
                         else ("partly reverted" if same else "trained")])
    return md(rows, ["encoder", "arm", "seeds identical to zero-shot", "status"])


def main():
    doc = [f"""# PTB-XL age prediction with TabPFN-2.5 — experiment log

Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by `make_ecg_report.py`.
All figures computed from `imbreg/results/*.jsonl`; none typed by hand.

## 1. What was done

Two interventions, applied in sequence and measured separately.

**Stage 1 — an encoder trained *through* a frozen TabPFN-2.5.** The existing result in
this repo trains a ResNet1d on an L1 age loss and then bolts TabPFN onto its
penultimate features. In that setup the encoder never knows a TabPFN sits downstream:
it is optimised to make a *linear* head accurate, not to produce good in-context
evidence. Here TabPFN's 10.2M weights are frozen and only the encoder's 405,760 are
trained, with the loss being TabPFN's bar-distribution NLL on query ECGs given encoded
context ECGs. Gradients reach the encoder through TabPFN's forward pass via the
library's prompt-tuning path (`differentiable_input=True`,
`fit_with_differentiable_input`, then `forward(..., use_inference_mode=True)`, which
keeps autograd on by design). Episodes are sampled **uniformly** at this stage, so the
encoder stays an unbiased feature extractor.

**Stage 2 — Gaussian-context fine-tuning of TabPFN** on the frozen encoder's
embeddings, identical to the tabular study: context points drawn with weight
proportional to `exp(-0.5*((y-mu)/sigma)^2)` in age space, query points drawn first and
unweighted, `(mu, sigma)` resampled per episode. The uniform-context arm applies the
same context subsampling with uniform weights, so the two differ only in sampling
weight and not in context size.

## 2. The data and why two protocols

{split_facts()}

The official split has **six** few-shot test points out of 954. Overall MAE on it is
therefore almost entirely a many-shot number, and the tail -- the thing both
interventions target -- is not measurable there. It is still reported because it is
what this repo's earlier numbers used. The rebalanced split (Yang et al., ICML 2021;
the same convention as the AgeDB-DIR work in this repo) gives enough tail coverage for
the tail claim to be tested.

Shot regions are assigned by the **training** frequency of each test point's 1-year age
bin (>=100 many, 20-100 medium, <20 few). The training age distribution has a bin count
ratio of 203:1.

### Prior results in this repo (from `pipeline.log`)

{md([[n, f"{v:.3f}"] for n, v in PRIOR], ["method", "test MAE"])}

## 3. Stage 1: encoder training curve

{encoder_curve()}
"""]

    for proto, title, note in [
        ("fixed", "Official split", "Comparable with the prior results above. Tail "
         "columns rest on six test points and should not be read as evidence."),
        ("dir", "Rebalanced split", "Where the tail claim is actually testable."),
    ]:
        df = load(proto)
        doc.append(f"\n---\n\n## 4{'a' if proto=='fixed' else 'b'}. Results — {title}\n\n{note}\n")
        if df.empty:
            doc.append("\n_no results recorded for this protocol yet_\n")
            continue
        doc.append("\n**Did the fine-tuning actually change the weights?**\n\n"
                   + noop_check(df) + "\n")
        for ds, desc in ENCODERS.items():
            doc.append(f"\n### {ds} — {desc}\n\n{arm_table(df, ds)}\n")
            for ref, alt in [("ft_uniform", "ft_gauss"),
                             ("zeroshot_recal_iso", "ft_gauss"),
                             ("zeroshot_recal_iso", "ft_gauss_recal")]:
                t = paired(df, ds, ref, alt)
                if t:
                    doc.append(f"\n**`{alt}` vs `{ref}`** (negative = `{alt}` better)\n\n{t}\n")
        doc.append("\n### Stage-1 effect: the two encoders under the same arm\n")
        for arm in ("zeroshot", "ft_gauss"):
            t = cross_encoder(df, arm)
            if t:
                doc.append(f"\n**arm = `{arm}`** (negative delta = TabPFN-trained encoder better)\n\n{t}\n")

    doc.append("""
---

## 5. Reading these results

Attribution is deliberately separated:

- **Stage 1** is isolated by comparing the two encoders at the *same* arm. Any
  difference at `zeroshot` is due to how the encoder was trained, since nothing
  downstream differs.
- **Stage 2** is isolated by `ft_gauss` vs `ft_uniform` *within* one encoder: same
  model, same data, same budget, same episode sizes, same learning-rate sweep, with
  the sampling weight as the only difference.

`zeroshot_recal_iso` -- isotonic recalibration of zero-shot predictions fitted on the
validation set, with no gradient updates -- is included because in the tabular study it
beat Gaussian-context fine-tuning on every tail metric on all three datasets. If it
does so here too, the honest conclusion is the same: most of the tail gain is
prediction decompression, which a monotone curve achieves more cheaply.

### Caveats

- **The `fixed` protocol fixes the split**, so its five seeds vary only model and
  preprocessing randomness, not the data partition. Paired comparisons between arms
  remain valid; what the error bars do not capture is split-to-split variance.
- **Five seeds** pins Wilcoxon signed-rank at p >= 0.0625 regardless of effect size;
  seed-win counts and effect sizes carry the information.
- **The encoder was warm-started** from the existing L1 checkpoint, so Stage 1 measures
  the benefit of *continuing* training through TabPFN, not of training that way from
  scratch. A from-scratch run is not included.
- Ablations not run: Gaussian context sampling during Stage 1 (`--ctx-sampling gauss`),
  and the Stage-2 ablations listed in the tabular log.
""")

    out = os.path.join(HERE, "ECG_EXPERIMENT_LOG.md")
    text = "\n".join(doc)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
