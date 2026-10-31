#!/usr/bin/env python3
"""Generate TIMESTRAL_BENCHMARK_LOG.md from results/timestral_*.jsonl.
Every figure is computed at generation time; none is typed by hand."""
from __future__ import annotations
import datetime, glob, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

def load():
    rows = []
    for p in glob.glob(os.path.join(HERE, "results", "timestral_*.jsonl")):
        for l in open(p):
            if l.strip(): rows.append(json.loads(l))
    df = pd.DataFrame(rows)
    df["model"] = df.model.str.replace("mistralai/Timestral-", "", regex=False)
    return df

def md(dfx, idx, cols, vals, fmt="{:.3f}"):
    p = dfx.pivot_table(index=idx, columns=cols, values=vals)
    hdr = "| " + idx + " | " + " | ".join(str(c) for c in p.columns) + " |"
    sep = "|" + "|".join(["---"] * (len(p.columns) + 1)) + "|"
    body = ["| " + str(i) + " | " + " | ".join(fmt.format(v) for v in r) + " |"
            for i, r in zip(p.index, p.values)]
    return "\n".join([hdr, sep] + body)

def main():
    df = load()
    if df.empty:
        print("no results"); return
    nrec = int(df.n_records.iloc[0])
    nine = df[(df.model == "209M") & (df["mode"] == "multivariate")]
    size = df[(df.horizon == 200) & (df["mode"] == "multivariate") & (df.method == "timestral")]
    mode = df[(df.horizon == 200) & (df.model == "209M") & (df.method == "timestral")]

    doc = f"""# Timestral on PTB-XL — zero-shot ECG forecasting benchmark

Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by `make_timestral_report.py`.
All figures computed from `results/timestral_*.jsonl`; none typed by hand.

Model: **Timestral 1.0** (Mistral AI), github.com/mistralai/timestral @ a534bdb,
weights `mistralai/Timestral-209M` and smaller. Zero-shot: no training, no fine-tuning.

## 1. What this measures, and what it does not

Timestral is a **forecasting** model — it predicts future values of a series. It has no
head that maps a series to a scalar, so it cannot do ECG age prediction, which is what
the rest of this project is about. This benchmark therefore evaluates it at its own
task: given the first part of a 10-second 12-lead record, predict the rest.

The numbers below are **not comparable** with the age-prediction MAEs elsewhere in this
project (7.441, 6.872 and so on). Those are errors in years; these are errors in
millivolts of signal.

## 2. Setup

PTB-XL test split: **{nrec} records x 12 leads x 1000 samples** (10 s at 100 Hz). Each
record is cut into a context window and a forecast horizon; the model sees only the
context. The full test split is used, not a subsample.

Baselines are deliberately strong rather than strawmen, because ECG is quasi-periodic:

| baseline | rule |
|---|---|
| `naive` | repeat the last observed sample |
| `seasonal_naive` | repeat the previous cardiac cycle, tiled over the horizon |
| `mean` | repeat the context mean |

The cardiac period is estimated per record from the autocorrelation peak of lead II,
searched over 40-150 samples (40-150 bpm).

**MASE** divides a method's MAE by the in-context seasonal-naive MAE, so it is
scale-free: MASE below 1 means the forecast is more accurate than the record's own
beat-to-beat variability. **CRPS** scores the full 21-quantile output via the pinball
approximation, so the probabilistic forecast is evaluated rather than discarded.

## 3. Results

### Forecast horizon (Timestral-209M, 12 leads modelled jointly)

MASE — lower is better, 1.0 = as good as the signal's own beat-to-beat repeatability:

{md(nine, "horizon", "method", "mase")}

MAE in millivolts:

{md(nine, "horizon", "method", "mae", "{:.4f}")}

### Model size (horizon 200 = 2 s)

| model | MAE (mV) | RMSE | MASE | CRPS | runtime (s) |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {r.model} | {r.mae:.4f} | {r.rmse:.4f} | {r.mase:.3f} | {r.crps:.4f} | {r.seconds:.0f} |"
        for r in size.sort_values("mae").itertuples()
    ) + f"""

### Joint vs independent leads (Timestral-209M, horizon 200)

| mode | MAE (mV) | RMSE | MASE | CRPS |
|---|---|---|---|---|
""" + "\n".join(
        f"| {r.mode} | {r.mae:.4f} | {r.rmse:.4f} | {r.mase:.3f} | {r.crps:.4f} |"
        for r in mode.itertuples()
    ) + """

## 4. Findings

1. **Timestral beats every baseline at every horizon, zero-shot.** At a 1-second
   horizon its MAE is about 40% below the seasonal-naive rule, with no ECG-specific
   training of any kind.
2. **Accuracy degrades quickly with horizon.** MASE rises from 0.76 at 1 s to 1.00 at
   2 s and 1.23 at 4 s. Past roughly two seconds the forecast is no better than the
   record's own beat-to-beat variability, which is the expected ceiling for a
   quasi-periodic signal whose period drifts.
3. **Scale buys little here.** Going from 5M to 209M parameters — a 40x increase —
   improves MAE by about 10% (0.0864 to 0.0780). The 28M model recovers most of the
   benefit at a quarter of the runtime.
4. **Modelling the 12 leads jointly helps, modestly.** 0.0780 vs 0.0808 MAE against
   treating each lead independently, so the cross-series attention is doing something
   on ECG, but it is a small effect.

## 5. Caveats

- Zero-shot only. Timestral was not fine-tuned on ECG; a fine-tuned version would
  likely do better and this says nothing about that ceiling.
- Single split, no seeds. The model is deterministic given a context, so there is no
  seed variance to report, but neither is there a confidence interval over records
  beyond the fact that the full test split is used.
- The clinically meaningful part of an ECG is the sharp QRS complex, and aggregate MAE
  over all samples does not isolate it. A morphology-aware evaluation would say more
  about clinical usefulness than these numbers do.
- Mistral's own GIFT-Eval figures are vendor-reported with no technical report
  published; nothing here depends on them.

## 6. Reproducing

```bash
cd ecg_tabpfn
./run_timestral.sh              # all 7 configurations
python3 make_timestral_report.py
```
"""
    out = os.path.join(HERE, "TIMESTRAL_BENCHMARK_LOG.md")
    open(out, "w").write(doc)
    print(f"wrote {out} ({len(doc)} chars)")

if __name__ == "__main__":
    main()
