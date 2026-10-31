# Timestral on PTB-XL — zero-shot ECG forecasting benchmark

Generated 2026-10-16 18:39 by `make_timestral_report.py`.
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

PTB-XL test split: **954 records x 12 leads x 1000 samples** (10 s at 100 Hz). Each
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

| horizon | mean | naive | seasonal_naive | timestral |
|---|---|---|---|---|
| 100 | 1.543 | 1.719 | 1.230 | 0.759 |
| 200 | 1.493 | 1.889 | 1.441 | 1.002 |
| 400 | 1.475 | 2.059 | 1.674 | 1.225 |

MAE in millivolts:

| horizon | mean | naive | seasonal_naive | timestral |
|---|---|---|---|---|
| 100 | 0.1204 | 0.1281 | 0.0994 | 0.0591 |
| 200 | 0.1143 | 0.1407 | 0.1148 | 0.0780 |
| 400 | 0.1083 | 0.1512 | 0.1279 | 0.0925 |

### Model size (horizon 200 = 2 s)

| model | MAE (mV) | RMSE | MASE | CRPS | runtime (s) |
|---|---|---|---|---|---|
| 209M | 0.0780 | 0.1721 | 1.002 | 0.0561 | 12 |
| 123M | 0.0792 | 0.1726 | 1.021 | 0.0573 | 8 |
| 28M | 0.0824 | 0.1751 | 1.065 | 0.0598 | 3 |
| 5M | 0.0864 | 0.1801 | 1.123 | 0.0632 | 2 |

### Joint vs independent leads (Timestral-209M, horizon 200)

| mode | MAE (mV) | RMSE | MASE | CRPS |
|---|---|---|---|---|
| multivariate | 0.0780 | 0.1721 | 1.002 | 0.0561 |
| univariate | 0.0808 | 0.1742 | 1.041 | 0.0585 |

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
