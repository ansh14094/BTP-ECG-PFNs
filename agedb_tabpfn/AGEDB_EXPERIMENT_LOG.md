# AgeDB-DIR age prediction with TabPFN-2.5 — experiment log

Generated 2026-10-24 16:54 by `make_agedb_report.py`.
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

| method | MAE |
|---|---|
| LDS | 7.67 |
| FDS | 7.69 |
| LDS + FDS | 7.47 |

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

| encoder | arm | seeds identical to zero-shot | status |
|---|---|---|---|
| `agedb_tabpfn_enc` | `ft_uniform` | 1/5 | partly reverted |
| `agedb_tabpfn_enc` | `ft_gauss` | 0/5 | trained |
| `agedb_tabpfn_enc` | `ft_gauss_recal` | 0/5 | trained |

Early stopping restores best-validation weights, so a run whose validation score never
improves silently reverts to the original checkpoint and the arm is just zero-shot
relabelled. Any comparison involving such an arm is vacuous.

## 4. Results

### agedb_l1_enc — ResNet-18 trained on L1 age loss (control)

| arm | seeds | MAE | RMSE | GM | SERA | MAE many | MAE medium | MAE few |
|---|---|---|---|---|---|---|---|---|
| LightGBM on embeddings | 1 | 8.174 ± nan | 10.622 ± nan | 5.017 ± nan | 32.896 ± nan | 6.907 ± nan | 10.606 ± nan | 13.417 ± nan |
| TabPFN-2.5 zero-shot | 1 | 8.186 ± nan | 10.626 ± nan | 5.058 ± nan | 32.947 ± nan | 6.912 ± nan | 10.641 ± nan | 13.425 ± nan |

### agedb_tabpfn_enc — ResNet-18 trained through frozen TabPFN-2.5

| arm | seeds | MAE | RMSE | GM | SERA | MAE many | MAE medium | MAE few |
|---|---|---|---|---|---|---|---|---|
| LightGBM on embeddings | 5 | 11.305 ± 0.042 | 13.969 ± 0.030 | 7.646 ± 0.098 | 60.573 ± 0.318 | 8.879 ± 0.082 | 15.993 ± 0.077 | 21.250 ± 0.073 |
| TabPFN-2.5 zero-shot | 5 | 11.335 ± 0.024 | 13.955 ± 0.024 | 7.696 ± 0.039 | 59.299 ± 0.160 | 8.913 ± 0.022 | 15.945 ± 0.037 | 21.466 ± 0.039 |
| zero-shot + linear recalibration | 5 | 10.701 ± 0.026 | 13.423 ± 0.030 | 7.096 ± 0.034 | 40.795 ± 0.251 | 9.383 ± 0.022 | 12.659 ± 0.060 | 17.730 ± 0.047 |
| zero-shot + isotonic recalibration | 5 | 10.618 ± 0.032 | 13.378 ± 0.038 | 6.934 ± 0.042 | 38.595 ± 0.223 | 9.419 ± 0.034 | 12.045 ± 0.046 | 17.996 ± 0.051 |
| TabPFN FT, uniform context | 5 | 11.262 ± 0.072 | 13.899 ± 0.059 | 7.679 ± 0.073 | 53.699 ± 4.001 | 9.138 ± 0.130 | 14.982 ± 0.653 | 21.036 ± 0.414 |
| TabPFN FT, Gaussian context | 5 | 10.687 ± 0.094 | 13.457 ± 0.051 | 7.056 ± 0.162 | 41.760 ± 3.325 | 9.334 ± 0.148 | 12.538 ± 0.687 | 18.351 ± 0.651 |
| Gaussian-context FT + recalibration | 5 | 10.668 ± 0.026 | 13.434 ± 0.032 | 7.024 ± 0.050 | 39.658 ± 0.330 | 9.413 ± 0.029 | 12.278 ± 0.050 | 18.068 ± 0.109 |

**`ft_gauss` vs `ft_uniform`** (negative = `ft_gauss` better)

| metric | ft_uniform | ft_gauss | rel delta | seeds won | p |  |
|---|---|---|---|---|---|---|
| mae | 11.262 | 10.687 | -5.1% | 5/5 | 0.0625 | WIN |
| rmse | 13.899 | 13.457 | -3.2% | 5/5 | 0.0625 | WIN |
| gm | 7.679 | 7.056 | -8.1% | 5/5 | 0.0625 | WIN |
| sera | 53.699 | 41.760 | -22.2% | 5/5 | 0.0625 | WIN |
| mae_many | 9.138 | 9.334 | +2.1% | 1/5 | 0.1250 | LOSS |
| mae_medium | 14.982 | 12.538 | -16.3% | 5/5 | 0.0625 | WIN |
| mae_few | 21.036 | 18.351 | -12.8% | 5/5 | 0.0625 | WIN |

**`ft_gauss` vs `zeroshot_recal_iso`** (negative = `ft_gauss` better)

| metric | zeroshot_recal_iso | ft_gauss | rel delta | seeds won | p |  |
|---|---|---|---|---|---|---|
| mae | 10.618 | 10.687 | +0.7% | 1/5 | 0.1250 | LOSS |
| rmse | 13.378 | 13.457 | +0.6% | 0/5 | 0.0625 | LOSS |
| gm | 6.934 | 7.056 | +1.8% | 0/5 | 0.0625 | LOSS |
| sera | 38.595 | 41.760 | +8.2% | 1/5 | 0.1250 | LOSS |
| mae_many | 9.419 | 9.334 | -0.9% | 4/5 | 0.3125 | WIN |
| mae_medium | 12.045 | 12.538 | +4.1% | 0/5 | 0.0625 | LOSS |
| mae_few | 17.996 | 18.351 | +2.0% | 1/5 | 0.1875 | LOSS |

**`ft_gauss_recal` vs `zeroshot_recal_iso`** (negative = `ft_gauss_recal` better)

| metric | zeroshot_recal_iso | ft_gauss_recal | rel delta | seeds won | p |  |
|---|---|---|---|---|---|---|
| mae | 10.618 | 10.668 | +0.5% | 1/5 | 0.1250 | LOSS |
| rmse | 13.378 | 13.434 | +0.4% | 0/5 | 0.0625 | LOSS |
| gm | 6.934 | 7.024 | +1.3% | 0/5 | 0.0625 | LOSS |
| sera | 38.595 | 39.658 | +2.8% | 0/5 | 0.0625 | LOSS |
| mae_many | 9.419 | 9.413 | -0.1% | 2/5 | 0.8125 | tie |
| mae_medium | 12.045 | 12.278 | +1.9% | 0/5 | 0.0625 | LOSS |
| mae_few | 17.996 | 18.068 | +0.4% | 2/5 | 0.3125 | tie |

### Stage-1 effect: the two encoders under the same arm

**arm = `zeroshot`** (negative = TabPFN-trained encoder better)

| metric | zeroshot | zeroshot | rel delta | seeds won | p |  |
|---|---|---|---|---|---|---|
| mae | 8.186 | 11.307 | +38.1% | 0/1 | 1.0000 | LOSS |
| rmse | 10.626 | 13.935 | +31.1% | 0/1 | 1.0000 | LOSS |
| gm | 5.058 | 7.667 | +51.6% | 0/1 | 1.0000 | LOSS |
| sera | 32.947 | 59.209 | +79.7% | 0/1 | 1.0000 | LOSS |
| mae_many | 6.912 | 8.888 | +28.6% | 0/1 | 1.0000 | LOSS |
| mae_medium | 10.641 | 15.917 | +49.6% | 0/1 | 1.0000 | LOSS |
| mae_few | 13.425 | 21.407 | +59.5% | 0/1 | 1.0000 | LOSS |

## 5. Caveats

- Five seeds; Wilcoxon cannot report below p = 0.0625 at n = 5, so seed-win counts and
  effect sizes carry the information, not the p-values.
- The encoders are ResNet-18 at 112x112, not the deeper backbones at 224x224 used by
  the published AgeDB-DIR numbers, so absolute MAEs are not directly comparable with
  them; the within-this-table comparisons are the meaningful ones.
- Gaussian context sampling is applied to TabPFN's fine-tuning episode sampler, not to
  its pre-training prior, which Prior Labs never released.
