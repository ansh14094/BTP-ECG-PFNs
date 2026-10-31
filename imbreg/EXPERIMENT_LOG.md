# Imbalance-aware fine-tuning of TabPFN-2.5 — experiment log

Generated 2026-10-08 16:43 by `make_report.py`.
Every figure below is computed from `results/*.jsonl`; none is typed by hand.

Model: **TabPFN-2.5** (`tabpfn-v2.5-regressor-v2.5_default.ckpt`, pinned explicitly —
the `tabpfn` package now defaults to v3). Package `tabpfn` 8.5.0. GPU: NVIDIA T400 4GB; NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.
Grid status: **complete**.

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

| dataset | n x d | target range | skew | rare mass (phi>=0.5) | train bin ratio | train/val/test | test many/med/few |
|---|---|---|---|---|---|---|---|
| `abalone` | 4177 x 8 | 1 - 29 | +1.11 | 0.163 | 659:1 | 3630/165/382 | 160/120/102 |
| `communities` | 1994 x 122 | 0 - 1 | +1.52 | 0.201 | 337:1 | 1571/123/300 | 96/112/92 |
| `wine_white` | 4898 x 11 | 3 - 9 | +0.16 | 0.254 | 2108:1 | 4429/150/319 | 180/120/19 |

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
episode (1024 rows)      <7:  99  7-8: 258  9-10: 353  11-12: 173  13+: 141

uniform context (410)   <7:  45  7-8: 106  9-10: 128  11-12:  72  13+:  59
gauss   context (410)   <7:   8  7-8:  91  9-10: 210  11-12:  87  13+:  14
query (204), identical  <7:  14  7-8:  54  9-10:  84  11-12:  29  13+:  23
```

Same 410 context rows and same 204 query rows in both arms — but the Gaussian context
holds far fewer tail animals, while the query still spans the full range. The model is
forced, every step, to predict targets its context under-represents.

### Training details

Per epoch the training pool is shuffled and cut into episodes of 1024 rows; query is
20% of each episode, context is 50% of the remainder. Loss is the bar-distribution
CRPS plus MSE on the query points. 60 epochs, early stopping on **validation** RMSE
(patience 8), AdamW, cosine schedule with warmup. Learning rate is swept over
{1e-5, 3e-5, 1e-4} **for both fine-tuning arms**, and each arm is reported at its
own validation-selected rate.

At evaluation the fine-tuned model receives the **full, unmodified training set** as
context — the biased sampling exists only during training, to shape the weights.

Fine-tuning adapts weights to one dataset's features and target scale, so each
dataset gets its own fine-tune from the original checkpoint. Nothing is transferred
between datasets.

---

## 3. Arms

| arm | name | detail |
|---|---|---|
| `lgbm` | LightGBM (tuned) | random search over 30 configs, selected on validation RMSE |
| `zeroshot` | TabPFN-2.5 zero-shot | no gradient updates; ensemble size matched to the fine-tuned arms |
| `zeroshot_recal` | TabPFN-2.5 + val recalibration (linear) | affine/isotonic correction fitted on the balanced validation set — an adversarial cheap baseline |
| `ctx_balanced` | TabPFN-2.5 + inverse-density context | inverse-density resampled context, inference only |
| `imcontext` | IM-Context on TabPFN-2.5 | per-query kNN context + inverse-density rows, after IM-Context (arXiv:2405.18202) |
| `ft_uniform` | TabPFN-2.5 + uniform-context FT | stock uniform context sampling at the same reduced context size — the arm that isolates the claim |
| `ft_gauss` | TabPFN-2.5 + Gaussian-context FT | Gaussian-weighted context sampling (the method) |

`ft_gauss` vs `ft_uniform` is the comparison that carries the claim. Comparing
`ft_gauss` against `zeroshot` alone would be uninterpretable, since any gain could
simply be "fine-tuning on your own data helps", which is already known.

---

## 4. Results

### Cross-dataset summary

| claim | abalone | communities | wine_white |
|---|---|---|---|
| Gaussian context beats uniform context (matched budget) | yes (-20.9%, 5/5) | yes (-4.1%, 5/5) | yes (-6.5%, 5/5) |
| ...and on the tail specifically | yes (-28.4%, 5/5) | yes (-7.5%, 5/5) | yes (-5.3%, 5/5) |
| Beats free isotonic recalibration (overall) | tie (+0.9%, 2/5) | no (+6.3%, 0/5) | no (+5.7%, 0/5) |
| Beats free isotonic recalibration (tail) | no (+15.0%, 0/5) | no (+20.6%, 0/5) | no (+17.4%, 0/5) |
| Preserves majority-region accuracy vs recalibration | yes (-18.8%, 5/5) | yes (-32.8%, 5/5) | yes (-26.1%, 5/5) |
| Combination beats recalibration alone (overall) | yes (-2.3%, 5/5) | no (+3.2%, 0/5) | yes (-3.9%, 5/5) |

Percentages are the mean paired difference relative to the reference arm; the fraction is the number of seeds on which the proposed arm won.

### abalone

| arm | seeds | lr(s) picked on val | MAE | RMSE | GM | SERA | MAE many | MAE medium | MAE few |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM (tuned) | 5 | - | 3.090 ± 0.056 | 4.176 ± 0.069 | 1.772 ± 0.061 | 16.239 ± 0.612 | 1.242 ± 0.095 | 2.691 ± 0.111 | 6.460 ± 0.098 |
| TabPFN-2.5 zero-shot | 5 | - | 2.942 ± 0.032 | 3.992 ± 0.020 | 1.671 ± 0.111 | 14.842 ± 0.182 | 1.170 ± 0.087 | 2.569 ± 0.032 | 6.163 ± 0.044 |
| TabPFN-2.5 + val recalibration (linear) | 5 | - | 2.224 ± 0.084 | 2.917 ± 0.085 | 1.377 ± 0.087 | 6.447 ± 0.345 | 1.744 ± 0.169 | 1.669 ± 0.124 | 3.630 ± 0.201 |
| TabPFN-2.5 + val recalibration (isotonic) | 5 | - | 2.184 ± 0.112 | 2.894 ± 0.119 | 0.860 ± 0.106 | 5.777 ± 0.368 | 1.949 ± 0.295 | 1.303 ± 0.127 | 3.588 ± 0.196 |
| TabPFN-2.5 + uniform-context FT | 5 | 0.0001,3e-05 | 2.786 ± 0.121 | 3.745 ± 0.118 | 1.618 ± 0.152 | 12.890 ± 0.837 | 1.208 ± 0.097 | 2.357 ± 0.117 | 5.767 ± 0.243 |
| TabPFN-2.5 + Gaussian-context FT | 5 | 0.0001,1e-05 | 2.202 ± 0.070 | 2.958 ± 0.080 | 1.289 ± 0.019 | 6.955 ± 0.306 | 1.584 ± 0.162 | 1.392 ± 0.105 | 4.127 ± 0.169 |
| Gaussian-context FT + val recalibration | 5 | 0.0001,3e-05 | 2.133 ± 0.102 | 2.814 ± 0.092 | 1.299 ± 0.123 | 5.700 ± 0.325 | 1.769 ± 0.200 | 1.343 ± 0.125 | 3.635 ± 0.225 |

**`ft_uniform` vs `zeroshot`** (negative = `ft_uniform` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | -0.156 | -5.3% | 5/5  (p=0.0625) |
| rmse | -0.247 | -6.2% | 5/5  (p=0.0625) |
| gm | -0.053 | -3.2% | 4/5  (p=0.4375) |
| sera | -1.952 | -13.2% | 5/5  (p=0.0625) |
| mae_many | +0.038 | +3.3% | 0/5  (p=0.0625) |
| mae_medium | -0.212 | -8.3% | 5/5  (p=0.0625) |
| mae_few | -0.396 | -6.4% | 5/5  (p=0.0625) |

**`ft_gauss` vs `ft_uniform`** (negative = `ft_gauss` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | -0.584 | -20.9% | 5/5  (p=0.0625) |
| rmse | -0.787 | -21.0% | 5/5  (p=0.0625) |
| gm | -0.329 | -20.3% | 5/5  (p=0.0625) |
| sera | -5.935 | -46.0% | 5/5  (p=0.0625) |
| mae_many | +0.376 | +31.1% | 0/5  (p=0.0625) |
| mae_medium | -0.965 | -40.9% | 5/5  (p=0.0625) |
| mae_few | -1.640 | -28.4% | 5/5  (p=0.0625) |

**`ft_gauss` vs `zeroshot_recal_iso`** (negative = `ft_gauss` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | +0.019 | +0.9% | 2/5  (p=0.4375) |
| rmse | +0.063 | +2.2% | 1/5  (p=0.1250) |
| gm | +0.429 | +49.8% | 0/5  (p=0.0625) |
| sera | +1.179 | +20.4% | 0/5  (p=0.0625) |
| mae_many | -0.366 | -18.8% | 5/5  (p=0.0625) |
| mae_medium | +0.089 | +6.8% | 1/5  (p=0.1250) |
| mae_few | +0.539 | +15.0% | 0/5  (p=0.0625) |

**`ft_gauss_recal` vs `zeroshot_recal_iso`** (negative = `ft_gauss_recal` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | -0.050 | -2.3% | 5/5  (p=0.0625) |
| rmse | -0.080 | -2.8% | 5/5  (p=0.0625) |
| gm | +0.439 | +51.1% | 0/5  (p=0.0625) |
| sera | -0.077 | -1.3% | 3/5  (p=0.3125) |
| mae_many | -0.181 | -9.3% | 5/5  (p=0.0625) |
| mae_medium | +0.040 | +3.1% | 1/5  (p=0.4375) |
| mae_few | +0.047 | +1.3% | 1/5  (p=0.4375) |

### communities

| arm | seeds | lr(s) picked on val | MAE | RMSE | GM | SERA | MAE many | MAE medium | MAE few |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM (tuned) | 5 | - | 0.165 ± 0.004 | 0.225 ± 0.004 | 0.094 ± 0.003 | 0.046 ± 0.002 | 0.065 ± 0.004 | 0.174 ± 0.006 | 0.257 ± 0.010 |
| TabPFN-2.5 zero-shot | 5 | - | 0.161 ± 0.004 | 0.222 ± 0.005 | 0.089 ± 0.004 | 0.045 ± 0.002 | 0.059 ± 0.002 | 0.168 ± 0.006 | 0.258 ± 0.006 |
| TabPFN-2.5 + val recalibration (linear) | 5 | - | 0.148 ± 0.003 | 0.195 ± 0.004 | 0.094 ± 0.007 | 0.031 ± 0.001 | 0.103 ± 0.010 | 0.139 ± 0.002 | 0.207 ± 0.004 |
| TabPFN-2.5 + val recalibration (isotonic) | 5 | - | 0.144 ± 0.003 | 0.194 ± 0.004 | 0.079 ± 0.012 | 0.030 ± 0.001 | 0.096 ± 0.010 | 0.143 ± 0.009 | 0.195 ± 0.009 |
| TabPFN-2.5 + uniform-context FT | 5 | 0.0001,1e-05,3e-05 | 0.160 ± 0.003 | 0.220 ± 0.004 | 0.089 ± 0.004 | 0.044 ± 0.002 | 0.060 ± 0.002 | 0.168 ± 0.006 | 0.254 ± 0.007 |
| TabPFN-2.5 + Gaussian-context FT | 5 | 0.0001,1e-05,3e-05 | 0.153 ± 0.002 | 0.211 ± 0.004 | 0.085 ± 0.005 | 0.039 ± 0.001 | 0.065 ± 0.003 | 0.162 ± 0.005 | 0.235 ± 0.001 |
| Gaussian-context FT + val recalibration | 5 | 0.0001,1e-05,3e-05 | 0.149 ± 0.003 | 0.195 ± 0.004 | 0.095 ± 0.005 | 0.031 ± 0.001 | 0.104 ± 0.010 | 0.140 ± 0.003 | 0.206 ± 0.003 |

**`ft_uniform` vs `zeroshot`** (negative = `ft_uniform` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | -0.001 | -0.6% | 1/5  (p=1.0000) |
| rmse | -0.001 | -0.6% | 1/5  (p=1.0000) |
| gm | -0.001 | -0.7% | 1/5  (p=1.0000) |
| sera | -0.001 | -1.6% | 1/5  (p=1.0000) |
| mae_many | +0.000 | +0.7% | 0/5  (p=1.0000) |
| mae_medium | +0.000 | +0.0% | 0/5  (p=1.0000) |
| mae_few | -0.004 | -1.5% | 1/5  (p=1.0000) |

**`ft_gauss` vs `ft_uniform`** (negative = `ft_gauss` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | -0.007 | -4.1% | 5/5  (p=0.0625) |
| rmse | -0.009 | -4.1% | 5/5  (p=0.0625) |
| gm | -0.004 | -4.3% | 4/5  (p=0.1875) |
| sera | -0.005 | -10.7% | 5/5  (p=0.0625) |
| mae_many | +0.005 | +8.3% | 0/5  (p=0.0625) |
| mae_medium | -0.006 | -3.6% | 5/5  (p=0.0625) |
| mae_few | -0.019 | -7.5% | 5/5  (p=0.0625) |

**`ft_gauss` vs `zeroshot_recal_iso`** (negative = `ft_gauss` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | +0.009 | +6.3% | 0/5  (p=0.0625) |
| rmse | +0.018 | +9.2% | 0/5  (p=0.0625) |
| gm | +0.006 | +8.1% | 1/5  (p=0.3125) |
| sera | +0.009 | +31.5% | 0/5  (p=0.0625) |
| mae_many | -0.032 | -32.8% | 5/5  (p=0.0625) |
| mae_medium | +0.018 | +12.7% | 0/5  (p=0.0625) |
| mae_few | +0.040 | +20.6% | 0/5  (p=0.0625) |

**`ft_gauss_recal` vs `zeroshot_recal_iso`** (negative = `ft_gauss_recal` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | +0.005 | +3.2% | 0/5  (p=0.0625) |
| rmse | +0.002 | +0.9% | 1/5  (p=0.1250) |
| gm | +0.016 | +20.9% | 0/5  (p=0.0625) |
| sera | +0.001 | +3.8% | 1/5  (p=0.1875) |
| mae_many | +0.008 | +7.9% | 1/5  (p=0.1250) |
| mae_medium | -0.004 | -2.7% | 3/5  (p=0.3125) |
| mae_few | +0.012 | +6.0% | 0/5  (p=0.0625) |

### wine_white

| arm | seeds | lr(s) picked on val | MAE | RMSE | GM | SERA | MAE many | MAE medium | MAE few |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM (tuned) | 5 | - | 0.789 ± 0.024 | 1.095 ± 0.028 | 0.257 ± 0.128 | 0.936 ± 0.060 | 0.362 ± 0.020 | 1.179 ± 0.077 | 2.370 ± 0.037 |
| TabPFN-2.5 zero-shot | 5 | - | 0.926 ± 0.012 | 1.168 ± 0.016 | 0.544 ± 0.026 | 1.067 ± 0.028 | 0.427 ± 0.014 | 1.432 ± 0.025 | 2.455 ± 0.031 |
| TabPFN-2.5 + val recalibration (linear) | 5 | - | 0.763 ± 0.024 | 0.998 ± 0.035 | 0.469 ± 0.025 | 0.723 ± 0.054 | 0.482 ± 0.018 | 0.990 ± 0.058 | 1.990 ± 0.078 |
| TabPFN-2.5 + val recalibration (isotonic) | 5 | - | 0.741 ± 0.029 | 0.978 ± 0.036 | 0.271 ± 0.133 | 0.650 ± 0.062 | 0.530 ± 0.060 | 0.871 ± 0.067 | 1.916 ± 0.068 |
| TabPFN-2.5 + uniform-context FT | 5 | 0.0001,3e-05 | 0.837 ± 0.012 | 1.098 ± 0.014 | 0.416 ± 0.042 | 0.940 ± 0.026 | 0.370 ± 0.021 | 1.295 ± 0.026 | 2.375 ± 0.028 |
| TabPFN-2.5 + Gaussian-context FT | 5 | 0.0001 | 0.783 ± 0.035 | 1.039 ± 0.043 | 0.414 ± 0.089 | 0.819 ± 0.068 | 0.391 ± 0.043 | 1.138 ± 0.061 | 2.250 ± 0.117 |
| Gaussian-context FT + val recalibration | 5 | 0.0001,3e-05 | 0.712 ± 0.024 | 0.959 ± 0.037 | 0.432 ± 0.032 | 0.652 ± 0.051 | 0.468 ± 0.025 | 0.886 ± 0.034 | 1.932 ± 0.094 |

**`ft_uniform` vs `zeroshot`** (negative = `ft_uniform` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | -0.089 | -9.6% | 5/5  (p=0.0625) |
| rmse | -0.070 | -6.0% | 5/5  (p=0.0625) |
| gm | -0.127 | -23.4% | 5/5  (p=0.0625) |
| sera | -0.127 | -11.9% | 5/5  (p=0.0625) |
| mae_many | -0.058 | -13.5% | 5/5  (p=0.0625) |
| mae_medium | -0.137 | -9.6% | 5/5  (p=0.0625) |
| mae_few | -0.080 | -3.2% | 5/5  (p=0.0625) |

**`ft_gauss` vs `ft_uniform`** (negative = `ft_gauss` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | -0.054 | -6.5% | 5/5  (p=0.0625) |
| rmse | -0.059 | -5.4% | 5/5  (p=0.0625) |
| gm | -0.002 | -0.5% | 3/5  (p=1.0000) |
| sera | -0.121 | -12.9% | 5/5  (p=0.0625) |
| mae_many | +0.022 | +5.9% | 1/5  (p=0.3125) |
| mae_medium | -0.157 | -12.1% | 5/5  (p=0.0625) |
| mae_few | -0.125 | -5.3% | 5/5  (p=0.0625) |

**`ft_gauss` vs `zeroshot_recal_iso`** (negative = `ft_gauss` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | +0.042 | +5.7% | 0/5  (p=0.0625) |
| rmse | +0.060 | +6.2% | 0/5  (p=0.0625) |
| gm | +0.143 | +52.8% | 1/5  (p=0.1875) |
| sera | +0.169 | +26.0% | 0/5  (p=0.0625) |
| mae_many | -0.139 | -26.1% | 5/5  (p=0.0625) |
| mae_medium | +0.267 | +30.6% | 0/5  (p=0.0625) |
| mae_few | +0.333 | +17.4% | 0/5  (p=0.0625) |

**`ft_gauss_recal` vs `zeroshot_recal_iso`** (negative = `ft_gauss_recal` better, 5 paired seeds)

| metric | abs delta | rel delta | seeds won |
|---|---|---|---|
| mae | -0.029 | -3.9% | 5/5  (p=0.0625) |
| rmse | -0.020 | -2.0% | 5/5  (p=0.0625) |
| gm | +0.161 | +59.3% | 1/5  (p=0.1250) |
| sera | +0.002 | +0.3% | 2/5  (p=1.0000) |
| mae_many | -0.062 | -11.7% | 5/5  (p=0.0625) |
| mae_medium | +0.015 | +1.7% | 1/5  (p=0.4375) |
| mae_few | +0.016 | +0.8% | 2/5  (p=0.6250) |

---

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
