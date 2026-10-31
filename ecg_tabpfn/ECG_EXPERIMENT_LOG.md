# PTB-XL age prediction with TabPFN-2.5 — experiment log

Generated 2026-10-16 17:10 by `make_ecg_report.py`.
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

| protocol | train/val/test | test many/medium/few |
|---|---|---|
| official (`fixed`) | 7566/963/954 | 668/280/6 |
| rebalanced (`dir`) | 7272/712/1499 | 740/580/179 |

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

| method | test MAE |
|---|---|
| ResNet1d-Wang, L1 loss (end-to-end) | 7.549 |
| frozen backbone features -> TabPFN | 7.441 |

## 3. Stage 1: encoder training curve

| step | train NLL | val MAE |
|---|---|---|
| 200 | 0.6714 | 7.3495 |
| 400 | 0.5730 | 7.3128 |
| 600 | 0.6947 | 7.3453 |
| 800 | 0.6617 | 7.3882 |
| 1000 | 0.6641 | 7.3871 |
| 1200 | 0.5068 | 7.3979 |
| 1400 | 0.6001 | 7.4328 |
| 1600 | 0.5736 | 7.3987 |
| 1800 | 0.5836 | 7.4191 |
| 2000 | 0.6064 | 7.3921 |

Best: val MAE **7.3128** at step 400.


---

## 4a. Results — Official split

Comparable with the prior results above. Tail columns rest on six test points and should not be read as evidence.


**Did the fine-tuning actually change the weights?**

| encoder | arm | seeds identical to zero-shot | status |
|---|---|---|---|
| `ptbxl_l1_enc` | `ft_uniform` | 0/5 | trained |
| `ptbxl_l1_enc` | `ft_gauss` | 2/5 | partly reverted |
| `ptbxl_l1_enc` | `ft_gauss_recal` | 0/5 | trained |
| `ptbxl_tabpfn_enc` | `ft_uniform` | 0/5 | trained |
| `ptbxl_tabpfn_enc` | `ft_gauss` | 5/5 | **VACUOUS - reverted to zero-shot** |
| `ptbxl_tabpfn_enc` | `ft_gauss_recal` | 0/5 | trained |


### ptbxl_l1_enc — L1-trained backbone (control)

| arm | seeds | MAE | RMSE | GM | SERA | MAE many | MAE medium | MAE few |
|---|---|---|---|---|---|---|---|---|
| LightGBM on embeddings | 5 | 7.501 ± 0.068 | 9.611 ± 0.048 | 4.819 ± 0.103 | 11.523 ± 1.334 | 6.795 ± 0.206 | 8.984 ± 0.620 | 17.015 ± 1.828 |
| TabPFN-2.5 zero-shot | 5 | 7.420 ± 0.005 | 9.574 ± 0.009 | 4.691 ± 0.027 | 9.635 ± 0.038 | 6.998 ± 0.004 | 8.290 ± 0.014 | 13.771 ± 0.035 |
| zero-shot + linear recalibration | 5 | 7.426 ± 0.005 | 9.487 ± 0.008 | 4.768 ± 0.038 | 10.425 ± 0.023 | 6.737 ± 0.005 | 8.884 ± 0.012 | 16.132 ± 0.024 |
| zero-shot + isotonic recalibration | 5 | 7.387 ± 0.011 | 9.468 ± 0.006 | 4.211 ± 0.391 | 10.367 ± 0.053 | 6.622 ± 0.015 | 9.055 ± 0.031 | 14.717 ± 0.192 |
| TabPFN FT, uniform context | 5 | 7.375 ± 0.026 | 9.534 ± 0.014 | 4.545 ± 0.036 | 9.985 ± 0.566 | 6.776 ± 0.077 | 8.650 ± 0.164 | 14.537 ± 0.432 |
| TabPFN FT, Gaussian context | 5 | 7.420 ± 0.005 | 9.574 ± 0.009 | 4.689 ± 0.027 | 9.634 ± 0.038 | 6.998 ± 0.003 | 8.291 ± 0.014 | 13.769 ± 0.036 |
| Gaussian-context FT + recalibration | 5 | 7.426 ± 0.004 | 9.487 ± 0.008 | 4.770 ± 0.038 | 10.423 ± 0.021 | 6.737 ± 0.004 | 8.884 ± 0.012 | 16.131 ± 0.024 |


**`ft_gauss` vs `ft_uniform`** (negative = `ft_gauss` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | +0.045 | +0.6% | 0/5 | 0.0625 | LOSS |
| rmse | +0.041 | +0.4% | 0/5 | 0.0625 | LOSS |
| gm | +0.145 | +3.2% | 0/5 | 0.0625 | LOSS |
| sera | -0.351 | -3.5% | 3/5 | 0.3125 | tie |
| mae_many | +0.223 | +3.3% | 0/5 | 0.0625 | LOSS |
| mae_medium | -0.360 | -4.2% | 5/5 | 0.0625 | WIN |
| mae_few | -0.768 | -5.3% | 5/5 | 0.0625 | WIN |


**`ft_gauss` vs `zeroshot_recal_iso`** (negative = `ft_gauss` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | +0.033 | +0.4% | 0/5 | 0.0625 | LOSS |
| rmse | +0.106 | +1.1% | 0/5 | 0.0625 | LOSS |
| gm | +0.478 | +11.4% | 0/5 | 0.0625 | LOSS |
| sera | -0.733 | -7.1% | 5/5 | 0.0625 | WIN |
| mae_many | +0.376 | +5.7% | 0/5 | 0.0625 | LOSS |
| mae_medium | -0.764 | -8.4% | 5/5 | 0.0625 | WIN |
| mae_few | -0.947 | -6.4% | 5/5 | 0.0625 | WIN |


**`ft_gauss_recal` vs `zeroshot_recal_iso`** (negative = `ft_gauss_recal` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | +0.039 | +0.5% | 0/5 | 0.0625 | LOSS |
| rmse | +0.019 | +0.2% | 0/5 | 0.0625 | LOSS |
| gm | +0.559 | +13.3% | 0/5 | 0.0625 | LOSS |
| sera | +0.056 | +0.5% | 1/5 | 0.1250 | LOSS |
| mae_many | +0.115 | +1.7% | 0/5 | 0.0625 | LOSS |
| mae_medium | -0.171 | -1.9% | 5/5 | 0.0625 | WIN |
| mae_few | +1.415 | +9.6% | 0/5 | 0.0625 | LOSS |


### ptbxl_tabpfn_enc — Encoder trained through frozen TabPFN-2.5

| arm | seeds | MAE | RMSE | GM | SERA | MAE many | MAE medium | MAE few |
|---|---|---|---|---|---|---|---|---|
| LightGBM on embeddings | 5 | 7.432 ± 0.007 | 9.539 ± 0.025 | 4.764 ± 0.127 | 10.585 ± 0.791 | 6.818 ± 0.158 | 8.717 ± 0.368 | 15.775 ± 1.350 |
| TabPFN-2.5 zero-shot | 5 | 7.421 ± 0.003 | 9.507 ± 0.002 | 4.724 ± 0.025 | 9.481 ± 0.019 | 6.972 ± 0.004 | 8.362 ± 0.018 | 13.527 ± 0.020 |
| zero-shot + linear recalibration | 5 | 7.416 ± 0.004 | 9.430 ± 0.004 | 4.872 ± 0.026 | 10.107 ± 0.007 | 6.723 ± 0.003 | 8.892 ± 0.013 | 15.728 ± 0.030 |
| zero-shot + isotonic recalibration | 5 | 7.485 ± 0.012 | 9.520 ± 0.016 | 4.207 ± 0.137 | 10.125 ± 0.069 | 6.760 ± 0.023 | 9.070 ± 0.028 | 14.315 ± 0.446 |
| TabPFN FT, uniform context | 5 | 7.368 ± 0.023 | 9.457 ± 0.017 | 4.687 ± 0.030 | 9.771 ± 0.395 | 6.788 ± 0.077 | 8.616 ± 0.112 | 13.827 ± 0.158 |
| TabPFN FT, Gaussian context | 5 | 7.421 ± 0.003 | 9.507 ± 0.002 | 4.724 ± 0.025 | 9.481 ± 0.019 | 6.972 ± 0.004 | 8.362 ± 0.018 | 13.527 ± 0.020 |
| Gaussian-context FT + recalibration | 5 | 7.416 ± 0.004 | 9.430 ± 0.004 | 4.872 ± 0.026 | 10.106 ± 0.007 | 6.723 ± 0.003 | 8.892 ± 0.013 | 15.728 ± 0.030 |


**`ft_gauss` vs `ft_uniform`** (negative = `ft_gauss` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | +0.052 | +0.7% | 0/5 | 0.0625 | LOSS |
| rmse | +0.050 | +0.5% | 0/5 | 0.0625 | LOSS |
| gm | +0.036 | +0.8% | 1/5 | 0.1875 | LOSS |
| sera | -0.290 | -3.0% | 4/5 | 0.1875 | WIN |
| mae_many | +0.184 | +2.7% | 0/5 | 0.0625 | LOSS |
| mae_medium | -0.254 | -2.9% | 5/5 | 0.0625 | WIN |
| mae_few | -0.300 | -2.2% | 5/5 | 0.0625 | WIN |


**`ft_gauss` vs `zeroshot_recal_iso`** (negative = `ft_gauss` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | -0.064 | -0.9% | 5/5 | 0.0625 | WIN |
| rmse | -0.013 | -0.1% | 4/5 | 0.1875 | WIN |
| gm | +0.517 | +12.3% | 0/5 | 0.0625 | LOSS |
| sera | -0.644 | -6.4% | 5/5 | 0.0625 | WIN |
| mae_many | +0.212 | +3.1% | 0/5 | 0.0625 | LOSS |
| mae_medium | -0.708 | -7.8% | 5/5 | 0.0625 | WIN |
| mae_few | -0.788 | -5.5% | 5/5 | 0.0625 | WIN |


**`ft_gauss_recal` vs `zeroshot_recal_iso`** (negative = `ft_gauss_recal` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | -0.069 | -0.9% | 5/5 | 0.0625 | WIN |
| rmse | -0.090 | -0.9% | 5/5 | 0.0625 | WIN |
| gm | +0.665 | +15.8% | 0/5 | 0.0625 | LOSS |
| sera | -0.018 | -0.2% | 3/5 | 0.4375 | tie |
| mae_many | -0.037 | -0.5% | 5/5 | 0.0625 | WIN |
| mae_medium | -0.178 | -2.0% | 5/5 | 0.0625 | WIN |
| mae_few | +1.413 | +9.9% | 0/5 | 0.0625 | LOSS |


### Stage-1 effect: the two encoders under the same arm


**arm = `zeroshot`** (negative delta = TabPFN-trained encoder better)

| metric | L1 encoder | TabPFN-trained encoder | rel delta | seeds won |
|---|---|---|---|---|
| mae | 7.420 | 7.421 | +0.0% | 3/5 |
| rmse | 9.574 | 9.507 | -0.7% | 5/5 |
| gm | 4.691 | 4.724 | +0.7% | 0/5 |
| sera | 9.635 | 9.481 | -1.6% | 5/5 |
| mae_many | 6.998 | 6.972 | -0.4% | 5/5 |
| mae_medium | 8.290 | 8.362 | +0.9% | 0/5 |
| mae_few | 13.771 | 13.527 | -1.8% | 5/5 |


**arm = `ft_gauss`** (negative delta = TabPFN-trained encoder better)

| metric | L1 encoder | TabPFN-trained encoder | rel delta | seeds won |
|---|---|---|---|---|
| mae | 7.420 | 7.421 | +0.0% | 3/5 |
| rmse | 9.574 | 9.507 | -0.7% | 5/5 |
| gm | 4.689 | 4.724 | +0.7% | 0/5 |
| sera | 9.634 | 9.481 | -1.6% | 5/5 |
| mae_many | 6.998 | 6.972 | -0.4% | 5/5 |
| mae_medium | 8.291 | 8.362 | +0.9% | 0/5 |
| mae_few | 13.769 | 13.527 | -1.8% | 5/5 |


---

## 4b. Results — Rebalanced split

Where the tail claim is actually testable.


**Did the fine-tuning actually change the weights?**

| encoder | arm | seeds identical to zero-shot | status |
|---|---|---|---|
| `ptbxl_l1_enc` | `ft_uniform` | 0/5 | trained |
| `ptbxl_l1_enc` | `ft_gauss` | 0/5 | trained |
| `ptbxl_l1_enc` | `ft_gauss_recal` | 0/5 | trained |
| `ptbxl_tabpfn_enc` | `ft_uniform` | 0/5 | trained |
| `ptbxl_tabpfn_enc` | `ft_gauss` | 0/5 | trained |
| `ptbxl_tabpfn_enc` | `ft_gauss_recal` | 0/5 | trained |


### ptbxl_l1_enc — L1-trained backbone (control)

| arm | seeds | MAE | RMSE | GM | SERA | MAE many | MAE medium | MAE few |
|---|---|---|---|---|---|---|---|---|
| LightGBM on embeddings | 5 | 7.473 ± 0.074 | 9.956 ± 0.095 | 4.691 ± 0.062 | 15.454 ± 1.029 | 5.716 ± 0.268 | 7.819 ± 0.200 | 13.615 ± 0.279 |
| TabPFN-2.5 zero-shot | 5 | 7.351 ± 0.076 | 9.811 ± 0.104 | 4.615 ± 0.060 | 15.017 ± 1.277 | 5.658 ± 0.214 | 7.627 ± 0.181 | 13.459 ± 0.247 |
| zero-shot + linear recalibration | 5 | 6.973 ± 0.082 | 9.449 ± 0.150 | 4.346 ± 0.065 | 13.798 ± 1.277 | 6.101 ± 0.232 | 6.871 ± 0.148 | 10.908 ± 0.520 |
| zero-shot + isotonic recalibration | 5 | 7.012 ± 0.113 | 9.476 ± 0.175 | 4.156 ± 0.267 | 13.614 ± 0.926 | 6.574 ± 0.337 | 6.211 ± 0.202 | 11.412 ± 0.459 |
| TabPFN FT, uniform context | 5 | 7.257 ± 0.094 | 9.722 ± 0.117 | 4.545 ± 0.103 | 14.232 ± 1.461 | 5.820 ± 0.335 | 7.314 ± 0.330 | 13.013 ± 0.307 |
| TabPFN FT, Gaussian context | 5 | 6.971 ± 0.099 | 9.455 ± 0.163 | 4.330 ± 0.068 | 13.084 ± 1.942 | 6.221 ± 0.356 | 6.594 ± 0.299 | 11.293 ± 0.566 |
| Gaussian-context FT + recalibration | 5 | 6.970 ± 0.102 | 9.429 ± 0.177 | 4.371 ± 0.075 | 13.372 ± 1.202 | 6.300 ± 0.276 | 6.506 ± 0.148 | 11.241 ± 0.463 |


**`ft_gauss` vs `ft_uniform`** (negative = `ft_gauss` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | -0.286 | -3.9% | 5/5 | 0.0625 | WIN |
| rmse | -0.266 | -2.7% | 5/5 | 0.0625 | WIN |
| gm | -0.215 | -4.7% | 5/5 | 0.0625 | WIN |
| sera | -1.148 | -8.1% | 4/5 | 0.1250 | WIN |
| mae_many | +0.401 | +6.9% | 1/5 | 0.1250 | LOSS |
| mae_medium | -0.719 | -9.8% | 5/5 | 0.0625 | WIN |
| mae_few | -1.720 | -13.2% | 5/5 | 0.0625 | WIN |


**`ft_gauss` vs `zeroshot_recal_iso`** (negative = `ft_gauss` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | -0.041 | -0.6% | 3/5 | 0.3125 | tie |
| rmse | -0.021 | -0.2% | 3/5 | 0.6250 | tie |
| gm | +0.174 | +4.2% | 1/5 | 0.1875 | LOSS |
| sera | -0.529 | -3.9% | 3/5 | 0.4375 | tie |
| mae_many | -0.354 | -5.4% | 5/5 | 0.0625 | WIN |
| mae_medium | +0.383 | +6.2% | 0/5 | 0.0625 | LOSS |
| mae_few | -0.120 | -1.0% | 4/5 | 0.4375 | WIN |


**`ft_gauss_recal` vs `zeroshot_recal_iso`** (negative = `ft_gauss_recal` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | -0.042 | -0.6% | 5/5 | 0.0625 | WIN |
| rmse | -0.047 | -0.5% | 4/5 | 0.1250 | WIN |
| gm | +0.215 | +5.2% | 1/5 | 0.1250 | LOSS |
| sera | -0.241 | -1.8% | 3/5 | 0.3125 | tie |
| mae_many | -0.274 | -4.2% | 5/5 | 0.0625 | WIN |
| mae_medium | +0.295 | +4.7% | 0/5 | 0.0625 | LOSS |
| mae_few | -0.171 | -1.5% | 3/5 | 0.3125 | tie |


### ptbxl_tabpfn_enc — Encoder trained through frozen TabPFN-2.5

| arm | seeds | MAE | RMSE | GM | SERA | MAE many | MAE medium | MAE few |
|---|---|---|---|---|---|---|---|---|
| LightGBM on embeddings | 5 | 7.389 ± 0.093 | 9.837 ± 0.115 | 4.680 ± 0.089 | 15.010 ± 0.913 | 5.707 ± 0.257 | 7.706 ± 0.147 | 13.317 ± 0.240 |
| TabPFN-2.5 zero-shot | 5 | 7.280 ± 0.094 | 9.715 ± 0.117 | 4.595 ± 0.124 | 14.469 ± 1.093 | 5.661 ± 0.246 | 7.530 ± 0.135 | 13.163 ± 0.291 |
| zero-shot + linear recalibration | 5 | 6.909 ± 0.095 | 9.361 ± 0.159 | 4.332 ± 0.079 | 13.307 ± 1.069 | 6.092 ± 0.263 | 6.802 ± 0.124 | 10.629 ± 0.474 |
| zero-shot + isotonic recalibration | 5 | 6.939 ± 0.112 | 9.375 ± 0.167 | 4.262 ± 0.168 | 12.930 ± 0.851 | 6.553 ± 0.314 | 6.142 ± 0.143 | 11.122 ± 0.360 |
| TabPFN FT, uniform context | 5 | 7.195 ± 0.118 | 9.622 ± 0.160 | 4.531 ± 0.141 | 13.849 ± 1.421 | 5.751 ± 0.256 | 7.356 ± 0.122 | 12.643 ± 0.414 |
| TabPFN FT, Gaussian context | 5 | 6.872 ± 0.109 | 9.347 ± 0.164 | 4.267 ± 0.074 | 11.826 ± 1.159 | 6.065 ± 0.213 | 6.593 ± 0.124 | 11.116 ± 0.544 |
| Gaussian-context FT + recalibration | 5 | 6.888 ± 0.087 | 9.320 ± 0.157 | 4.326 ± 0.070 | 12.817 ± 1.016 | 6.261 ± 0.232 | 6.442 ± 0.128 | 10.923 ± 0.339 |


**`ft_gauss` vs `ft_uniform`** (negative = `ft_gauss` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | -0.323 | -4.5% | 5/5 | 0.0625 | WIN |
| rmse | -0.275 | -2.9% | 5/5 | 0.0625 | WIN |
| gm | -0.263 | -5.8% | 5/5 | 0.0625 | WIN |
| sera | -2.022 | -14.6% | 5/5 | 0.0625 | WIN |
| mae_many | +0.314 | +5.5% | 0/5 | 0.0625 | LOSS |
| mae_medium | -0.762 | -10.4% | 5/5 | 0.0625 | WIN |
| mae_few | -1.528 | -12.1% | 5/5 | 0.0625 | WIN |


**`ft_gauss` vs `zeroshot_recal_iso`** (negative = `ft_gauss` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | -0.067 | -1.0% | 4/5 | 0.1250 | WIN |
| rmse | -0.028 | -0.3% | 4/5 | 0.1875 | WIN |
| gm | +0.006 | +0.1% | 3/5 | 1.0000 | tie |
| sera | -1.103 | -8.5% | 5/5 | 0.0625 | WIN |
| mae_many | -0.488 | -7.4% | 5/5 | 0.0625 | WIN |
| mae_medium | +0.452 | +7.4% | 0/5 | 0.0625 | LOSS |
| mae_few | -0.006 | -0.1% | 3/5 | 0.8125 | tie |


**`ft_gauss_recal` vs `zeroshot_recal_iso`** (negative = `ft_gauss_recal` better)

| metric | abs delta | rel delta | seeds won | wilcoxon p |  |
|---|---|---|---|---|---|
| mae | -0.052 | -0.7% | 4/5 | 0.1250 | WIN |
| rmse | -0.055 | -0.6% | 4/5 | 0.1250 | WIN |
| gm | +0.064 | +1.5% | 3/5 | 0.8125 | tie |
| sera | -0.113 | -0.9% | 4/5 | 0.3125 | WIN |
| mae_many | -0.292 | -4.5% | 5/5 | 0.0625 | WIN |
| mae_medium | +0.300 | +4.9% | 0/5 | 0.0625 | LOSS |
| mae_few | -0.199 | -1.8% | 4/5 | 0.1875 | WIN |


### Stage-1 effect: the two encoders under the same arm


**arm = `zeroshot`** (negative delta = TabPFN-trained encoder better)

| metric | L1 encoder | TabPFN-trained encoder | rel delta | seeds won |
|---|---|---|---|---|
| mae | 7.351 | 7.280 | -1.0% | 5/5 |
| rmse | 9.811 | 9.715 | -1.0% | 5/5 |
| gm | 4.615 | 4.595 | -0.4% | 4/5 |
| sera | 15.017 | 14.469 | -3.6% | 5/5 |
| mae_many | 5.658 | 5.661 | +0.1% | 2/5 |
| mae_medium | 7.627 | 7.530 | -1.3% | 5/5 |
| mae_few | 13.459 | 13.163 | -2.2% | 5/5 |


**arm = `ft_gauss`** (negative delta = TabPFN-trained encoder better)

| metric | L1 encoder | TabPFN-trained encoder | rel delta | seeds won |
|---|---|---|---|---|
| mae | 6.971 | 6.872 | -1.4% | 5/5 |
| rmse | 9.455 | 9.347 | -1.1% | 5/5 |
| gm | 4.330 | 4.267 | -1.4% | 4/5 |
| sera | 13.084 | 11.826 | -9.6% | 4/5 |
| mae_many | 6.221 | 6.065 | -2.5% | 4/5 |
| mae_medium | 6.594 | 6.593 | -0.0% | 3/5 |
| mae_few | 11.293 | 11.116 | -1.6% | 3/5 |


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

---

## Table 1 — cumulative effect of the two modifications (completed 2026-10-29)

PTB-XL age regression, DIR-balanced evaluation protocol, 5 seeds, fine-tuning
learning rate selected per seed on validation RMSE (same rule as every other
table in this log).

Encoders: **E1** = ResNet1d-Wang trained directly on age (L1 loss).
**E2** = same backbone trained *through* the frozen stock TabPFN-2.5.
**E2.1** = trained through the frozen *Gaussian-fine-tuned* TabPFN.
**E2.2** = trained through the frozen *Gaussian+Uniform-fine-tuned* TabPFN.

| TabPFN-2.5 | Encoder | MAE | Abs. reduction | Improvement |
|---|---|---|---|---|
| Frozen | E1 | 7.351 | — | — |
| Frozen | E2 | 7.280 | 0.071 | 1.0% |
| Uniform fine-tuning | E1 | 7.257 | 0.095 | 1.3% |
| Uniform fine-tuning | E2 | 7.195 | 0.157 | 2.1% |
| **Gaussian fine-tuning** | **E2** | **6.872** | **0.479** | **6.5%** |
| Gaussian fine-tuning | E1 | 6.971 | 0.380 | 5.2% |
| Gaussian fine-tuning | E2.1 | 7.018 | 0.333 | 4.5% |
| Gaussian + Uniform fine-tuning | E1 | 7.105 | 0.247 | 3.4% |
| Gaussian + Uniform fine-tuning | E2 | 7.033 | 0.318 | 4.3% |
| Gaussian + Uniform fine-tuning | E2.2 | 7.182 | 0.170 | 2.3% |

Baseline for "abs. reduction" is Frozen/E1 = 7.351.

### Paired per-seed comparisons (n=5)

| Comparison | mean Δ MAE | seeds won |
|---|---|---|
| Gauss+Uniform vs Uniform (E1) | −0.152 | 5/5 |
| Gauss+Uniform vs Uniform (E2) | −0.162 | 5/5 |
| Gauss+Uniform vs Gaussian (E1) | +0.134 | 0/5 |
| Gauss+Uniform vs Gaussian (E2) | +0.161 | 0/5 |
| E2.1 vs E2, Gaussian arm | +0.146 | 1/5 |
| E2.2 vs E2, Gauss+Uniform arm | +0.149 | 0/5 |

Wilcoxon signed-rank cannot reach p<0.05 at n=5 (floor p=0.0625), so seed-win
counts are reported instead.

### Findings

**1. The ordering is monotone: Uniform < Gaussian+Uniform < Gaussian.**
Mixing half a uniform context back in recovers part of the Gaussian benefit
(beats pure uniform 5/5 on both encoders) but strictly loses to pure Gaussian
(0/5 on both). Diluting the narrow-context episodes weakens the training
signal rather than regularising it. The hypothesis that mixing would combine
the strengths of both is not supported.

**2. Co-adapting the encoder to an already-fine-tuned TabPFN does not help.**
E2.1 and E2.2 are both worse than E2, which was trained through the *stock*
frozen model. This is a genuine negative result, not a training failure:
- identical warm start, split, step budget (2000) and eval cadence for E2, E2.1, E2.2
- fine-tuned weights verified loaded (191/191 tensors, missing=0 unexpected=0)
- best-validation checkpoint selection in all three runs

The encoder-training traces show why. Best validation MAE during Stage 1 was
E2 7.313 @ step 400, E2.1 7.863 @ step 200, E2.2 7.646 @ step 400; the E2.1/E2.2
runs peak almost immediately and degrade thereafter, and their training loss
sits higher throughout without descending. Training through the fine-tuned model
gives the encoder a worse optimisation signal, so the saved checkpoint is barely
moved from the warm start — which is why E2.1's frozen baseline (7.446) is closer
to E1's (7.351) than to E2's (7.280).

**Best configuration remains Gaussian fine-tuning on E2: MAE 6.872, a 6.5%
reduction over the frozen/E1 baseline.** Neither modification added in this
round improves on it.

### Age-band breakdown — Gaussian + Uniform fine-tuning

Same runs as Table 1 (DIR-balanced protocol, 5 seeds, val-RMSE-selected LR),
re-scored by absolute age band instead of by train-frequency shot region.
Test points per band: age≤30 = 340, 30<age<60 = 580, age≥60 = 579.
Values are MAE in years; Δ is against the frozen (zero-shot) model on the same encoder.

| Encoder | Arm | age ≤ 30 | 30 < age < 60 | age ≥ 60 | Overall |
|---|---|---|---|---|---|
| E1 | Frozen | 7.298 | 6.000 | 8.737 | 7.351 |
| E1 | **Gauss+Uniform** | **6.484** | **6.542** | **8.033** | **7.105** |
| E1 | *Δ* | *−0.814* | *+0.542* | *−0.704* | *−0.246* |
| E2 | Frozen | 7.144 | 5.975 | 8.668 | 7.280 |
| E2 | **Gauss+Uniform** | **6.283** | **6.604** | **7.903** | **7.033** |
| E2 | *Δ* | *−0.861* | *+0.629* | *−0.765* | *−0.247* |
| E2.2 | Frozen | 7.233 | 6.027 | 8.856 | 7.393 |
| E2.2 | **Gauss+Uniform** | **6.687** | **6.666** | **7.988** | **7.182** |
| E2.2 | *Δ* | *−0.546* | *+0.639* | *−0.868* | *−0.211* |

**The entire gain comes from the two tails; the middle band gets worse.**
On all three encoders Gauss+Uniform improves age≤30 by 0.55-0.86 years and
age≥60 by 0.70-0.87 years, while degrading 30<age<60 by 0.54-0.64 years. This is
the intended trade: PTB-XL's training ages concentrate in the middle, so the
narrow-context episodes buy tail accuracy at the head's expense. The net is
positive under the balanced protocol because the tails hold 919 of 1499 test
points — under the official split, where the middle dominates, the same trade
would net out much smaller.

**Pure Gaussian vs Gaussian + Uniform, by band (positive Δ = pure Gaussian better).**
This is the same-encoder contrast; E2.2 has no pure-Gaussian arm, so only E1 and E2 appear.

| Encoder | Band | Gaussian | Gauss+Uniform | Δ | Gaussian wins |
|---|---|---|---|---|---|
| E1 | Overall | **6.971** | 7.105 | +0.134 | **5/5** |
| E1 | age ≤ 30 | 6.484 | 6.484 | −0.000 | 2/5 |
| E1 | 30 < age < 60 | 6.612 | 6.542 | −0.070 | 1/5 |
| E1 | age ≥ 60 | **7.616** | 8.033 | **+0.416** | **5/5** |
| E2 | Overall | **6.872** | 7.033 | +0.161 | **5/5** |
| E2 | age ≤ 30 | 5.990 | 6.283 | +0.294 | 4/5 |
| E2 | 30 < age < 60 | 6.544 | 6.604 | +0.060 | 3/5 |
| E2 | age ≥ 60 | 7.719 | 7.903 | +0.184 | 3/5 |

**Only two of these claims survive the seed noise.** Pure Gaussian beats the
mixture overall on both encoders, 5/5 seeds. On E1 that deficit is almost
entirely the age≥60 band: +0.416 years, 5/5 seeds, every seed between +0.28 and
+0.63. So on E1, putting uniform context back specifically gives back the
old-age tail.

Everything else per-band is within noise and must not be reported as a finding:

- E1 age≤30 shows Δ = −0.0001, which is a **coincidence of averaging, not a tie**.
  The per-seed deltas are [+0.982, +0.860, −0.739, −0.429, −0.675] and cancel.
  The band is noisy at roughly ±1 year per seed.
- E1 30<age<60 flips sign across seeds (1/5).
- E2's three bands all lean the same way as its overall number but none is
  unanimous; age≥60 is 3/5 with two seeds favouring the mixture. The E1
  old-age-tail mechanism therefore does **not** replicate cleanly on E2, and
  localising E2's deficit to a band would need more than 5 seeds.


---

## Experiment 4 — predictive uncertainty of the best model (2026-10-29)

Gaussian fine-tuning on encoder E2, DIR protocol, 5 seeds, 7,495 test predictions.
`calibration_experiment.py` re-runs that exact cell and additionally dumps, per test
point, the standard deviation of TabPFN's full predictive distribution, taken from
`FullSupportBarDistribution.variance` (not a quantile spread: the outermost bins
carry half-normal tails of infinite support, and a Gaussian quantile estimate
understated sigma by ~20% on a controlled check). Figure: `imbreg/figures/04_uncertainty_calibration.png`.

Reproduction is not bit-exact -- GPU fine-tuning is non-deterministic -- but all five
seeds land within 0.04 years (<=0.6%) of the recorded MAE, with the sign of the
difference varying by seed:

| seed | MAE reproduced | MAE recorded | diff | mean sigma |
|---|---|---|---|---|
| 0 | 6.8036 | 6.7746 | +0.029 | 7.405 |
| 1 | 6.8484 | 6.8092 | +0.039 | 7.388 |
| 2 | 6.8360 | 6.7964 | +0.040 | 7.339 |
| 3 | 6.9659 | 6.9912 | −0.025 | 7.185 |
| 4 | 6.9703 | 6.9902 | −0.020 | 7.287 |

### Headline: the model is over-confident, and almost all of it is the age>80 tail

Interval coverage is below nominal at every level:

| interval | nominal | observed |
|---|---|---|
| ±1σ | 68.3% | 62.4% |
| ±1.645σ | 90.0% | 84.5% |
| ±1.96σ | 95.0% | 90.5% |
| ±2.576σ | 99.0% | 96.2% |

Mean realised RMSE within a sigma bin is 1.26x the sigma claimed for that bin.

Broken out by age, the global number is almost entirely one band:

| band | n | mean abs. error | mean sigma | error/sigma | ±1σ coverage |
|---|---|---|---|---|---|
| age < 30 | 1600 | 5.89 | 6.50 | 0.91 | 64.1% |
| 30-60 | 3000 | 6.56 | 8.06 | 0.81 | 68.2% |
| 60-80 | 2000 | 5.73 | 7.21 | 0.79 | 71.1% |
| **age > 80** | **895** | **12.32** | **6.55** | **1.88** | **20.4%** |

Below 80 the model is well calibrated to slightly conservative (coverage 64-71%
against a nominal 68.3%). Above 80 the absolute error roughly doubles to 12.3 years
while sigma does not rise -- it *falls* slightly, to 6.55 -- so a nominal 68%
interval covers **20%** of those patients.

### Does sigma predict error at all?

Globally Spearman(sigma, |error|) = 0.183, which looks weak. That number is
misleading: the relationship is non-monotone across bands. Restricted to age >= 75 it
is **0.685**. And Spearman(sigma, true age) = **-0.099**: sigma does not rise with age
at all. The model has no idea that elderly patients are the hard case.

### What this means

Gaussian fine-tuning reduced tail *error* (Table 1, and the age-band breakdown above),
but it did **not** fix tail *uncertainty*. On the oldest patients the model is
confidently wrong, which is the dangerous failure mode for anything clinical --
a wide, honest error bar would be far safer than a narrow, wrong one.

This is a limitation of the method as it stands, not a result in its favour, and it
should be reported as such. It also suggests a concrete next step: the current loss
rewards accurate point predictions on query points drawn uniformly; nothing in the
objective asks the predictive *distribution* to widen where the context was thin.
