# Does the Gaussian-context idea transfer to another in-context model?

Generated 2026-10-24 16:54 by `make_tabicl_report.py`.
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

| dataset | TabPFN-2.5: ft_gauss vs ft_uniform | TabICL v2: ft_gauss vs ft_uniform |
|---|---|---|
| `abalone` | -20.9% (5/5) yes | -23.3% (5/5) yes |
| `communities` | -4.1% (5/5) yes | -4.3% (5/5) yes |
| `wine_white` | -6.5% (5/5) yes | -9.0% (5/5) yes |
| `ptbxl_tabpfn_enc` | -1.5% (2/5) tie | -4.7% (4/5) yes |
| `agedb_tabpfn_enc` | -5.1% (5/5) yes | — |

Negative percentages are improvements; the fraction is the number of seeds on which Gaussian sampling won. Seed-win counts rather than p-values, since Wilcoxon cannot go below p = 0.0625 at five seeds.

## Absolute numbers, both models side by side

### abalone

| arm | seeds | mae | rmse | sera | mae_many | mae_medium | mae_few |
|---|---|---|---|---|---|---|---|
| TabPFN zeroshot | 5 | 2.942 | 3.992 | 14.842 | 1.170 | 2.569 | 6.163 |
| TabICL zeroshot | 5 | 2.956 | 4.011 | 15.025 | 1.146 | 2.595 | 6.219 |
| TabPFN ft_uniform | 5 | 2.786 | 3.745 | 12.890 | 1.208 | 2.357 | 5.767 |
| TabICL ft_uniform | 5 | 2.807 | 3.815 | 13.501 | 1.173 | 2.374 | 5.879 |
| TabPFN ft_gauss | 5 | 2.202 | 2.958 | 6.955 | 1.584 | 1.392 | 4.127 |
| TabICL ft_gauss | 5 | 2.153 | 2.863 | 6.092 | 1.673 | 1.846 | 3.266 |

### communities

| arm | seeds | mae | rmse | sera | mae_many | mae_medium | mae_few |
|---|---|---|---|---|---|---|---|
| TabPFN zeroshot | 5 | 0.161 | 0.222 | 0.045 | 0.059 | 0.168 | 0.258 |
| TabICL zeroshot | 5 | 0.160 | 0.221 | 0.045 | 0.059 | 0.168 | 0.255 |
| TabPFN ft_uniform | 5 | 0.160 | 0.220 | 0.044 | 0.060 | 0.168 | 0.254 |
| TabICL ft_uniform | 5 | 0.150 | 0.206 | 0.037 | 0.068 | 0.163 | 0.220 |
| TabPFN ft_gauss | 5 | 0.153 | 0.211 | 0.039 | 0.065 | 0.162 | 0.235 |
| TabICL ft_gauss | 5 | 0.144 | 0.198 | 0.032 | 0.073 | 0.164 | 0.192 |

### wine_white

| arm | seeds | mae | rmse | sera | mae_many | mae_medium | mae_few |
|---|---|---|---|---|---|---|---|
| TabPFN zeroshot | 5 | 0.926 | 1.168 | 1.067 | 0.427 | 1.432 | 2.455 |
| TabICL zeroshot | 5 | 0.785 | 1.090 | 0.936 | 0.335 | 1.195 | 2.453 |
| TabPFN ft_uniform | 5 | 0.837 | 1.098 | 0.940 | 0.370 | 1.295 | 2.375 |
| TabICL ft_uniform | 5 | 0.783 | 1.090 | 0.937 | 0.333 | 1.192 | 2.464 |
| TabPFN ft_gauss | 5 | 0.783 | 1.039 | 0.819 | 0.391 | 1.138 | 2.250 |
| TabICL ft_gauss | 5 | 0.713 | 1.021 | 0.797 | 0.334 | 1.019 | 2.364 |

### ptbxl_tabpfn_enc

| arm | seeds | mae | rmse | sera | mae_many | mae_medium | mae_few |
|---|---|---|---|---|---|---|---|
| TabPFN zeroshot | 10 | 7.351 | 9.611 | 11.975 | 6.316 | 7.946 | 13.345 |
| TabICL zeroshot | 5 | 7.258 | 9.709 | 14.510 | 5.636 | 7.495 | 13.196 |
| TabPFN ft_uniform | 5 | 7.279 | 9.506 | 11.318 | 6.289 | 8.137 | 13.364 |
| TabICL ft_uniform | 5 | 7.231 | 9.678 | 14.668 | 5.663 | 7.447 | 13.013 |
| TabPFN ft_gauss | 5 | 7.167 | 9.413 | 10.397 | 6.546 | 7.680 | 12.453 |
| TabICL ft_gauss | 5 | 6.895 | 9.406 | 11.726 | 6.044 | 6.724 | 10.962 |

### agedb_tabpfn_enc

| arm | seeds | mae | rmse | sera | mae_many | mae_medium | mae_few |
|---|---|---|---|---|---|---|---|
| TabPFN zeroshot | 5 | 11.335 | 13.955 | 59.299 | 8.913 | 15.945 | 21.466 |
| TabICL zeroshot | 5 | 11.548 | 14.186 | 64.719 | 8.900 | 16.766 | 22.117 |
| TabPFN ft_uniform | 5 | 11.262 | 13.899 | 53.699 | 9.138 | 14.982 | 21.036 |
| TabPFN ft_gauss | 5 | 10.687 | 13.457 | 41.760 | 9.334 | 12.538 | 18.351 |

## Reading this

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
