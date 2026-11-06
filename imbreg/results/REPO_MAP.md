# Repository map, metric definitions and splits

Written for whoever has to check a number in the paper against the code that produced
it. Deliverable 1 of `paper_completion_plan.md` section 7.

Last updated 6 Nov 2026.

---

## 1. Pipelines

| stage | code | what it produces |
|---|---|---|
| ECG encoder, supervised | `train.py` + `resnet1d_wang_fds.py` | ResNet1d-Wang trained on an L1 age loss. Also the Task A baseline. |
| ECG encoder, through a frozen ICL model | `ecg_tabpfn/train_encoder.py` (TabPFN), `ecg_tabpfn/train_encoder_tabicl.py` (TabICL) | the same backbone, trained so a frozen in-context model predicts well from its features |
| embedding dump | `ecg_tabpfn/extract_embeddings.py` | `ecg_tabpfn/results/emb_*.npz`, one 128-d vector per record |
| prior-level continued pre-training | `tabicl_prior/train_gauss_prior.py` + `gauss_prior.py` | `tabicl_prior/runs/<mode>/step-3000.ckpt` |
| evaluation harness | `imbreg/run.py` | per-prediction CSVs + one JSONL row per (dataset, arm, seed) |
| metrics | `eval/compute_metrics.py` | `imbreg/results/metrics/*.csv` |
| significance | `eval/significance.py` | paired bootstrap CIs and per-seed win counts |
| tables / figures | `eval/make_tables.py`, `eval/make_figures.py` | `paper/tables/*.tex`, `paper/figs/*.pdf` |

Metrics are computed in exactly one place, `eval/compute_metrics.py`, reading saved
predictions. No metric is computed inside a training loop.

## 2. The encoders

| | backbone | trained through |
|---|---|---|
| E1 | ResNet1d-Wang | nothing: supervised L1 age loss |
| E2 | same | frozen stock TabPFN-2.5 |
| E3 | same | frozen stock TabICL |
| E4 | same | frozen Gaussian-prior TabICL |
| E5 | same | frozen mixed-prior TabICL |

E3, E4 and E5 all start from E1's checkpoint, so the comparison between them is the
frozen body and nothing else. E5 has no row in the paper's tables; it is reported in
`STATUS.md` only.

## 3. The prediction bodies

All four are TabICL. `stock` is the released regressor; the other three are that
same checkpoint given 3000 further pre-training steps on TabICL's **own synthetic
prior**, differing only in which rows of each synthetic dataset land before the
positional context/query cut:

| mode | rule | role |
|---|---|---|
| `uniform` | unbiased permutation | matched control: separates "the ordering helped" from "3000 more steps helped" |
| `gauss` | Gaussian weight on the target, centre and width redrawn per episode | the intervention |
| `mixed` | half Gaussian, half uniform | dilution ablation |
| `target_balanced` | probability proportional to 1 / (count of the row's target bin), 50 equal-width bins | Task B control: does a plain balancing scheme do the same job? |

No body has ever seen any evaluation dataset.

## 4. Splits

**PTB-XL.** `data/1000_timesteps.csv` carries a `split` column: 7566 train / 963 val
/ 954 test. `train.py` and `ecg_tabpfn/ecg_data.py` both read that column, so the
ResNet baselines and the encoders share the identical partition.

`imbreg/data.py:load_embedding_npz` concatenates train+val+test into one 9483-row
table and records the official boundaries as `fixed_split`. Two protocols then run
on top of it:

- **`fixed`** uses the official partition. Test = the 954 records, natural age
  distribution, seen by no encoder. Seeds vary model randomness only, not the split.
- **`dir`** (`imbreg/data.py:dir_split`) ignores the official boundaries and
  re-partitions all 9483 records: each 1-year age bin donates up to `per_bin_test`
  rows to test and `per_bin_val` to val, leaving the rest as train. Test is therefore
  near-uniform over age — this is what the existing code calls **balanced
  evaluation**, confirmed by reading the function, not inferred from the numbers.

  **This protocol leaks.** See `STATUS.md` section 1.

**Tabular.** Abalone, Communities and Wine Quality White have no official split and
use `dir` throughout.

| dataset | n | features | bin width | per-bin val / test |
|---|---|---|---|---|
| abalone | 4177 | 8 | 1.0 | 10 / 20 |
| communities | 1994 | 122 | 0.05 | 8 / 16 |
| wine_white | 4898 | 11 | 1.0 | 30 / 60 |
| ptbxl_* | 9483 | 128 | 1.0 | 10 / 20 |

## 5. Metric definitions

### PTB-XL age bands (plan section 1, non-overlapping)
`young` age < 30 · `middle` 30 ≤ age < 60 · `old` age ≥ 60.
Age exactly 60 falls in `old`. About 20 test records per DIR seed sit on that edge.

### PTB-XL metrics, years
- **per-band MAE** — mean absolute error within each band.
- **natural** — MAE over the test split as it occurs. Measured under `fixed`, whose
  954 test records carry PTB-XL's real age distribution (13% young, 55% middle, 32%
  old).
- **band_macro** — plain mean of the three band MAEs. An explicit balancing that does
  not depend on how any test set was resampled.
- **balanced** — MAE under `dir`, i.e. on the resampled near-uniform test set. This is
  the pre-existing definition, reported unchanged.
- **GM** — `exp(mean(log(|y_pred - y_true| + 1e-8)))`, per the plan.

`natural` and `balanced` are the same quantity measured on different test sets, so
the protocol column distinguishes them rather than a separate metric name.

### Tabular shot regions
`imbreg/metrics.py:shot_regions`, following Yang et al. (2021): bin the target, count
**training** rows per bin, then label each test row by its bin's training frequency —
many-shot > 100, few-shot < 20, medium-shot in between. Thresholds are
`MANY_SHOT_MIN = 100` and `FEW_SHOT_MAX = 20`, matching the plan's expectation.

### SERA
Also computed (`imbreg/metrics.py`), following Ribeiro & Moniz (2020), with the
relevance function fitted on **training** targets only. Not used in the paper's
tables; retained because neither convention is universally accepted.

## 6. Hyperparameter selection

Fine-tuning arms sweep learning rate over 1e-5, 3e-5, 1e-4. For each
(dataset, arm, seed) the run with the lowest **validation** RMSE is selected. The
learning rate lives at `config.lr` in the JSONL row, not at the top level. The test
split is read once, for final evaluation.
