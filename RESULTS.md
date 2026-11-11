# Results — Gaussian context sampling for imbalanced regression

Complete results record for `paper_completion_plan.md`.

**Provenance.** Every metric in sections 4–12 was recomputed for this document from
the saved prediction files in `imbreg/results/preds/` through
`eval/compute_metrics.py`; none is copied from an earlier report, a log summary or a
previous version of this file. Two tables are exceptions and are *diagnostic
measurements*, not metrics over the prediction files — the seen/unseen leak
breakdown in section 3 and the sampler verification in section 10.1. Both are quoted
from the recorded diagnostic runs in `imbreg/results/STATUS.md` and are marked where
they appear.

| | |
|---|---|
| Compiled | 11 November 2026 |
| Repository | <https://github.com/ansh14094/BTP-ECG-PFNs> |
| Prediction files | 1161 |
| Metric rows (`all_metrics.csv`) | 1161 |
| Seeds per reported cell | 5 (0–4) |
| `TODO` cells across the four tables | 2 (ResNet Balanced — see section 7) |
| Significance comparisons reporting `ok` | 19 of 19 |

**Dispersion convention.** All `mean ± sd` figures use the **population** standard
deviation (`ddof = 0`) over the 5 seeds, matching `eval/make_tables.py`. Some earlier
notes in `imbreg/results/STATUS.md` quote the sample standard deviation (`ddof = 1`),
which is larger — e.g. the ResNet vanilla baseline is 7.477 ± 0.085 here and
7.477 ± 0.095 there. Same data, different convention.

**Read section 2 before quoting any PTB-XL number**, and section 12 before quoting
Table 2 row 4.

---

## Contents

1. [Status against the plan's deliverables](#1-status-against-the-plans-deliverables)
2. [Evaluation protocol and data](#2-evaluation-protocol-and-data)
3. [Data hygiene: the DIR protocol leaks](#3-data-hygiene-the-dir-protocol-leaks)
4. [Reference-value cross-check](#4-reference-value-cross-check)
5. [Table 1a — TabICL, synthetic-prior continued pre-training](#5-table-1a--tabicl-synthetic-prior-continued-pre-training)
6. [Table 1b — TabPFN-2.5, per-dataset fine-tuning](#6-table-1b--tabpfn-25-per-dataset-fine-tuning)
7. [Table 2 — PTB-XL age prediction](#7-table-2--ptb-xl-age-prediction)
8. [Table 3 — encoder against prediction head](#8-table-3--encoder-against-prediction-head)
9. [Task A — ResNet baselines](#9-task-a--resnet-baselines)
10. [Task B — target-balanced controls](#10-task-b--target-balanced-controls)
11. [Task C — clean-split re-evaluation and the size of the leak](#11-task-c--clean-split-re-evaluation-and-the-size-of-the-leak)
12. [Task D — significance tests](#12-task-d--significance-tests)
13. [Figures](#13-figures)
14. [Corrections applied while compiling this record](#14-corrections-applied-while-compiling-this-record)
15. [What the results support, and what they do not](#15-what-the-results-support-and-what-they-do-not)
16. [Limitations](#16-limitations)
17. [Reproduction](#17-reproduction)

---

## 1. Status against the plan's deliverables

Plan section 7 checklist, verified by file existence and content:

| # | Deliverable | Status |
|---|---|---|
| 1 | `results/REPO_MAP.md` — pipelines, metric definitions, splits | present |
| 2 | `results/STATUS.md` — run status, discrepancies, hygiene checks | present |
| 3 | `results/preds/*.csv` | 1161 files |
| 4 | `results/metrics/*.csv` | 15 files incl. `all_metrics.csv`, `significance.csv` |
| 5 | `paper/tables/table1a.tex` | present, 0 TODO |
| 6 | `paper/tables/table1b.tex` | present, 0 TODO |
| 7 | `paper/tables/table2.tex` | present, 2 TODO (ResNet Balanced, section 7) |
| 8 | `paper/tables/table3.tex` | present, 0 TODO |
| 9 | `paper/figs/fig1_method.pdf` | present, vector |
| 10 | `paper/figs/fig2_pred_vs_true.pdf` | present, vector |
| 11 | `paper/main.tex` compiling to ≤ 4 pages excl. references | compiles, **3 pages**, 0 overfull boxes |
| 12 | `paper/RESULTS_SUMMARY.md` | present, machine-generated |

Experiment tasks:

| Task | What | Status |
|---|---|---|
| A | ResNet vanilla ×5 seeds, RankSim+ConR ×5 seeds, unified protocol | complete |
| B1 | `target_balanced` prior, 3000 steps, 3 tabular datasets ×5 seeds | complete |
| B2 | TabPFN per-dataset `target_balanced` on Abalone ×5 seeds ×3 LRs | complete (gate satisfied: B1 landed on day 4) |
| C | 5 encoders × 4 TabICL bodies × 5 seeds, clean `fixed` split | complete (100 cells) |
| D | Paired bootstrap + per-seed win counts | complete (19 rows) |

**Nothing from the plan is outstanding.** The paper prose is not part of this record.

---

## 2. Evaluation protocol and data

### 2.1 Datasets and test-set sizes

| Dataset | n | Features | Protocol | Test n | Many / Medium / Few |
|---|---|---|---|---|---|
| Abalone | 4177 | 8 | `dir` | 382 | 160 / 120 / 102 |
| Communities | 1994 | 122 | `dir` | 300 | 96 / 112 / 92 |
| Wine Quality White | 4898 | 11 | `dir` | 319 | 180 / 120 / **19** |
| PTB-XL (embeddings) | 9483 | 128 | `fixed` | 954 | — |
| PTB-XL (embeddings) | 9483 | 128 | `dir` | 1499 | — |

> **Caveat.** The Wine few-shot column rests on **19 test rows**. Differences in that
> column are not resolvable at this sample size and should not carry an argument.

PTB-XL age bands, official test split (954 records):

| Band | Definition | n | Share |
|---|---|---|---|
| Young | age < 30 | 125 | 13.1% |
| Middle | 30 ≤ age < 60 | 526 | 55.1% |
| Old | age ≥ 60 | 303 | 31.8% |

Under `dir` the test set is 1499 records, 320 / 600 / 579 — near-uniform by
construction.

### 2.2 Metric definitions

Implemented once, in `eval/compute_metrics.py`, from saved predictions. No metric is
computed inside a training loop.

| Metric | Definition |
|---|---|
| Per-band MAE | mean absolute error within an age band |
| Natural MAE | MAE over the official test split as it occurs (55% middle-aged) |
| Band-macro MAE | plain mean of the three band MAEs; test-set age mix cancels |
| Balanced MAE | MAE under `dir`, i.e. on the resampled near-uniform test set. This is the **pre-existing** definition in the code, confirmed by reading `imbreg/data.py:dir_split`, not inferred from the numbers |
| GM | `exp(mean(log(abs(y_pred − y_true) + 1e-8)))` |
| Shot regions | bin the target, count **training** rows per bin; many > 100, few < 20, medium in between (Yang et al., 2021) |

`natural` and `balanced` are the same quantity on different test sets, so the
protocol column distinguishes them rather than a separate metric name.

**Band-macro and natural MAE can disagree in sign.** They do, for the headline
PTB-XL comparison (section 7.2). Any claim must state which weighting it uses.

### 2.3 Two intervention levels — not interchangeable

| Level | Code | Arms | What is trained |
|---|---|---|---|
| **Dataset-level** | `imbreg/biased_split.py` | `ft_uniform`, `ft_gauss`, `ft_mixed`, `ft_target_balanced` | TabPFN-2.5 fine-tuned on the target dataset |
| **Prior-level** | `tabicl_prior/` | `ticl_prior_uniform`, `ticl_prior_gauss`, `ticl_prior_mixed`, `ticl_prior_target_balanced` | TabICL given 3000 further pre-training steps on its **own synthetic prior**; it never sees an evaluation dataset |

The two produce opposite conclusions. Quoting a number from one as evidence for the
other is the single easiest mistake to make with this repository.

### 2.4 Context sampler

`μ = quantile(y, U(0.05, 0.95))`, `σ = std(y) · exp(U(log 0.15, log 1.5))`,
`w_i ∝ exp(−½((y_i − μ)/σ)²)`, selected by Gumbel-top-k in log space (Gaussian
weights underflow to zero several σ from the mean). `target_balanced` instead uses
`w_i ∝ 1 / count(bin(y_i))` over 50 equal-width bins, the same rule at both levels.

### 2.5 Model selection

Fine-tuning arms sweep learning rate over 1e-5, 3e-5, 1e-4. For each
(dataset, arm, seed) the run with the lowest **validation** RMSE is selected. Test is
read once, for final evaluation. Verified for every arm in section 14.1.

---

## 3. Data hygiene: the DIR protocol leaks

This is the plan's stop-rule check, and it failed. Reported before further runs.

`imbreg/data.py:load_embedding_npz` concatenates the official PTB-XL train (7566),
validation (963) and test (954) splits into one 9483-row table. `dir_split` then
re-partitions **all** of it. Every encoder (E1–E5) was fitted on the official 7566-row
train split, so the DIR test set is mostly records the encoder already trained on:

| Seed | DIR test n | Seen by the encoder |
|---|---|---|
| 0 | 1499 | 1207 (80.5%) |
| 1 | 1499 | 1229 (82.0%) |

Measured on the DIR test set, split by whether a record was in the encoder's training
split (diagnostic run, quoted from `imbreg/results/STATUS.md` §1.3 — not recomputed
here, since it needs the per-record split membership rather than the metrics):

| Cell | Seen | Unseen | Gap |
|---|---|---|---|
| E1 + stock | 7.207 | 7.862 | −0.656 |
| E1 + Gaussian | 7.033 | 7.717 | −0.684 |
| E3 + stock | 7.012 | 7.748 | −0.736 |
| E3 + Gaussian | 6.902 | 7.639 | −0.736 |

**Two different gaps, both real — do not conflate them:**

| Quantity | Value |
|---|---|
| *Within* `dir`, seen vs unseen records | 0.66 – 0.74 years |
| *Between* protocols, `fixed` minus `dir`, band-macro | **+0.278 years** (section 11) |

The between-protocol gap is smaller because `dir`'s test set is only ~80%
contaminated *and* is harder within each band (the per-bin cap over-samples extreme
ages). The two effects push in opposite directions.

**Action.** All PTB-XL results for the paper are reported under the `fixed`
protocol — the official 954-record test set, seen by no encoder. The `dir` numbers
are retained and shown beside the clean ones in Table 3, so the gap is visible rather
than hidden. The tabular benchmarks are unaffected: they have no pre-trained encoder
in front of them (section 14.2).

**A note on `fixed` seeds.** Under `fixed` the partition is the official one and is
identical for every seed; seeds vary model and preprocessing randomness only.
Reported standard deviations therefore capture run-to-run variation, **not**
split-to-split variation. This is stated rather than worked around, because the
official split is the only uncontaminated one available.

---

## 4. Reference-value cross-check

Plan section 4 supplies previously reported values for sanity checking, with
instructions to flag any rerun differing by more than 0.1 MAE. All 18 were recomputed
through the new pipeline from saved predictions.

| Check | Reference | Recomputed | Diff |
|---|---|---|---|
| Abalone TabICL few-shot, zero-shot | 6.2190 | 6.2190 | 0.0000 |
| Abalone TabICL few-shot, Gaussian (per-dataset FT) | 3.2660 | 3.2661 | 0.0001 |
| Communities TabICL few-shot, zero-shot | 0.2552 | 0.2552 | 0.0000 |
| Communities TabICL few-shot, Gaussian (per-dataset FT) | 0.1924 | 0.1924 | 0.0000 |
| Wine TabICL few-shot, zero-shot | 2.4527 | 2.4527 | 0.0000 |
| Wine TabICL few-shot, Gaussian (per-dataset FT) | 2.3639 | 2.3639 | 0.0000 |
| Abalone TabPFN overall, zero-shot | 2.9420 | 2.9423 | 0.0003 |
| Abalone TabPFN overall, Gaussian FT | 2.2020 | 2.2025 | 0.0005 |
| Communities TabPFN overall, zero-shot | 0.1610 | 0.1606 | 0.0004 |
| Communities TabPFN overall, Gaussian FT | 0.1530 | 0.1530 | 0.0000 |
| Wine TabPFN overall, zero-shot | 0.9260 | 0.9259 | 0.0001 |
| Wine TabPFN overall, Gaussian FT | 0.7830 | 0.7829 | 0.0001 |
| PTB-XL balanced, E2 + stock TabPFN | 7.2800 | 7.2804 | 0.0004 |
| PTB-XL balanced, E2 + Gaussian TabPFN | 6.8720 | 6.8723 | 0.0003 |
| PTB-XL balanced, E3 + stock TabICL | 7.1520 | 7.1518 | 0.0002 |
| PTB-XL balanced, E3 + Gaussian TabICL | 7.0420 | 7.0419 | 0.0001 |
| PTB-XL balanced, E1 + stock TabICL | 7.3300 | 7.3304 | 0.0004 |
| PTB-XL balanced, E1 + Gaussian TabICL | 7.1620 | 7.1619 | 0.0001 |

**18 of 18 reproduce. Zero flags. Largest absolute difference 0.0005.** This
validates the new metric code against whatever produced the original numbers.

### 4.1 But three of the reference values describe the wrong experiment

Plan section 4 heads Table 1a *"TabICL, synthetic-prior fine-tuning only (main
result)"* and offers these few-shot values for it: Abalone 6.219 → 3.266,
Communities 0.2552 → 0.1924, Wine 2.4527 → 2.3639.

Those three "Gaussian" values reproduce exactly from arm `ticl_ft_gauss`, which is
**per-dataset fine-tuning on the target data** — not the synthetic-prior run. The
actual prior-level arms give the opposite result (section 5). Table 1a is generated
from the real prior-level numbers; the reference values are not comparable to it.

---

## 5. Table 1a — TabICL, synthetic-prior continued pre-training

`dir` protocol, 5 seeds, mean ± sd. Lower is better. Bold = best in column within
each dataset block. **`uniform context` is the matched control**: same 3000 extra
steps, unbiased row order. It isolates "the ordering helped" from "more steps
helped".

### Abalone

| Arm | Many | Medium | Few | Overall | GM |
|---|---|---|---|---|---|
| zero-shot | 1.146 ± 0.081 | 2.595 ± 0.031 | 6.219 ± 0.030 | 2.956 ± 0.038 | 1.649 ± 0.085 |
| uniform context | 1.148 ± 0.082 | **2.578 ± 0.031** | **6.189 ± 0.031** | **2.943 ± 0.039** | **1.645 ± 0.077** |
| target-balanced context | 1.163 ± 0.054 | 3.226 ± 0.027 | 7.233 ± 0.032 | 3.432 ± 0.023 | 1.989 ± 0.064 |
| Gaussian context | **1.128 ± 0.087** | 2.618 ± 0.024 | 6.385 ± 0.037 | 3.000 ± 0.028 | 1.665 ± 0.075 |

### Communities

| Arm | Many | Medium | Few | Overall | GM |
|---|---|---|---|---|---|
| zero-shot | 0.0587 ± 0.0013 | **0.1680 ± 0.0054** | 0.2552 ± 0.0083 | 0.1598 ± 0.0040 | 0.0898 ± 0.0068 |
| uniform context | 0.0585 ± 0.0014 | 0.1683 ± 0.0055 | **0.2549 ± 0.0082** | **0.1598 ± 0.0040** | **0.0896 ± 0.0072** |
| target-balanced context | **0.0514 ± 0.0023** | 0.2126 ± 0.0059 | 0.3552 ± 0.0024 | 0.2047 ± 0.0020 | 0.1135 ± 0.0025 |
| Gaussian context | 0.0577 ± 0.0015 | 0.1737 ± 0.0066 | 0.2854 ± 0.0037 | 0.1708 ± 0.0031 | 0.0937 ± 0.0042 |

### Wine Quality White

| Arm | Many | Medium | Few | Overall | GM |
|---|---|---|---|---|---|
| zero-shot | **0.335 ± 0.008** | **1.195 ± 0.060** | **2.453 ± 0.024** | **0.785 ± 0.020** | **0.256 ± 0.008** |
| uniform context | 0.337 ± 0.009 | 1.199 ± 0.059 | 2.456 ± 0.025 | 0.787 ± 0.019 | 0.267 ± 0.009 |
| target-balanced context | 0.345 ± 0.009 | 1.275 ± 0.055 | 2.535 ± 0.019 | 0.825 ± 0.019 | 0.285 ± 0.013 |
| Gaussian context | 0.349 ± 0.011 | 1.280 ± 0.051 | 2.517 ± 0.022 | 0.828 ± 0.018 | 0.353 ± 0.010 |

### 5.1 Reading

**This is a negative result and must be reported as one.**

- Gaussian context improves few-shot MAE against its matched control on **0 of 3**
  datasets (6.189→6.385, 0.2549→0.2854, 2.456→2.517).
- Gaussian context improves overall MAE against its matched control on **0 of 3**.
- The three-way ordering is the same on all three datasets:
  **uniform < Gaussian < target-balanced** on overall MAE
  (Abalone 2.943 / 3.000 / 3.432; Communities 0.1598 / 0.1708 / 0.2047;
  Wine 0.787 / 0.828 / 0.825 — Wine is the one inversion, within noise).
- Gaussian beats target-balanced (section 12, comparison 4) **only because
  target-balanced is the worst of the three**. It does not beat the control it has to
  beat. Comparison 4 must never be quoted without this table beside it.
- Wine GM degrades disproportionately: 0.256 → 0.353, a 38% relative increase, much
  larger than the 5% increase in overall MAE. GM weights small errors heavily, so the
  Gaussian arm is losing precision on easy, frequent-region rows.

The honest statement for the tabular benchmarks: **any non-uniform reordering of the
prior's context hurts, and it hurts more the further from uniform it goes.**

---

## 6. Table 1b — TabPFN-2.5, per-dataset fine-tuning

`dir` protocol, 5 seeds, learning rate selected per seed on validation RMSE.

### Abalone

| Arm | Many | Medium | Few | Overall |
|---|---|---|---|---|
| zero-shot | **1.170 ± 0.077** | 2.569 ± 0.028 | 6.163 ± 0.040 | 2.942 ± 0.029 |
| uniform fine-tuning | 1.208 ± 0.086 | 2.357 ± 0.105 | 5.767 ± 0.217 | 2.786 ± 0.108 |
| **Gaussian fine-tuning** | 1.584 ± 0.145 | **1.392 ± 0.094** | **4.127 ± 0.151** | **2.202 ± 0.062** |
| target-balanced fine-tuning | 1.188 ± 0.082 | 2.563 ± 0.026 | 6.092 ± 0.148 | 2.929 ± 0.039 |

### Communities

| Arm | Many | Medium | Few | Overall |
|---|---|---|---|---|
| zero-shot | **0.0592 ± 0.0015** | 0.1677 ± 0.0049 | 0.2577 ± 0.0053 | 0.1606 ± 0.0035 |
| uniform fine-tuning | 0.0597 ± 0.0016 | 0.1678 ± 0.0051 | 0.2538 ± 0.0062 | 0.1596 ± 0.0026 |
| **Gaussian fine-tuning** | 0.0646 ± 0.0031 | **0.1617 ± 0.0042** | **0.2348 ± 0.0012** | **0.1530 ± 0.0020** |

### Wine Quality White

| Arm | Many | Medium | Few | Overall |
|---|---|---|---|---|
| zero-shot | 0.427 ± 0.013 | 1.432 ± 0.022 | 2.455 ± 0.027 | 0.926 ± 0.011 |
| uniform fine-tuning | **0.370 ± 0.019** | 1.295 ± 0.023 | 2.375 ± 0.025 | 0.837 ± 0.011 |
| **Gaussian fine-tuning** | 0.391 ± 0.038 | **1.138 ± 0.055** | **2.250 ± 0.104** | **0.783 ± 0.031** |

### 6.1 Reading

**This is the paper's positive result.** Against the matched uniform control:

| Dataset | Few-shot: uniform → Gaussian | Overall: uniform → Gaussian |
|---|---|---|
| Abalone | 5.767 → **4.127** (−28.4%) | 2.786 → **2.202** (−21.0%) |
| Communities | 0.2538 → **0.2348** (−7.5%) | 0.1596 → **0.1530** (−4.1%) |
| Wine | 2.375 → **2.250** (−5.3%) | 0.837 → **0.783** (−6.5%) |

Gaussian fine-tuning improves few-shot MAE on **3 of 3** datasets and overall MAE on
**3 of 3**, against the control that received the identical budget.

**The trade-off is explicit and must be reported.** The many-shot column gets *worse*
on all three datasets (Abalone 1.208 → 1.584, +31%; Communities 0.0597 → 0.0646;
Wine 0.370 → 0.391). The method moves error from the tails into the bulk. On these
three datasets the net is favourable; that is not guaranteed in general, and on
PTB-XL it is not (section 7).

---

## 7. Table 2 — PTB-XL age prediction

MAE in years. 5 seeds. Every column is the clean `fixed` protocol — the official
954-record test split — **except Balanced**, which is the pre-existing definition:
MAE on the resampled near-uniform `dir` test set, and therefore leaky (section 3).

| Model | < 30 | 30–60 | ≥ 60 | Natural | Balanced | Band-macro | GM |
|---|---|---|---|---|---|---|---|
| ResNet (vanilla) | 8.533 ± 0.215 | **6.784 ± 0.169** | 8.244 ± 0.558 | 7.477 ± 0.085 | *TODO* | 7.854 ± 0.097 | 4.863 ± 0.081 |
| ResNet + RankSim + ConR | 8.399 ± 0.504 | 6.847 ± 0.174 | 8.425 ± 0.317 | 7.551 ± 0.138 | *TODO* | 7.890 ± 0.123 | 4.905 ± 0.110 |
| E2 + stock TabPFN-2.5 | 7.659 ± 0.013 | 7.246 ± 0.006 | 7.628 ± 0.010 | 7.421 ± 0.003 | 7.280 ± 0.084 | 7.511 ± 0.006 | 4.728 ± 0.015 |
| E2 + Gaussian TabPFN-2.5 *(see 7.3)* | 7.663 ± 0.017 | 7.244 ± 0.006 | 7.629 ± 0.010 | 7.421 ± 0.003 | **6.872 ± 0.097** | 7.512 ± 0.007 | 4.731 ± 0.014 |
| E3 + stock TabICL | 7.600 ± 0.020 | 7.152 ± 0.005 | 7.601 ± 0.006 | **7.353 ± 0.002** | 7.152 ± 0.090 | 7.451 ± 0.005 | **4.716 ± 0.038** |
| E3 + Gaussian TabICL | **7.059 ± 0.053** | 7.849 ± 0.061 | **7.342 ± 0.053** | 7.585 ± 0.017 | 7.042 ± 0.120 | **7.417 ± 0.017** | 4.903 ± 0.058 |

**The two Balanced cells marked `TODO` cannot be filled from saved predictions.** The
`dir` split is defined over the 9483-row embedding table, and the ResNet baselines
were only ever run on the official 954-record test split, so no ResNet predictions
exist for the remaining records. Filling them needs a fresh inference pass over all
9483 records with the five Task A checkpoints — and the result would be leaky in
exactly the way section 3 describes, since the ResNet trained on 7566 of them.

**The best Balanced value, 6.872, is the row that section 7.3 shows is the
un-fine-tuned model on the clean split.** The Balanced column is the only place in
this table where Gaussian TabPFN appears to beat stock, and it is the leaky one.

### 7.1 The in-context stack beats the convolutional baseline at the tails only

Against ResNet vanilla, E3 + Gaussian TabICL is better by 1.47 years in the young
band and 0.90 years in the old band, and worse by 1.07 years in the middle band.
Significance in section 12, comparison 3: all four intervals exclude zero.

### 7.2 Band-macro and natural MAE disagree in sign — state which you mean

For E3, stock → Gaussian:

| Weighting | stock | Gaussian | Direction |
|---|---|---|---|
| Band-macro (bands weighted equally) | 7.451 | **7.417** | Gaussian **better** by 0.034 |
| Natural (55% middle-aged) | **7.353** | 7.585 | Gaussian **worse** by 0.232 |

Both are correct. The Gaussian arm improves both tails and degrades the middle; the
middle dominates PTB-XL's natural age distribution. **On a tail-weighted view the
method is roughly neutral; on the natural distribution it is significantly worse.**
A claim that does not name its weighting is not checkable.

### 7.3 Row 4 is not a fine-tuned model

**E2 + Gaussian TabPFN-2.5 under the clean `fixed` protocol is byte-identical to the
zero-shot model in all 5 seeds** (md5 match on every prediction file; Δ MAE exactly
0.0000, section 12 comparison 2a).

Mechanism, not a code bug: `arm_finetune` runs with `early_stopping=True,
patience=8`. Validation RMSE never improved over the starting model, so early
stopping returned the un-fine-tuned one. The matched control did train
(`ft_uniform` = 7.372, genuinely different predictions), so the loop works — Gaussian
fine-tuning specifically failed to find an improving step here.

Consequence: **the dataset-level result of section 6 does not transfer to PTB-XL
under the clean protocol.** Under the leaky `dir` protocol it appears to
(7.280 → 6.872, an apparent 0.41-year gain), which is one more reason not to quote
`dir`. The row stays in the table because removing it would hide a negative result,
but it must carry this note or it reads as a fine-tuned model that coincidentally
matches zero-shot to three decimals.

### 7.4 The imbalanced-regression baseline does not help

RankSim + ConR is worse than plain L1 on natural MAE (7.551 vs 7.477), on band-macro
(7.890 vs 7.854), on GM, and in two of the three bands. With 5 seeds and this spread
the difference is not significant — but there is no gain to claim.

---

## 8. Table 3 — encoder against prediction head

Natural MAE (official split) and Balanced MAE (DIR split), years, 5 seeds.
**The two column groups are not comparable to each other** — the DIR columns are
optimistic by roughly 0.28 years band-macro (section 11) because that split re-uses
encoder training data.

| Encoder | Natural, stock head | Natural, Gaussian head | Balanced, stock head | Balanced, Gaussian head |
|---|---|---|---|---|
| E1 (supervised L1) | 7.428 ± 0.008 | 7.654 ± 0.041 | 7.330 ± 0.067 | 7.162 ± 0.070 |
| E2 (through TabPFN) | 7.428 ± 0.004 | 7.648 ± 0.015 | 7.258 ± 0.090 | 7.108 ± 0.080 |
| E3 (through stock TabICL) | **7.353 ± 0.002** | **7.585 ± 0.017** | **7.152 ± 0.090** | **7.042 ± 0.120** |
| E4 (through Gaussian TabICL) | 7.469 ± 0.004 | 7.617 ± 0.007 | 7.384 ± 0.080 | 7.392 ± 0.108 |

**The sign of the Gaussian head's effect flips between the two protocols** — it helps
on DIR for E1/E2/E3 and hurts on the official split for all four. That flip is the
leak, not a property of the method.

E3 is the best encoder in all four columns. Training the encoder through a frozen
stock TabICL beats training it through a supervised L1 loss (E1), through TabPFN
(E2), or through the Gaussian-prior TabICL (E4). **E4 is worse than E3**: matching
the encoder's training body to the Gaussian prediction body does not help.

### 8.1 E5, omitted from the planned tables

E5 (encoder trained through the mixed-prior body) was run and evaluated but has no
row in any planned table. Clean-split values, for completeness:

| Encoder | stock | uniform | gauss | mixed |
|---|---|---|---|---|
| E5, natural MAE | 7.440 | 7.443 | 7.566 | 7.468 |
| E5, band-macro | 7.588 | 7.590 | 7.584 | 7.579 |

### 8.2 Full clean (`fixed`) matrix, natural MAE / band-macro

All 5 encoders × 4 bodies, for reference. Format `natural / band-macro`.

| Encoder | stock | uniform | gauss | mixed |
|---|---|---|---|---|
| E1 | 7.428 / 7.548 | 7.433 / 7.550 | 7.654 / 7.499 | 7.447 / 7.562 |
| E2 | 7.428 / 7.519 | 7.427 / 7.516 | 7.648 / 7.476 | 7.440 / 7.522 |
| E3 | 7.353 / 7.451 | 7.355 / 7.450 | 7.585 / 7.417 | 7.373 / 7.452 |
| E4 | 7.469 / 7.640 | 7.469 / 7.638 | 7.617 / 7.671 | 7.497 / 7.626 |
| E5 | 7.440 / 7.588 | 7.443 / 7.590 | 7.566 / 7.584 | 7.468 / 7.579 |

Note that `stock` and `uniform` agree to within 0.005 natural MAE in every row: 3000
extra steps on an unbiased prior change nothing. Whatever the Gaussian arm does, it
is the row ordering doing it, not the extra training.

---

## 9. Task A — ResNet baselines

Retrained from scratch, 5 seeds, on the identical official PTB-XL partition used by
every encoder (`data/1000_timesteps.csv`, `split` column — verified by record ID).

Per-seed natural MAE (years):

| Arm | s0 | s1 | s2 | s3 | s4 | Mean ± sd |
|---|---|---|---|---|---|---|
| vanilla | 7.493 | 7.479 | 7.342 | 7.461 | 7.610 | **7.477 ± 0.085** |
| RankSim + ConR | 7.416 | 7.797 | 7.578 | 7.425 | 7.542 | **7.551 ± 0.138** |

The original reference run reported 7.5485. The new vanilla mean sits within one
standard deviation of it — the agreement being checked for.

### 9.1 Three bugs had to be fixed before Task A ran at all

All ten runs had previously failed in about one second each, twice, for different
reasons:

1. **protobuf.** `train.py` imports `tensorboard_logger`, whose bundled `*_pb2.py`
   were generated against protobuf 3.x; the installed version is 7.36.2, so the
   import dies with *"Descriptors cannot be created directly"*. Fixed with protobuf's
   documented workaround, `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, set inside
   `run_task_a.sh` — deliberately **not** by downgrading protobuf, which would risk
   breaking TabICL and TabPFN for every other experiment and for the other user of
   this shared workstation.
2. **GPU selection.** `train.py` wraps the model in `DataParallel` over every visible
   device. This machine has an RTX PRO 6000 (95 GB) *and* a T400 (3.6 GB), so every
   batch was split across both and the small card set the pace. Pinning to the large
   card took training from 0.20 s/batch to 0.00–0.01 s/batch. `nvidia-smi` numbers by
   PCI bus (T400 = 0) while torch defaults to fastest-first (RTX PRO = 0), so
   `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set to disambiguate.
3. **Checkpoint path.** The script reconstructed the run directory name by hand and
   got it wrong twice: `train.py` appends `--store_name` to the model name, producing
   a *double* underscore, and for `--ranking`/`--conr` it also embeds every
   regulariser hyperparameter. All ten runs trained correctly and were then reported
   as "no checkpoint". The script now globs for the directory.

**Consequence of (2) to disclose:** the baselines were retrained on one GPU rather
than two. Batch-norm statistics are computed differently under single-GPU, so these
runs are not bit-identical to the original reference run.

### 9.2 A code fix that affects what the baseline measures

`train.py` and `resnet1d_wang_fds.py` gained a `return_features` path. RankSim and
ConR previously received features **only when FDS was enabled**, which would have
silently made every regularised run an FDS run too, confounding the comparison.

---

## 10. Task B — target-balanced controls

The purpose of both arms is to answer one reviewer objection: *"the Gaussian only
helps because it over-samples rare targets; an explicit inverse-density sampler would
do the same."*

### 10.1 Sampler verification, before use

Measured on a deliberately skewed pool (diagnostic run, quoted from
`imbreg/results/STATUS.md` §8):

| Mode | Rare-target share of context |
|---|---|
| pool baseline | 0.0385 |
| `uniform` | 0.0382 |
| `target_balanced` | **0.0785** |
| `gauss` | 0.0000 |

`target_balanced` roughly doubles the rare-target share where `uniform` tracks the
pool. It does what it claims. (`gauss` reads 0.0000 because on this pool the drawn
centre fell away from the rare region — the Gaussian concentrates, it does not
balance.)

### 10.2 B1 — prior level (TabICL, 3000 steps)

Results are in Table 1a (section 5). `target_balanced` is the **worst** of the four
arms on overall MAE on all three datasets, and on few-shot MAE on all three.

**Stated limitation on B1's leverage.** TabICL draws `train_size` at 79–81% of rows,
so a without-replacement reweighting can only drop the most common ~20%. Measured on
a skewed pool it lifts the rare-target fraction from 0.076 to 0.095, where the
Gaussian arm cuts it to 0.016. The arm is a faithful implementation of the plan's
spec; its leverage at this level is limited by the prior, and that limit belongs
beside the result.

### 10.3 B2 — dataset level (TabPFN, Abalone)

Gate satisfied: B1 finished on day 4. Required new code — `target_balanced` added to
`imbreg/biased_split.py` (50 equal-width bins, matching
`tabicl_prior/gauss_prior.py` so both levels measure the same rule) and wired as arm
`ft_target_balanced`.

Per-seed few-shot MAE, learning rate selected on validation:

| Arm | s0 | s1 | s2 | s3 | s4 | Mean ± sd | Overall MAE |
|---|---|---|---|---|---|---|---|
| zero-shot | 6.176 | 6.109 | 6.142 | 6.158 | 6.228 | 6.163 ± 0.040 | 2.942 ± 0.029 |
| uniform FT | 5.581 | 6.068 | 5.876 | 5.462 | 5.846 | 5.767 ± 0.217 | 2.786 ± 0.108 |
| **Gaussian FT** | 3.963 | 4.305 | 4.008 | 4.045 | 4.314 | **4.127 ± 0.151** | **2.202 ± 0.062** |
| target-balanced FT | 6.176 | 6.109 | 6.142 | 5.806 | 6.228 | 6.092 ± 0.148 | 2.929 ± 0.039 |

- Gaussian beats uniform in **5/5** seeds and target-balanced in **5/5**, with
  non-overlapping per-seed ranges (Gaussian max 4.314 < uniform min 5.462).
- Target-balanced beats uniform in **0/5**.

**The balancing hypothesis fails.** Concentrating the context on a drawn region does
something that flattening it across the range does not.

### 10.4 Read the target-balanced row with its mechanism attached

Its per-seed values above are **identical to zero-shot in 4 of 5 seeds**, and its
predictions are byte-identical to zero-shot there. Same mechanism as section 7.3:
validation RMSE never improved, so early stopping (patience 8) returned the
un-fine-tuned model.

| | |
|---|---|
| (seed, lr) cells that reverted to the stock model | **14 of 15** |
| the exception | seed 3 at lr 1e-4 |

So the honest reading of 10.3 is stronger than it first looks: *target-balanced
fine-tuning does not merely underperform the Gaussian — on this dataset it fails to
improve on not fine-tuning at all, at every learning rate tried.*

**What this does not license:** "inverse-density context sampling cannot work". It
was tested at one context size, three learning rates, 60 epochs, on one dataset,
inside TabPFN's fine-tuning loop with early stopping. A wider sweep could find a
setting where it trains. The supported claim is the narrow one: *under the identical
protocol that makes Gaussian sampling work, it does not.*

---

## 11. Task C — clean-split re-evaluation and the size of the leak

The 100-cell clean matrix (5 encoders × 4 TabICL bodies × 5 seeds, official PTB-XL
split) is complete, so the leak predicted in section 3 can be measured rather than
inferred.

### 11.1 The prediction holds

Band-macro compared between protocols for identical encoder × arm cells (band-macro
so the test set's age mix cancels):

| | |
|---|---|
| Cells where the clean split is worse | **20 of 20** |
| Mean gap | **+0.278 years** |
| Range | +0.192 to +0.371 |

### 11.2 The Gaussian effect shrinks by about two thirds once the leak is removed

Gaussian prior against its matched uniform control, band-macro, positive = Gaussian
better:

| Encoder | `dir` (leaky) | `fixed` (clean) |
|---|---|---|
| E1 | +0.158 | +0.051 |
| E2 | +0.145 | +0.040 |
| E3 | +0.073 | +0.034 |
| E4 | −0.077 | −0.033 |
| E5 | −0.045 | +0.005 |
| **Mean** | **+0.051** | **+0.019** |

+0.019 years is roughly one week of age. It is positive in 4 of 5 encoders, but **it
is not a magnitude any claim should rest on**, and on the natural weighting the same
comparison is negative (section 7.2).

---

## 12. Task D — significance tests

`eval/significance.py`. Paired bootstrap, 10,000 resamples, 95% CI of
Δ MAE = MAE(method) − MAE(reference). **Negative = method better.** Plus per-seed win
counts.

Two bootstrap variants, because the protocols differ and they are not the same test:

- Under `fixed`, the test split is the official partition and identical across seeds,
  so predictions are averaged per record first, then bootstrapped — as the plan asks.
- Under `dir`, every seed draws a different test set, so there is no common row set;
  the bootstrap runs within each seed and the interval endpoints are averaged.

The variant used is printed with every row.

| Comparison | Band | Δ MAE | 95% CI | Sig. | Wins |
|---|---|---|---|---|---|
| **1.** E3 + Gaussian TabICL vs E3 + stock | overall | **+0.218** | [+0.130, +0.308] | yes | 0/5 |
| | young | **−0.551** | [−0.699, −0.403] | yes | 5/5 |
| | middle | **+0.683** | [+0.586, +0.785] | yes | 0/5 |
| | old | **−0.272** | [−0.454, −0.087] | yes | 5/5 |
| **2a.** E2 + Gaussian TabPFN vs E2 + **stock** *(plan reference)* | overall | **0.000** | [0.000, 0.000] | no | 0/5 |
| | young | 0.000 | [0.000, 0.000] | no | 0/5 |
| | middle | 0.000 | [0.000, 0.000] | no | 0/5 |
| | old | 0.000 | [0.000, 0.000] | no | 0/5 |
| **2b.** E2 + Gaussian TabPFN vs E2 + **uniform FT** *(matched control)* | overall | **+0.057** | [+0.026, +0.087] | yes | 0/5 |
| | young | −0.069 | [−0.141, +0.003] | no | 3/5 |
| | middle | **+0.115** | [+0.080, +0.152] | yes | 0/5 |
| | old | +0.008 | [−0.058, +0.074] | no | 2/5 |
| **3.** Best ICL (E3 + Gaussian TabICL) vs ResNet vanilla | overall | **+0.401** | [+0.184, +0.620] | yes | 1/5 |
| | young | **−1.337** | [−1.890, −0.744] | yes | 5/5 |
| | middle | **+1.369** | [+1.106, +1.633] | yes | 0/5 |
| | old | **−0.561** | [−0.953, −0.174] | yes | 5/5 |
| **4.** Abalone: prior Gaussian vs target-balanced | overall | **−0.432** | [−0.487, −0.376] | yes | 5/5 |
| **4.** Communities: prior Gaussian vs target-balanced | overall | **−0.034** | [−0.039, −0.028] | yes | 5/5 |
| **4.** Wine: prior Gaussian vs target-balanced | overall | +0.003 | [−0.007, +0.013] | no | 0/5 |

All 19 rows report `ok`; none is `MISSING`.

**Reading notes.**

- **Comparison 2a's zero is the result**, not a missing value: the Gaussian-fine-tuned
  TabPFN *is* the stock model under the clean split (section 7.3). The interval has
  zero width because the predictions are byte-identical.
- **Comparison 4 is comparing two losing arms.** Both are worse than the uniform
  control (section 5.1). Gaussian "wins" because target-balanced is worse still.
- **Comparison 1 and 3 share one shape**, which is the finding of the PTB-XL half of
  this study: large significant gains in both tails, a larger loss in the middle, and
  a net loss overall because the middle dominates the natural distribution.
- The plan specifies comparison 2's reference as stock TabPFN; the matched control is
  the more informative reference. Both are reported rather than one silently
  substituted for the other (section 14.3).

---

## 13. Figures

| File | Content |
|---|---|
| `paper/figs/fig1_method.pdf` | Method schematic. Left: target histogram with 2–3 Gaussian context-sampling curves at different drawn mean/variance. Right: queries spread uniformly over the target range. No data claims. |
| `paper/figs/fig2_pred_vs_true.pdf` | Prediction vs truth, two panels: Abalone (TabICL zero-shot vs Gaussian) and PTB-XL (E3 + stock vs E3 + Gaussian TabICL). Binned means of `y_pred` against `y_true`, ±1 sd shading, `y = x` line, so reduced pull toward the mean in the tails is visible. Seed-averaged; stated in the caption. |

Both are vector PDF, single-column width (3.3 in), 8 pt, colour-blind-safe palette.
Fig. 2's PTB-XL panel is drawn from the clean `fixed` split.

---

## 14. Corrections applied while compiling this record

Four issues were found while recomputing these numbers. All four are fixed in code
and the tables, significance file and PDF were regenerated. They are recorded here
because they changed published numbers.

### 14.1 Inconsistent model selection inside Table 1b

`eval/make_tables.py:LR_SWEPT` listed the arms whose learning-rate sweep must be
resolved by selecting the best validation RMSE. `ft_target_balanced` (added for Task
B2) was **missing from that set**, so its cell silently *averaged* the three learning
rates while every other fine-tuning row in the same table *selected* one. Two
different model-selection rules in one table is not a fair comparison.

Fixed by adding the arm to `LR_SWEPT`. Effect on the Abalone target-balanced row:

| Metric | Was (LR-averaged) | Now (validation-selected) |
|---|---|---|
| Many | 1.176 ± 0.077 | 1.188 ± 0.082 |
| Medium | 2.567 ± 0.027 | 2.563 ± 0.026 |
| Few | 6.139 ± 0.063 | **6.092 ± 0.148** |
| Overall | 2.938 ± 0.030 | **2.929 ± 0.039** |

The conclusion is unchanged — target-balanced still loses to uniform and to Gaussian
— but the published figure was wrong.

A sweep for the same fault found no other affected row: the remaining multi-tag arms
are either genuinely swept and listed (`ft_gauss`, `ft_uniform`, `ft_mixed`,
`ticl_ft_gauss`, `ticl_ft_uniform`) or carry duplicate tags that are **identical
re-evaluations** of one model, where averaging is a no-op (verified to 4 decimals for
`ptbxl_l1_enc` and `ptbxl_tabpfn_enc` under `dir`, tags `dir_matrix` and
`dir_ticl_base`).

### 14.2 A diagnostic run was contaminating the ResNet baseline

Tag `taskA_probe` is a **single-seed** re-evaluation of the *original, pre-existing*
ResNet checkpoint, kept for the reference-value sanity check — it reproduces 7.5485
exactly. It is not one of the five Task A seeds. Because the table generator grouped
by (dataset, protocol, arm, seed) without excluding it, it was being averaged into
seed 0 of the reported 5-seed baseline.

Fixed with an explicit `EXCLUDE_TAGS` set. Effect on Table 2, row 1:

| Metric | Was | Now |
|---|---|---|
| < 30 | 8.603 ± 0.312 | **8.533 ± 0.215** |
| 30–60 | 6.726 ± 0.177 | **6.784 ± 0.169** |
| ≥ 60 | 8.334 ± 0.553 | **8.244 ± 0.558** |
| Natural | 7.483 ± 0.087 | **7.477 ± 0.085** |
| Band-macro | 7.888 ± 0.129 | **7.854 ± 0.097** |
| GM | 4.869 ± 0.080 | **4.863 ± 0.081** |

This made the baseline look slightly worse than it is, i.e. it biased *in favour of*
the paper's method. The corrected baseline is used in significance comparison 3.

### 14.3 A significance row was labelled with a reference it did not use

The row reading *"2. E2+gauss TabPFN vs E2+stock TabPFN"* in fact compared against
`ft_uniform`, the matched control — not stock. The plan asks for stock. Rather than
change one for the other, both are now computed and each is labelled with the
reference it actually uses (2a and 2b in section 12). Running the plan's version is
what exposed section 7.3: Δ is exactly zero because the arm never trained.

### 14.4 A degenerate confidence interval was reported as significant

Significance was computed as `not (lo < 0 < hi)`. For comparison 2a the bootstrap
interval is `[0.000, 0.000]` — zero width, because the two arms' predictions are
identical — and a strict inequality marked it **significant**. Changed to
`not (lo <= 0 <= hi)`, so an interval whose endpoint touches zero reads as not
significant. This is the conservative reading and it changes no other row.

### 14.5 Not a bug, but it changes how a row must be read

E2 + Gaussian TabPFN-2.5 on PTB-XL under the clean split is the un-fine-tuned model
(section 7.3), as is the Abalone target-balanced arm in 14 of 15 cells
(section 10.4). Both are early-stopping reverting to the starting checkpoint. The
rows are kept — deleting them would hide negative results — and both now carry the
mechanism in their captions.

---

## 15. What the results support, and what they do not

The paper's one-line story is that Gaussian context sampling improves rare-target
regression, holding (a) per dataset for TabPFN-2.5, (b) using only TabICL's synthetic
prior, and (c) on PTB-XL ECG age prediction. Against the final numbers:

| Claim | Verdict | Evidence |
|---|---|---|
| **(a)** Per-dataset TabPFN fine-tuning | **Supported** | Few-shot MAE improves vs the matched uniform control on 3/3 datasets (−28.4%, −7.5%, −5.3%); overall MAE on 3/3. §6 |
| **(a′)** The gain is not just rare-target over-sampling | **Supported** | Explicit inverse-density sampler is beaten 5/5 seeds and fails to beat zero-shot at all. §10.3–10.4 |
| **(a″)** (a) transfers to PTB-XL | **Not supported** | Under the clean split the arm reverted to zero-shot; Δ = 0.000. The apparent 0.41-year gain exists only on the leaky protocol. §7.3 |
| **(b)** Prior-level, synthetic prior only | **Contradicted on tabular** | Worse than the matched control on 3/3 datasets, few-shot and overall. §5.1 |
| **(b)** Prior-level, on PTB-XL | **Too small to claim** | Band-macro +0.019 years (≈ one week) vs control, positive in 4/5 encoders; the same comparison is significantly *negative* on natural MAE. §11.2, §7.2 |
| **(c)** PTB-XL ECG age prediction | **Mixed, and must be stated as a trade-off** | Tails improve significantly (young −0.551, old −0.272); middle degrades more (+0.683); net natural MAE significantly worse (+0.218). §12 |
| In-context stack beats the ResNet baseline | **Only at the tails** | Young −1.337, old −0.561, middle +1.369, overall +0.401 — all significant. §12 |
| Imbalanced-regression baselines help | **No** | RankSim + ConR is worse than plain L1 overall and in 2/3 bands. §7.4 |

### 15.1 The strongest reviewer objections, stated unprompted

1. **"Your headline claim (b) is contradicted by your own Table 1a."** The
   prior-level arm loses to its matched control on every tabular dataset. The paper
   cannot claim the effect holds from the synthetic prior alone. The defensible
   version is narrower: *dataset-level* Gaussian context sampling works; the
   prior-level version does not transfer to these benchmarks.
2. **"Your PTB-XL gain depends on which average you take."** Band-macro says +0.034
   for E3, natural MAE says −0.232, and both come from the same predictions. Pick one,
   justify it, and report the other.
3. **"Two of your table rows are not the models their labels claim."** Sections 7.3
   and 10.4. Disclosed, but a reviewer will find them.
4. **"Your encoders are not converged."** E3/E4/E5 all peaked at the first evaluation
   (step 200) and early-stopped at 1800 — effectively "warm start plus 200 steps". The
   encoder comparison in Table 3 is therefore a comparison of barely-trained encoders.
5. **"The Wine few-shot column has 19 test rows."** No difference in that column is
   resolvable.
6. **"The effect size on PTB-XL is one week of age."** +0.019 years band-macro. Even
   taking the favourable weighting, this is not a result to build a contribution on.

The honest framing the numbers do support: **the dataset-level result is real,
sizeable and survives its strongest control; the prior-level and ECG results do not
replicate it, and that asymmetry is itself the finding.**

---

## 16. Limitations

Stated rather than hidden. Carried from `imbreg/results/STATUS.md` §5 and extended.

1. **The DIR protocol leaks** (§3). All PTB-XL headline numbers use the clean
   official split; DIR numbers appear only beside them, labelled.
2. **`fixed` seeds do not vary the split**, so reported sd is run-to-run variation
   only, not split-to-split (§3).
3. **The encoders are not converged.** E3, E4 and E5 recorded their best validation
   score at the first evaluation (step 200) and degraded thereafter, early-stopping at
   1800. They differ from one another by up to 0.23 years, so training did real work,
   but a converged comparison needs a finer evaluation interval.
4. **Two arms silently reverted to their starting checkpoint** via early stopping
   (§7.3, §10.4). A wider learning-rate or patience sweep might train them.
5. **`target_balanced` leverage is capped at the prior level.** TabICL draws
   `train_size` at 79–81% of rows, so a without-replacement reweighting can only drop
   the most common ~20% (§10.2).
6. **Wine few-shot rests on 19 test rows** (§2.1).
7. **Single-GPU retraining** changes batch-norm statistics, so Task A runs are not
   bit-identical to the original reference run (§9.1).
8. **E5 has no row in any planned table**; its numbers are in §8.1 and
   `all_metrics.csv`.
9. **AgeDB is incomplete and out of scope.** Results exist for some arms; it was not
   rerun and is not part of this paper.
10. **One dataset for B2.** The dataset-level target-balanced control was run on
    Abalone only, per the plan's own specification.

---

## 17. Reproduction

```bash
# environment
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # tensorboard_logger vs protobuf 7.x
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1                          # the RTX PRO 6000, by PCI order

# the GPU is shared: every job must take the mutex
./gpu_lock.sh <command>

# full chain, detached so it survives a disconnect
setsid nohup ./run_all.sh >> run_all.log 2>&1 < /dev/null &
```

Regenerating everything downstream of the saved predictions needs no GPU:

```bash
python3 eval/compute_metrics.py     # preds/*.csv       -> metrics/*.csv
python3 eval/significance.py        # preds/*.csv       -> metrics/significance.csv
python3 eval/make_tables.py         # all_metrics.csv   -> paper/tables/*.tex
python3 eval/make_figures.py        # preds/*.csv       -> paper/figs/*.pdf
python3 eval/make_summary.py        # all_metrics.csv   -> paper/RESULTS_SUMMARY.md
cd paper && pdflatex main.tex && pdflatex main.tex
```

### Where things live

| Path | Content |
|---|---|
| `CODE_MAP.md` | whole-repository guide: data flow, every file's role, arm table, traps |
| `README.md` | findings first, with the leak caveat up front |
| `HANDOFF.md` | state of work, stage sentinels, how to resume |
| `imbreg/results/REPO_MAP.md` | pipelines, metric definitions, splits |
| `imbreg/results/STATUS.md` | per-run status, hygiene checks, discrepancies |
| `imbreg/results/preds/*.csv` | every prediction, `id, y_true, y_pred, split` |
| `imbreg/results/metrics/all_metrics.csv` | 1161 rows, one per prediction file |
| `imbreg/results/metrics/significance.csv` | Task D output |
| `paper/tables/*.tex`, `paper/figs/*.pdf` | generated; never hand-edited |

### Naming

Prediction files are `<dataset>_<tag>_<protocol>_<arm>_s<seed>.csv`. The `tag`
separates runs of the same arm (learning rate, evaluation sweep); `protocol` is
`fixed` or `dir`; `arm` names the intervention. `ticl_prior_*` is prior-level,
`ft_*` is dataset-level — section 2.3.
