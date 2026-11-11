# Run status, hygiene checks and discrepancies

Deliverable 2 of `paper_completion_plan.md` section 7. Updated as runs land.

Last updated 6 Nov 2026, 17:40 IST.

---

## 1. Data-hygiene check — ONE FAILURE, and it is load-bearing

Plan rule 5 requires that fine-tuning use only train, model selection only
validation, and test be touched once. Checked for every pipeline.

### 1.1 PASS — learning-rate selection
Fine-tuning arms sweep 1e-5 / 3e-5 / 1e-4 and select on `val_rmse`, which
`imbreg/run.py` computes on the validation split. Test is scored once per cell. The
selection code reads `config.lr` from the JSONL, never a test metric.

### 1.2 PASS — the tabular benchmarks
Abalone, Communities and Wine have no pre-trained encoder in front of them, and the
prior-level TabICL models were trained only on TabICL's synthetic prior. Nothing in
those pipelines has seen an evaluation row before test time.

### 1.3 FAIL — the `dir` protocol on PTB-XL re-uses encoder training data

`imbreg/data.py:load_embedding_npz` concatenates the official train, validation and
test splits into one 9483-row table. `dir_split` then re-partitions **all** of it to
build a balanced test set. But every encoder in this study (E1–E5) was fitted on the
official 7566-row train split. So the DIR test set is mostly records the encoder
already trained on, with their labels:

| seed | DIR test n | of which seen by the encoder |
|---|---|---|
| 0 | 1499 | 1207 (80.5%) |
| 1 | 1499 | 1229 (82.0%) |

Measured directly, on the DIR test set, splitting by whether a record was in the
encoder's training split:

| cell | seen by encoder | unseen | gap |
|---|---|---|---|
| E1 + stock | 7.207 | 7.862 | **−0.656** |
| E1 + Gaussian | 7.033 | 7.717 | **−0.684** |
| E3 + stock | 7.012 | 7.748 | **−0.736** |
| E3 + Gaussian | 6.902 | 7.639 | **−0.736** |

The contamination is worth 0.66–0.74 years. The effect the paper is measuring
(Gaussian vs its matched control on PTB-XL) is 0.095 years. **The leak is about
seven times the size of the signal.**

It does not invalidate the paired comparisons — all arms share the same contaminated
test set, so the *ranking* of arms is still meaningful — but every absolute PTB-XL
number under `dir` is optimistic, and any comparison against a model trained on a
different split is not sound.

**Action taken.** Per the plan's stop rule this was reported before further runs.
All PTB-XL results for the paper are being regenerated under the `fixed` protocol,
whose test set is the official 954 records, seen by nothing. The `dir` numbers are
retained and reported as the "balanced" column in Table 3 only, beside the clean
"natural" column, so the gap between them is visible rather than hidden.

### 1.4 NOTE — `fixed` seeds do not vary the split
Under `fixed` the partition is the official one and identical for every seed; seeds
vary model and preprocessing randomness only. Reported standard deviations therefore
capture run-to-run variation, not split-to-split variation. This is stated rather
than worked around, because the official split is the only uncontaminated one
available.

---

## 2. Reference-value cross-check

Plan section 4 lists previously reported values for sanity checking, with
instructions to flag any rerun differing by more than 0.1 MAE. All 14 were
recomputed through the new `eval/compute_metrics.py` pipeline, from the saved
prediction files.

**All reproduce. Zero flags.** Largest absolute difference 0.0005. (Recounted on
11 Nov 2026: the plan lists **18** reference values, not 14. All 18 reproduce; the
full table is in `RESULTS.md` section 4.)

This validates the new metric code against whatever produced the original numbers.

### 2.1 But Table 1a's reference values are from the wrong experiment

Plan section 4 heads Table 1a *"TabICL, synthetic-prior fine-tuning only (main
result)"* and offers these few-shot MAE values for it:

> Abalone 6.219 → 3.266 · Communities 0.2552 → 0.1924 · Wine 2.4527 → 2.3639

Those three "Gaussian" values reproduce exactly from the arm `ticl_ft_gauss`, which
is **per-dataset fine-tuning on the target data** — not the synthetic-prior run. The
actual prior-level arms give the opposite result:

| few-shot MAE | zero-shot | prior uniform (control) | prior **Gaussian** |
|---|---|---|---|
| Abalone | 6.219 | 6.189 | **6.385** |
| Communities | 0.2552 | 0.2549 | **0.2854** |
| Wine | 2.4527 | 2.4560 | **2.5175** |

Prior-level Gaussian context is **worse than its matched control on all three
tabular datasets**, in the few-shot region and overall. The paper's claim (b) —
that the effect holds using only the synthetic prior — is not supported outside
PTB-XL. Table 1a is being generated from the real prior-level numbers.

---

## 3. Run status

| task | what | status |
|---|---|---|
| — | `eval/compute_metrics.py`, unified protocol | **done** |
| — | `results/REPO_MAP.md` | **done** |
| B1 train | `target_balanced` prior, 3000 steps | **running** |
| C | PTB-XL clean (`fixed`) re-evaluation, 5 encoders × 4 bodies × 5 seeds | queued |
| A | ResNet vanilla ×5, RankSim+ConR ×5 | queued |
| B1 eval | `target_balanced` on 3 tabular + PTB-XL; abalone TabPFN zero-shot preds | queued |
| D | `eval/significance.py` | **code done**, awaits C and A |
| figs | `fig1_method.pdf`, `fig2_pred_vs_true.pdf` | **done** (fig 2 regenerates on clean data) |
| tables | `table1a/1b/2/3.tex` | **generating**, TODO cells until runs land |
| paper | `main.tex` compiles | **done**, 2 pages with placeholders |

Jobs are serialised through `gpu_lock.sh`. The workstation GPU is shared with
another user who was holding 72–85 GB of 98 GB throughout; our prior-training job
holds about 2.8 GB.

---

## 4. Code changes made for this work

| file | change | why |
|---|---|---|
| `tabicl_prior/gauss_prior.py` | added `target_balanced_row_order` | Task B control arm |
| `imbreg/run.py` | prediction files now carry `id` and `split`; registered `target_balanced` | a prediction could not previously be traced to its record, which is how 1.3 stayed hidden |
| `train.py` | added `--seed`; appends to `store_name` | there was no seed control at all, and the plan needs 5 |
| `train.py`, `resnet1d_wang_fds.py` | added `return_features` | RankSim and ConR only received features when FDS was on, which would have made every regularised run an FDS run too |

---

## 5. Known gaps, stated rather than hidden

- **E5** (encoder trained through the mixed-prior body) exists and was evaluated,
  but has no row in any planned table. Its results are in `all_metrics.csv`.
- **AgeDB** results exist for some arms and are incomplete for others. AgeDB is not
  part of this paper and was not rerun.
- **Task B2** (TabPFN per-dataset `target_balanced` on Abalone) is conditional on B1
  finishing early, per the plan's own stop rule.
- **The encoders are not converged.** E3, E4 and E5 all recorded their best
  validation score at the first evaluation (step 200) and degraded thereafter,
  early-stopping at 1800. They are effectively "warm start plus 200 steps". They
  differ from one another by up to 0.23 years so the training did real work, but a
  converged comparison would need a finer evaluation interval.
- **`target_balanced` is constrained by the prior's train-size range.** TabICL draws
  `train_size` at 79–81% of rows, so a without-replacement reweighting can only drop
  the most common ~20%. Measured on a skewed pool, it lifts the rare-target fraction
  from 0.076 to 0.095 where the Gaussian arm cuts it to 0.016. The arm is a faithful
  implementation of the plan's spec; its leverage is limited by the prior, and that
  limit should be stated if it underperforms.

---

## 6. Clean-split results — added 10 November 2026

The 100-cell clean matrix (5 encoders × 4 TabICL bodies × 5 seeds, official PTB-XL
split) is complete. The leak analysis in section 1.3 can now be checked against it
rather than inferred.

### 6.1 The leak prediction holds, at about half the naive magnitude

Comparing `band_macro` (mean of the three age-band MAEs, so the test set's age mix
cancels) between protocols, for identical encoder × arm × seed cells:

| | |
|---|---|
| cells where the clean split is worse | **20 of 20** |
| mean gap | **+0.278 years** |
| range | +0.192 to +0.371 |

Section 1.3 measured a *within*-`dir` gap of 0.66–0.74 years between records the
encoder had seen and records it had not. The between-protocol gap is smaller, and
should be: `dir`'s test set is only ~80% contaminated, and it is also harder within
each age band because the per-bin cap over-samples extreme ages. Those two effects
push in opposite directions. Both numbers are real; they are not the same quantity,
and neither should be quoted as "the size of the leak" without saying which it is.

### 6.2 The Gaussian effect shrinks by about two thirds once the leak is removed

Gaussian prior against its matched uniform control, `band_macro`, positive = Gaussian
better:

| encoder | `dir` (leaky) | `fixed` (clean) |
|---|---|---|
| E1 `ptbxl_l1_enc` | +0.158 | +0.051 |
| E2 `ptbxl_tabpfn_enc` | +0.145 | +0.040 |
| E3 `ptbxl_enc_e3` | +0.073 | +0.034 |
| E4 `ptbxl_enc_e4` | −0.077 | −0.033 |
| E5 `ptbxl_enc_e5` | −0.045 | +0.005 |
| **mean** | **+0.051** | **+0.019** |

+0.019 years is roughly one week of age. It is positive in 4 of 5 encoders, but it is
not a magnitude any claim should rest on.

### 6.3 What the effect actually is, on clean data

Paired bootstrap, 10k resamples, seed-averaged, E3 encoder, Gaussian-prior TabICL
against stock TabICL (`imbreg/results/metrics/significance.csv`):

| band | Δ MAE (years) | 95% CI | seeds won | significant |
|---|---|---|---|---|
| young (<30) | **−0.551** | [−0.699, −0.403] | 5/5 | yes |
| middle (30–60) | **+0.683** | [+0.586, +0.785] | 0/5 | yes |
| old (≥60) | **−0.272** | [−0.454, −0.087] | 5/5 | yes |
| overall, natural mix | **+0.218** | [+0.130, +0.308] | 0/5 | yes |

Negative = Gaussian better. Both tails improve significantly and the bulk degrades
significantly. Because the natural PTB-XL test set is 55% middle-aged, the net effect
on overall MAE is a significant **worsening** of 0.218 years.

This reconciles with 6.2 rather than contradicting it: `band_macro` weights the three
bands equally, so the tail gains nearly cancel the middle loss; overall MAE weights by
the natural distribution, where the middle band dominates. **Any claim about this
method must state which weighting it is using.** On a tail-weighted view it is
roughly neutral; on the natural distribution it is worse.

### 6.4 Still outstanding

`clean_tabpfn_lr3e-5` (6 cells) and `clean_tabpfn_lr1e-4` (10 cells) were still
running when the chain died on 7 November and have been restarted. Task A (ResNet
baselines) and Task B1 (`target_balanced` arm) have not run at all. Three of the four
significance comparisons are therefore still `MISSING`.

---

## 7. All runs complete — 10 November 2026

Tasks A, B1 and C have all finished. Every cell in all four tables is populated;
`grep TODO paper/tables/*.tex` returns nothing.

### 7.1 Three bugs that had to be fixed to get Task A to run

Task A had never executed successfully. All ten runs failed in about one second
each, twice, for different reasons:

1. **protobuf.** `train.py` imports `tensorboard_logger`, whose bundled `*_pb2.py`
   files were generated against protobuf 3.x. The installed version is 7.36.2, so the
   import dies with *"Descriptors cannot be created directly"*. Fixed with protobuf's
   documented workaround, `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, set inside
   `run_task_a.sh` — not by downgrading protobuf, which would risk breaking TabICL and
   TabPFN for every other experiment and for the other user of this workstation.
2. **GPU selection.** `train.py` wraps the model in `DataParallel` over every visible
   device. This machine has an RTX PRO 6000 (95 GB) *and* a T400 (3.6 GB), so every
   batch was split across both and the small card set the pace. Pinning to the large
   card took training from 0.20 s/batch to 0.00–0.01 s/batch, roughly twenty-fold.
   Note `nvidia-smi` numbers by PCI bus (T400 = 0) while torch defaults to
   fastest-first (RTX PRO = 0); `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set to disambiguate.
3. **Checkpoint path.** The script reconstructed the run directory name by hand and
   got it wrong twice: `train.py` appends `--store_name` to the model name, producing a
   *double* underscore, and for `--ranking`/`--conr` it also embeds every regulariser
   hyperparameter. All ten runs trained correctly and were then reported as
   "no checkpoint". The script now globs for the directory instead.

Because of (2), the baselines were retrained on one GPU rather than two. Batch-norm
statistics are computed differently under single-GPU, so these runs are not
bit-identical to the original reference run.

### 7.2 The baselines are sane

| arm | test L1 (years), mean ± sd over 5 seeds |
|---|---|
| ResNet vanilla | **7.477 ± 0.095** |
| ResNet + RankSim + ConR | 7.551 ± 0.155 |

The original reference run reported 7.5485. The new vanilla mean sits within one
standard deviation of it, which is the agreement that was being checked for.

**The imbalance methods do not help.** RankSim + ConR is *worse* than plain L1 on
overall error and worse in two of the three age bands. With 5 seeds and this spread
the difference is not significant, but there is certainly no gain to claim.

### 7.3 The target-balanced control: read it carefully

`significance.csv` reports comparison 4 as Gaussian *beating* `target_balanced` on
abalone (−0.432, significant) and communities (−0.034, significant). **On its own
this is misleading.** The three-way comparison on overall MAE under `dir`:

| dataset | uniform control | Gaussian | target-balanced |
|---|---|---|---|
| abalone | **2.943** | 3.000 | 3.432 |
| communities | **0.1598** | 0.1708 | 0.2047 |
| wine_white | **0.787** | 0.828 | 0.825 |

The ordering is the same on all three: **uniform < Gaussian < target-balanced**, i.e.
the unbiased control is best and the most aggressive reweighting is worst. Gaussian
beats `target_balanced` only because `target_balanced` is the worst arm of the three.
It does not beat the control it has to beat.

The honest statement for the tabular benchmarks is therefore: *any* non-uniform
reordering of the prior's context hurts, and it hurts more the further from uniform
it goes.

### 7.4 In-context stack versus the ResNet baseline

Best in-context cell (E3 + Gaussian-prior TabICL) against ResNet vanilla, paired
bootstrap, official split:

| band | Δ MAE (years) | 95% CI | significant |
|---|---|---|---|
| young (<30) | **−1.337** | [−1.899, −0.744] | yes |
| old (≥60) | **−0.561** | [−0.948, −0.167] | yes |
| middle (30–60) | **+1.369** | [+1.103, +1.632] | yes |
| overall, natural mix | **+0.401** | [+0.185, +0.616] | yes |

Negative = in-context better. The same shape as everywhere else in this study: large,
significant gains in both tails, a larger loss in the middle, and a net loss overall
because the middle dominates the natural distribution.

---

## 8. Task B2 — added 10 November 2026

The plan gates B2 on "only if B1 is done by day 5". B1 finished on 10 November,
day 4, so the gate was satisfied and B2 was run. It needed new code: the
dataset-level sampler had only `uniform`, `gauss` and `mixed`, so `target_balanced`
was added to `imbreg/biased_split.py` (50 equal-width bins, matching
`tabicl_prior/gauss_prior.py` so the two levels measure the same rule) and wired as
the `ft_target_balanced` arm.

The sampler was checked before use, on a deliberately skewed pool:

| mode | rare-target share of context |
|---|---|
| pool baseline | 0.0385 |
| `uniform` | 0.0382 |
| `target_balanced` | 0.0785 |
| `gauss` | 0.0000 |

It does what it claims: roughly doubles the rare-target share, where `uniform`
tracks the pool.

### 8.1 Result: the balancing hypothesis fails

Abalone few-shot MAE, learning rate selected per seed on validation:

| arm | few-shot MAE | overall MAE |
|---|---|---|
| uniform fine-tuning | 5.767 ± 0.217 | 2.786 ± 0.108 |
| **Gaussian fine-tuning** | **4.127 ± 0.151** | **2.202 ± 0.062** |
| target-balanced fine-tuning | 6.092 ± 0.148 | 2.929 ± 0.039 |

Gaussian beats uniform in 5 of 5 seeds and target-balanced in 5 of 5, with
non-overlapping per-seed ranges — about 11.7 pooled standard deviations apart.
Target-balanced beats uniform in 0 of 5.

This is the strongest evidence in the study for the dataset-level claim. The obvious
reviewer objection to the Gaussian's few-shot gain is that it merely over-samples
rare targets; an explicit inverse-density sampler would then reproduce it. It does
not. Concentrating the context on a drawn region is doing something that flattening
it across the range does not.

### 8.2 But read the target-balanced row with its mechanism attached

Its predictions are **byte-identical to zero-shot** for most seeds. That is not a
broken arm: validation RMSE never improved over the starting model, so early stopping
(patience 8) returned the un-fine-tuned model.

| | |
|---|---|
| (seed, lr) cells that reverted to the stock model | **14 of 15** |
| the exception | seed 3 at lr 1e-4 |

So the honest reading of 8.1 is: *target-balanced fine-tuning does not merely
underperform the Gaussian — on this dataset it fails to improve on not fine-tuning at
all, at every learning rate tried.* The table caption states this so the row is not
mistaken for a fine-tuned model that happens to match zero-shot.

What this does **not** license is "inverse-density context sampling cannot work". It
was tested at one context size, three learning rates, 60 epochs, on one dataset,
inside TabPFN's fine-tuning loop with early stopping. A wider sweep could find a
setting where it trains. The claim supported is the narrow one: under the identical
protocol that makes Gaussian sampling work, it does not.


---

## 9. Corrections found while compiling `RESULTS.md` — 11 November 2026

Recomputing every table from the prediction files for `RESULTS.md` surfaced four
issues. All are fixed in code; `paper/tables/*.tex`, `significance.csv`,
`paper/RESULTS_SUMMARY.md` and `paper/main.pdf` were regenerated.

### 9.1 `ft_target_balanced` was missing from `LR_SWEPT`

`eval/make_tables.py` resolves a learning-rate sweep by selecting the lowest
validation RMSE, for arms listed in `LR_SWEPT`. The Task B2 arm was not in that set,
so its Table 1b cell **averaged** its three learning rates while every other
fine-tuning row **selected** one. Two model-selection rules in one table.

Fixed. Abalone target-balanced row, few-shot 6.139 ± 0.063 -> **6.092 ± 0.148**,
overall 2.938 ± 0.030 -> **2.929 ± 0.039**. Section 8.1 above already quoted the
validation-selected values, so that section was right and the table was wrong.

No other row is affected: the remaining multi-tag arms are either listed in
`LR_SWEPT` or carry duplicate tags that are identical re-evaluations of one model.

### 9.2 `taskA_probe` was contaminating the ResNet vanilla baseline

`taskA_probe` is a single-seed re-evaluation of the ORIGINAL pre-existing checkpoint
(it reproduces 7.5485), kept for the reference-value check. It is not one of the five
Task A seeds, but it was being averaged into seed 0 of the reported baseline.

Fixed with an `EXCLUDE_TAGS` set. Table 2 row 1: natural 7.483 -> **7.477**,
band_macro 7.888 -> **7.854**, young 8.603 -> **8.533**, middle 6.726 -> **6.784**,
old 8.334 -> **8.244**, GM 4.869 -> **4.863**. The error made the baseline look worse
than it is, i.e. it biased in favour of our method. Significance comparison 3 now
uses the corrected baseline.

### 9.3 Significance comparison 2 was labelled with a reference it did not use

The row read "E2+gauss TabPFN vs E2+stock TabPFN" but compared against `ft_uniform`.
Both references are now computed and labelled separately (2a vs stock, the plan's
reference; 2b vs uniform fine-tuning, the matched control).

### 9.4 A zero-width bootstrap interval was reported as significant

`not (lo < 0 < hi)` marks `[0.000, 0.000]` as significant. Changed to
`not (lo <= 0 <= hi)`. No other row changes.

### 9.5 The finding that 9.3 exposed

Running the plan's own comparison 2a returns **delta = 0.0000, CI [0, 0]**:
**E2 + Gaussian TabPFN under the clean `fixed` protocol is byte-identical to the
zero-shot model in all 5 seeds** (md5 verified). Same early-stopping revert as the
Task B2 target-balanced arm in section 8.2. The matched control `ft_uniform` did
train (7.372 vs 7.421), so the loop works.

Consequence: **the dataset-level result does not transfer to PTB-XL under the clean
protocol.** Under `dir` it appears to (7.280 -> 6.872), which is one more reason the
`dir` numbers must not be quoted. Table 2 row 4 is kept, with the mechanism in its
caption, because deleting it would hide a negative result.
