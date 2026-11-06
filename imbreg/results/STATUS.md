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

**All 14 reproduce. Zero flags.** Largest absolute difference 0.0005.

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
