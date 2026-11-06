# Task: Finish experiments, tables and figures for a 4-page student workshop paper

**Deadline:** 20 Nov 2026. Target: all results final by 14 Nov, writing done by 19 Nov.

**Paper story (one line):** Sampling the *context* from a Gaussian over the target while fine-tuning a tabular in-context learning (ICL) model improves rare-target regression. This holds (a) per dataset for TabPFN-2.5, (b) using only the synthetic prior for TabICL v2, and (c) on PTB-XL ECG age prediction.

---

## 0. Ground rules (read first)

1. **Never invent or estimate a number.** Every value in a table must come from a saved prediction file or log produced by code in this repo. If a run fails or is missing, write `TODO` in the table and record why in `results/STATUS.md`.
2. **Discover before changing.** First map the repo: training scripts, the context-sampling code, evaluation code, data splits and checkpoints. Write a short summary to `results/REPO_MAP.md`. Do not refactor working code; add new options behind flags.
3. **Save raw predictions** for every run as `results/preds/<run_id>_seed<k>.csv` with columns `id, y_true, y_pred, split`. All metrics are computed from these files by one script (`eval/compute_metrics.py`), never inside training loops.
4. **Seeds:** 5 seeds (0–4) for every row that appears in the paper.
5. **Data hygiene:** fine-tuning uses only the training split; learning rate and early stopping use only the validation split; the test split is touched once, for final evaluation. Verify this for every existing pipeline and report it in `results/STATUS.md` (especially the per-dataset TabPFN fine-tuning, Exp 3).
6. **Do not change an existing metric definition silently.** If the existing "balanced evaluation" or many/medium/few-shot definition differs from Section 1, keep the existing one, add the Section 1 version alongside it, and document both.
7. Ask the user (do not guess) if a required checkpoint, dataset or definition cannot be found.

---

## 1. Unified evaluation protocol (applies to every result)

**PTB-XL age bands (non-overlapping):**
- Young: age < 30
- Middle: 30 ≤ age < 60
- Old: age ≥ 60

**PTB-XL metrics, all in years:**
- Per-band MAE (mean absolute error) for the three bands.
- **Natural MAE:** MAE over all test samples, as they occur.
- **Band-macro MAE:** plain average of the three per-band MAEs.
- **Existing "balanced" MAE:** reproduce whatever the current code calls balanced evaluation, and document its exact definition in `REPO_MAP.md`. (Previously reported overall values fit band weights of roughly 0.23 / 0.39 / 0.39, so it is likely sample-weighted on a resampled test set. Confirm this from the code.)
- **GM (geometric-mean error):** `exp(mean(log(|y_pred - y_true| + 1e-8)))`.

**Tabular benchmarks (Abalone, Communities, Wine Quality White):**
- Many-, medium- and few-shot MAE, using the repo's existing region definition. Document that definition (expected: target bins with more than 100 training samples are many-shot, fewer than 20 are few-shot, the rest medium-shot, following Yang et al. 2021, "Delving into Deep Imbalanced Regression").
- Overall MAE and GM on the same test set the existing results used.

**Reporting:** mean ± standard deviation over 5 seeds, with 3 decimals for PTB-XL and Abalone/Wine, and 4 decimals for Communities.

---

## 2. Experiments to run

### Task A: Rerun ResNet baselines under the unified protocol (highest priority)
- Models: ResNet (`RESENT_WANG` label in old results) **Vanilla** and **RankSim + CONR**.
- Use exactly the same PTB-XL train/validation/test split as the TabPFN/TabICL experiments. Verify this by comparing record IDs, and log the check.
- Output: per-band, natural, band-macro, balanced and GM metrics, 5 seeds.
- Reason: old ResNet numbers used different band edges (< 30, > 60) and possibly a different test set, so they are not comparable.

### Task B: Target-balanced context control (new)
Add a context-sampling mode `target_balanced` next to the existing `uniform`, `gaussian` and `mixed` modes:
- Histogram the target values of the candidate context pool (use the same binning the code already uses for its Gaussian sampler if one exists; otherwise 50 equal-width bins).
- Sample context rows without replacement with probability proportional to 1 / (count of the row's bin).
- Keep everything else identical to the Gaussian run: context size, query sampling, number of steps, learning rate, optimizer.

Runs:
- **B1:** TabICL v2 continued training on its own synthetic prior with `target_balanced` context, using the same step count (3,000) and settings as the Gaussian-context run. Evaluate on Abalone, Communities and Wine (5 seeds).
- **B2 (only if B1 is done by day 5):** TabPFN-2.5 per-dataset fine-tuning on Abalone with `target_balanced` context.

### Task C: Seeds and missing metrics for existing TabICL results
Re-evaluate or rerun with 5 seeds, saving predictions:
- **Tabular:** TabICL zero-shot, uniform-context, Gaussian-context, target-balanced (from B1). Add Overall MAE and GM.
- **PTB-XL:** encoders E1, E2, E3, E4, each with stock and Gaussian-context TabICL prediction heads.
  - E1 = encoder trained with supervised L1 age loss.
  - E2 = encoder trained through frozen stock TabPFN-2.5.
  - E3 = encoder trained through frozen stock TabICL.
  - E4 = encoder trained through frozen Gaussian-context TabICL.
- **PTB-XL TabPFN:** E2 + stock TabPFN-2.5, and E2 + Gaussian-fine-tuned TabPFN-2.5, under the unified protocol (natural MAE and GM are currently missing).

If existing checkpoints are available, re-evaluate them instead of retraining, but confirm which seed each checkpoint corresponds to.

### Task D: Significance tests
Script `eval/significance.py`:
- **Paired bootstrap over test samples:** 10,000 resamples. Report the 95% confidence interval of ΔMAE = MAE(method) − MAE(reference), using seed-averaged predictions per sample.
- **Per-seed win count:** in how many of the 5 seeds the method beats the reference.

Comparisons:
1. E3 + Gaussian TabICL vs E3 + stock TabICL (overall and per band).
2. E2 + Gaussian TabPFN vs E2 + stock TabPFN (overall and per band).
3. Best ICL configuration vs ResNet Vanilla (overall and per band).
4. TabICL Gaussian vs TabICL target-balanced (few-shot region, each tabular dataset).

---

## 3. Figures

- **Fig. 1 (method schematic):** vector PDF made with matplotlib. Left panel: target histogram of the dataset with a Gaussian context-sampling curve overlaid (show 2–3 different mean/variance draws). Right panel: queries spread uniformly over the target range. No data claims in this figure.
- **Fig. 2 (prediction vs truth):** two panels: Abalone (TabICL zero-shot vs Gaussian) and PTB-XL (E3 + stock vs E3 + Gaussian TabICL). Use binned means of `y_pred` against `y_true` with ±1 standard deviation shading, plus the y = x line, so that reduced pull toward the mean in the tails is visible. Use seed 0, or average predictions over seeds; state which in the caption.
- Style: single-column width (3.3 in), 8 pt font, colour-blind-safe palette, save as `paper/figs/*.pdf`.

---

## 4. Tables to generate (LaTeX via `eval/make_tables.py`)

Read from `results/metrics/*.csv` and write to `paper/tables/*.tex`. Bold the best value per column within each block. Use `TODO` for missing cells.

**Table 1a — TabICL, synthetic-prior fine-tuning only (main result)**

Rows: zero-shot, uniform-context, target-balanced-context, Gaussian-context, for each of the 3 datasets.
Columns: Many, Medium, Few, Overall, GM.

**Table 1b — TabPFN-2.5 per-dataset fine-tuning**

Rows: zero-shot, Gaussian fine-tuning (and target-balanced for Abalone if B2 is done), for each of the 3 datasets.
Columns: Many, Medium, Few, Overall.

**Table 2 — PTB-XL age prediction (MAE in years)**

Rows:
1. ResNet Vanilla
2. ResNet + RankSim + CONR
3. E2 + stock TabPFN-2.5
4. E2 + Gaussian TabPFN-2.5 (fine-tuned on PTB-XL)
5. E3 + stock TabICL
6. E3 + Gaussian TabICL (synthetic prior only)

Columns: < 30, 30–60, ≥ 60, Natural, Balanced, GM.
Footnote: the bootstrap 95% confidence intervals from Task D for the key comparisons.

**Table 3 — Encoder training vs prediction head (PTB-XL, TabICL)**

Rows: E1, E2, E3, E4. Columns: stock head, Gaussian head (Natural MAE and Balanced MAE).

**Reference values** previously reported, for sanity checks only: never copy them into the tables. Flag any rerun that differs by more than 0.1 MAE in `STATUS.md`.

- Tabular, TabICL few-shot MAE: Abalone zero-shot 6.219 → Gaussian 3.266; Communities 0.2552 → 0.1924; Wine 2.4527 → 2.3639.
- Tabular, TabPFN overall MAE: Abalone 2.942 → 2.202; Communities 0.161 → 0.153; Wine 0.926 → 0.783.
- PTB-XL, balanced MAE:
  - E2 + stock TabPFN 7.280; E2 + Gaussian TabPFN 6.872.
  - E3 + stock TabICL 7.152; E3 + Gaussian TabICL 7.042.
  - E1 + stock TabICL 7.330; E1 + Gaussian TabICL 7.162.
  - E4 + stock TabICL 7.384; E4 + Gaussian TabICL 7.392.

---

## 5. Paper skeleton (LaTeX)

Create `paper/main.tex`, using the venue template if it is present in the repo; otherwise use a generic two-column 4-page article and leave a `% TODO: venue template` note. Include the tables and figures above.

Sections, each with a target length and placeholder text only (bullets of what goes there, no invented claims):
1. **Introduction** (~0.6 page): imbalanced regression; tabular ICL models as regressors; contribution bullets.
2. **Method** (~0.7 page): context vs query sampling; Gaussian context with random mean and variance; the two regimes (per-dataset fine-tuning, synthetic-prior fine-tuning); the ECG encoder trained through a frozen ICL model.
3. **Experiments** (~2.3 pages): Tables 1–3 and Fig. 2. Include an explicit trade-off paragraph: frequent-region and 30–60 band errors can get worse.
4. **Conclusion and limitations** (~0.2 page).

In `paper/RESULTS_SUMMARY.md`, write one factual sentence per table (for example, "Gaussian context reduces few-shot MAE on all three datasets") generated strictly from the final numbers, including the cases where the method does worse.

---

## 6. Schedule and stop rules

| Day (from 7 Nov) | Deliverable |
|---|---|
| 1 | `REPO_MAP.md`, data-hygiene check, protocol implemented in `compute_metrics.py` |
| 1–3 | Task A (ResNet reruns) |
| 1–5 | Task B1 (target-balanced TabICL) |
| 3–6 | Task C (seeds and metrics) |
| 6–7 | Task D (significance), Figs 1–2, all tables generated |
| 8 | `STATUS.md` final, `main.tex` compiles with all tables and figures |

**Stop rules:**
- If B1 is not finished by day 6, mark the target-balanced rows `TODO` and continue; it becomes a stated limitation.
- If any data-hygiene check fails (test or validation data used in fine-tuning or model selection), stop and report it before running anything else.

## 7. Final deliverables checklist
- [ ] `results/REPO_MAP.md`: pipelines, metric definitions, splits
- [ ] `results/STATUS.md`: every run's status, discrepancies against the reference values, hygiene checks
- [ ] `results/preds/*.csv` and `results/metrics/*.csv`
- [ ] `paper/tables/table1a.tex`, `table1b.tex`, `table2.tex`, `table3.tex`
- [ ] `paper/figs/fig1_method.pdf`, `fig2_pred_vs_true.pdf`
- [ ] `paper/main.tex` compiling to 4 pages or fewer, excluding references
- [ ] `paper/RESULTS_SUMMARY.md`
