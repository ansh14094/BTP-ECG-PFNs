# Code map

Written for whoever picks this repository up next — someone who has not seen it
before and needs to re-run an experiment, check a number, or add an arm.

Read this file top to bottom once. It is the only document that covers the whole
repository; everything else is either a narrative log of one experiment or a
reference for one part.

Companion documents:

| document | answers |
|---|---|
| [`README.md`](README.md) | what the project is and what was found |
| [`imbreg/results/REPO_MAP.md`](imbreg/results/REPO_MAP.md) | exact metric and split definitions, for checking a published number |
| [`imbreg/results/STATUS.md`](imbreg/results/STATUS.md) | data-hygiene audit, known failures, what is still missing |
| [`HANDOFF.md`](HANDOFF.md) | what was running when work stopped, and how to resume it |
| [`paper_completion_plan.md`](paper_completion_plan.md) | the deliverables the current round of work is aimed at |

---

## 1. The one-minute version

The repository answers one question: **if a tabular in-context model's context set is
sampled from a Gaussian over the target instead of uniformly, does it predict the
rare tails of the target distribution better?**

That question is asked at two levels, and the distinction is the single most
important thing to understand before touching anything:

- **Dataset level** — the context is Gaussian-sampled while *fine-tuning on the
  evaluation dataset itself*. Code: [`imbreg/biased_split.py`](imbreg/biased_split.py).
  Arms named `ft_*` and `ticl_ft_*`.
- **Prior level** — the released TabICL regressor gets 3000 further pre-training
  steps on *TabICL's own synthetic prior*, with the rows of each synthetic dataset
  reordered so the context/query cut lands on a target-concentrated context. Code:
  [`tabicl_prior/`](tabicl_prior/). Arms named `ticl_prior_*`. The resulting model
  has never seen any evaluation dataset.

Numbers from the two levels are not interchangeable and have been confused before —
see `STATUS.md` section 2.1, where three values in the completion plan turned out to
come from the dataset-level run while being presented as prior-level results.

## 2. Data flow

```
raw data                encoder                embeddings            evaluation
────────                ───────                ──────────            ──────────

PTB-XL signals ─┬─> train.py (L1)          ─┐
 data/*.npy     │   resnet1d_wang_fds.py    │
                │                           ├─> ecg_tabpfn/              imbreg/run.py
                ├─> ecg_tabpfn/             │   extract_embeddings.py ─>  │
                │   train_encoder.py        │   emb_*.npz (128-d)         │  arms x seeds
                │   (through frozen TabPFN) │                             │
                │                           │                             ├─> results/preds/*.csv
                └─> ecg_tabpfn/             │                             └─> results/*.jsonl
                    train_encoder_tabicl.py ┘                                     │
                    (through frozen TabICL)                                       │
                                                                                  v
AgeDB images ─────> agedb_tabpfn/train_encoder.py ─> emb_*.npz           eval/compute_metrics.py
                                                                                  │
tabular CSVs ─────────────────────────────────────> imbreg/data.py                │
 (auto-downloaded from UCI)                                                       v
                                                            eval/significance.py, make_tables.py,
TabICL synthetic prior ─> tabicl_prior/train_gauss_prior.py        make_figures.py, make_summary.py
                          + gauss_prior.py (the reordering)                       │
                          ─> runs/<mode>/step-3000.ckpt                           v
                                                                      paper/tables/*.tex, figs/*.pdf
```

Two rules this layout enforces, both deliberate:

1. **Every modality goes through `imbreg/run.py`.** ECG and face images become
   embedding tables, so they run the same arms, metrics and aggregation as the
   tabular datasets. There is no parallel evaluation implementation to drift.
2. **Metrics are computed in exactly one place**, `eval/compute_metrics.py`, from
   saved per-prediction CSVs. No metric is computed inside a training loop, and no
   table reads a training log.

## 3. Directory map

| path | contents | in git? |
|---|---|---|
| `*.py` (root) | ResNet1d ECG training stack — the supervised baseline and the imbalance-method baselines | yes |
| `imbreg/` | the evaluation harness: datasets, split protocols, arms, metrics | yes |
| `tabicl_prior/` | prior-level continued pre-training of TabICL | code yes, checkpoints no |
| `ecg_tabpfn/` | PTB-XL encoders trained through a frozen in-context model | code yes, embeddings no |
| `agedb_tabpfn/` | AgeDB-DIR face-age counterpart of `ecg_tabpfn/` | code yes, embeddings no |
| `eval/` | the reporting pipeline: metrics, significance, tables, figures | yes |
| `paper/` | LaTeX source, generated tables and figures | yes |
| `data/` | PTB-XL signals and the metadata CSV | **no** (PhysioNet licence) |
| `third_party/` | vendored upstream source and bundled weights | **no** |
| `checkpoint_*/` | model checkpoints | **no** (size) |
| `*_LOG.md` | narrative logs, one per experiment line | yes |

Everything excluded is listed with its reason in [`.gitignore`](.gitignore). Nothing
excluded is needed to *read* a result — only to regenerate one.

## 4. Entry points, file by file

### 4.1 Root — the ResNet1d ECG stack

This is the conventional supervised pipeline, inherited from the imbalanced-regression
literature and used as the Task A baseline.

| file | role |
|---|---|
| `train.py` | **main entry point.** Trains ResNet1d-Wang / Inception1d on PTB-XL age. Flags select the imbalance method: `--fds`, `--ranking` (RankSim), `--conr` (ConR), `--reweight`, `--lds`. `--seed` sets all four RNGs *and* appends `_seed<n>` to the store name. |
| `resnet1d_wang_fds.py` | the backbone. `return_features=True` makes `forward` return `(out, encoding)` without enabling FDS. |
| `inception1d_fds.py` | alternative backbone, same interface |
| `fds.py` | Feature Distribution Smoothing (Yang et al. 2021) |
| `ranksim.py`, `ranking.py` | RankSim batchwise ranking regulariser and its differentiable sort |
| `conr.py` | ConR contrastive regulariser (Keramati et al. 2023; upstream copyright header retained) |
| `loss.py` | loss assembly — pulls in RankSim and ConR |
| `datasets.py` | PTB-XL `Dataset`, label-distribution smoothing, reweighting |
| `build_ptbxl_csv.py` | builds `data/1000_timesteps.csv` (columns: `split`, `age`, `path`, `index`) from pre-processed PTB-XL |
| `utils.py` | checkpointing, logging, meters |
| `error_vs_age.py` | per-age-bin MAE for a saved checkpoint |
| `sanity_check.py` | standalone repository sanity check (data, model, method, expected-vs-observed) |
| `tabpfn_feature_experiment.py` | the original two-stage probe: backbone features → TabPFN |
| `make_table1.py` | an earlier table builder; **superseded by `eval/make_tables.py`** |
| `notify_on_done.py` | watches for a pipeline to finish and emails the result; reads credentials from `~/.config/btp_mail.json`, never prints them |

**Trap.** Before the fix recorded in `STATUS.md` section 4, `train.py` had *no seed
control at all* and the model only returned features when FDS was on — so
`--ranking`/`--conr` silently forced FDS. Both are fixed; do not reintroduce them.

### 4.2 `imbreg/` — the evaluation harness

| file | role |
|---|---|
| `run.py` | **main entry point.** One process = one dataset × arms × seeds. Writes per-prediction CSVs and one JSONL row per cell. |
| `data.py` | dataset loaders (`LOADERS`) and the split protocols `dir_split` / `fixed_split` |
| `biased_split.py` | the Gaussian context sampler, dataset level |
| `metrics.py` | shot regions (Yang et al. 2021) and SERA (Ribeiro & Moniz 2020) |
| `tabicl_split.py` | TabICL-side context/query handling |
| `analyze.py`, `verdict.py`, `check_cells.py` | ad-hoc inspection of the JSONL results |
| `make_matrix_csv.py`, `matrix_age_bands.py`, `age_band_report.py`, `ticl_age_bands.py` | the age-band matrix tables |
| `make_report.py`, `make_tabicl_report.py` | narrative report generators |
| `plot_*.py` | exploratory plots (imbalance, μ/σ draws, loss curves, calibration, scatter) |
| `calibration_experiment.py`, `loss_curve_experiment.py`, `prior_trajectory.py` | side experiments |
| `run_*.sh`, `resume_all.sh` | shell drivers for grids of `run.py` calls |

**`run.py` command line:**

```
--dataset       one key of imbreg/data.py:LOADERS
--arms          comma-separated (see 4.3)
--seeds         comma-separated, default 0,1,2,3,4
--protocol      dir | fixed | natural
--tag           free-form label; becomes part of the output filename
--epochs        fine-tuning epochs (default 60)
--lr            fine-tuning learning rate (default 1e-5)
--ctx-plus-query, --ctx-keep-frac, --center-mode, --query-mode,
--sigma-min, --sigma-max    Gaussian sampler parameters
--n-trials-lgbm optuna trials for the LightGBM baseline
```

`run.py` **resumes**: a `(arm, seed)` already present in the target JSONL is skipped.
To force a re-run, change `--tag` or delete the JSONL rows — this has bitten before.

### 4.3 The arms

Dispatched in `imbreg/run.py` around line 440.

| arm | what it is |
|---|---|
| `lgbm` | LightGBM baseline, Optuna-tuned on validation |
| `zeroshot` | stock TabPFN-2.5, no fine-tuning |
| `zeroshot_recal`, `zeroshot_recal_iso` | the same, with linear / isotonic recalibration |
| `ft_uniform` | TabPFN fine-tuned, uniform context — **the matched control** |
| `ft_gauss` | TabPFN fine-tuned, Gaussian context — the dataset-level intervention |
| `ft_mixed` | half Gaussian, half uniform |
| `ft_gauss_recal` | `ft_gauss` plus recalibration |
| `ctx_balanced` | context balanced over target bins |
| `imcontext` | imbalance-aware context baseline |
| `ticl_zeroshot` | stock TabICL, no fine-tuning |
| `ticl_ft_uniform`, `ticl_ft_gauss` | TabICL fine-tuned, dataset level |
| `ticl_prior_uniform` | prior level, unbiased permutation — **the matched control** |
| `ticl_prior_gauss` | prior level, Gaussian — the intervention |
| `ticl_prior_mixed` | prior level, half Gaussian half uniform |
| `ticl_prior_target_balanced` | prior level, inverse bin frequency — the control that makes the Gaussian claim falsifiable |

`ticl_prior_<mode>` loads `tabicl_prior/runs/<mode>/step-3000.ckpt`. A missing
checkpoint fails that arm only; `run.py` catches per-arm exceptions and continues.

**Always report an intervention against its matched control, never against the stock
model** — the control exists to separate "the ordering helped" from "3000 more steps
helped".

### 4.4 `tabicl_prior/` — prior-level pre-training

| file | role |
|---|---|
| `gauss_prior.py` | the row-reordering functions: `gauss_row_order`, `target_balanced_row_order`, `_gumbel_top_k`. **This is the method.** |
| `train_gauss_prior.py` | strips `--reorder_mode` / `--reorder_seed` out of `argv`, patches the reordering into TabICL's prior, then hands the rest to TabICL's own trainer |
| `run_arm.sh` | `./run_arm.sh <mode> <steps> <outdir>` — the architecture flags are copied verbatim from TabICL's `train_v2_reg_stage3.sh` and **must** match the released checkpoint |
| `run_both.sh`, `trial.sh` | multi-arm and timing-trial drivers |

Selection is Gumbel-top-k **in log space**: raw Gaussian weights underflow to zero a
few σ out, which silently collapses the sampler. Do not "simplify" it to
`torch.multinomial` on the weights.

Modes: `gauss`, `uniform`, `mixed`, `target_balanced`. Adding a fifth means a branch
in `gauss_prior.py`'s dispatch, the mode name in `train_gauss_prior.py`'s validation
tuple, and an entry in `imbreg/run.py`'s `PRIOR_CKPT`.

Each checkpoint is 218 MB — over GitHub's per-file limit, hence gitignored.

### 4.5 `ecg_tabpfn/` and `agedb_tabpfn/` — encoders

| file | role |
|---|---|
| `ecg_data.py` | PTB-XL loading; signals (~363 MB fp32) are held resident on GPU so an episode is one index operation |
| `train_encoder.py` | trains the encoder end-to-end through a **frozen TabPFN-2.5** |
| `train_encoder_tabicl.py` | the same through a **frozen TabICL** — one encoder per prior checkpoint |
| `finetune_and_save.py` | fine-tunes TabPFN on embeddings and saves the weights (needed for E2.1/E2.2) |
| `extract_embeddings.py` | freezes an encoder and dumps 128-d embeddings per split as `.npz` |
| `timestral_encoder.py`, `timestral_bench.py` | Timestral's forecasting encoder used in place of ours |
| `agedb_tabpfn/*` | the AgeDB-DIR counterpart; images resized once to 112×112, held as uint8; embeddings are 512-d |

The five PTB-XL encoders, all sharing E1's warm start so the only difference is the
frozen body they were trained through:

| | trained through | `LOADERS` key |
|---|---|---|
| E1 | nothing (supervised L1) | `ptbxl_l1_enc` |
| E2 | frozen stock TabPFN-2.5 | `ptbxl_tabpfn_enc` |
| E3 | frozen stock TabICL | `ptbxl_enc_e3` |
| E4 | frozen Gaussian-prior TabICL | `ptbxl_enc_e4` |
| E5 | frozen mixed-prior TabICL | `ptbxl_enc_e5` |

E2.1 / E2.2 (`ptbxl_enc_e21`, `ptbxl_enc_e22`) are an earlier line: encoders trained
through a TabPFN that had itself been fine-tuned with biased context sampling.

### 4.6 `eval/` — the reporting pipeline

Run in this order; each consumes the previous one's output.

| file | consumes | produces |
|---|---|---|
| `predict_resnet.py` | a ResNet checkpoint | a prediction CSV in the harness's own format, so ResNet rows and in-context rows are scored by identical code |
| `compute_metrics.py` | `imbreg/results/preds/*.csv` | `imbreg/results/metrics/*.csv` — **the single source of truth for every number** |
| `significance.py` | the prediction CSVs | paired bootstrap (10k) confidence intervals and per-seed win counts |
| `make_tables.py` | the metrics CSVs | `paper/tables/table{1a,1b,2,3}.tex`; writes a literal `TODO` for any cell whose run has not landed |
| `make_figures.py` | the metrics CSVs | `paper/figs/fig1_method.pdf`, `fig2_pred_vs_true.pdf` |
| `make_summary.py` | the metrics CSVs | `paper/RESULTS_SUMMARY.md`, with direction words chosen by the sign of the measured difference |
| `check_palette.py` | — | OKLab + Viénot 1999 dichromat simulation of the figure palette |

`make_tables.py` selects the fine-tuning learning rate per `(dataset, arm, seed)` by
**lowest validation RMSE**, read from `config.lr` in the JSONL row. The test split is
read once, at final evaluation.

## 5. Naming conventions

**Prediction files** — `imbreg/results/preds/<dataset>_<tag>_<protocol>_<arm>_s<seed>.csv`,
columns `id, y_true, y_pred, split`. `id` is the row's index into the dataset *as
loaded*, which is what makes it possible to ask whether a test row was in the
encoder's own training split.

`compute_metrics.py` parses that filename with a regex that matches dataset names
against `LOADERS`, **longest first**. A non-greedy `.+?` was used once and truncated
`ptbxl_enc_e3` to `ptbxl` and `wine_white` to `wine`, silently collapsing twelve
datasets into five. Any new dataset must be registered in `LOADERS`.

**JSONL rows** — `imbreg/results/<dataset>_<tag>.jsonl`, one row per
`(dataset, arm, seed)`, carrying all metrics plus `config` (epochs, lr, sampler
parameters) and `seconds`.

**Tags in current use** — `clean_matrix` (PTB-XL on the official split),
`clean_tabpfn*` (TabPFN arms on the official split), `ticl_base` (tabular prior-level),
`baseline_preds` (regenerated zero-shot predictions), `taskA` (ResNet baselines),
`matrix` / `main` / `baseline` / `lr*` (earlier rounds).

**Prior checkpoints** — `tabicl_prior/runs/<mode>/step-3000.ckpt`.

## 6. Protocols — and the one that leaks

| protocol | split |
|---|---|
| `fixed` | the dataset's official partition. PTB-XL: 7566 / 963 / 954, natural age distribution. Seeds vary model randomness only, not the split. |
| `dir` | ignores official boundaries and re-partitions **all** rows: each target bin donates up to `per_bin_test` rows to test, `per_bin_val` to val, rest to train. Test is near-uniform over the target. |
| `natural` | natural distribution without rebalancing |

> **`dir` leaks on PTB-XL.** It resamples all 9483 records, but every encoder was
> fitted on the official 7566-row train split — so 80–82% of the `dir` test set is
> data the encoder trained on. Measured directly, seen records score 0.66–0.74 years
> better than unseen ones, against a Gaussian effect of 0.095 years. **The
> contamination is about seven times the signal.** Full numbers in `STATUS.md`
> section 1.3.
>
> PTB-XL results are therefore reported under `fixed`. The `dir` numbers are retained
> and labelled, not deleted, because `dir` is what this codebase has always called
> "balanced evaluation".
>
> Tabular datasets have no official split, so `dir` is correct there and no encoder is
> involved. AgeDB's official val/test are already balanced over age bins, so it uses
> `fixed`.

Metric definitions — age bands, `natural` / `band_macro` / `balanced`, geometric mean,
shot regions, SERA — are in [`imbreg/results/REPO_MAP.md`](imbreg/results/REPO_MAP.md)
section 5 and are not repeated here.

## 7. Running things

### Environment

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`tabicl==2.2.0` is pinned: it is the first release shipping both `tabicl/prior/` (the
synthetic dataset generators) and `tabicl/train/` (the pre-training CLI). The
prior-level experiment cannot run without them.

### Data

Not in version control, and not ours to redistribute.

- **PTB-XL** — PhysioNet. Pre-process, then `python3 build_ptbxl_csv.py` to produce
  `data/1000_timesteps.csv`.
- **AgeDB-DIR** — request from the AgeDB authors; follow
  [`UPSTREAM_README_agedb_dir.md`](UPSTREAM_README_agedb_dir.md).
- **Abalone, Communities and Crime, Wine Quality** — downloaded automatically from
  UCI on first use into `imbreg/data_cache/`.

### The GPU mutex — use it

The workstation GPU is **shared with another user** and is usually near capacity.
Every job must serialise:

```bash
. ./gpu_lock.sh
gpu_lock          # blocks in the kernel until the GPU frees
```

`flock` on `.gpu.lock`, released automatically when the holding shell exits, so a
crashed job cannot wedge the queue. It bounds *our* concurrency to one; it cannot see
the other user's processes.

**Never identify a job with `pgrep -f <script name>`.** That pattern matches the
waiting shell itself and any diagnostic command mentioning the script. It cost 14
hours on 2026-10-30. Kill by PID, read state from lock files and sentinels.

### Drivers

Each stage takes its own `flock` and writes a `.<stage>_done` sentinel, so every
script below is safe to re-run — finished stages return immediately.

```bash
./run_all.sh                       # the ordered chain: Task C, then A, then B1
./run_taskC_clean.sh               # PTB-XL encoder x body matrix, official split
./run_task_a.sh                    # ResNet baselines, 2 arms x 5 seeds, 90 epochs
./run_taskB1_eval.sh               # target_balanced arm on every dataset
./run_prior_target_balanced.sh     # 3000-step prior training for that arm (~4 h)
```

Then regenerate every reported number:

```bash
python3 eval/compute_metrics.py
python3 eval/significance.py
python3 eval/make_tables.py
python3 eval/make_figures.py
python3 eval/make_summary.py
cd paper && pdflatex main.tex
```

### A worked example

```bash
. ./gpu_lock.sh && gpu_lock
cd imbreg
python3 -u run.py --dataset ptbxl_enc_e3 --protocol fixed --tag my_experiment \
    --arms ticl_zeroshot,ticl_prior_uniform,ticl_prior_gauss --seeds 0,1,2,3,4
cd .. && python3 eval/compute_metrics.py
```

## 8. How to extend it

**A new dataset** — add a loader to `imbreg/data.py:LOADERS` returning a `Dataset`
with `name`, `X`, `y`, `bin_width`. The name must be the `LOADERS` key or the
filename regex in `compute_metrics.py` will mis-parse it. Set `bin_width` so that
`dir_split` can fill its per-bin quotas.

**A new arm** — write `arm_<name>(Xtr, ytr, Xva, yva, X_eval, seed, ...) -> ndarray`
in `imbreg/run.py` and add a branch to the dispatch. It must return predictions for
`X_eval`, which is `concat([Xte, Xva])` — validation predictions come back in the same
array so model selection can happen on validation and never on test.

**A new prior mode** — a `*_row_order` function in `tabicl_prior/gauss_prior.py`, its
dispatch branch, the name in `train_gauss_prior.py`'s validation tuple, and an entry
in `imbreg/run.py:PRIOR_CKPT`. Then train it with
`cd tabicl_prior && ./run_arm.sh <mode> 3000 runs/<mode>`.

**A new table or figure** — extend `eval/make_tables.py` / `make_figures.py`. Read
from `imbreg/results/metrics/*.csv`, never from a JSONL row or a training log,
so there stays exactly one place where a metric is defined.

## 9. Traps, collected

1. **`dir` leaks on PTB-XL** (section 6). The largest single issue in the repository.
2. **`run.py` resumes silently.** An existing `(arm, seed)` in the JSONL is skipped.
   Change `--tag` to force a re-run.
3. **The filename regex must match dataset names longest-first** (section 5).
4. **Gumbel-top-k must stay in log space** (section 4.4).
5. **`--ranking`/`--conr` used to force FDS on.** Fixed via `return_features`; do not
   undo it.
6. **`train.py` had no seed control.** Now `--seed` sets all four RNGs and appends
   `_seed<n>` to the store name, so seeds no longer overwrite each other.
7. **Never `pgrep -f` a script name** (section 7).
8. **Checkpoints, datasets and raw logs are gitignored.** A fresh clone cannot
   reproduce a number without first obtaining the data and re-training.
9. **The prior trainer has exited 0 after a failed run.** Trust
   `runs/<mode>/step-3000.ckpt` existing, not the exit code — `run_prior_target_balanced.sh`
   already does.

## 10. Credit

The imbalanced-regression protocol, the DIR splits and the AgeDB pipeline follow
Yang et al., *Delving into Deep Imbalanced Regression*, ICML 2021. ConR follows
Keramati et al., 2023, with the upstream copyright header retained in `conr.py`.
SERA follows Ribeiro & Moniz, 2020. TabICL and TabPFN are the work of their
respective authors; this repository contains no copy of either model's weights.
