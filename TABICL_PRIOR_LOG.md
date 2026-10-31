# Moving the intervention from fine-tuning to the prior

Started 2026-10-29. Training side complete; evaluation queued.

## What this experiment is

Every result up to now applies Gaussian context sampling **during fine-tuning on a
target dataset**. The obvious reviewer question is whether that is a property of the
pre-trained model at all, or only of how we adapt it. TabPFN-2.5 cannot answer it:
Prior Labs ship the checkpoint but not the prior generator.

TabICL can. Version 2.2.0 ships `tabicl/prior/` (the structural-causal-model dataset
generators) *and* `tabicl/train/` (the full pre-training CLI), so the prior that made
the model is runnable. We do **not** train a new model from scratch — we take the
released regressor checkpoint and continue pre-training it on its own prior for 3000
further steps, changing one thing: which rows become the in-context set.

## How the intervention is expressed

TabICL splits each synthetic dataset into context and query **positionally**: the
dataset carries a `train_size` drawn uniformly in `[min_train_size, max_train_size]`,
and rows `[:train_size]` are the context. Which rows land there is just the order the
generator emitted them in.

So nothing about the sampler, the model, or the loss needs touching. We reorder rows.
Rows whose target sits near a drawn centre mu move to the front and become context;
tail rows fall past the cut into the query. Same (mu, sigma) priors as the fine-tuning
work, redrawn every dataset:

    mu    = quantile(y, U(0.05, 0.95))
    sigma = std(y) * exp(U(log 0.15, log 1.5))
    w_i  ~  exp(-0.5 * ((y_i - mu) / sigma)^2)

with Gumbel-top-k selection, because Gaussian weights underflow to exactly zero a few
sigma out and ordinary weighted sampling then degenerates.

Code: `tabicl_prior/gauss_prior.py` (the dataset subclass), `train_gauss_prior.py`
(launcher), `run_arm.sh` / `run_both.sh` (the two arms).

## The control

`--reorder_mode uniform` performs an unbiased permutation of the same shape and cost.
The control arm is therefore not "no permutation" but "a permutation that carries no
target information" — the two arms are matched in every operation performed, and
differ in the ordering rule alone. Same SCM, same hyperparameter sampling, same
sequence lengths, same `train_size` draws, same 3000 steps, same seed.

## Run configuration

Derived from TabICL's released stage-3 script.

| | |
|---|---|
| warm start | `tabicl-regressor-v2-20260212.ckpt`, `--only_load_model True` |
| steps | 3000 per arm |
| prior | `graph_scm` |
| `max_seq_len` | 3000 |
| batch | `--batch_size 8 --micro_batch_size 1` |
| optimiser | Muon |
| objective | `--regression_method quantile --num_quantiles 999` |
| train-size draw | `--min_train_size 0.79 --max_train_size 0.81` |
| memory | `--recompute True` (gradient checkpointing) |

Wall clock: uniform 16:25–19:57, gauss 19:57–22:36 (~2 h 40 each, 3.2 s/step, 3.3 GB).

## Training-side outcome

Both arms completed 3000/3000 steps.

Mean training pinball loss by step window:

| steps | uniform | gauss |
|---|---|---|
| 1–249 | 0.0600 | 0.1979 |
| 250–499 | 0.0548 | 0.1463 |
| 500–999 | 0.0559 | 0.1361 |
| 1000–1499 | 0.0584 | 0.1307 |
| 1500–1999 | 0.0575 | 0.1308 |
| 2000–2499 | 0.0545 | 0.1274 |
| 2500–3000 | 0.0552 | 0.1247 |

**These two columns must not be compared to each other.** The arms do not share a
query distribution: under Gaussian ordering the context is target-concentrated, so the
queries are disproportionately tail points and the prediction problem is genuinely
harder. A higher pinball loss in the gauss column is what a harder query set looks
like, not evidence of a worse model. The only reading the columns support is the
within-column one: both arms descended (gauss −37% from its own start, uniform −8%),
so both trained.

Whether the intervention helped is a question only the downstream evaluation answers.

## Weight-space check

Relative L2 distance between flattened state dicts (347 tensors, 28,544,991 params):

| pair | distance |
|---|---|
| stock → uniform | 0.00453 |
| stock → gauss | 0.00681 |
| uniform ↔ gauss | 0.00816 |

Both arms moved from stock, and the arm-to-arm distance exceeds either arm's distance
from stock, i.e. the two updates are close to orthogonal rather than one being a
scaled version of the other. This matters because an earlier attempt at this
experiment silently ran **both** arms in gauss mode — the `reorder_mode` class
attribute did not survive the DataLoader's `spawn` start method, so each worker
re-imported the module and got the default back. It was caught only because the two
arms' losses were identical. The fix copies the setting onto the instance in
`__init__`, where pickling carries it; this distance table is the standing check that
the control is a control.

## Evaluation (queued)

`run_prior_eval.sh` adds two arms, `ticl_prior_uniform` and `ticl_prior_gauss`, to
`imbreg/run.py`. Both are **zero-shot**: no gradient step is taken on the evaluation
dataset, so the only difference from the existing `ticl_zeroshot` arm is which weights
are loaded, and the whole of any gap is attributable to the continued pre-training.

Datasets: `abalone`, `communities`, `wine_white`, `ptbxl_l1_enc`, `ptbxl_tabpfn_enc`
(DIR protocol) and `agedb_tabpfn_enc` (`fixed`), 5 seeds each, appended to the
existing `*_ticl_base.jsonl` so all three weight sets sit in one file per dataset.

### What would count as a result

- **gauss < uniform on tail metrics (`mae_few`, SERA) across datasets** — the
  intervention transfers to the prior, and the claim stops being about fine-tuning.
- **gauss ≈ uniform** — the effect needs the target dataset's own targets and is an
  adaptation technique, not a pre-training one. This is a publishable negative and
  should be reported as such.
- **both < stock** — says only that 3000 extra steps on the prior help; it is the
  gauss−uniform gap that isolates the intervention.

A single composite checkpoint at step 3000 is one point on a trajectory; intermediate
checkpoints every 250 steps were kept, so a step-vs-performance curve is available if
the step-3000 comparison is ambiguous.
