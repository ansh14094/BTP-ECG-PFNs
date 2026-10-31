#!/usr/bin/env python3
"""Imbalanced-regression benchmark for TabPFN-2.5, with and without biased-context
fine-tuning.

Arms
----
lgbm        LightGBM, hyperparameters chosen by random search on the validation set.
zeroshot    TabPFN-2.5 out of the box (no gradient updates).
ft_uniform  TabPFN-2.5 fine-tuned with the stock uniform context/query sampler, but
            at the *same reduced context size* as ft_gauss. This is the arm that
            matters: without it, any gain from ft_gauss could simply be the effect of
            fine-tuning at all.
ft_gauss    TabPFN-2.5 fine-tuned with Gaussian-weighted context sampling and an
            independently drawn query set (biased_split.py).

ft_uniform and ft_gauss share every hyperparameter, the same seeds, the same episode
count, the same early-stopping criterion and the same validation set. The sampling
weight is the only difference.

All arms select on the DIR-balanced validation set, so model selection itself is not
biased toward the majority target region -- and is identical across arms.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch

# Bound CPU threads. Several of these processes run concurrently (one per dataset or
# per encoder/protocol cell); unbounded, each grabs every core and the machine ends up
# ~2x oversubscribed, which measurably slowed fine-tune cells from 146 s to 262 s.
_THREADS = int(os.environ.get("RUN_N_THREADS", "10"))
torch.set_num_threads(_THREADS)

import data as data_mod
import metrics as metrics_mod

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_V25 = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
# Checkpoints from continued pre-training of TabICL on its own prior, where the
# positional context/query cut was made to land on a Gaussian-weighted context
# (`gauss`) or on an unbiased permutation (`uniform`). See tabicl_prior/.
PRIOR_CKPT = {m: os.path.join(HERE, os.pardir, "tabicl_prior", "runs", m, "step-3000.ckpt")
              for m in ("uniform", "gauss", "mixed")}


def prep_frame(df: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    """Give categorical columns pandas 'category' dtype (TabPFN and LightGBM both
    detect categoricals from dtype)."""
    out = df.copy()
    for c in cat_features:
        out[c] = out[c].astype("category")
    return out


# --------------------------------------------------------------------------- arms


def arm_lgbm(Xtr, ytr, Xva, yva, Xte, seed, n_trials=30):
    """LightGBM with a random search on the validation set.

    Tuned rather than left at defaults so the comparison against TabPFN is not
    rigged in TabPFN's favour.
    """
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    best, best_rmse = None, np.inf
    for _ in range(n_trials):
        params = dict(
            objective="l2",
            n_estimators=int(rng.integers(100, 1200)),
            learning_rate=float(np.exp(rng.uniform(np.log(0.005), np.log(0.3)))),
            num_leaves=int(rng.integers(7, 128)),
            min_child_samples=int(rng.integers(2, 60)),
            subsample=float(rng.uniform(0.5, 1.0)),
            subsample_freq=1,
            colsample_bytree=float(rng.uniform(0.5, 1.0)),
            reg_lambda=float(np.exp(rng.uniform(np.log(1e-3), np.log(10.0)))),
            random_state=seed,
            verbose=-1,
            # Bound the thread count. LightGBM grabs every core by default, so
            # several of these running concurrently oversubscribe the machine badly
            # (four concurrent searches on 48 cores turned a 54 s arm into >60 min).
            n_jobs=int(os.environ.get("LGBM_N_JOBS", "8")),
        )
        m = lgb.LGBMRegressor(**params).fit(Xtr, ytr)
        rmse = float(np.sqrt(np.mean((m.predict(Xva) - yva) ** 2)))
        if rmse < best_rmse:
            best_rmse, best = rmse, m
    return best.predict(Xte)


def arm_zeroshot(Xtr, ytr, Xva, yva, Xte, seed, n_estimators=8):
    """TabPFN-2.5 with no gradient updates.

    `n_estimators` is matched to the fine-tuned arms' final-inference ensemble size
    so the arms differ in weights, not in ensemble budget.
    """
    from tabpfn import TabPFNRegressor

    reg = TabPFNRegressor(model_path=CKPT_V25, device="cuda",
                          n_estimators=n_estimators, auto_scale_n_estimators=False,
                          random_state=seed)
    reg.fit(Xtr, ytr)
    return reg.predict(Xte)


def _numeric_matrix(Xtr, Xte):
    """One-hot + standardize, for nearest-neighbour retrieval in feature space."""
    from sklearn.preprocessing import StandardScaler

    both = pd.concat([Xtr, Xte], axis=0)
    both = pd.get_dummies(both, dummy_na=False)
    both = both.fillna(both.median(numeric_only=True)).astype(float)
    scaler = StandardScaler().fit(both.iloc[:len(Xtr)])
    Z = scaler.transform(both)
    return Z[:len(Xtr)], Z[len(Xtr):]


def _inverse_density_pick(y: np.ndarray, size: int, rng, n_bins: int = 20) -> np.ndarray:
    """Sample indices with probability inversely proportional to target-bin frequency."""
    edges = np.linspace(y.min(), y.max() + 1e-12, n_bins + 1)
    b = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(b, minlength=n_bins).astype(float)
    counts[counts == 0] = 1.0
    w = 1.0 / counts[b]
    w = w / w.sum()
    size = min(size, len(y))
    return rng.choice(len(y), size=size, replace=False, p=w)


def arm_ft_gauss_recal(Xtr, ytr, Xva, yva, Xte, seed, **kw):
    """Gaussian-context fine-tuning followed by the same validation recalibration.

    Exists to separate two explanations of the fine-tuning gain. If plain
    recalibration of the zero-shot model already matches `ft_gauss`, then the useful
    part of the fine-tuning may be nothing more than decompressing predictions -- in
    which case stacking recalibration on top of it should add little. If instead this
    arm clearly beats recalibration alone, the fine-tuning is contributing something
    a monotone output transform cannot.
    """
    n_te = kw.pop("n_te")
    pred_all = arm_finetune(Xtr, ytr, Xva, yva, Xte, seed, mode="gauss", **kw)
    pred_all = np.asarray(pred_all, float)
    p_te, p_va = pred_all[:n_te], pred_all[n_te:]
    a, b = np.polyfit(p_va, yva, 1)          # fitted on validation only
    return np.concatenate([a * p_te + b, a * p_va + b])


def arm_tabicl_zeroshot(Xtr, ytr, Xva, yva, Xte, seed, n_estimators=8):
    """TabICL (v2 regressor) with no gradient updates — the TabICL analogue of
    the `zeroshot` arm."""
    from tabicl import TabICLRegressor

    reg = TabICLRegressor(n_estimators=n_estimators, device="cuda", random_state=seed)
    reg.fit(Xtr, ytr)
    return reg.predict(Xte)


def arm_tabicl_prior(Xtr, ytr, Xva, yva, Xte, seed, *, mode, n_estimators=8):
    """Zero-shot TabICL, but from a checkpoint whose PRIOR was reordered.

    No gradient step is taken on this dataset. The only difference from
    `ticl_zeroshot` is which weights are loaded, so whatever this arm gains or loses
    is attributable to the continued pre-training alone -- and `mode="uniform"` is
    the matched control, trained for the same 3000 steps on the same prior with an
    unbiased permutation in place of the Gaussian one.
    """
    from tabicl import TabICLRegressor

    path = PRIOR_CKPT[mode]
    if not os.path.exists(path):
        raise FileNotFoundError(f"prior checkpoint missing: {path}")
    reg = TabICLRegressor(model_path=path, n_estimators=n_estimators,
                          device="cuda", random_state=seed)
    reg.fit(Xtr, ytr)
    return reg.predict(Xte)


def arm_tabicl_finetune(Xtr, ytr, Xva, yva, Xte, seed, *, mode, epochs, lr,
                        ctx_keep_frac, sigma_range, center_mode, n_estimators=8):
    """Fine-tune TabICL with the episode sampler swapped, mirroring the TabPFN arms.

    TabICL's splitter is a module-level function that its data pipeline calls by name,
    so rebinding the module attribute before `fit` redirects every episode split
    without forking the library — the same technique used for TabPFN.
    """
    from tabicl._finetune import FinetunedTabICLRegressor
    import tabicl._finetune.data as ticl_data

    import tabicl_split

    original = ticl_data._split_ctx_query
    tabicl_split.reset_episode_counter()
    ticl_data._split_ctx_query = tabicl_split.make_split_fn(
        mode=mode, ctx_keep_frac=ctx_keep_frac,
        sigma_range=sigma_range, center_mode=center_mode)
    try:
        # Episode size caps peak memory: TabICL attends over the whole context, so
        # a 10k-row episode on a 512-dim embedding needs ~70 GB and OOMs. AgeDB
        # (12,208 x 512) must run smaller; PTB-XL (7,566 x 128) is fine at the
        # default. Both arms of a comparison always share this value.
        reg = FinetunedTabICLRegressor(
            epochs=epochs, learning_rate=lr, device="cuda", random_state=seed,
            n_estimators_inference=n_estimators, early_stopping=True, patience=8,
            finetune_ctx_query_ratio=0.2,
            max_data_size=int(os.environ.get("TICL_MAX_DATA_SIZE", "10000")),
        )
        # Pass the balanced validation set explicitly, as for the TabPFN arms, so
        # early stopping is not decided on a majority-dominated internal split.
        reg.fit(Xtr, ytr, X_val=Xva, y_val=yva)
        return reg.predict(Xte)
    finally:
        ticl_data._split_ctx_query = original


def arm_zeroshot_recal(Xtr, ytr, Xva, yva, Xte, seed, kind="linear", n_estimators=8):
    """Zero-shot TabPFN-2.5 plus a monotone recalibration fitted on the balanced val set.

    This is the cheapest possible way to exploit a balanced validation set: a model
    that regresses toward the majority region produces compressed predictions, and an
    affine (or isotonic) correction can undo much of that compression without any
    gradient updates. It is included deliberately as an adversarial baseline -- if it
    matches biased-context fine-tuning, then the fine-tuning is not the thing doing
    the work and the paper should say so.
    """
    from tabpfn import TabPFNRegressor

    reg = TabPFNRegressor(model_path=CKPT_V25, device="cuda", n_estimators=n_estimators,
                          auto_scale_n_estimators=False, random_state=seed)
    reg.fit(Xtr, ytr)
    p_va = np.asarray(reg.predict(Xva), float)
    p_te = np.asarray(reg.predict(Xte), float)

    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip").fit(p_va, yva)
        return cal.predict(p_te)
    a, b = np.polyfit(p_va, yva, 1)   # least squares on the balanced val set
    return a * p_te + b


def arm_ctx_balanced(Xtr, ytr, Xva, yva, Xte, seed, frac=0.5, n_estimators=8):
    """Inference-only: give TabPFN-2.5 an inverse-density-resampled context.

    This is the regression analogue of the downsampling remedy that
    arXiv:2605.21742 found effective for class imbalance in PFNs. No weights change;
    only which training rows enter the context.
    """
    from tabpfn import TabPFNRegressor

    rng = np.random.default_rng(seed)
    idx = _inverse_density_pick(ytr, int(len(ytr) * frac), rng)
    reg = TabPFNRegressor(model_path=CKPT_V25, device="cuda", n_estimators=n_estimators,
                          auto_scale_n_estimators=False, random_state=seed)
    reg.fit(Xtr.iloc[idx], ytr[idx])
    return reg.predict(Xte)


def arm_imcontext(Xtr, ytr, Xva, yva, Xte, seed, k_near=128, k_inv=128, n_estimators=2):
    """Inference-only reimplementation of IM-Context (arXiv:2405.18202) on TabPFN-2.5.

    For each query the context is the union of its `k_near` nearest training
    neighbours in feature space and `k_inv` rows drawn from an inverse-density
    resampling of the training set. The original paper used GPT-2-style in-context
    regressors and the older 6-layer PFN of Mueller et al., not TabPFN-2.5; this is
    their context-construction rule carried over to the model under study, so that
    the fine-tuning results have the right prior-work comparison rather than only a
    citation.

    `n_estimators` is lower here purely for runtime: this arm refits per query.
    """
    from tabpfn import TabPFNRegressor
    from sklearn.neighbors import NearestNeighbors

    Ztr, Zte = _numeric_matrix(Xtr, Xte)
    k_near = min(k_near, len(ytr))
    nn = NearestNeighbors(n_neighbors=k_near).fit(Ztr)
    _, neigh = nn.kneighbors(Zte)

    rng = np.random.default_rng(seed)
    inv_pool = _inverse_density_pick(ytr, min(len(ytr), 4 * k_inv), rng)

    preds = np.empty(len(Xte), dtype=float)
    for i in range(len(Xte)):
        extra = rng.choice(inv_pool, size=min(k_inv, len(inv_pool)), replace=False)
        ctx = np.unique(np.concatenate([neigh[i], extra]))
        reg = TabPFNRegressor(model_path=CKPT_V25, device="cuda",
                              n_estimators=n_estimators, auto_scale_n_estimators=False,
                              random_state=seed)
        reg.fit(Xtr.iloc[ctx], ytr[ctx])
        preds[i] = float(reg.predict(Xte.iloc[[i]])[0])
    return preds


def arm_finetune(Xtr, ytr, Xva, yva, Xte, seed, *, mode, epochs, lr,
                 ctx_plus_query, ctx_keep_frac, center_mode, query_mode,
                 sigma_range, n_estimators=8, out_dir=None):
    """Fine-tune TabPFN-2.5, with the episode sampler swapped for ours.

    The stock sampler is a module-level function that the training loop wraps in
    `functools.partial` each epoch, so rebinding the module attribute before `fit`
    redirects every episode split without forking the library.
    """
    from functools import partial

    import tabpfn.finetuning.finetuned_base as fb
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

    import biased_split

    original = fb._ratio_ctx_query_split
    biased_split.reset_episode_counter()
    fb._ratio_ctx_query_split = partial(
        biased_split.imbalanced_ctx_query_split,
        mode=mode,
        ctx_keep_frac=ctx_keep_frac,
        center_mode=center_mode,
        query_mode=query_mode,
        sigma_range=sigma_range,
    )
    try:
        reg = FinetunedTabPFNRegressor(
            device="cuda",
            model_version=ModelVersion.V2_5,
            epochs=epochs,
            learning_rate=lr,
            n_finetune_ctx_plus_query_samples=ctx_plus_query,
            finetune_ctx_query_split_ratio=0.2,
            # All three kept equal: `use_fixed_preprocessing_seed` (on by default)
            # is only well-defined when they match, and it also makes the finetuned
            # arms' inference ensemble the same size as the zero-shot arm's.
            n_estimators_finetune=n_estimators,
            n_estimators_validation=n_estimators,
            n_estimators_final_inference=n_estimators,
            random_state=seed,
            early_stopping=True,
            early_stopping_patience=8,
        )
        # Passing the DIR-balanced val set explicitly keeps early stopping from
        # being decided on a majority-dominated internal split.
        reg.fit(Xtr, ytr, X_val=Xva, y_val=yva,
                output_dir=None if out_dir is None else out_dir)
        return reg.predict(Xte)
    finally:
        fb._ratio_ctx_query_split = original


# ---------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="abalone")
    ap.add_argument("--arms", default="lgbm,zeroshot,ft_uniform,ft_gauss")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--protocol", default="dir",
                    choices=["dir", "natural", "fixed"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ctx-plus-query", type=int, default=1024,
                    help="episode size; smaller means more gradient steps per epoch")
    ap.add_argument("--ctx-keep-frac", type=float, default=0.5)
    ap.add_argument("--center-mode", default="random_quantile",
                    choices=["random_quantile", "mode"])
    ap.add_argument("--query-mode", default="uniform", choices=["uniform", "balanced"])
    ap.add_argument("--sigma-min", type=float, default=0.15)
    ap.add_argument("--sigma-max", type=float, default=1.5)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n-trials-lgbm", type=int, default=30)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")
    sigma_range = (args.sigma_min, args.sigma_max)

    ds = data_mod.load(args.dataset)
    X_all = prep_frame(ds.X, ds.cat_features)
    results_path = os.path.join(HERE, "results", f"{args.dataset}_{args.tag}.jsonl")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Resumability: skip (arm, seed) pairs this tag has already recorded, so a run
    # killed by a reboot picks up where it stopped instead of repeating finished work.
    done: set[tuple[str, int]] = set()
    if os.path.exists(results_path):
        for line in open(results_path):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r["arm"], int(r["seed"])))
    if done:
        print(f"resuming: {len(done)} (arm, seed) cells already complete", flush=True)

    print(f"dataset={ds.name} X={X_all.shape} cat={ds.cat_features} "
          f"protocol={args.protocol} arms={arms} seeds={seeds}", flush=True)

    for seed in seeds:
        if args.protocol == "fixed":
            if ds.fixed_split is None:
                raise ValueError(f"{ds.name} has no fixed split")
            # The official partition is identical for every seed; seeds then vary
            # only model/preprocessing randomness, not the data split. Paired
            # comparisons across arms remain valid; what they no longer capture is
            # split-to-split variance, which is reported as such.
            tr, va, te = ds.fixed_split
        elif args.protocol == "dir":
            tr, va, te = data_mod.dir_split(ds.y, ds.bin_width, seed=seed,
                                           per_bin_val=ds.per_bin_val,
                                           per_bin_test=ds.per_bin_test)
        else:
            tr, va, te = data_mod.natural_split(len(ds.y), seed=seed)

        Xtr, Xva, Xte = X_all.iloc[tr], X_all.iloc[va], X_all.iloc[te]
        ytr, yva, yte = ds.y[tr], ds.y[va], ds.y[te]
        phi = metrics_mod.RelevanceFunction(ytr)

        print(f"\n[seed {seed}] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

        # Arms predict on test and val in one call (val first is avoided so that
        # `imcontext`'s per-row loop keeps test rows contiguous). Validation metrics
        # are recorded so that hyperparameters such as the fine-tuning learning rate
        # can be selected on val, never on test.
        X_eval = pd.concat([Xte, Xva], axis=0)
        n_te = len(Xte)

        for arm in arms:
            if (arm, seed) in done:
                print(f"  {arm:11s} skipped (already recorded)", flush=True)
                continue
            t0 = time.time()
            try:
                if arm == "lgbm":
                    pred = arm_lgbm(Xtr, ytr, Xva, yva, X_eval, seed)
                elif arm == "zeroshot":
                    pred = arm_zeroshot(Xtr, ytr, Xva, yva, X_eval, seed)
                elif arm == "ticl_zeroshot":
                    pred = arm_tabicl_zeroshot(Xtr, ytr, Xva, yva, X_eval, seed)
                elif arm.startswith("ticl_prior_"):
                    pred = arm_tabicl_prior(
                        Xtr, ytr, Xva, yva, X_eval, seed,
                        mode=arm.removeprefix("ticl_prior_"))
                elif arm in ("ticl_ft_uniform", "ticl_ft_gauss"):
                    pred = arm_tabicl_finetune(
                        Xtr, ytr, Xva, yva, X_eval, seed,
                        mode=("uniform" if arm == "ticl_ft_uniform" else "gauss"),
                        epochs=args.epochs, lr=args.lr,
                        ctx_keep_frac=args.ctx_keep_frac,
                        sigma_range=sigma_range, center_mode=args.center_mode)
                elif arm == "zeroshot_recal":
                    pred = arm_zeroshot_recal(Xtr, ytr, Xva, yva, X_eval, seed)
                elif arm == "zeroshot_recal_iso":
                    pred = arm_zeroshot_recal(Xtr, ytr, Xva, yva, X_eval, seed,
                                              kind="isotonic")
                elif arm == "ctx_balanced":
                    pred = arm_ctx_balanced(Xtr, ytr, Xva, yva, X_eval, seed)
                elif arm == "imcontext":
                    pred = arm_imcontext(Xtr, ytr, Xva, yva, X_eval, seed)
                elif arm == "ft_gauss_recal":
                    pred = arm_ft_gauss_recal(
                        Xtr, ytr, Xva, yva, X_eval, seed, n_te=n_te,
                        epochs=args.epochs, lr=args.lr,
                        ctx_plus_query=args.ctx_plus_query,
                        ctx_keep_frac=args.ctx_keep_frac,
                        center_mode=args.center_mode,
                        query_mode=args.query_mode,
                        sigma_range=sigma_range)
                elif arm in ("ft_uniform", "ft_gauss", "ft_mixed"):
                    pred = arm_finetune(
                        Xtr, ytr, Xva, yva, X_eval, seed,
                        mode={"ft_uniform": "uniform", "ft_gauss": "gauss",
                              "ft_mixed": "mixed"}[arm],
                        epochs=args.epochs, lr=args.lr,
                        ctx_plus_query=args.ctx_plus_query,
                        ctx_keep_frac=args.ctx_keep_frac,
                        center_mode=args.center_mode,
                        query_mode=args.query_mode,
                        sigma_range=sigma_range,
                    )
                else:
                    raise ValueError(f"unknown arm {arm!r}")
            except Exception:
                print(f"  {arm}: FAILED\n{traceback.format_exc()}", flush=True)
                continue

            pred_all = np.asarray(pred, float)
            pred, pred_va = pred_all[:n_te], pred_all[n_te:]
            res = metrics_mod.evaluate(yte, pred, ytr, phi, bin_width=ds.bin_width)
            res["val_mae"] = float(np.mean(np.abs(pred_va - yva)))
            res["val_rmse"] = float(np.sqrt(np.mean((pred_va - yva) ** 2)))

            # Keep raw predictions so the *mechanism* can be inspected later (e.g.
            # how much of the gain is reduced regression-to-the-mean), not just the
            # summary metrics.
            pred_dir = os.path.join(HERE, "results", "preds")
            os.makedirs(pred_dir, exist_ok=True)
            pd.DataFrame({"y_true": yte, "y_pred": pred}).to_csv(
                os.path.join(pred_dir,
                             f"{ds.name}_{args.tag}_{args.protocol}_{arm}_s{seed}.csv"),
                index=False)
            rec = dict(dataset=ds.name, arm=arm, seed=seed, protocol=args.protocol,
                       tag=args.tag, seconds=round(time.time() - t0, 1),
                       config=dict(epochs=args.epochs, lr=args.lr,
                                   ctx_plus_query=args.ctx_plus_query,
                                   ctx_keep_frac=args.ctx_keep_frac,
                                   center_mode=args.center_mode,
                                   query_mode=args.query_mode,
                                   sigma_range=list(sigma_range)),
                       **res)
            with open(results_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {arm:11s} MAE={res['mae']:.4f} RMSE={res['rmse']:.4f} "
                  f"GM={res['gm']:.4f} SERA={res['sera']:.3f} | "
                  f"many={res['mae_many']:.3f} med={res['mae_medium']:.3f} "
                  f"few={res['mae_few']:.3f} ({rec['seconds']}s)", flush=True)

    print(f"\nresults -> {results_path}")


if __name__ == "__main__":
    main()
