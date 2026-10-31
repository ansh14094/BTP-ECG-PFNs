#!/usr/bin/env python3
"""Dedicated experiment: instrumented TabPFN-2.5 fine-tuning, uniform vs Gaussian
context, recording the full training and validation trajectory.

Why this is a separate run rather than a re-analysis of the earlier ones: the original
runs only printed tqdm progress bars to stdout, and TabPFN emits its per-epoch
validation numbers through Python `logging`, which never reached those files. Trying to
recover curves from that scrollback produced obviously corrupt values (losses of 1e13),
so the honest fix is to re-run with proper instrumentation rather than to patch up a
bad parse.

TabPFN exposes a `FinetuningLogger` protocol for exactly this. `JsonlLogger` below
implements it and records, per run:

    per step   train/loss, train/lr, train/epoch, train/global_step
    per epoch  mean training loss and every validation metric TabPFN computes

Everything else is identical to the main study -- same episode sizes, same context
subsampling in both arms, same early stopping on validation -- so these curves describe
the runs the headline results came from, not a different configuration.
"""
from __future__ import annotations

import argparse, json, os, time, warnings
from functools import partial

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
import data as data_mod          # noqa: E402
import metrics as metrics_mod    # noqa: E402

CKPT_V25 = "tabpfn-v2.5-regressor-v2.5_default.ckpt"


class JsonlLogger:
    """Implements tabpfn.finetuning.logging.FinetuningLogger, writing one JSON object
    per event so the full trajectory survives the process."""

    def __init__(self, path: str, run_meta: dict):
        self.path = path
        self.meta = run_meta
        self.f = None
        self.n_steps = 0
        self.n_epochs = 0

    def setup(self, config: dict) -> None:
        self.f = open(self.path, "a")
        self._w({"event": "setup",
                 "config": {k: str(v) for k, v in config.items()
                            if isinstance(v, (int, float, str, bool, type(None)))}})

    def _w(self, obj: dict) -> None:
        if self.f:
            self.f.write(json.dumps({**self.meta, **obj}) + "\n")
            self.f.flush()

    def log_step(self, metrics: dict, step: int) -> None:
        self.n_steps += 1
        self._w({"event": "step", "step": step, **metrics})

    def log_epoch(self, metrics: dict, step: int) -> None:
        self.n_epochs += 1
        self._w({"event": "epoch", "step": step, **metrics})

    def finish(self) -> None:
        self._w({"event": "finish", "n_step_logs": self.n_steps, "n_epoch_logs": self.n_epochs})
        if self.f:
            self.f.close(); self.f = None


def run_one(ds, tr, va, te, X_all, arm, seed, args, out_path):
    """One fine-tuning run, fully instrumented. Returns final test metrics."""
    import tabpfn.finetuning.finetuned_base as fb
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
    import biased_split

    mode = "uniform" if arm == "ft_uniform" else "gauss"
    meta = {"dataset": ds.name, "arm": arm, "seed": seed, "lr": args.lr,
            "epochs_configured": args.epochs, "mode": mode}
    logger = JsonlLogger(out_path, meta)

    original = fb._ratio_ctx_query_split
    biased_split.reset_episode_counter()
    fb._ratio_ctx_query_split = partial(
        biased_split.imbalanced_ctx_query_split, mode=mode,
        ctx_keep_frac=args.ctx_keep_frac, center_mode="random_quantile",
        query_mode="uniform", sigma_range=(0.15, 1.5))
    t0 = time.time()
    try:
        reg = FinetunedTabPFNRegressor(
            device="cuda", model_version=ModelVersion.V2_5,
            epochs=args.epochs, learning_rate=args.lr,
            n_finetune_ctx_plus_query_samples=args.ctx_plus_query,
            finetune_ctx_query_split_ratio=0.2,
            n_estimators_finetune=8, n_estimators_validation=8,
            n_estimators_final_inference=8,
            random_state=seed, early_stopping=True, early_stopping_patience=8,
            experiment_logger=logger,
        )
        reg.fit(X_all.iloc[tr], ds.y[tr], X_val=X_all.iloc[va], y_val=ds.y[va])
        pred = np.asarray(reg.predict(X_all.iloc[te]), float)
    finally:
        fb._ratio_ctx_query_split = original
        logger.finish()
    wall = time.time() - t0

    phi = metrics_mod.RelevanceFunction(ds.y[tr])
    res = metrics_mod.evaluate(ds.y[te], pred, ds.y[tr], phi, bin_width=ds.bin_width)
    return {**meta, "wall_seconds": round(wall, 1), **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="abalone")
    ap.add_argument("--protocol", default="dir", choices=["dir", "fixed", "natural"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ctx-plus-query", type=int, default=1024)
    ap.add_argument("--ctx-keep-frac", type=float, default=0.5)
    ap.add_argument("--tag", default="losscurve")
    args = ap.parse_args()

    ds = data_mod.load(args.dataset)
    X_all = ds.X.copy()
    for c in ds.cat_features:
        X_all[c] = X_all[c].astype("category")

    curves = os.path.join(HERE, "results", f"curves_{args.dataset}_{args.tag}.jsonl")
    summary = os.path.join(HERE, "results", f"curvesummary_{args.dataset}_{args.tag}.jsonl")
    os.makedirs(os.path.dirname(curves), exist_ok=True)

    # Resume-safe: the curves file is appended to, never truncated, so a restart
    # keeps the trajectories of runs that already finished.
    done = set()
    if os.path.exists(summary):
        for l in open(summary):
            if l.strip():
                r = json.loads(l); done.add((r["arm"], r["seed"]))

    print(f"dataset={ds.name} protocol={args.protocol} epochs={args.epochs} lr={args.lr}", flush=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        if args.protocol == "fixed":
            tr, va, te = ds.fixed_split
        elif args.protocol == "dir":
            tr, va, te = data_mod.dir_split(ds.y, ds.bin_width, seed=seed,
                                            per_bin_val=ds.per_bin_val,
                                            per_bin_test=ds.per_bin_test)
        else:
            tr, va, te = data_mod.natural_split(len(ds.y), seed=seed)
        print(f"\n[seed {seed}] train={len(tr)} val={len(va)} test={len(te)}", flush=True)
        for arm in ("ft_uniform", "ft_gauss"):
            if (arm, seed) in done:
                print(f"  {arm}: skipped"); continue
            r = run_one(ds, tr, va, te, X_all, arm, seed, args, curves)
            with open(summary, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  {arm:11s} MAE={r['mae']:.4f} few={r['mae_few']:.3f} "
                  f"({r['wall_seconds']}s)", flush=True)

    print(f"\ncurves  -> {curves}\nsummary -> {summary}")
    print("LOSS_CURVE_EXPERIMENT_COMPLETE")


if __name__ == "__main__":
    main()
