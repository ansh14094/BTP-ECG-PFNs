#!/usr/bin/env python3
"""Fine-tune TabPFN-2.5 on an encoder's embeddings and SAVE the resulting weights.

Needed for encoders E2.1 and E2.2. Those are trained the same way E2 was -- gradients
flowing into the encoder through a FROZEN TabPFN -- except the frozen model is no
longer the stock checkpoint but a TabPFN that has already been fine-tuned with biased
context sampling. There is a chicken-and-egg here: fine-tuning needs embeddings and
embeddings need an encoder, so the chain starts from E1, the directly-trained encoder.

    E1 embeddings -> fine-tune TabPFN (gauss | mixed) -> freeze -> train E2.1 / E2.2

Early stopping restores best-validation weights, so a run that never improves reverts
to the original checkpoint and would silently hand back the stock model. This script
checks the saved weights against the pristine ones and refuses to write a no-op.
"""
from __future__ import annotations

import argparse, os, sys, warnings
from functools import partial

import numpy as np, torch
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "imbreg"))
import data as data_mod      # noqa: E402
import biased_split          # noqa: E402

CKPT_V25 = "tabpfn-v2.5-regressor-v2.5_default.ckpt"


def pristine_state():
    from tabpfn import TabPFNRegressor
    r = TabPFNRegressor(model_path=CKPT_V25, device="cuda", n_estimators=1,
                        auto_scale_n_estimators=False, random_state=0)
    r.fit(np.random.randn(64, 4), np.random.randn(64))
    return {k: v.detach().cpu().clone() for k, v in r.model_.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl_l1_enc", help="embedding set to fine-tune on")
    ap.add_argument("--protocol", default="dir", choices=["dir", "fixed"])
    ap.add_argument("--mode", required=True, choices=["gauss", "mixed", "uniform"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ctx-plus-query", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-early-stopping", action="store_true",
                    help="guarantee the weights move; use when a no-op would be useless")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import tabpfn.finetuning.finetuned_base as fb
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

    ds = data_mod.load(args.dataset)
    X = ds.X.copy()
    for c in ds.cat_features:
        X[c] = X[c].astype("category")
    if args.protocol == "fixed":
        tr, va, _ = ds.fixed_split
    else:
        tr, va, _ = data_mod.dir_split(ds.y, ds.bin_width, seed=args.seed,
                                       per_bin_val=ds.per_bin_val,
                                       per_bin_test=ds.per_bin_test)
    print(f"{args.dataset}: train={len(tr)} val={len(va)}  mode={args.mode} "
          f"epochs={args.epochs} lr={args.lr}", flush=True)

    base_sd = pristine_state()

    original = fb._ratio_ctx_query_split
    biased_split.reset_episode_counter()
    fb._ratio_ctx_query_split = partial(
        biased_split.imbalanced_ctx_query_split, mode=args.mode,
        ctx_keep_frac=0.5, center_mode="random_quantile",
        query_mode="uniform", sigma_range=(0.15, 1.5))
    try:
        reg = FinetunedTabPFNRegressor(
            device="cuda", model_version=ModelVersion.V2_5,
            epochs=args.epochs, learning_rate=args.lr,
            n_finetune_ctx_plus_query_samples=args.ctx_plus_query,
            finetune_ctx_query_split_ratio=0.2,
            n_estimators_finetune=8, n_estimators_validation=8,
            n_estimators_final_inference=8,
            random_state=args.seed,
            early_stopping=not args.no_early_stopping,
            early_stopping_patience=8)
        reg.fit(X.iloc[tr], ds.y[tr], X_val=X.iloc[va], y_val=ds.y[va])
    finally:
        fb._ratio_ctx_query_split = original

    sd = {k: v.detach().cpu().clone()
          for k, v in reg.finetuned_estimator_.model_.state_dict().items()}
    diff = sum(not torch.allclose(base_sd[k], sd[k]) for k in sd if k in base_sd)
    delta = sum(float((base_sd[k].float() - sd[k].float()).abs().sum())
                for k in sd if k in base_sd)
    print(f"weights moved: {diff}/{len(sd)} tensors, total abs delta {delta:.2f}", flush=True)
    if diff == 0:
        print("REFUSING to save: early stopping reverted this to the stock checkpoint, "
              "so it would be indistinguishable from the frozen baseline. "
              "Re-run with --no-early-stopping or a different learning rate.", flush=True)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": sd, "mode": args.mode, "dataset": args.dataset,
                "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                "tensors_changed": diff, "abs_delta": delta}, args.out)
    print(f"wrote {args.out}", flush=True)
    print("FINETUNE_SAVE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
