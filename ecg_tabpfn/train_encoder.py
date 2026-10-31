#!/usr/bin/env python3
"""Stage 1: train a raw-ECG encoder end-to-end through a FROZEN TabPFN-2.5.

Motivation
----------
The existing two-stage result in this repo trains a ResNet1d on an L1 age loss, then
bolts TabPFN on top of its penultimate features (test MAE 7.441 vs 7.549 for the
ResNet alone). In that setup the encoder never knows a TabPFN sits downstream: it is
optimised to make a *linear* head accurate, not to produce features that are good
in-context evidence.

This script closes that loop. TabPFN's weights are frozen; the encoder is trained so
that TabPFN, given encoded context examples, predicts the age of encoded query
examples well. Gradients reach the encoder through TabPFN's forward pass via the
library's prompt-tuning path (`differentiable_input=True`).

Mechanism (verified before writing this):
    reg = TabPFNRegressor(differentiable_input=True, fit_mode="fit_preprocessors")
    reg.fit_with_differentiable_input(X_ctx, y_ctx)       # torch tensors
    logits, _, _ = reg.forward(X_q, use_inference_mode=True)
    loss = reg.znorm_space_bardist_(logits.T, y_q_std)
    loss.backward()                                       # reaches the encoder

`use_inference_mode=True` is correct here and not a bug: the library disables
torch.inference_mode whenever `differentiable_input=True`, so this selects the
standard (non-batched) executor while leaving autograd on.

Episode sampling here is UNIFORM by default. The imbalance-aware idea is applied in
stage 2, to TabPFN's own fine-tuning, so that the encoder stays an unbiased feature
extractor and the two interventions can be attributed separately. `--ctx-sampling
gauss` applies the same Gaussian context weighting at this stage too, as an ablation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, BASE)

import ecg_data  # noqa: E402
from resnet1d_wang_fds import resnet1d_wang_fds  # noqa: E402
from inception1d_fds import inception1d_fds  # noqa: E402

CKPT_V25 = "tabpfn-v2.5-regressor-v2.5_default.ckpt"


# ------------------------------------------------------------------ encoder


def build_encoder(name: str, warm_start: str | None, device: str) -> nn.Module:
    """ResNet1d-Wang / Inception1d with the age head removed, so forward() returns
    the 128-d pooled encoding instead of a scalar prediction."""
    model = (resnet1d_wang_fds(input_channels=12, fds=False) if name == "resnet1d_wang"
             else inception1d_fds(input_channels=12, fds=False))
    if warm_start:
        ckpt = torch.load(warm_start, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"warm start from {warm_start} "
              f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    model.linear = nn.Identity()          # forward() now yields the encoding
    return model.to(device)


# ------------------------------------------------------------- episode sampling


def _gumbel_top_k(log_w: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """Weighted sampling without replacement (Gumbel-top-k).

    Used instead of multinomial-without-replacement because the Gaussian weights
    underflow to zero several sigma out, which that API cannot handle -- exactly the
    strong-imbalance regime this is for.
    """
    u = torch.rand(log_w.shape, device=log_w.device, generator=gen).clamp_(1e-20, 1.0)
    g = -torch.log(-torch.log(u))
    return torch.topk(log_w + g, k).indices


def sample_episode(y: torch.Tensor, n_ctx: int, n_qry: int, gen: torch.Generator,
                   mode: str = "uniform", sigma_range=(0.15, 1.5)):
    """Draw (context_idx, query_idx) for one training step.

    The query set is drawn FIRST and without weighting, so its age marginal is
    untouched; only the context is biased under `mode="gauss"`. Context and query
    sizes are identical across modes, so the two differ solely in sampling weight.
    """
    n = y.shape[0]
    perm = torch.randperm(n, device=y.device, generator=gen)
    q_idx = perm[:n_qry]
    pool = perm[n_qry:]

    if mode == "uniform":
        return pool[:n_ctx], q_idx
    if mode != "gauss":
        raise ValueError(f"unknown mode {mode!r}")

    y_pool = y[pool]
    std = y_pool.std()
    if std < 1e-8:
        return pool[:n_ctx], q_idx
    q = torch.rand(1, device=y.device, generator=gen).item() * 0.9 + 0.05
    mu = torch.quantile(y_pool, q)
    lo, hi = np.log(sigma_range[0]), np.log(sigma_range[1])
    sigma = std * float(np.exp(np.random.default_rng(
        int(torch.randint(0, 2**31 - 1, (1,), device=y.device, generator=gen).item())
    ).uniform(lo, hi)))
    log_w = -0.5 * ((y_pool - mu) / sigma) ** 2
    return pool[_gumbel_top_k(log_w, min(n_ctx, len(pool)), gen)], q_idx


# ------------------------------------------------------------------ evaluation


@torch.no_grad()
def encode_all(enc: nn.Module, X: torch.Tensor, batch: int = 256) -> torch.Tensor:
    enc.eval()
    out = [enc(X[i:i + batch]) for i in range(0, len(X), batch)]
    enc.train()
    return torch.cat(out, dim=0)


@torch.no_grad()
def evaluate(enc, X_ctx, y_ctx, X_eval, y_eval, device, n_estimators=4, seed=0,
             tabpfn_weights=None):
    """Fit a clean (non-differentiable) TabPFN on the encoded training set and score.

    A separate estimator is used rather than the differentiable one so that evaluation
    follows the ordinary inference path and cannot be affected by prompt-tuning state.
    """
    from tabpfn import TabPFNRegressor

    Ec = encode_all(enc, X_ctx).float().cpu().numpy()
    Ee = encode_all(enc, X_eval).float().cpu().numpy()
    reg = TabPFNRegressor(model_path=CKPT_V25, device=device, n_estimators=n_estimators,
                          auto_scale_n_estimators=False, random_state=seed)
    reg.fit(Ec, y_ctx.cpu().numpy())
    if tabpfn_weights is not None:
        # Score through the same fine-tuned model the encoder trains against.
        reg.model_.load_state_dict(tabpfn_weights, strict=False)
    pred = np.asarray(reg.predict(Ee), dtype=float)
    true = y_eval.cpu().numpy().astype(float)
    return float(np.mean(np.abs(pred - true))), pred, true


# ---------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="resnet1d_wang",
                    choices=["resnet1d_wang", "inception1d"])
    ap.add_argument("--warm-start", default=os.path.join(
        BASE, "checkpoint_full",
        "1000_timesteps_resnet1d_wang_fds_full_adam_l1_0.001_32", "ckpt.best.pth.tar"))
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-ctx", type=int, default=512)
    ap.add_argument("--n-qry", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=8, help="evals without improvement")
    ap.add_argument("--ctx-sampling", default="uniform", choices=["uniform", "gauss"])
    ap.add_argument("--tabpfn-weights", default=None,
                    help="checkpoint from finetune_and_save.py. The frozen model the "
                         "encoder trains through becomes that FINE-TUNED TabPFN "
                         "instead of the stock one -- this is what distinguishes "
                         "encoders E2.1 / E2.2 from E2.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="enc_tabpfn")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    splits = ecg_data.to_device(ecg_data.load_split_arrays(), dev)
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    print(f"train={len(ytr)} val={len(yva)} test={len(splits['test'][1])}", flush=True)

    enc = build_encoder(args.encoder,
                        None if args.from_scratch else args.warm_start, dev)

    from tabpfn import TabPFNRegressor
    reg = TabPFNRegressor(model_path=CKPT_V25, device=dev, n_estimators=1,
                          auto_scale_n_estimators=False, random_state=args.seed,
                          differentiable_input=True, fit_mode="fit_preprocessors")

    # Weights are loaded after the first fit_with_differentiable_input builds
    # `model_`; see the `pending_weights` handling in the training loop below.
    pending_weights = None
    if args.tabpfn_weights:
        ck = torch.load(args.tabpfn_weights, map_location="cpu", weights_only=False)
        pending_weights = ck["state_dict"]
        print(f"frozen TabPFN will be the fine-tuned one from {args.tabpfn_weights} "
              f"(mode={ck.get('mode')}, {ck.get('tensors_changed')} tensors moved)",
              flush=True)

    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    ckpt_path = os.path.join(HERE, "ckpt", f"{args.tag}.pt")
    log_path = os.path.join(HERE, "logs", f"{args.tag}_curve.jsonl")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    eval_weights = pending_weights
    best_mae, bad_evals, frozen = float("inf"), 0, False
    t0 = time.time()

    for step in range(1, args.steps + 1):
        c_idx, q_idx = sample_episode(ytr, args.n_ctx, args.n_qry, gen,
                                      mode=args.ctx_sampling)
        Xc, Xq = enc(Xtr[c_idx]), enc(Xtr[q_idx])
        yc, yq = ytr[c_idx], ytr[q_idx]

        reg.fit_with_differentiable_input(Xc, yc)
        if pending_weights is not None:
            missing, unexpected = reg.model_.load_state_dict(pending_weights, strict=False)
            print(f"loaded fine-tuned TabPFN weights "
                  f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
            pending_weights = None
        if not frozen:
            # Freeze TabPFN after the first fit builds `models_`; only the encoder
            # is trained. Done once, after init, so nothing is left trainable.
            for p in reg.model_.parameters():
                p.requires_grad_(False)
            n_enc = sum(p.numel() for p in enc.parameters() if p.requires_grad)
            n_pfn = sum(p.numel() for p in reg.model_.parameters())
            print(f"encoder trainable params={n_enc:,}  TabPFN frozen params={n_pfn:,}",
                  flush=True)
            frozen = True

        logits, _, _ = reg.forward(Xq, use_inference_mode=True)
        yq_std = (yq - reg.y_train_mean_) / reg.y_train_std_
        loss = reg.znorm_space_bardist_.to(dev)(
            logits.transpose(0, 1).float(), yq_std.float()).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step % 50 == 0:
            print(f"step {step:5d}/{args.steps}  loss={loss.item():.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            mae, _, _ = evaluate(enc, Xtr, ytr, Xva, yva, dev, seed=args.seed,
                                 tabpfn_weights=eval_weights)
            improved = mae < best_mae - 1e-4
            print(f"  [eval] step {step}  val MAE={mae:.4f}"
                  f"{'  <-- best' if improved else ''}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": float(loss.item()),
                                    "val_mae": mae, "tag": args.tag}) + "\n")
            if improved:
                best_mae, bad_evals = mae, 0
                torch.save({"state_dict": enc.state_dict(), "step": step,
                            "val_mae": mae, "args": vars(args)}, ckpt_path)
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    print(f"early stop at step {step} "
                          f"({bad_evals} evals without improvement)", flush=True)
                    break

    print(f"\nBEST val MAE={best_mae:.4f} -> {ckpt_path}", flush=True)
    print(f"ENCODER_TRAINING_COMPLETE tag={args.tag}", flush=True)


if __name__ == "__main__":
    main()
