#!/usr/bin/env python3
"""Stage 1, TabICL edition: train a raw-ECG encoder end-to-end through a FROZEN TabICL.

This is `train_encoder.py` with the in-context body swapped from TabPFN-2.5 to
TabICL, so that encoders can be trained through each of the three prior-level
checkpoints and compared:

    E3  body = stock released TabICL
    E4  body = TabICL continued-pretrained with GAUSSIAN context ordering
    E5  body = TabICL continued-pretrained with MIXED (half Gaussian) ordering

The bodies differ only in a 3000-step continuation on TabICL's OWN synthetic prior.
None of them has ever seen PTB-XL, which is the point: the earlier TabPFN version of
this experiment (E2.1/E2.2) used bodies that had been fine-tuned on PTB-XL itself,
and the stated reason those failed was that the encoder was co-adapting to a body
already specialised to the data. That explanation cannot apply here.

Mechanism
---------
TabPFN needs its prompt-tuning API to accept differentiable inputs. TabICL does not:
`TabICL.forward(X, y_train)` is an ordinary nn.Module forward over a float tensor, so
gradients reach the encoder with no special path. The raw module is used directly
rather than the sklearn wrapper, because the wrapper's preprocessing goes through
numpy and would sever the graph.

    quantiles = body(X, y_train)                    # (B, n_qry, Q) raw quantiles
    loss      = pinball(quantiles, y_qry_std, alpha)
    loss.backward()                                 # reaches the encoder

X is (1, n_ctx + n_qry, D) with the context rows FIRST, because TabICL splits
context from query positionally and takes `train_size` from `y_train.shape[1]`.
Targets are z-normalised by the CONTEXT's mean and standard deviation, and the query
targets use that same scaler -- copied from `tabicl/_finetune/data.py`, which is what
the model was trained to expect.

Episode sampling is UNIFORM, as it was for E2, so the encoder stays an unbiased
feature extractor and the only thing that varies across E3/E4/E5 is the body.

    python3 train_encoder_tabicl.py --body <ckpt|stock> --tag enc_ticl_gauss
"""
from __future__ import annotations

import argparse
import glob
import inspect
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

PRIOR_RUNS = os.path.join(BASE, "tabicl_prior", "runs")


def stock_ckpt() -> str:
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--jingang--TabICL/snapshots/*/tabicl-regressor-v2-*.ckpt"))
    if not hits:
        raise FileNotFoundError("stock TabICL checkpoint not in the HF cache")
    return hits[0]


def resolve_body(spec: str) -> str:
    """'stock' | 'gauss' | 'uniform' | 'mixed' | an explicit .ckpt path."""
    if spec == "stock":
        return stock_ckpt()
    if spec in ("gauss", "uniform", "mixed"):
        return os.path.join(PRIOR_RUNS, spec, "step-3000.ckpt")
    return spec


def load_body(path: str, device: str) -> nn.Module:
    """Build the raw TabICL module from a checkpoint and freeze it."""
    from tabicl._model.tabicl import TabICL

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    allowed = set(inspect.signature(TabICL.__init__).parameters) - {"self"}
    body = TabICL(**{k: v for k, v in cfg.items() if k in allowed})
    body.load_state_dict(ck["state_dict"], strict=True)
    for p in body.parameters():
        p.requires_grad_(False)
    body.to(device)
    # Train mode selects TabICL's raw-quantile path, which is what the pinball loss
    # consumes; the weights are frozen either way.
    body.train()
    print(f"frozen TabICL body from {path}  "
          f"({sum(p.numel() for p in body.parameters()):,} params)", flush=True)
    return body


def build_encoder(name: str, warm_start: str | None, device: str) -> nn.Module:
    """Identical to train_encoder.py, so E3-E5 share E1/E2's backbone and warm start."""
    model = (resnet1d_wang_fds(input_channels=12, fds=False) if name == "resnet1d_wang"
             else inception1d_fds(input_channels=12, fds=False))
    if warm_start:
        ckpt = torch.load(warm_start, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"warm start from {warm_start} "
              f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    model.linear = nn.Identity()
    return model.to(device)


def sample_episode(n: int, n_ctx: int, n_qry: int, gen: torch.Generator, device: str):
    """Uniform episode: query drawn first, context from what is left."""
    perm = torch.randperm(n, device=device, generator=gen)
    return perm[n_qry:n_qry + n_ctx], perm[:n_qry]


@torch.no_grad()
def encode_all(enc: nn.Module, X: torch.Tensor, batch: int = 256) -> torch.Tensor:
    enc.eval()
    out = [enc(X[i:i + batch]) for i in range(0, len(X), batch)]
    enc.train()
    return torch.cat(out, dim=0)


@torch.no_grad()
def evaluate(enc, X_ctx, y_ctx, X_eval, y_eval, device, body_path, n_estimators=4, seed=0):
    """Score through the ordinary TabICL inference path, using the same body.

    A fresh TabICLRegressor is used rather than the training module so evaluation
    cannot be affected by train-mode behaviour, and so the number reported is one the
    downstream harness would reproduce.
    """
    from tabicl import TabICLRegressor

    Ec = encode_all(enc, X_ctx).float().cpu().numpy()
    Ee = encode_all(enc, X_eval).float().cpu().numpy()
    reg = TabICLRegressor(model_path=body_path, n_estimators=n_estimators,
                          device=device, random_state=seed)
    reg.fit(Ec, y_ctx.cpu().numpy())
    pred = np.asarray(reg.predict(Ee), dtype=float)
    return float(np.mean(np.abs(pred - y_eval.cpu().numpy().astype(float))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="resnet1d_wang",
                    choices=["resnet1d_wang", "inception1d"])
    # The SAME warm start E1 and E2 used. E3-E5 must begin from these weights or
    # the comparison measures the starting point rather than the body.
    ap.add_argument("--warm-start", default=os.path.join(
        BASE, "checkpoint_full",
        "1000_timesteps_resnet1d_wang_fds_full_adam_l1_0.001_32", "ckpt.best.pth.tar"))
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--body", default="stock",
                    help="stock | gauss | uniform | mixed | path to a .ckpt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-ctx", type=int, default=512)
    ap.add_argument("--n-qry", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="enc_ticl_stock")
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
    body_path = resolve_body(args.body)
    body = load_body(body_path, dev)

    from tabicl._finetune.regressor import _pinball_loss

    n_enc = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"encoder trainable params={n_enc:,}", flush=True)

    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    ckpt_path = os.path.join(HERE, "ckpt", f"{args.tag}.pt")
    log_path = os.path.join(HERE, "logs", f"{args.tag}_curve.jsonl")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    best_mae, bad_evals = float("inf"), 0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        c_idx, q_idx = sample_episode(len(ytr), args.n_ctx, args.n_qry, gen, dev)
        Ec, Eq = enc(Xtr[c_idx]), enc(Xtr[q_idx])
        yc, yq = ytr[c_idx].float(), ytr[q_idx].float()

        # Context first: TabICL splits positionally and reads train_size off y_train.
        X = torch.cat([Ec, Eq], dim=0).unsqueeze(0)
        mean, std = yc.mean(), yc.std().clamp_min(1e-8)
        y_train = ((yc - mean) / std).unsqueeze(0)
        y_qry = ((yq - mean) / std).unsqueeze(0)

        quantiles = body(X, y_train)
        q = quantiles.shape[-1]
        alpha = torch.linspace(0.0, 1.0, q + 2, device=quantiles.device,
                               dtype=quantiles.dtype)[1:-1]
        loss = _pinball_loss(quantiles, y_qry, alpha)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step % 50 == 0:
            print(f"step {step:5d}/{args.steps}  loss={loss.item():.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            mae = evaluate(enc, Xtr, ytr, Xva, yva, dev, body_path, seed=args.seed)
            improved = mae < best_mae - 1e-4
            print(f"  [eval] step {step}  val MAE={mae:.4f}"
                  f"{'  <-- best' if improved else ''}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": float(loss.item()),
                                    "val_mae": mae, "tag": args.tag,
                                    "body": body_path}) + "\n")
            if improved:
                best_mae, bad_evals = mae, 0
                torch.save({"state_dict": enc.state_dict(), "step": step,
                            "val_mae": mae, "body": body_path,
                            "args": vars(args)}, ckpt_path)
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    print(f"early stop at step {step}", flush=True)
                    break

    print(f"\nBEST val MAE={best_mae:.4f} -> {ckpt_path}", flush=True)
    print(f"ENCODER_TRAINING_COMPLETE tag={args.tag}", flush=True)


if __name__ == "__main__":
    main()
