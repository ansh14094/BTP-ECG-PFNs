#!/usr/bin/env python3
"""Stage 1 for AgeDB-DIR: train a face-image encoder, either through a frozen
TabPFN-2.5 or on a plain L1 age loss.

Both objectives live in one script on purpose. For PTB-XL the control encoder was an
existing checkpoint trained at some earlier time with its own schedule, so the
encoder comparison there was not perfectly matched. Here both encoders start from the
same ImageNet weights and get the same optimiser, schedule, step budget and episode
sampling; only the loss differs:

    --objective l1       ResNet-18 + linear head, L1 on age        (control)
    --objective tabpfn   ResNet-18 features -> frozen TabPFN-2.5,  (treatment)
                         bar-distribution NLL on query images

That makes "does training through TabPFN produce better features than training for a
linear head?" a clean, single-variable question.

Gradients reach the encoder through TabPFN via the library's prompt-tuning path
(`differentiable_input=True`), verified before this was written.
"""
from __future__ import annotations

import argparse, json, os, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agedb_data  # noqa: E402

CKPT_V25 = "tabpfn-v2.5-regressor-v2.5_default.ckpt"


def build_encoder(device: str, pretrained: bool = True):
    """ImageNet ResNet-18 with the classifier removed -> 512-d embedding."""
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m.to(device), feat_dim


def _gumbel_top_k(log_w, k, gen):
    u = torch.rand(log_w.shape, device=log_w.device, generator=gen).clamp_(1e-20, 1.0)
    return torch.topk(log_w + (-torch.log(-torch.log(u))), k).indices


def sample_episode(y, n_ctx, n_qry, gen, mode="uniform", sigma_range=(0.15, 1.5)):
    """Query drawn first and unweighted; context optionally Gaussian-weighted in age."""
    n = y.shape[0]
    perm = torch.randperm(n, device=y.device, generator=gen)
    q_idx, pool = perm[:n_qry], perm[n_qry:]
    if mode == "uniform":
        return pool[:n_ctx], q_idx
    y_pool = y[pool]
    std = y_pool.std()
    if std < 1e-8:
        return pool[:n_ctx], q_idx
    q = torch.rand(1, device=y.device, generator=gen).item() * 0.9 + 0.05
    mu = torch.quantile(y_pool, q)
    sigma = std * float(np.exp(np.random.default_rng(
        int(torch.randint(0, 2**31 - 1, (1,), device=y.device, generator=gen).item())
    ).uniform(np.log(sigma_range[0]), np.log(sigma_range[1]))))
    log_w = -0.5 * ((y_pool - mu) / sigma) ** 2
    return pool[_gumbel_top_k(log_w, min(n_ctx, len(pool)), gen)], q_idx


@torch.no_grad()
def encode_all(enc, X, batch=512):
    enc.eval()
    out = [enc(agedb_data.normalize(X[i:i + batch])) for i in range(0, len(X), batch)]
    enc.train()
    return torch.cat(out, 0)


@torch.no_grad()
def eval_tabpfn(enc, Xc, yc, Xe, ye, device, n_estimators=4, seed=0):
    """Score the encoder the way it will actually be used: features -> TabPFN."""
    from tabpfn import TabPFNRegressor
    Ec = encode_all(enc, Xc).float().cpu().numpy()
    Ee = encode_all(enc, Xe).float().cpu().numpy()
    reg = TabPFNRegressor(model_path=CKPT_V25, device=device, n_estimators=n_estimators,
                          auto_scale_n_estimators=False, random_state=seed)
    reg.fit(Ec, yc.cpu().numpy())
    pred = np.asarray(reg.predict(Ee), float)
    return float(np.mean(np.abs(pred - ye.cpu().numpy().astype(float))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="tabpfn", choices=["tabpfn", "l1"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--n-qry", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--ctx-sampling", default="uniform", choices=["uniform", "gauss"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tag = args.tag or f"enc_{args.objective}"

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    splits = agedb_data.to_device(agedb_data.load_split_arrays(), dev)
    Xtr, ytr = splits["train"]; Xva, yva = splits["val"]
    print(f"train={len(ytr)} val={len(yva)} test={len(splits['test'][1])}", flush=True)

    enc, feat_dim = build_encoder(dev)
    head = nn.Linear(feat_dim, 1).to(dev) if args.objective == "l1" else None
    params = list(enc.parameters()) + (list(head.parameters()) if head else [])
    print(f"encoder feat_dim={feat_dim} trainable={sum(p.numel() for p in params):,}", flush=True)

    reg = None
    if args.objective == "tabpfn":
        from tabpfn import TabPFNRegressor
        reg = TabPFNRegressor(model_path=CKPT_V25, device=dev, n_estimators=1,
                              auto_scale_n_estimators=False, random_state=args.seed,
                              differentiable_input=True, fit_mode="fit_preprocessors")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    ckpt_path = os.path.join(HERE, "ckpt", f"{tag}.pt")
    log_path = os.path.join(HERE, "logs", f"{tag}_curve.jsonl")

    best, bad, frozen = float("inf"), 0, False
    t0 = time.time()
    for step in range(1, args.steps + 1):
        c_idx, q_idx = sample_episode(ytr, args.n_ctx, args.n_qry, gen, mode=args.ctx_sampling)

        if args.objective == "l1":
            # Matched budget: the control sees the same images per step as the
            # treatment, just scored with L1 through a linear head.
            idx = torch.cat([c_idx, q_idx])
            pred = head(enc(agedb_data.normalize(Xtr[idx]))).squeeze(-1)
            loss = torch.abs(pred - ytr[idx]).mean()
        else:
            Xc = enc(agedb_data.normalize(Xtr[c_idx]))
            Xq = enc(agedb_data.normalize(Xtr[q_idx]))
            reg.fit_with_differentiable_input(Xc, ytr[c_idx])
            if not frozen:
                for p in reg.model_.parameters():
                    p.requires_grad_(False)
                print(f"TabPFN frozen: {sum(p.numel() for p in reg.model_.parameters()):,} params",
                      flush=True)
                frozen = True
            logits, _, _ = reg.forward(Xq, use_inference_mode=True)
            yq_std = (ytr[q_idx] - reg.y_train_mean_) / reg.y_train_std_
            loss = reg.znorm_space_bardist_.to(dev)(
                logits.transpose(0, 1).float(), yq_std.float()).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        if step % 50 == 0:
            print(f"step {step:5d}/{args.steps} loss={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            # Both objectives are scored the same way -- features into TabPFN -- so the
            # comparison is on the quantity we actually care about, not on each
            # objective's own training loss.
            mae = eval_tabpfn(enc, Xtr, ytr, Xva, yva, dev, seed=args.seed)
            imp = mae < best - 1e-4
            print(f"  [eval] step {step} val MAE(TabPFN)={mae:.4f}{'  <-- best' if imp else ''}",
                  flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": float(loss.item()),
                                    "val_mae": mae, "tag": tag}) + "\n")
            if imp:
                best, bad = mae, 0
                torch.save({"state_dict": enc.state_dict(), "step": step,
                            "val_mae": mae, "args": vars(args)}, ckpt_path)
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"early stop at step {step}", flush=True); break

    print(f"\nBEST val MAE={best:.4f} -> {ckpt_path}", flush=True)
    print(f"ENCODER_TRAINING_COMPLETE tag={tag}", flush=True)


if __name__ == "__main__":
    main()
