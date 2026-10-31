#!/usr/bin/env python3
"""Use Timestral's encoder as the ECG feature extractor instead of our own ResNet.

Timestral is a forecasting model, so its quantile head predicts the next samples --
not age. But everything below that head is a general time-series encoder, and the
question is whether that representation transfers to PTB-XL age regression.

What this does, per record:
  * take the 12 leads as one multivariate series (group_ids all equal, so leads
    attend to each other -- the same `multivariate` mode the forecasting benchmark
    used, which was the stronger of the two there)
  * run `representation` -> `transformer`, exactly as `_predict_context_only` does,
    and keep the ENCODER tokens instead of passing them to the quantile head
  * mean-pool over time patches, then over leads, giving one d_model vector

The encoder is frozen and has never seen an age label, so this is an honest test of
the representation, not of a trained predictor. Note that makes it a deliberately
unequal comparison against E1, which was trained on age directly -- see the log.

Emits two .npz files consumable by ../imbreg:
  emb_timestral.npz      full d_model features (768 for the 209M model)
  emb_timestral_pca.npz  reduced to 128 dims to match E1/E2's width exactly.
                         PCA is fitted on TRAIN ONLY and applied to val/test, so no
                         test information reaches the transform.

    python3 timestral_encoder.py --model mistralai/Timestral-209M
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecg_data  # noqa: E402

PATCH_MULTIPLE_LEN = 992  # 31 patches of 32; 1000 is not a multiple of patch_size


@torch.no_grad()
def encode_batch(model, x: torch.Tensor) -> torch.Tensor:
    """(B, 12, T) raw ECG -> (B, d_model) pooled encoder representation.

    Mirrors TimestralModel._predict_context_only up to the transformer, then stops:
    the quantile head is forecasting-specific and is deliberately not applied.
    """
    from timestral.masks import (build_forecast_positions, build_forecast_time_mask,
                                 build_two_stream_variate_mask, compute_enc_key_validity,
                                 group_first_observed_patch)
    from timestral.layers import TwoStreamMask

    c = model.config
    ps = c.patch_size
    b, n_leads, t = x.shape
    assert t % ps == 0, f"context {t} must be a multiple of patch_size {ps}"
    ll, h = t // ps, 1

    # Pack (B, leads, T) into Timestral's (R, T) row layout, one group per record so
    # that a record's 12 leads attend each other but records stay independent.
    inputs = x.reshape(b * n_leads, t)
    observed = torch.ones_like(inputs, dtype=torch.bool)
    group_ids = torch.arange(b, device=x.device).repeat_interleave(n_leads)

    scaled, loc, scale = model.scale(inputs, observed)
    x_enc = model.representation(scaled, observed)
    x_dec = model.representation.forecast_tokens(b * n_leads, h)

    obs_per_patch = observed.unflatten(-1, (ll, ps)).any(dim=-1)
    key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
    time_mask = build_forecast_time_mask(ll, h, key_valid, x.device)
    variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h)
    unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
    first_obs = group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
    positions = build_forecast_positions(ll, h, first_obs)

    xx = torch.cat([x_enc, x_dec], dim=1)
    mask = time_mask
    if c.split_context_horizon_attn:
        mask = TwoStreamMask(mask=time_mask, enc_len=x_enc.shape[1])
    hidden = model.transformer(xx, mask, variate_mask, positions)

    enc_tokens = hidden[:, : x_enc.shape[1], :]          # (B*leads, ll, d_model)
    pooled = enc_tokens.float().mean(dim=1)              # over time patches
    return pooled.reshape(b, n_leads, -1).mean(dim=1)    # over leads -> (B, d_model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Timestral-209M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "emb_timestral.npz"))
    ap.add_argument("--out-pca", default=os.path.join(HERE, "results", "emb_timestral_pca.npz"))
    args = ap.parse_args()

    from timestral import TimestralForecaster

    t0 = time.time()
    fc = TimestralForecaster.from_pretrained(args.model, device=args.device)
    model = fc.model if hasattr(fc, "model") else fc
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"loaded {args.model} in {time.time() - t0:.1f}s  params={n_par/1e6:.1f}M  "
          f"d_model={model.config.d_model} patch={model.config.patch_size}", flush=True)

    splits = ecg_data.load_split_arrays()
    out: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        if split not in splits:
            continue
        X, y = splits[split]
        X = X[:, :, :PATCH_MULTIPLE_LEN]
        embs = []
        t1 = time.time()
        for i in range(0, len(X), args.batch):
            xb = torch.from_numpy(X[i:i + args.batch]).to(args.device)
            embs.append(encode_batch(model, xb).cpu().numpy())
            if i and i % (args.batch * 50) == 0:
                print(f"  {split} {i}/{len(X)}  ({time.time() - t1:.0f}s)", flush=True)
        E = np.concatenate(embs, axis=0).astype(np.float32)
        out[f"X_{split}"] = E
        out[f"y_{split}"] = y.astype(float)
        print(f"{split:6s} {E.shape}  ({time.time() - t1:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    print("wrote", args.out, flush=True)

    # Dimension-matched variant. Fitting PCA on train only matters: fitting it on all
    # splits would leak test structure into the features and silently inflate results.
    from sklearn.decomposition import PCA

    pca = PCA(n_components=args.pca_dim, random_state=0).fit(out["X_train"])
    red = {f"y_{s}": out[f"y_{s}"] for s in ("train", "val", "test") if f"y_{s}" in out}
    for s in ("train", "val", "test"):
        if f"X_{s}" in out:
            red[f"X_{s}"] = pca.transform(out[f"X_{s}"]).astype(np.float32)
    np.savez(args.out_pca, **red)
    print(f"wrote {args.out_pca}  (PCA {args.pca_dim} dims fitted on train only, "
          f"explained variance {pca.explained_variance_ratio_.sum():.3f})", flush=True)


if __name__ == "__main__":
    main()
