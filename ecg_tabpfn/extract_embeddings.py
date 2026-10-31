#!/usr/bin/env python3
"""Stage 2: freeze the trained encoder and dump 128-d embeddings for every split.

Produces an .npz consumable by the imbalanced-regression harness in ../imbreg, so the
ECG experiment runs through exactly the same arms, metrics and aggregation as the
tabular datasets rather than a parallel implementation that could drift.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, BASE)

import ecg_data  # noqa: E402
from train_encoder import build_encoder, encode_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt", "enc_main.pt"),
                    help="encoder checkpoint from train_encoder.py; "
                         "omit --ckpt and pass --baseline-ckpt for the L1 backbone")
    ap.add_argument("--baseline-ckpt", default=None,
                    help="use the original L1-trained backbone instead (the control)")
    ap.add_argument("--encoder", default="resnet1d_wang")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "emb_enc_main.npz"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    splits = ecg_data.to_device(ecg_data.load_split_arrays(), dev)

    if args.baseline_ckpt:
        # The control encoder: trained on an L1 age loss, never aware of TabPFN.
        enc = build_encoder(args.encoder, args.baseline_ckpt, dev)
    else:
        enc = build_encoder(args.encoder, None, dev)
        ck = torch.load(args.ckpt, map_location="cpu")
        enc.load_state_dict(ck["state_dict"])
        print(f"loaded {args.ckpt} (step {ck.get('step')}, val MAE {ck.get('val_mae'):.4f})")
    enc.eval()

    out = {}
    for split, (X, y) in splits.items():
        E = encode_all(enc, X).float().cpu().numpy()
        out[f"X_{split}"] = E
        out[f"y_{split}"] = y.cpu().numpy().astype(float)
        print(f"{split:6s} embeddings {E.shape}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
