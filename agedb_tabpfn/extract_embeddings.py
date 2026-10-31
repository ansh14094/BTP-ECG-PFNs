#!/usr/bin/env python3
"""Stage 2 for AgeDB: freeze an encoder and dump 512-d embeddings per split,
in the .npz layout the imbreg harness consumes."""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import agedb_data                                   # noqa: E402
from train_encoder import build_encoder, encode_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    enc, _ = build_encoder(args.device, pretrained=False)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    enc.load_state_dict(ck["state_dict"])
    enc.eval()
    print(f"loaded {args.ckpt} (step {ck.get('step')}, val MAE {ck.get('val_mae'):.4f})")

    splits = agedb_data.to_device(agedb_data.load_split_arrays(), args.device)
    out = {}
    for s, (X, y) in splits.items():
        E = encode_all(enc, X).float().cpu().numpy()
        out[f"X_{s}"], out[f"y_{s}"] = E, y.cpu().numpy().astype(float)
        print(f"  {s:6s} {E.shape}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
