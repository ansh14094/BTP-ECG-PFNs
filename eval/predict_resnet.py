#!/usr/bin/env python3
"""Task A: direct ResNet predictions on PTB-XL, written in the harness format.

The ResNet baselines are the comparison point for everything else in Table 2, so
they must be measured on the same test records, with the same metric code, as the
in-context-learning arms. This script loads a checkpoint trained by ../train.py,
predicts age for the OFFICIAL PTB-XL test split, and writes

    imbreg/results/preds/ptbxl_resnet_<tag>_fixed_<arm>_s<seed>.csv

with columns id, y_true, y_pred, split -- exactly what eval/compute_metrics.py
consumes, so no metric is computed twice by two different pieces of code.

The official partition is used deliberately. The `dir` protocol resamples a balanced
test set out of ALL records, and because every encoder in this study was trained on
the official train split, roughly 80% of that resampled test set consists of records
the encoder was fitted on (see results/STATUS.md). The official split's test records
were seen by nothing.

    python3 eval/predict_resnet.py --ckpt <path> --arm vanilla --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ecg_tabpfn"))

import ecg_data  # noqa: E402
from resnet1d_wang_fds import resnet1d_wang_fds  # noqa: E402

PRED = os.path.join(ROOT, "imbreg", "results", "preds")


def load_model(ckpt_path: str, device: str) -> nn.Module:
    """Rebuild the trained regressor, head included -- we want its age output."""
    model = resnet1d_wang_fds(input_channels=12, fds=False)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    # train.py wraps the model in DataParallel, which prefixes every key.
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "running_" not in k and "num_batches" not in k]
    if missing:
        raise SystemExit(f"checkpoint does not fit the model; missing {missing[:6]}")
    print(f"loaded {ckpt_path} (epoch {ck.get('epoch')}, unexpected={len(unexpected)})")
    return model.to(device).eval()


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        out.append(model(X[i:i + batch]).squeeze(-1).float().cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", required=True, help="vanilla | ranksim_conr")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="taskA")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    splits = ecg_data.load_split_arrays()
    model = load_model(args.ckpt, args.device)

    # ids are offsets into train+val+test concatenated in that order, matching
    # imbreg/data.py:load_embedding_npz, so a prediction here lines up with the
    # same record in the embedding datasets.
    start = len(splits["train"][1]) + len(splits["val"][1])
    X, y = splits["test"]
    pred = predict(model, torch.from_numpy(X).float().to(args.device))

    os.makedirs(PRED, exist_ok=True)
    out = os.path.join(PRED,
                       f"ptbxl_resnet_{args.tag}_fixed_{args.arm}_s{args.seed}.csv")
    pd.DataFrame({"id": np.arange(start, start + len(y)), "y_true": y.astype(float),
                  "y_pred": pred.astype(float), "split": "test"}).to_csv(out, index=False)
    mae = float(np.mean(np.abs(pred - y)))
    print(f"wrote {out}  n={len(y)}  MAE={mae:.4f}")


if __name__ == "__main__":
    main()
