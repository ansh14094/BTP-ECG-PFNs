"""AgeDB-DIR loading for the encoder + TabPFN experiments.

Mirrors ecg_tabpfn/ecg_data.py so the two modalities run through the same pipeline.
Images are variable-sized square face crops (75-239 px); they are resized once to
112x112 and held in memory as uint8 (~620 MB for 16,488 images), which removes the
DataLoader from the training loop -- an episode then costs one index operation, and
this loop resamples a fresh episode every step.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = "/mnt/storage/dataset_LBP's/agedb_data_orginal"
CSV = os.path.join(ROOT, "agedb.csv")
IMG_SIZE = 112
# ImageNet statistics, since the encoder is initialised from ImageNet weights.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_split_arrays(cache: str | None = None):
    """Return {split: (X_uint8 (N,3,112,112), y_float (N,))}.

    AgeDB-DIR's val and test sets are already balanced across ages by construction
    (Yang et al., ICML 2021), so the official split is used as-is -- unlike PTB-XL,
    no rebalancing is needed to make tail error measurable.
    """
    cache = cache or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "results", f"agedb_{IMG_SIZE}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {s: (d[f"X_{s}"], d[f"y_{s}"]) for s in ("train", "val", "test")}

    df = pd.read_csv(CSV)
    out = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].reset_index(drop=True)
        X = np.empty((len(sub), 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        for i, p in enumerate(sub["path"]):
            im = Image.open(os.path.join(ROOT, p)).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            X[i] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
        out[split] = (X, sub["age"].to_numpy(dtype=np.float32))
        print(f"  {split}: {X.shape}", flush=True)
    np.savez(cache, **{f"X_{s}": v[0] for s, v in out.items()},
             **{f"y_{s}": v[1] for s, v in out.items()})
    return out


def to_device(splits: dict, device: str = "cuda"):
    """Images stay uint8 on the GPU; normalisation happens per batch in `normalize`."""
    return {k: (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device))
            for k, (X, y) in splits.items()}


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 (B,3,H,W) -> ImageNet-normalised float32."""
    x = x_uint8.float().div_(255.0)
    m = torch.as_tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.as_tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s
