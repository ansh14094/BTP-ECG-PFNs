"""PTB-XL ECG loading for the TabPFN encoder experiments.

The raw signals total ~363 MB as float32, which fits comfortably in GPU memory. Holding
them resident removes the DataLoader from the training loop entirely: an episode is
then just an index operation, which matters because this loop samples a fresh
context/query episode every single step rather than iterating a fixed batch order.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "data", "1000_timesteps.csv")


def load_split_arrays(csv_path: str = CSV):
    """Return {split: (X, y)} with X as (N, 12, 1000) float32 and y as (N,) float32.

    The CSV addresses rows inside per-split .npy files by (path, index); signals are
    stored (N, 1000, 12) and transposed here to channels-first for Conv1d.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["age"])
    out = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        arrays, ages = [], []
        for path, grp in sub.groupby("path"):
            mm = np.load(path, mmap_mode="r")
            idx = grp["index"].to_numpy(dtype=int)
            arrays.append(np.asarray(mm[idx], dtype=np.float32))
            ages.append(grp["age"].to_numpy(dtype=np.float32))
        X = np.concatenate(arrays, axis=0).transpose(0, 2, 1)   # (N, 12, 1000)
        y = np.concatenate(ages, axis=0)
        out[split] = (np.ascontiguousarray(X), y)
    return out


def to_device(splits: dict, device: str = "cuda"):
    """Move signals and ages onto the GPU once, up front."""
    return {
        k: (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device))
        for k, (X, y) in splits.items()
    }
