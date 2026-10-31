#!/usr/bin/env python3
"""
Builds the `<data_dir>/1000_timesteps.csv` metadata file that datasets.py / train.py
expect (columns: split, age, path, index), pointing at the pre-processed PTB-XL
arrays in SOURCE_DIR. The .npy arrays are used in place (not copied); `path` is
stored as an absolute path so `--data_dir` can be any local folder.
"""
import os
import numpy as np
import pandas as pd

SOURCE_DIR = "/mnt/storage/Harinandan shukla Phd/ECG_project/DATASET/ptbxl_data"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_CSV = os.path.join(OUT_DIR, "1000_timesteps.csv")

SPLITS = {
    "train": "x_train_ptbxl.npy",
    "val": "x_val_ptbxl.npy",
    "test": "x_test_ptbxl.npy",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for split, x_name in SPLITS.items():
        x_path = os.path.join(SOURCE_DIR, x_name)
        y_path = os.path.join(SOURCE_DIR, x_name.replace("x_", "y_", 1))

        x_shape = np.load(x_path, mmap_mode="r").shape
        y = np.load(y_path)
        assert x_shape[0] == len(y), f"{split}: x has {x_shape[0]} rows, y has {len(y)}"

        for i, age in enumerate(y):
            rows.append({"split": split, "age": float(age), "path": x_path, "index": i})
        print(f"{split}: {len(y)} samples from {x_path}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
