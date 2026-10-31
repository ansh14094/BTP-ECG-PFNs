#!/usr/bin/env python3
"""Emit the encoder x body matrix as two paste-ready CSV blocks.

Block A is the encoder axis at a FIXED prediction body (stock TabICL), so the only
thing varying down the rows is which frozen model the ECG encoder was trained
through. Block B is the full matrix: for each encoder, all four prediction bodies.

The distinction matters and is easy to lose in a spreadsheet: "Gaussian-prior
TabICL" means two different experiments depending on the axis. On the encoder axis
it is the model the encoder was trained THROUGH; on the body axis it is the model
that made the PREDICTION. Both column headers say which.

Numbers are pooled over all five seeds, matching the awk verification.
"""
from __future__ import annotations
import glob, os
import numpy as np

PRED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "preds")
TAG = "dir_matrix"
ENC = [("E1", "ptbxl_l1_enc",     "none (supervised L1 age loss)"),
       ("E2", "ptbxl_tabpfn_enc", "frozen stock TabPFN-2.5"),
       ("E3", "ptbxl_enc_e3",     "frozen stock TabICL"),
       ("E4", "ptbxl_enc_e4",     "frozen Gaussian-prior TabICL (3000 steps)"),
       ("E5", "ptbxl_enc_e5",     "frozen mixed-prior TabICL (3000 steps)")]
BODY = [("ticl_zeroshot",      "Stock TabICL"),
        ("ticl_prior_uniform", "Uniform-prior TabICL (control, 3000 steps)"),
        ("ticl_prior_gauss",   "Gaussian-prior TabICL (3000 steps)"),
        ("ticl_prior_mixed",   "Mixed-prior TabICL (3000 steps)")]


def maes(ds: str, arm: str) -> tuple[float, float, float, float]:
    """Pooled MAE over every seed: overall, <30, [30,60), >=60."""
    y, p = [], []
    for f in sorted(glob.glob(os.path.join(PRED, f"{ds}_{TAG}_dir_{arm}_s*.csv"))):
        a = np.loadtxt(f, delimiter=",", skiprows=1)
        y.append(a[:, 0]); p.append(a[:, 1])
    y, p = np.concatenate(y), np.concatenate(p)
    e = np.abs(p - y)
    return (e.mean(), e[y < 30].mean(), e[(y >= 30) & (y < 60)].mean(), e[y >= 60].mean())


def main() -> None:
    out = []
    out.append("BLOCK A - encoder axis. Prediction body is stock TabICL in every row;")
    out.append("only the model the ECG encoder was trained through changes.")
    out.append("Encoder,Encoder trained through,Age < 30 MAE,Age 30-60 MAE,Age >= 60 MAE,Overall MAE")
    for code, ds, trained in ENC:
        o, b1, b2, b3 = maes(ds, "ticl_zeroshot")
        out.append(f"{code},{trained},{b1:.3f},{b2:.3f},{b3:.3f},{o:.3f}")

    out.append("")
    out.append("BLOCK B - full matrix. Every encoder under every prediction body, zero-shot.")
    out.append("No body has ever seen PTB-XL.")
    out.append("Encoder,Encoder trained through,Prediction body,"
               "Age < 30 MAE,Age 30-60 MAE,Age >= 60 MAE,Overall MAE")
    for code, ds, trained in ENC:
        for arm, bname in BODY:
            o, b1, b2, b3 = maes(ds, arm)
            out.append(f"{code},{trained},{bname},{b1:.3f},{b2:.3f},{b3:.3f},{o:.3f}")

    out.append("")
    out.append("Test set: PTB-XL, DIR-rebalanced, 1499 patients per seed "
               "(320 age<30 / 600 age 30-60 / 579 age>=60), 5 seeds pooled.")
    out.append("Age exactly 60 is counted in the >=60 band. Lower MAE is better.")
    txt = "\n".join(out) + "\n"
    with open(os.path.join(os.path.dirname(PRED), "matrix_age_bands.csv"), "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
