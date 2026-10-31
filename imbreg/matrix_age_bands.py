#!/usr/bin/env python3
"""The encoder x body matrix, as mean absolute error split by age band.

Rows are encoders, columns are the frozen TabICL body used at prediction time:

    E1  ResNet1d-Wang trained directly on age (L1 loss)
    E2  same backbone trained through the frozen stock TabPFN-2.5
    E3  same backbone trained through the frozen stock TabICL
    E4  same backbone trained through TabICL continued-pretrained, GAUSSIAN ordering
    E5  same backbone trained through TabICL continued-pretrained, MIXED ordering

    stock / uniform / gauss / mixed  -- the body at evaluation, zero-shot

No body has seen PTB-XL: uniform, gauss and mixed differ from stock only by a
3000-step continuation on TabICL's own synthetic prior. `uniform` is the matched
control for `gauss` -- same prior, same steps, unbiased ordering -- so the pair
separates "the Gaussian ordering helped" from "3000 more steps helped".

Reads the per-prediction CSVs the harness saved; nothing is re-run.

    python3 matrix_age_bands.py
"""
from __future__ import annotations
import glob, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "results", "preds")
TAG = "dir_matrix"

BANDS = [("age < 30", lambda y: y < 30),
         ("30-60",    lambda y: (y >= 30) & (y < 60)),
         ("age >= 60", lambda y: y >= 60)]
ENCODERS = [("E1", "ptbxl_l1_enc"), ("E2", "ptbxl_tabpfn_enc"),
            ("E3", "ptbxl_enc_e3"), ("E4", "ptbxl_enc_e4"), ("E5", "ptbxl_enc_e5")]
BODIES = [("stock", "ticl_zeroshot"), ("uniform", "ticl_prior_uniform"),
          ("gauss", "ticl_prior_gauss"), ("mixed", "ticl_prior_mixed")]
SEEDS = [0, 1, 2, 3, 4]


def cell(dsname: str, arm: str):
    """Per-seed (overall MAE, per-band MAE) for one encoder x body cell."""
    overall, bands = [], {b: [] for b, _ in BANDS}
    for s in SEEDS:
        f = os.path.join(PRED, f"{dsname}_{TAG}_dir_{arm}_s{s}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        y, p = d.y_true.to_numpy(float), d.y_pred.to_numpy(float)
        overall.append(float(np.mean(np.abs(p - y))))
        for bname, fn in BANDS:
            m = fn(y)
            if m.sum():
                bands[bname].append(float(np.mean(np.abs(p[m] - y[m]))))
    return overall, bands


def main() -> None:
    grid = {}
    for ecode, dsname in ENCODERS:
        for bcode, arm in BODIES:
            grid[(ecode, bcode)] = cell(dsname, arm)

    for title, key in [("OVERALL mean absolute error (years)", None)] + \
                      [(f"MAE, {b}", b) for b, _ in BANDS]:
        print(f"\n{'='*74}\n{title}   5 seeds, zero-shot")
        hdr = f"  {'encoder':10s}" + "".join(f"{b:>14s}" for b, _ in BODIES)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for ecode, _ in ENCODERS:
            cells = []
            for bcode, _ in BODIES:
                ov, bd = grid[(ecode, bcode)]
                v = ov if key is None else bd[key]
                cells.append(f"{np.mean(v):10.3f}    " if v else f"{'--':>10s}    ")
            print(f"  {ecode:10s}" + "".join(cells))

    print(f"\n{'='*74}\nPaired gauss vs uniform body, per encoder (the controlled comparison)")
    print(f"  {'encoder':10s}{'overall':>22s}" + "".join(f"{b:>22s}" for b, _ in BANDS))
    for ecode, _ in ENCODERS:
        ou, bu = grid[(ecode, "uniform")]
        og, bg = grid[(ecode, "gauss")]
        row = f"  {ecode:10s}"
        for u, g in [(ou, og)] + [(bu[b], bg[b]) for b, _ in BANDS]:
            if len(u) == len(g) and len(u):
                d = np.array(g) - np.array(u)
                row += f"{d.mean():+15.3f} ({int((d < 0).sum())}/{len(d)})"
            else:
                row += f"{'--':>22s}"
        print(row)

    missing = [(e, b) for (e, b), (ov, _) in grid.items() if len(ov) < len(SEEDS)]
    if missing:
        print(f"\nINCOMPLETE cells ({len(missing)}): {sorted(missing)}")


if __name__ == "__main__":
    main()
