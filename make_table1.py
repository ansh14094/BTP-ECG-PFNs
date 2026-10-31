#!/usr/bin/env python3
"""Build Table 1: PTB-XL cumulative effect of the two improvements.

Rows are (TabPFN variant x encoder). Every number is computed from
imbreg/results/*.jsonl with the fine-tuning learning rate selected per seed on
validation, exactly as the other reports do. Cells whose runs have not completed are
printed as "pending" rather than omitted, so the table never looks finished when it
is not.
"""
from __future__ import annotations
import glob, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "imbreg", "results")

ENCODERS = {
    "ptbxl_l1_enc":     ("E1",   "Encoder trained directly (L1 age loss)"),
    "ptbxl_tabpfn_enc": ("E2",   "Tuned on frozen TabPFN"),
    "ptbxl_enc_e21":    ("E2.1", "Tuned on Gaussian-fine-tuned frozen TabPFN"),
    "ptbxl_enc_e22":    ("E2.2", "Tuned on Gauss+Uniform-fine-tuned frozen TabPFN"),
}
VARIANTS = {"zeroshot": "Frozen", "ft_uniform": "Uniform fine-tuning",
            "ft_gauss": "Gaussian fine-tuning", "ft_mixed": "Gaussian + Uniform fine-tuning"}
# The rows the study actually asks for, in order.
ROWS = [("zeroshot", "ptbxl_l1_enc"), ("zeroshot", "ptbxl_tabpfn_enc"),
        ("ft_uniform", "ptbxl_l1_enc"), ("ft_uniform", "ptbxl_tabpfn_enc"),
        ("ft_gauss", "ptbxl_l1_enc"), ("ft_gauss", "ptbxl_tabpfn_enc"),
        ("ft_gauss", "ptbxl_enc_e21"),
        ("ft_mixed", "ptbxl_l1_enc"), ("ft_mixed", "ptbxl_tabpfn_enc"),
        ("ft_mixed", "ptbxl_enc_e22")]


def load():
    rows = []
    for ds in ENCODERS:
        for p in glob.glob(os.path.join(RES, f"{ds}_*.jsonl")):
            for l in open(p):
                if l.strip():
                    rows.append(json.loads(l))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df.protocol == "dir") & (df.tag.str.startswith("dir_"))]
    df = df.drop_duplicates(subset=["dataset", "arm", "seed", "tag"], keep="last")
    ft, rest = df[df.arm.str.startswith("ft_")], df[~df.arm.str.startswith("ft_")]
    if ft.empty or "val_rmse" not in ft:
        return rest
    best = ft.sort_values("val_rmse").groupby(["dataset", "arm", "seed"], as_index=False).first()
    return pd.concat([rest, best], ignore_index=True)


def main():
    df = load()
    if df.empty:
        print("no results"); return
    base = None
    out = []
    for arm, ds in ROWS:
        x = df[(df.dataset == ds) & (df.arm == arm)]
        code, _ = ENCODERS[ds]
        if x.empty:
            out.append([VARIANTS[arm], code, "pending", "", "", 0]); continue
        mae = float(x.mae.mean())
        if base is None:
            base = mae
        out.append([VARIANTS[arm], code, f"{mae:.3f}",
                    "—" if mae == base else f"{base - mae:.3f}",
                    "—" if mae == base else f"{100*(base-mae)/base:.1f}%", len(x)])

    w = [32, 7, 9, 12, 14, 7]
    hdr = ["TabPFN-2.5", "Encoder", "MAE", "Abs. reduction", "Improvement", "seeds"]
    print("\nTable 1 — PTB-XL: cumulative effect of the two improvements "
          "(balanced evaluation, val-selected LR)\n")
    print("".join(h.ljust(x) for h, x in zip(hdr, w)))
    print("-" * sum(w))
    for r in out:
        print("".join(str(c).ljust(x) for c, x in zip(r, w)))
    pend = sum(1 for r in out if r[2] == "pending")
    print(f"\n{len(out)-pend}/{len(out)} cells complete"
          + (f"; {pend} pending" if pend else "; TABLE COMPLETE"))

    with open(os.path.join(HERE, "table1_ptbxl.json"), "w") as f:
        json.dump([dict(zip(hdr, r)) for r in out], f, indent=2)


if __name__ == "__main__":
    main()
