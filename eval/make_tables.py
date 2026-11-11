#!/usr/bin/env python3
"""Generate the paper's LaTeX tables from saved metrics. Nothing is typed by hand.

Reads `imbreg/results/metrics/all_metrics.csv` (produced by compute_metrics.py from
the prediction files) and writes `paper/tables/*.tex`.

Cells report mean +- standard deviation over the 5 seeds, to 3 decimals, or 4 for
Communities whose target is on a 0-1 scale. A cell with no data is written TODO,
never estimated and never carried over from an earlier run.

Learning-rate selection. Fine-tuning arms were swept over 1e-5, 3e-5, 1e-4 under
separate tags. For each (dataset, arm, seed) the tag with the lowest VALIDATION
RMSE is selected. Validation scores live only in the run JSONL, so that one file is
read for model selection; every reported test number still comes from the prediction
files via compute_metrics.py.

Protocols. PTB-XL tables are built on the `fixed` protocol -- the official PTB-XL
partition -- because the `dir` protocol resamples its test set out of all records
and so re-uses roughly 80% of the encoder's own training data. Table 3 deliberately
shows both, since the gap between them is the size of that contamination.

    python3 eval/make_tables.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MET = os.path.join(ROOT, "imbreg", "results", "metrics", "all_metrics.csv")
JSONL = os.path.join(ROOT, "imbreg", "results")
OUT = os.path.join(ROOT, "paper", "tables")

SEEDS = (0, 1, 2, 3, 4)
# Arms whose runs were swept over learning rates under separate tags, so a cell
# must SELECT one tag on validation RMSE rather than average the sweep.
# `ft_target_balanced` (Task B2) was missing here, which silently averaged its
# three learning rates while every other fine-tuning row selected one -- an
# inconsistent model-selection rule inside the same table.
LR_SWEPT = {"ft_gauss", "ft_uniform", "ft_mixed", "ticl_ft_gauss",
            "ticl_ft_uniform", "ft_target_balanced"}

# Diagnostic runs that must never enter a reported cell. `taskA_probe` is a
# single-seed re-evaluation of the ORIGINAL pre-existing ResNet checkpoint, kept
# for the reference-value sanity check (it reproduces 7.5485). It is not one of
# the five Task A seeds, and averaging it into seed 0 biased the baseline.
EXCLUDE_TAGS = {"taskA_probe"}
TODO = r"\textit{TODO}"


# ----------------------------------------------------------------- selection
def best_tags() -> dict:
    """(dataset, protocol, arm, seed) -> tag with the lowest validation RMSE."""
    best: dict = {}
    for f in glob.glob(os.path.join(JSONL, "*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "arm" not in r or r.get("val_rmse") is None:
                continue
            k = (r.get("dataset"), r.get("protocol"), r["arm"], int(r.get("seed", -1)))
            v = float(r["val_rmse"])
            if k not in best or v < best[k][0]:
                best[k] = (v, r.get("tag"))
    return {k: t for k, (_, t) in best.items()}


BEST = best_tags()


def cell(df: pd.DataFrame, dataset: str, protocol: str, arm: str,
         metric: str, tag: str | None = None) -> np.ndarray:
    """Per-seed values for one cell, after learning-rate selection where it applies."""
    sub = df[(df.dataset == dataset) & (df.protocol == protocol) & (df.arm == arm)]
    sub = sub[~sub.tag.isin(EXCLUDE_TAGS)]
    if tag is not None:
        sub = sub[sub.tag == tag]
    if sub.empty or metric not in sub.columns:
        return np.array([])
    vals = []
    for s in SEEDS:
        rows = sub[sub.seed == s]
        if rows.empty:
            continue
        if len(rows) > 1 and arm in LR_SWEPT:
            want = BEST.get((dataset, protocol, arm, s))
            if want is not None and (rows.tag == want).any():
                rows = rows[rows.tag == want]
        v = rows[metric].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v):
            vals.append(float(v[0]) if len(v) == 1 else float(np.mean(v)))
    return np.asarray(vals)


def fmt(vals: np.ndarray, nd: int) -> str:
    if len(vals) == 0:
        return TODO
    return f"{vals.mean():.{nd}f}\\,$\\pm$\\,{vals.std(ddof=0):.{nd}f}"


def bold_best(col: list[str], raw: list[np.ndarray]) -> list[str]:
    """Bold the lowest mean in a column; all metrics here are errors."""
    means = [v.mean() if len(v) else np.inf for v in raw]
    if all(np.isinf(m) for m in means):
        return col
    i = int(np.argmin(means))
    col = list(col)
    if col[i] != TODO:
        col[i] = r"\textbf{" + col[i] + "}"
    return col


def write(name: str, body: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        fh.write(body)
    print("wrote", p)


def table_block(df, dataset, rows, metrics, protocol, nd):
    """One dataset's block: list of (label, arm) against the metric columns."""
    raw = {m: [cell(df, dataset, protocol, arm, m) for _, arm in rows] for m in metrics}
    cols = {m: bold_best([fmt(v, nd) for v in raw[m]], raw[m]) for m in metrics}
    out = []
    for i, (label, _) in enumerate(rows):
        out.append(" & ".join([label] + [cols[m][i] for m in metrics]) + r" \\")
    return out


# ------------------------------------------------------------------- tables
DATASETS = [("Abalone", "abalone", 3), ("Communities", "communities", 4),
            ("Wine Quality White", "wine_white", 3)]


def table1a(df) -> None:
    rows = [("zero-shot", "ticl_zeroshot"),
            ("uniform context", "ticl_prior_uniform"),
            ("target-balanced context", "ticl_prior_target_balanced"),
            ("Gaussian context", "ticl_prior_gauss")]
    metrics = ["mae_many", "mae_medium", "mae_few", "mae", "gm"]
    body = [r"\begin{tabular}{lccccc}", r"\toprule",
            r"& Many & Medium & Few & Overall & GM \\"]
    for title, ds, nd in DATASETS:
        body += [r"\midrule", r"\multicolumn{6}{l}{\textit{" + title + r"}} \\"]
        body += table_block(df, ds, rows, metrics, "dir", nd)
    body += [r"\bottomrule", r"\end{tabular}"]
    write("table1a.tex", "\n".join(body) + "\n")


def table1b(df) -> None:
    rows = [("zero-shot", "zeroshot"),
            ("uniform fine-tuning", "ft_uniform"),
            ("Gaussian fine-tuning", "ft_gauss")]
    metrics = ["mae_many", "mae_medium", "mae_few", "mae"]
    body = [r"\begin{tabular}{lcccc}", r"\toprule",
            r"& Many & Medium & Few & Overall \\"]
    for title, ds, nd in DATASETS:
        body += [r"\midrule", r"\multicolumn{5}{l}{\textit{" + title + r"}} \\"]
        # Task B2 (the dataset-level target-balanced control) is specified on Abalone
        # only, so the row is emitted only there. Adding it everywhere would print
        # TODO cells for datasets the plan never asked to be run.
        ds_rows = rows + [("target-balanced fine-tuning", "ft_target_balanced")] \
            if ds == "abalone" else rows
        body += table_block(df, ds, ds_rows, metrics, "dir", nd)
    body += [r"\bottomrule", r"\end{tabular}"]
    write("table1b.tex", "\n".join(body) + "\n")


T2_ROWS = [
    ("ResNet (vanilla)", "ptbxl_resnet", "vanilla"),
    ("ResNet + RankSim + ConR", "ptbxl_resnet", "ranksim_conr"),
    ("E2 + stock TabPFN-2.5", "ptbxl_tabpfn_enc", "zeroshot"),
    ("E2 + Gaussian TabPFN-2.5", "ptbxl_tabpfn_enc", "ft_gauss"),
    ("E3 + stock TabICL", "ptbxl_enc_e3", "ticl_zeroshot"),
    ("E3 + Gaussian TabICL", "ptbxl_enc_e3", "ticl_prior_gauss"),
]


# Plan section 4 specifies Table 2's columns as <30 / 30-60 / >=60 / Natural /
# Balanced / GM, and plan section 1 additionally requires band-macro. "Balanced" is
# the pre-existing definition: MAE on the resampled near-uniform `dir` test set. It
# is therefore the one column read from a DIFFERENT protocol than the rest of the
# row, which is why each column carries its own protocol here.
T2_COLS = [
    (r"$<$30",       "fixed", "mae_young"),
    (r"30--60",      "fixed", "mae_middle"),
    (r"$\geq$60",    "fixed", "mae_old"),
    ("Natural",      "fixed", "mae"),
    ("Balanced",     "dir",   "mae"),
    ("Band-macro",   "fixed", "band_macro"),
    ("GM",           "fixed", "gm"),
]


def table2(df) -> None:
    # The ResNet baselines were only ever evaluated on the official 954-record test
    # split, so they have no `dir` predictions and their Balanced cells come out TODO.
    raw = [[cell(df, ds, proto, arm, m) for _, ds, arm in T2_ROWS]
           for _, proto, m in T2_COLS]
    cols = [bold_best([fmt(v, 3) for v in col], col) for col in raw]
    spec = "l" + "c" * len(T2_COLS)
    head = " & ".join([""] + [h for h, _, _ in T2_COLS]) + r" \\"
    body = [r"\begin{tabular}{" + spec + "}", r"\toprule", head, r"\midrule"]
    for i, (label, _, _) in enumerate(T2_ROWS):
        body.append(" & ".join([label] + [c[i] for c in cols]) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("table2.tex", "\n".join(body) + "\n")


T3_ROWS = [("E1 (supervised L1)", "ptbxl_l1_enc"),
           ("E2 (through TabPFN)", "ptbxl_tabpfn_enc"),
           ("E3 (through stock TabICL)", "ptbxl_enc_e3"),
           ("E4 (through Gaussian TabICL)", "ptbxl_enc_e4")]


def table3(df) -> None:
    specs = [("ticl_zeroshot", "fixed", "mae"), ("ticl_prior_gauss", "fixed", "mae"),
             ("ticl_zeroshot", "dir", "mae"), ("ticl_prior_gauss", "dir", "mae")]
    raw = [[cell(df, ds, pr, arm, m) for _, ds in T3_ROWS] for arm, pr, m in specs]
    cols = [bold_best([fmt(v, 3) for v in col], col) for col in raw]
    body = [r"\begin{tabular}{lcccc}", r"\toprule",
            r"& \multicolumn{2}{c}{Natural (official split)}"
            r" & \multicolumn{2}{c}{Balanced (DIR split)} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"Encoder & stock head & Gaussian head & stock head & Gaussian head \\",
            r"\midrule"]
    for i, (label, _) in enumerate(T3_ROWS):
        body.append(" & ".join([label] + [c[i] for c in cols]) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("table3.tex", "\n".join(body) + "\n")


def main() -> None:
    df = pd.read_csv(MET)
    table1a(df); table1b(df); table2(df); table3(df)
    n_todo = sum(open(os.path.join(OUT, f)).read().count(TODO)
                 for f in ("table1a.tex", "table1b.tex", "table2.tex", "table3.tex"))
    print(f"\n{n_todo} TODO cells remain across the four tables")


if __name__ == "__main__":
    main()
