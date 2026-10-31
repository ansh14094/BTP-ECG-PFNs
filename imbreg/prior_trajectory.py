#!/usr/bin/env python3
"""Is the step-3000 prior result a stable point or an artefact of where we stopped?

The two prior arms were compared at step 3000 only. Intermediate checkpoints were
kept every 250 steps, so the same zero-shot evaluation can be run along the
trajectory. If gauss's gain on PTB-XL grows with steps while its loss on communities
also grows, the two are the same mechanism pulling in opposite directions on
different data. If either is non-monotone, step 3000 is a coincidence and the
headline needs weakening.

Zero-shot only -- no gradient step is taken on the evaluation dataset, so the sole
difference between rows is which weights are loaded.

    python3 prior_trajectory.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = os.path.join(HERE, os.pardir, "tabicl_prior", "runs")
STEPS = [1000, 2000, 3000]
SEEDS = [0, 1, 2]           # three seeds: this is a shape check, not a headline number
CASES = [("ptbxl_tabpfn_enc", "dir"), ("communities", "dir")]
OUT = os.path.join(HERE, "results", "prior_trajectory.jsonl")


def main() -> None:
    import data as data_mod
    import metrics as metrics_mod
    from tabicl import TabICLRegressor
    from run import prep_frame

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["dataset"], r["arm"], r["step"], r["seed"]))

    fh = open(OUT, "a")
    for ds_name, proto in CASES:
        ds = data_mod.load(ds_name)
        # Same preprocessing path as the harness, so these numbers sit on the same
        # scale as the step-3000 results in *_ticl_base.jsonl.
        X_all = prep_frame(ds.X, ds.cat_features)
        for seed in SEEDS:
            tr, va, te = data_mod.dir_split(ds.y, ds.bin_width, seed=seed,
                                            per_bin_val=ds.per_bin_val,
                                            per_bin_test=ds.per_bin_test)
            Xtr, Xte = X_all.iloc[tr], X_all.iloc[te]
            ytr, yte = ds.y[tr], ds.y[te]
            phi = metrics_mod.RelevanceFunction(ytr)
            for step in STEPS:
                for arm in ("uniform", "gauss"):
                    key = (ds_name, arm, step, seed)
                    if key in done:
                        continue
                    ck = os.path.join(RUNS, arm, f"step-{step}.ckpt")
                    if not os.path.exists(ck):
                        print("missing", ck, flush=True); continue
                    reg = TabICLRegressor(model_path=ck, n_estimators=8,
                                          device="cuda", random_state=seed)
                    reg.fit(Xtr, ytr)
                    pred = np.asarray(reg.predict(Xte), float)
                    m = metrics_mod.evaluate(yte, pred, ytr, phi,
                                             bin_width=ds.bin_width)
                    rec = {"dataset": ds_name, "protocol": proto, "arm": arm,
                           "step": step, "seed": seed, **m}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    print(f"{ds_name:22s} {arm:8s} step{step:5d} seed{seed} "
                          f"MAE={m['mae']:.4f} SERA={m['sera']:.4f} few={m['mae_few']:.4f}",
                          flush=True)
    fh.close()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
