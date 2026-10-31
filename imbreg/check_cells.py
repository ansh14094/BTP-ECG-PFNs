#!/usr/bin/env python3
"""Exit non-zero unless every (arm, seed) cell a run was asked for is on disk.

`run.py` catches per-cell exceptions so that one failed arm does not abandon the
rest of a sweep. The cost is that it exits 0 even when every cell inside it failed,
so a driver script's `|| FAILED=1` never fires and a completion sentinel gets
written over a run that produced nothing. That happened twice in this project --
once to the prior-training arms, once to AgeDB. Checking the recorded rows is the
only honest completion test.

    python3 check_cells.py results/agedb_l1_enc_ticl_lr1e-5.jsonl \
        --arms ticl_ft_uniform,ticl_ft_gauss --seeds 0,1,2,3,4
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    arms = a.arms.split(",")
    seeds = {int(s) for s in a.seeds.split(",")}

    bad = 0
    for f in a.files:
        by: dict[str, set[int]] = defaultdict(set)
        if os.path.exists(f):
            for line in open(f):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    by[r["arm"]].add(int(r["seed"]))
        for arm in arms:
            miss = sorted(seeds - by.get(arm, set()))
            if miss:
                print(f"INCOMPLETE {os.path.basename(f)} {arm} missing seeds {miss}")
                bad += 1
    if bad:
        print(f"{bad} incomplete (file, arm) groups")
        return 1
    print(f"complete: {len(a.files)} file(s) x {len(arms)} arm(s) x {len(seeds)} seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
