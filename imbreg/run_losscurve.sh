#!/usr/bin/env bash
# Instrumented uniform-vs-Gaussian fine-tuning runs, for the training-curve analysis.
# Sequential: one GPU job at a time.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .losscurve.lock
flock -n 9 || { echo "[$(date -Is)] already running"; exit 0; }
# The three standard imbalanced-regression benchmarks from the main study.
for SPEC in "abalone dir" "communities dir" "wine_white dir"; do
    set -- $SPEC
    echo "[$(date -Is)] === $1 ($2) ==="
    python3 -u loss_curve_experiment.py --dataset "$1" --protocol "$2" \
        --seeds 0,1,2 --epochs 60 --lr 1e-4 --tag losscurve \
        || echo "[$(date -Is)] $1 FAILED"
done
echo "[$(date -Is)] LOSSCURVE_ALL_COMPLETE"
touch .losscurve_done
