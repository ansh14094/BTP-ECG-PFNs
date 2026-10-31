#!/usr/bin/env bash
# Zero-shot Timestral forecasting benchmark on PTB-XL, full test split (954 records).
#
# Three axes, because a single configuration would not say much:
#   horizon      how far ahead it can forecast before the quasi-periodic ECG structure
#                stops helping (1 s / 2 s / 4 s)
#   model size   whether the 40x parameter range from 5M to 209M buys anything on ECG
#   mode         12 leads jointly (Timestral models cross-series dependencies) vs each
#                lead independently -- isolates whether the joint modelling helps here
#
# Resumable: a config whose results file already exists is skipped.
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs results
exec 9> .timestral.lock
flock -n 9 || { echo "[$(date -Is)] another timestral run holds the lock"; exit 0; }

run() {  # model, mode, horizon
    local m="$1" mode="$2" h="$3"
    local tag="$(basename "$m")_${mode}_h${h}"
    [ -f "results/timestral_${tag}.jsonl" ] && { echo "[$(date -Is)] skip ${tag}"; return; }
    echo "[$(date -Is)] === ${tag} ==="
    python3 -u timestral_bench.py --model "$m" --mode "$mode" \
        --context $((1000 - h)) --horizon "$h" --batch 16 --tag "$tag" \
        || echo "[$(date -Is)] ${tag} FAILED"
}

# Horizon sweep at the flagship size.
for H in 100 200 400; do run mistralai/Timestral-209M multivariate "$H"; done
# Model-size sweep at the middle horizon.
for M in 5M 28M 123M; do run "mistralai/Timestral-${M}" multivariate 200; done
# Joint vs independent leads.
run mistralai/Timestral-209M univariate 200

echo "[$(date -Is)] TIMESTRAL_SWEEP_COMPLETE"
touch .timestral_done
