#!/usr/bin/env bash
# All stages for ONE dataset. Datasets write to disjoint results files and never
# interact, so running one of these per dataset concurrently is safe and uses the
# GPU far better than the sequential driver did (each job needs under 2 GB).
#
# Usage:   ./run_one.sh <dataset>
# Resumable: run.py skips (arm, seed) cells already recorded, so a kill or a reboot
# mid-run costs at most the cell in flight.

set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs results

DS="${1:?usage: run_one.sh <dataset>}"
SEEDS="${SEEDS:-0,1,2,3,4}"
EPOCHS="${EPOCHS:-60}"
LRS="${LRS:-1e-5 3e-5 1e-4}"
PY=python3

# One lock per dataset, so a @reboot relaunch cannot collide with a manual one but
# the three datasets still run in parallel.
exec 9> ".one_${DS}.lock"
if ! flock -n 9; then
    echo "[$(date -Is)] ${DS}: another run holds the lock; exiting"; exit 0
fi
if [ -f ".done_${DS}" ]; then
    echo "[$(date -Is)] ${DS}: already complete"; exit 0
fi

echo "[$(date -Is)] ########## ${DS}: start ##########"

# Reference arms, including the recalibration baselines that turned out to be the
# ones to beat.
$PY run.py --dataset "$DS" --seeds "$SEEDS" --tag baseline \
    --arms lgbm,zeroshot,zeroshot_recal,zeroshot_recal_iso \
    || echo "[$(date -Is)] ${DS}: baseline FAILED (continuing)"

# Both fine-tuning arms plus the combination get the same sweep; each is reported at
# its own validation-selected rate.
for LR in $LRS; do
    echo "[$(date -Is)] ${DS}: lr=${LR}"
    $PY run.py --dataset "$DS" --seeds "$SEEDS" --epochs "$EPOCHS" \
        --tag "lr${LR}" --arms ft_uniform,ft_gauss,ft_gauss_recal --lr "$LR" \
        || echo "[$(date -Is)] ${DS}: lr${LR} FAILED (continuing)"
done

echo "[$(date -Is)] ########## ${DS}: DATASET_COMPLETE ##########"
touch ".done_${DS}"
