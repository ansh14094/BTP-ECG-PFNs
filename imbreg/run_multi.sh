#!/usr/bin/env bash
# Multi-dataset replication of the TabPFN-2.5 biased-context fine-tuning result.
#
# One dataset is not enough to claim the method works, so this re-runs the whole
# comparison on additional imbalanced-regression tasks. Both fine-tuning arms get
# the SAME learning-rate sweep and are each reported at their own validation-selected
# learning rate -- at lr=1e-5 uniform fine-tuning was effectively inert on abalone,
# and reporting an inert baseline would rig the comparison.
#
# Detached launch:  setsid nohup ./run_multi.sh < /dev/null >> logs/multi.log 2>&1 & disown
# Watch:            tail -f logs/multi.log
# Done when:        grep -q MULTI_COMPLETE logs/multi.log
#
# Resumable: run.py skips (arm, seed) cells already recorded, so a reboot mid-run
# resumes rather than recomputing.

set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs results

exec 9> .multi.lock
if ! flock -n 9; then
    echo "[$(date -Is)] another multi run holds the lock; exiting"
    exit 0
fi

# A @reboot relaunch of an already-finished grid should be a no-op rather than a
# re-run; run.py would skip every cell anyway, but this exits immediately.
if [ -f .multi_done ]; then
    echo "[$(date -Is)] .multi_done present; nothing to do"
    exit 0
fi

DATASETS="${DATASETS:?set DATASETS, e.g. 'abalone communities wine_white'}"
SEEDS="${SEEDS:-0,1,2,3,4}"
EPOCHS="${EPOCHS:-60}"
LRS="${LRS:-1e-5 3e-5 1e-4}"
PY=python3

for DS in $DATASETS; do
    echo "[$(date -Is)] ########## dataset ${DS} ##########"

    echo "[$(date -Is)] === ${DS}: reference arms ==="
    $PY run.py --dataset "$DS" --seeds "$SEEDS" --tag baseline \
        --arms lgbm,zeroshot,zeroshot_recal \
        || echo "[$(date -Is)] ${DS}/baseline FAILED (continuing)"

    for LR in $LRS; do
        echo "[$(date -Is)] === ${DS}: lr=${LR} ==="
        $PY run.py --dataset "$DS" --seeds "$SEEDS" --epochs "$EPOCHS" \
            --tag "lr${LR}" --arms ft_uniform,ft_gauss --lr "$LR" \
            || echo "[$(date -Is)] ${DS}/lr${LR} FAILED (continuing)"
    done
done

echo "[$(date -Is)] MULTI_COMPLETE"
touch .multi_done
