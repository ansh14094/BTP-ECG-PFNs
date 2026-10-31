#!/usr/bin/env bash
# Follow-up arms, run after run_multi.sh completes.
#
# Why these exist: on abalone, plain TabPFN-2.5 plus a two-parameter affine
# recalibration fitted on the balanced validation set BEAT Gaussian-context
# fine-tuning on every tail metric (MAE +2.6%, SERA +26.7%, few-shot +23.6%, 0/5
# seeds). That makes one question decisive for interpreting the whole study:
#
#   Is the fine-tuning doing anything a monotone output transform cannot?
#
# ft_gauss_recal stacks the same recalibration on top of the fine-tuned model. If it
# clearly beats recalibration alone, the fine-tuning contributes beyond decompressing
# predictions. If it does not, the honest conclusion is that recalibration is the
# method and the fine-tuning is unnecessary.
#
# zeroshot_recal_iso is the isotonic version, to check the linear fit is not the
# limiting factor in the baseline.

set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs results

exec 8> .followup.lock
if ! flock -n 8; then
    echo "[$(date -Is)] another followup run holds the lock; exiting"
    exit 0
fi
if [ -f .followup_done ]; then
    echo "[$(date -Is)] .followup_done present; nothing to do"
    exit 0
fi

DATASETS="${DATASETS:-abalone communities wine_white}"
SEEDS="${SEEDS:-0,1,2,3,4}"
EPOCHS="${EPOCHS:-60}"
LRS="${LRS:-1e-5 3e-5 1e-4}"
PY=python3

for DS in $DATASETS; do
    echo "[$(date -Is)] ########## followup: ${DS} ##########"
    $PY run.py --dataset "$DS" --seeds "$SEEDS" --tag baseline \
        --arms zeroshot_recal_iso \
        || echo "[$(date -Is)] ${DS}/iso FAILED (continuing)"
    for LR in $LRS; do
        $PY run.py --dataset "$DS" --seeds "$SEEDS" --epochs "$EPOCHS" \
            --tag "lr${LR}" --arms ft_gauss_recal --lr "$LR" \
            || echo "[$(date -Is)] ${DS}/ft_gauss_recal lr${LR} FAILED (continuing)"
    done
done

echo "[$(date -Is)] FOLLOWUP_COMPLETE"
touch .followup_done
