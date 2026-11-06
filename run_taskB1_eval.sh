#!/usr/bin/env bash
# Task B1 evaluation: the target-balanced prior arm on every dataset in the paper.
#
# This arm is the control that makes the Gaussian claim falsifiable. `gauss`
# concentrates the context on one drawn region of target space; `target_balanced`
# flattens it across the whole range by inverse bin frequency. If Gaussian context
# sampling only works because it happens to over-sample rare targets, this arm will
# match it. If the concentration itself is what matters, it will not.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .taskb1.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .taskb1_done ] && { echo "[$(date -Is)] already complete"; exit 0; }

CKPT=tabicl_prior/runs/target_balanced/step-3000.ckpt
if [ ! -f "$CKPT" ]; then
    echo "[$(date -Is)] $CKPT not present -- training has not finished; skipping"
    exit 0
fi

. ./gpu_lock.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN_N_THREADS=6 OMP_NUM_THREADS=6
log() { echo "[$(date -Is)] $*"; }
FAILED=0

gpu_lock
cd imbreg || exit 1

# Tabular: the DIR protocol, matching how the other prior arms were measured.
for DS in abalone communities wine_white; do
    log "### $DS [dir] ticl_prior_target_balanced"
    python3 -u run.py --dataset "$DS" --protocol dir --tag ticl_base \
        --arms ticl_prior_target_balanced --seeds 0,1,2,3,4 \
        || { log "$DS FAILED"; FAILED=1; }
done

# PTB-XL: the clean official split, same tag as the rest of the clean matrix.
for DS in ptbxl_l1_enc ptbxl_tabpfn_enc ptbxl_enc_e3 ptbxl_enc_e4 ptbxl_enc_e5; do
    log "### $DS [fixed] ticl_prior_target_balanced"
    python3 -u run.py --dataset "$DS" --protocol fixed --tag clean_matrix \
        --arms ticl_prior_target_balanced --seeds 0,1,2,3,4 \
        || { log "$DS FAILED"; FAILED=1; }
done

# Table 1b needs TabPFN zero-shot predictions, and abalone's were never saved to
# disk -- only the summary row survives, and the plan requires every table value to
# come from a prediction file. Re-running zero-shot inference regenerates them.
log "### abalone [dir] TabPFN zeroshot (predictions were never saved)"
python3 -u run.py --dataset abalone --protocol dir --tag baseline_preds \
    --arms zeroshot --seeds 0,1,2,3,4 || { log "abalone zeroshot FAILED"; FAILED=1; }

cd ..
if [ "$FAILED" -eq 0 ]; then
    touch .taskb1_done; log "TASK_B1_EVAL_COMPLETE"
else
    log "TASK_B1_EVAL_INCOMPLETE -- sentinel not written"
fi
