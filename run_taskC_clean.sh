#!/usr/bin/env bash
# Task C, clean protocol: the encoder x body matrix re-evaluated on the OFFICIAL
# PTB-XL test split instead of the DIR resample.
#
# Why this exists. The `dir` protocol builds a balanced test set by resampling from
# ALL 9483 records. Every encoder here was fitted on the official train split (7566
# of them), so ~80% of that DIR test set is records the encoder saw, with their
# labels. Measured directly, those records score 0.66-0.74 years better than unseen
# ones -- roughly seven times the effect the paper is trying to measure. The `fixed`
# protocol uses the official partition, whose 954 test records were seen by nothing.
#
# The DIR numbers are kept and reported as the "balanced" column, which is what the
# existing code has always meant by balanced evaluation; these fixed-protocol numbers
# are the "natural" column. Both appear in Table 2, with the caveat recorded in
# results/STATUS.md.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .taskc.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .taskc_done ] && { echo "[$(date -Is)] already complete"; exit 0; }

. ./gpu_lock.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN_N_THREADS=6 OMP_NUM_THREADS=6
log() { echo "[$(date -Is)] $*"; }
FAILED=0

gpu_lock
cd imbreg || exit 1

# TabICL bodies: stock plus the three prior-level continuations.
ARMS=ticl_zeroshot,ticl_prior_uniform,ticl_prior_gauss,ticl_prior_mixed
for DS in ptbxl_l1_enc ptbxl_tabpfn_enc ptbxl_enc_e3 ptbxl_enc_e4 ptbxl_enc_e5; do
    log "### $DS [fixed] $ARMS"
    python3 -u run.py --dataset "$DS" --protocol fixed --tag clean_matrix \
        --arms "$ARMS" --seeds 0,1,2,3,4 || { log "$DS FAILED"; FAILED=1; }
done

# TabPFN rows of Table 2: E2 with the stock model and with the Gaussian fine-tuned
# one. Natural MAE and GM for these were never recorded.
log "### ptbxl_tabpfn_enc [fixed] TabPFN arms"
python3 -u run.py --dataset ptbxl_tabpfn_enc --protocol fixed --tag clean_tabpfn \
    --arms zeroshot --seeds 0,1,2,3,4 || { log "tabpfn zeroshot FAILED"; FAILED=1; }
for LR in 1e-5 3e-5 1e-4; do
    python3 -u run.py --dataset ptbxl_tabpfn_enc --protocol fixed \
        --tag "clean_tabpfn_lr${LR}" --arms ft_uniform,ft_gauss --seeds 0,1,2,3,4 \
        --lr "$LR" || { log "tabpfn lr$LR FAILED"; FAILED=1; }
done

cd ..
if [ "$FAILED" -eq 0 ]; then
    touch .taskc_done; log "TASK_C_COMPLETE"
else
    log "TASK_C_INCOMPLETE -- sentinel not written"
fi
