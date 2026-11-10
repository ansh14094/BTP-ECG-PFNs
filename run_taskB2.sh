#!/usr/bin/env bash
# Task B2: TabPFN-2.5 per-dataset fine-tuning on Abalone with target-balanced context.
#
# Gated in paper_completion_plan.md on "only if B1 is done by day 5". B1 completed
# 10 November, which is day 4, so the gate is satisfied.
#
# This is the dataset-level twin of B1. B1 asked whether a target-balanced *prior*
# does what the Gaussian prior does; B2 asks the same of per-dataset fine-tuning,
# which is the regime where Gaussian sampling actually showed a gain on the tabular
# benchmarks. If the Gaussian's few-shot win there is really just "rare targets got
# over-sampled", this arm should reproduce it.
#
# Settings are copied from the ft_gauss/ft_uniform runs so the comparison is clean:
# epochs 60, ctx_plus_query 1024, ctx_keep_frac 0.5, the same three learning rates
# under the same tags, selected on validation exactly as the other arms are.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .taskb2.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .taskb2_done ] && { echo "[$(date -Is)] already complete"; exit 0; }

. ./gpu_lock.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN_N_THREADS=6 OMP_NUM_THREADS=6
log() { echo "[$(date -Is)] $*"; }
FAILED=0

gpu_lock
cd imbreg || exit 1

for LR in 1e-5 3e-5 1e-4; do
    log "### abalone [dir] ft_target_balanced lr=$LR"
    python3 -u run.py --dataset abalone --protocol dir --tag "lr${LR}" \
        --arms ft_target_balanced --seeds 0,1,2,3,4 --lr "$LR" \
        || { log "lr$LR FAILED"; FAILED=1; }
done

cd ..
if [ "$FAILED" -eq 0 ]; then
    touch .taskb2_done; log "TASK_B2_COMPLETE"
else
    log "TASK_B2_INCOMPLETE -- sentinel not written"
fi
