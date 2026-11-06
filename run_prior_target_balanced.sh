#!/usr/bin/env bash
# Task B1: fourth prior-level arm -- inverse-density ("target_balanced") context.
#
# The question this arm answers: is the Gaussian doing something a plain balancing
# scheme would not? `gauss` concentrates the context on one drawn region of target
# space; `target_balanced` instead flattens it across the whole range by sampling
# each row with probability proportional to 1 / (count of its target bin).
#
# A win for `gauss` only counts if it beats THIS, not merely the uniform control.
# Same generator, same 3000 steps, same seed, same flags as the other three arms.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .priortb.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .priortb_done ] && { echo "[$(date -Is)] already complete"; exit 0; }

. ./gpu_lock.sh
gpu_lock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date -Is)] $*"; }
cd tabicl_prior || exit 1
log "===== arm target_balanced (3000 steps) ====="
if bash run_arm.sh target_balanced 3000 runs/target_balanced; then
    # Trust the checkpoint, not the exit code: the trainer has exited 0 over a
    # failed run before.
    if [ -f runs/target_balanced/step-3000.ckpt ]; then
        touch runs/target_balanced/.done; cd ..; touch .priortb_done
        log "PRIOR_TARGET_BALANCED_COMPLETE"
    else
        cd ..; log "PRIOR_TARGET_BALANCED_INCOMPLETE -- no step-3000.ckpt"
    fi
else
    cd ..; log "PRIOR_TARGET_BALANCED_FAILED"
fi
