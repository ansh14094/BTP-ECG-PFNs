#!/usr/bin/env bash
# Third prior-level arm: context half Gaussian-weighted, half uniform ("mixed").
#
# Completes the set. The three continued-pre-training arms differ only in which
# rows of each synthetic dataset land before the positional context/query cut:
#   uniform  unbiased permutation            (the matched control)
#   gauss    Gaussian-weighted on the target (the intervention)
#   mixed    half Gaussian, half uniform     (does diluting it help or hurt?)
# Same generator, same 3000 steps, same seed, same flags as the other two.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .priormixed.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .priormixed_done ] && { echo "[$(date -Is)] already complete"; exit 0; }

. ./gpu_lock.sh
gpu_lock

log() { echo "[$(date -Is)] $*"; }
cd tabicl_prior || exit 1
log "===== arm mixed (3000 steps) ====="
if bash run_arm.sh mixed 3000 runs/mixed; then
    # Trust the checkpoint, not the exit code: the trainer has exited 0 over a
    # failed run before.
    if [ -f runs/mixed/step-3000.ckpt ]; then
        touch runs/mixed/.done; cd ..; touch .priormixed_done
        log "PRIOR_MIXED_COMPLETE"
    else
        cd ..; log "PRIOR_MIXED_INCOMPLETE -- no step-3000.ckpt"
    fi
else
    cd ..; log "PRIOR_MIXED_FAILED"
fi
