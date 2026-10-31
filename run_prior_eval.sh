#!/usr/bin/env bash
# Evaluate the two prior-level continued-pretraining checkpoints.
#
# Three weight sets, ZERO gradient steps on any evaluation dataset:
#   ticl_zeroshot        stock released TabICL regressor          (already recorded)
#   ticl_prior_uniform   + 3000 steps on TabICL's own prior, rows unbiasedly permuted
#   ticl_prior_gauss     + 3000 steps on the same prior, context Gaussian-weighted
#
# The two new arms differ from each other in the row-ordering rule alone -- same SCM,
# same hyperparameters, same step count, same seed -- so any gap between them is the
# intervention and nothing else. Results append to the existing *_ticl_base.jsonl so
# all three sit in one file per dataset.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .priorval.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .priorval_done ] && { echo "[$(date -Is)] already complete"; exit 0; }

# One GPU job at a time. Concurrent fine-tunes caused an OOM cascade earlier in
# this project; waiting costs hours, a collision costs the run.
while pgrep -f "run\.py --dataset" > /dev/null || pgrep -f train_gauss_prior > /dev/null; do
    echo "[$(date -Is)] GPU busy; waiting..."; sleep 120
done

export RUN_N_THREADS=6 OMP_NUM_THREADS=6
log() { echo "[$(date -Is)] $*"; }
cd imbreg || exit 1
FAILED=0
ARMS=ticl_prior_uniform,ticl_prior_gauss

go() {  # dataset protocol tag [extra]
    log "### $1 [$2] -> $3"
    TICL_MAX_DATA_SIZE="${4:-10000}" python3 -u run.py --dataset "$1" --protocol "$2" \
        --tag "$3" --arms "$ARMS" --seeds 0,1,2,3,4 \
        || { log "$1 FAILED"; FAILED=1; }
}

go abalone          dir   ticl_base
go communities      dir   ticl_base
go wine_white       dir   ticl_base
go ptbxl_l1_enc     dir   dir_ticl_base
go ptbxl_tabpfn_enc dir   dir_ticl_base
go agedb_tabpfn_enc fixed ticl_base 4096

cd ..
[ "$FAILED" -eq 0 ] && { touch .priorval_done; log "PRIOR_EVAL_COMPLETE"; } \
    || log "PRIOR_EVAL_INCOMPLETE -- sentinel not written"
