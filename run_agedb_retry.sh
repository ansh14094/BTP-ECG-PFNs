#!/usr/bin/env bash
# Re-run the AgeDB TabICL cells that OOMed, at a smaller episode size.
#
# What happened: TICL_MAX_DATA_SIZE=4096 still left the process holding ~65 GB, and
# the GPU is shared with another user who grew to ~29 GB, so allocations failed.
# 2048 halves the context the model attends over.
#
# Partial result files are DELETED rather than topped up: a file holding some seeds
# at 4096 and others at 2048 is not a matched comparison. The two agedb_tabpfn_enc
# cells that completed cleanly at 4096 (lr 1e-5 and 3e-5, both arms, all 5 seeds)
# are left alone -- within each of those files both arms share the same value, which
# is what the gauss-vs-uniform claim needs.
#
# Completion is decided by check_cells.py reading the recorded rows, NOT by exit
# codes: run.py catches per-cell exceptions and exits 0 even when every cell failed.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .agedbretry.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .agedbretry_done ] && { echo "[$(date -Is)] already complete"; exit 0; }

. ./gpu_lock.sh
gpu_lock

export RUN_N_THREADS=6 OMP_NUM_THREADS=6 TICL_MAX_DATA_SIZE=2048
# The failures reported 4.85 GB reserved-but-unallocated; expandable segments let
# the allocator reuse that instead of fragmenting into an OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date -Is)] $*"; }
cd imbreg || exit 1
ARMS=ticl_ft_uniform,ticl_ft_gauss

# Clear the partials so every seed in a file shares one episode size.
rm -f results/agedb_tabpfn_enc_ticl_lr1e-4.jsonl \
      results/agedb_l1_enc_ticl_lr1e-5.jsonl \
      results/agedb_l1_enc_ticl_lr3e-5.jsonl \
      results/agedb_l1_enc_ticl_lr1e-4.jsonl

TODO=("agedb_tabpfn_enc 1e-4" "agedb_l1_enc 1e-5" "agedb_l1_enc 3e-5" "agedb_l1_enc 1e-4")
for entry in "${TODO[@]}"; do
    set -- $entry; DS="$1"; LR="$2"
    log "### $DS lr=$LR (max_data_size=$TICL_MAX_DATA_SIZE)"
    python3 -u run.py --dataset "$DS" --protocol fixed --tag "ticl_lr${LR}" \
        --arms "$ARMS" --seeds 0,1,2,3,4 --lr "$LR"
done

log "### completeness check"
FILES=(results/agedb_tabpfn_enc_ticl_lr1e-4.jsonl
       results/agedb_l1_enc_ticl_lr1e-5.jsonl
       results/agedb_l1_enc_ticl_lr3e-5.jsonl
       results/agedb_l1_enc_ticl_lr1e-4.jsonl)
if python3 check_cells.py "${FILES[@]}" --arms "$ARMS" --seeds 0,1,2,3,4; then
    cd ..; touch .agedbretry_done; log "AGEDB_RETRY_COMPLETE"
else
    cd ..; log "AGEDB_RETRY_INCOMPLETE -- sentinel not written"
fi
