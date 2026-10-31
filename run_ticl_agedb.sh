#!/usr/bin/env bash
# AgeDB TabICL fine-tuning, with the episode size capped.
#
# Why the cap: AgeDB is 12,208 x 512. TabICL attends over the entire context, so the
# default max_data_size=10000 needs ~70 GB of GPU and OOMs -- that, not the earlier
# concurrency, is the real reason these cells have never completed. 4096 keeps a
# large context while fitting comfortably. Both arms share the value, so the
# uniform-vs-Gaussian comparison stays matched.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .ticlagedb.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .ticlagedb_done ] && { echo "[$(date -Is)] already complete"; exit 0; }

# Never share the GPU with the prior-level training run.
while pgrep -f train_gauss_prior > /dev/null; do
    echo "[$(date -Is)] prior run owns the GPU; waiting..."; sleep 120
done

export RUN_N_THREADS=6 OMP_NUM_THREADS=6 TICL_MAX_DATA_SIZE=4096
log() { echo "[$(date -Is)] $*"; }
cd imbreg || exit 1
FAILED=0
for DS in agedb_tabpfn_enc agedb_l1_enc; do
    log "### $DS zero-shot"
    python3 -u run.py --dataset "$DS" --protocol fixed --tag ticl_base \
        --arms ticl_zeroshot --seeds 0,1,2,3,4 || { log "$DS base FAILED"; FAILED=1; }
    for LR in 1e-5 3e-5 1e-4; do
        log "### $DS lr=$LR"
        python3 -u run.py --dataset "$DS" --protocol fixed --tag "ticl_lr${LR}" \
            --arms ticl_ft_uniform,ticl_ft_gauss --seeds 0,1,2,3,4 --lr "$LR" \
            || { log "$DS lr$LR FAILED"; FAILED=1; }
    done
done
cd ..
[ "$FAILED" -eq 0 ] && { touch .ticlagedb_done; log "TICL_AGEDB_COMPLETE"; } \
    || log "TICL_AGEDB_INCOMPLETE -- sentinel not written"
