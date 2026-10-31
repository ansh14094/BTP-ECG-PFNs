#!/usr/bin/env bash
# Fill the missing TabICL cells so the TabICL table matches the TabPFN one.
#
# Missing before this ran:
#   ptbxl_l1_enc (E1)   no TabICL arms at all
#   agedb_tabpfn_enc    zero-shot only -- the fine-tuning arms died of CUDA OOM on
#                       2026-10-24 ("free: 405 MB, total: 102 GB") during the
#                       concurrency cascade. Not a code fault; they were simply
#                       starved. Re-run sequentially.
#   agedb_l1_enc        nothing
#
# Protocols follow what each dataset already used: PTB-XL is DIR-rebalanced, AgeDB's
# official val/test are already balanced over age bins so it stays on `fixed`.
#
# STRICTLY SEQUENTIAL, and nice'd: the prior-level continued-training run owns the
# GPU and must not be starved by this.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .ticlgaps.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .ticlgaps_done ] && { echo "[$(date -Is)] already complete"; exit 0; }

export RUN_N_THREADS=6 OMP_NUM_THREADS=6 LGBM_N_JOBS=4
log() { echo "[$(date -Is)] $*"; }
cd imbreg || exit 1

run() {  # dataset, protocol, tagprefix, arms, extra
    local DS="$1" PROTO="$2" TAG="$3" ARMS="$4"; shift 4
    log "### $DS [$PROTO] $TAG :: $ARMS"
    nice -n 10 python3 -u run.py --dataset "$DS" --protocol "$PROTO" --tag "$TAG" \
        --arms "$ARMS" --seeds 0,1,2,3,4 "$@" || log "$DS $TAG FAILED"
}

# ---- PTB-XL E1: TabICL has never been run on this encoder -------------------
run ptbxl_l1_enc dir dir_ticl_base ticl_zeroshot
for LR in 1e-5 3e-5 1e-4; do
    run ptbxl_l1_enc dir "dir_ticl_lr${LR}" ticl_ft_uniform,ticl_ft_gauss --lr "$LR"
done

# ---- AgeDB (TabPFN encoder): fine-tuning arms lost to the OOM cascade --------
for LR in 1e-5 3e-5 1e-4; do
    run agedb_tabpfn_enc fixed "ticl_lr${LR}" ticl_ft_uniform,ticl_ft_gauss --lr "$LR"
done

# ---- AgeDB (L1 encoder): never run at all -----------------------------------
run agedb_l1_enc fixed ticl_base ticl_zeroshot
for LR in 1e-5 3e-5 1e-4; do
    run agedb_l1_enc fixed "ticl_lr${LR}" ticl_ft_uniform,ticl_ft_gauss --lr "$LR"
done

cd ..
touch .ticlgaps_done
log "TICL_GAPS_COMPLETE"
