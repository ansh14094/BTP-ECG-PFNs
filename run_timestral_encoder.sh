#!/usr/bin/env bash
# Experiment 5: Timestral's encoder in place of our ResNet encoder, on PTB-XL age.
#
# Two feature sets, both from the frozen Timestral-209M encoder:
#   ptbxl_timestral_pca   128 dims, PCA fitted on train only -- width-matched to E1/E2
#   ptbxl_timestral_enc   full 768 dims -- shows whether the reduction cost anything
#
# Arms mirror the Table 1 rows so the comparison against E1/E2 is like-for-like:
# frozen TabPFN, then uniform and Gaussian fine-tuning, swept over the same three
# learning rates and selected on validation.
#
# STRICTLY SEQUENTIAL -- concurrent GPU fine-tunes caused an OOM cascade earlier.
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs
exec 9> .tsenc.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }

if [ -f .tsenc_done ]; then
    echo "[$(date -Is)] .tsenc_done present -- already complete, exiting."
    exit 0
fi

export RUN_N_THREADS=10 OMP_NUM_THREADS=10 LGBM_N_JOBS=8
log() { echo "[$(date -Is)] $*"; }

cd imbreg || exit 1

for DS in ptbxl_timestral_pca ptbxl_timestral_enc; do
    log "########## $DS : baselines ##########"
    python3 -u run.py --dataset "$DS" --protocol dir --tag dir_baseline \
        --arms lgbm,zeroshot,zeroshot_recal_iso --seeds 0,1,2,3,4 \
        || log "$DS baselines FAILED"

    for LRTAG in 1e-5 3e-5 1e-4; do
        log "########## $DS : fine-tuning lr=$LRTAG ##########"
        python3 -u run.py --dataset "$DS" --protocol dir --tag "dir_lr${LRTAG}" \
            --arms ft_uniform,ft_gauss --seeds 0,1,2,3,4 --lr "$LRTAG" \
            || log "$DS lr$LRTAG FAILED"
    done
done

cd ..
touch .tsenc_done
log "TSENC_COMPLETE"
