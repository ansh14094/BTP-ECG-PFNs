#!/usr/bin/env bash
# Experiment 2 orchestrator: AgeDB-DIR + TabICL.
#
# Everything runs SEQUENTIALLY after the AgeDB encoders finish. That is deliberate:
# an earlier attempt ran six GPU jobs at once and the two AgeDB encoders alone take
# 60 GB of the 95 GB card (ResNet-18 activations for 1280 images per step), while a
# TabICL fine-tune on the PTB-XL embeddings wants another 30 GB. The result was a
# cascade of CUDA OOM failures. Memory headroom, not wall-clock, is the binding
# constraint here.
#
# Resumable at every stage; safe to re-run or fire at @reboot.
#   setsid nohup ./run_experiment2.sh < /dev/null >> logs_exp2.log 2>&1 & disown

set -u
cd "$(dirname "$0")" || exit 1
exec 8> .exp2.lock
flock -n 8 || { echo "[$(date -Is)] another exp2 run holds the lock"; exit 0; }
say() { echo "[$(date -Is)] $*"; }

# ---- wait for the AgeDB encoders (launched separately, may already be done) ----
while pgrep -f "train_encoder.py --objective" > /dev/null; do sleep 60; done
say "AgeDB encoders finished"

for OBJ in tabpfn l1; do
    CK="agedb_tabpfn/ckpt/enc_${OBJ}.pt"
    OUT="agedb_tabpfn/results/emb_${OBJ}.npz"
    if [ -f "$OUT" ]; then say "embeddings ${OBJ}: present"; continue; fi
    [ -f "$CK" ] || { say "MISSING checkpoint $CK"; continue; }
    say "extracting embeddings: ${OBJ}"
    python3 -u agedb_tabpfn/extract_embeddings.py --ckpt "$CK" --out "$OUT" \
        || say "extract ${OBJ} FAILED"
done

# ---- AgeDB arms (TabPFN side) -------------------------------------------------
# AgeDB-DIR's official val/test are already age-balanced, so `fixed` is correct and
# no rebalanced variant is needed.
cd imbreg || exit 1
for DS in agedb_l1_enc agedb_tabpfn_enc; do
    [ -f "../agedb_tabpfn/.done_${DS}" ] && { say "${DS}: done"; continue; }
    say "########## ${DS} ##########"
    python3 run.py --dataset "$DS" --protocol fixed --seeds 0,1,2,3,4 \
        --tag baseline --arms lgbm,zeroshot,zeroshot_recal,zeroshot_recal_iso \
        || say "${DS}/baseline FAILED"
    for LR in 1e-5 3e-5 1e-4; do
        say "${DS}: lr=${LR}"
        python3 run.py --dataset "$DS" --protocol fixed --seeds 0,1,2,3,4 \
            --epochs 30 --tag "lr${LR}" \
            --arms ft_uniform,ft_gauss,ft_gauss_recal --lr "$LR" \
            || say "${DS}/lr${LR} FAILED"
    done
    touch "../agedb_tabpfn/.done_${DS}"
done

# ---- TabICL, one dataset at a time -------------------------------------------
for SPEC in "abalone dir" "communities dir" "wine_white dir" \
            "ptbxl_tabpfn_enc dir" "agedb_tabpfn_enc fixed"; do
    set -- $SPEC
    [ -f ".done_ticl_$1" ] && { say "TabICL $1: done"; continue; }
    say "########## TabICL $1 ##########"
    PROTO="$2" ./run_tabicl.sh "$1" || say "TabICL $1 FAILED"
done

# ---- reports ------------------------------------------------------------------
cd ..
say "generating reports"
python3 -u agedb_tabpfn/make_agedb_report.py  || say "AgeDB report FAILED"
python3 -u imbreg/make_tabicl_report.py       || say "TabICL report FAILED"
say "EXPERIMENT2_COMPLETE"
touch .exp2_done
