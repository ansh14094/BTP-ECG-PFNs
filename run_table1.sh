#!/usr/bin/env bash
# Fill Table 1: PTB-XL cumulative effect, all TabPFN variants x all encoders.
#
#   TabPFN variants : Frozen | Uniform FT | Gaussian FT | Gaussian+Uniform FT
#   Encoders        : E1   directly trained (L1 age loss)
#                     E2   trained through FROZEN stock TabPFN
#                     E2.1 trained through frozen GAUSSIAN-fine-tuned TabPFN
#                     E2.2 trained through frozen GAUSS+UNIFORM-fine-tuned TabPFN
#
# E2.1/E2.2 need a fine-tuned TabPFN before their encoder can be trained, and
# fine-tuning needs embeddings, so the chain starts from E1.
#
# STRICTLY SEQUENTIAL. Two encoder trainings plus a TabPFN fine-tune will not fit on
# the card together -- an earlier attempt at concurrency produced a cascade of CUDA
# OOM failures. Every stage skips its own completed output, so this is safe to
# re-run and safe at @reboot.
#
#   setsid nohup ./run_table1.sh < /dev/null >> logs_table1.log 2>&1 & disown

set -u

# Table complete -> nothing to do. Without this, the @reboot entry would restart
# the whole pipeline and re-train the E2.1/E2.2 encoders over their checkpoints.
# Delete .table1_done to force a genuine re-run.
if [ -f "$(dirname "$0")/.table1_done" ]; then
    echo "[$(date -Is)] .table1_done present -- Table 1 ALREADY COMPLETE, exiting."
    exit 0
fi
cd "$(dirname "$0")" || exit 1
exec 7> .table1.lock
flock -n 7 || { echo "[$(date -Is)] another table1 run holds the lock"; exit 0; }
say() { echo "[$(date -Is)] $*"; }
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 RUN_N_THREADS=10

SEEDS="${SEEDS:-0,1,2,3,4}"
LRS="${LRS:-1e-5 3e-5 1e-4}"

arms_for() {  # dataset, arm-list, tag-prefix
    local ds="$1" arms="$2"
    for LR in $LRS; do
        say "${ds}: ${arms} lr=${LR}"
        ( cd imbreg && python3 run.py --dataset "$ds" --protocol dir --seeds "$SEEDS" \
            --epochs 30 --tag "dir_lr${LR}" --arms "$arms" --lr "$LR" ) \
            || say "${ds}/${arms}/lr${LR} FAILED"
    done
}

# ---- Phase 1: Gaussian+Uniform arm on the encoders that already exist -------
say "########## Phase 1: ft_mixed on E1 and E2 ##########"
arms_for ptbxl_l1_enc      ft_mixed
arms_for ptbxl_tabpfn_enc  ft_mixed

# ---- Phase 2: fine-tuned TabPFN weights, from E1 embeddings ----------------
say "########## Phase 2: fine-tuned TabPFN checkpoints ##########"
for M in gauss mixed; do
    OUT="ecg_tabpfn/ckpt/tabpfn_ft_${M}_E1.pt"
    if [ -f "$OUT" ]; then say "weights ${M}: present"; continue; fi
    say "fine-tuning TabPFN (${M}) on E1 embeddings"
    # Early stopping off: a reverted no-op would be indistinguishable from the
    # stock frozen model and would make E2.1/E2.2 meaningless.
    python3 -u ecg_tabpfn/finetune_and_save.py --dataset ptbxl_l1_enc --protocol dir \
        --mode "$M" --epochs 30 --lr 1e-4 --no-early-stopping --out "$OUT" \
        || say "finetune_and_save ${M} FAILED"
done

# ---- Phase 3: encoders E2.1 and E2.2 ---------------------------------------
say "########## Phase 3: encoders E2.1 / E2.2 ##########"
train_enc() {  # tag, weights
    local tag="$1" w="$2"
    if grep -q "ENCODER_TRAINING_COMPLETE" "ecg_tabpfn/logs/${tag}.log" 2>/dev/null; then
        say "encoder ${tag}: done"; return
    fi
    [ -f "$w" ] || { say "encoder ${tag}: SKIPPED, missing $w"; return; }
    # Resume from this encoder's own best checkpoint if a previous attempt was cut
    # short. The machine rebooted mid-training on 23 Oct and the AgeDB encoder was
    # silently carried forward at step 250 of 3000, which invalidated that whole
    # comparison; warm-starting means a reboot costs minutes, not the run.
    local WARM=""
    [ -f "ecg_tabpfn/ckpt/${tag}.pt" ] && WARM="--warm-start ../ecg_tabpfn/ckpt/${tag}.pt" \
        && say "encoder ${tag}: resuming from its existing checkpoint"
    say "training encoder ${tag}"
    ( cd ecg_tabpfn && python3 -u train_encoder.py --steps 3000 --n-ctx 2048 --n-qry 512 \
        --lr 1e-4 --eval-every 200 --patience 8 --ctx-sampling uniform \
        --tag "$tag" --tabpfn-weights "../$w" $WARM ) >> "ecg_tabpfn/logs/${tag}.log" 2>&1 \
        || say "encoder ${tag} FAILED"
}
train_enc enc_e21 ecg_tabpfn/ckpt/tabpfn_ft_gauss_E1.pt
train_enc enc_e22 ecg_tabpfn/ckpt/tabpfn_ft_mixed_E1.pt

# ---- Phase 4: embeddings ----------------------------------------------------
say "########## Phase 4: embeddings ##########"
for SPEC in "enc_e21 emb_enc_e21" "enc_e22 emb_enc_e22"; do
    set -- $SPEC
    [ -f "ecg_tabpfn/results/$2.npz" ] && { say "$2: present"; continue; }
    [ -f "ecg_tabpfn/ckpt/$1.pt" ] || { say "$2: SKIPPED, no checkpoint"; continue; }
    python3 -u ecg_tabpfn/extract_embeddings.py --ckpt "ecg_tabpfn/ckpt/$1.pt" \
        --out "ecg_tabpfn/results/$2.npz" || say "extract $2 FAILED"
done

# ---- Phase 5: final arms on E2.1 / E2.2 ------------------------------------
say "########## Phase 5: arms on E2.1 / E2.2 ##########"
for SPEC in "ptbxl_enc_e21 ft_gauss" "ptbxl_enc_e22 ft_mixed"; do
    set -- $SPEC
    [ -f "ecg_tabpfn/results/emb_${1#ptbxl_enc_}.npz" ] 2>/dev/null
    ( cd imbreg && python3 run.py --dataset "$1" --protocol dir --seeds "$SEEDS" \
        --tag dir_baseline --arms zeroshot ) || say "$1/zeroshot FAILED"
    arms_for "$1" "$2"
done

say "########## Phase 6: table ##########"
python3 -u make_table1.py || say "table generation FAILED"
say "TABLE1_COMPLETE"
touch .table1_done
