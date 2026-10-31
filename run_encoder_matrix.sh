#!/usr/bin/env bash
# The encoder x body matrix.
#
# Three new encoders, each the SAME ResNet1d-Wang backbone from the same warm start,
# trained end-to-end through a different FROZEN TabICL body:
#   E3  stock released TabICL
#   E4  TabICL continued-pretrained with Gaussian context ordering
#   E5  TabICL continued-pretrained with mixed (half Gaussian) ordering
#
# Then every encoder is evaluated under every body, zero-shot. No body has ever seen
# PTB-XL: all three differ only by a 3000-step continuation on TabICL's own synthetic
# prior, so the whole matrix is dataset-independent on the model side.
#
# E1 and E2 (the TabPFN-era encoders) are evaluated under the same four bodies too.
# Their embeddings already exist, evaluation is zero-shot and cheap, and without them
# there is nothing to say whether a TabICL-trained encoder beats the ones we had.
#
# The uniform body is carried as the matched control for gauss: same step count, same
# prior, unbiased ordering. Without it, "gauss helped" cannot be separated from
# "3000 more steps helped".
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .encmatrix.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .encmatrix_done ] && { echo "[$(date -Is)] already complete"; exit 0; }

. ./gpu_lock.sh
log() { echo "[$(date -Is)] $*"; }

ENC_DIR=ecg_tabpfn
STEPS="${ENC_STEPS:-3000}"
FAILED=0

# ---------------------------------------------------------------- stage 1
# E5 needs the mixed checkpoint, which trains in its own job. Wait for the file
# rather than for a process name.
wait_for_ckpt() {
    local f="$1" waited=0
    while [ ! -f "$f" ]; do
        if [ "$waited" -ge 21600 ]; then log "TIMEOUT waiting for $f"; return 1; fi
        sleep 120; waited=$((waited + 120))
    done
}

train_one() {  # tag, body
    local TAG="$1" BODY="$2"
    if [ -f "$ENC_DIR/ckpt/${TAG}.pt" ]; then
        log "### $TAG already trained; skipping"; return 0
    fi
    log "### training $TAG (body=$BODY, $STEPS steps)"
    ( cd "$ENC_DIR" && python3 -u train_encoder_tabicl.py \
        --body "$BODY" --tag "$TAG" --steps "$STEPS" --seed 0 ) \
        || { log "$TAG FAILED"; FAILED=1; return 1; }
    [ -f "$ENC_DIR/ckpt/${TAG}.pt" ] || { log "$TAG produced no checkpoint"; FAILED=1; }
}

extract_one() {  # tag, outname
    local TAG="$1" OUT="$2"
    if [ -f "$ENC_DIR/results/${OUT}" ]; then
        log "### $OUT already extracted; skipping"; return 0
    fi
    log "### extracting embeddings for $TAG -> $OUT"
    ( cd "$ENC_DIR" && python3 -u extract_embeddings.py \
        --ckpt "ckpt/${TAG}.pt" --out "results/${OUT}" ) \
        || { log "$OUT extraction FAILED"; FAILED=1; }
}

gpu_lock
train_one enc_ticl_e3 stock
train_one enc_ticl_e4 gauss
wait_for_ckpt tabicl_prior/runs/mixed/step-3000.ckpt && train_one enc_ticl_e5 mixed

extract_one enc_ticl_e3 emb_enc_e3.npz
extract_one enc_ticl_e4 emb_enc_e4.npz
extract_one enc_ticl_e5 emb_enc_e5.npz

# ---------------------------------------------------------------- stage 2
# Zero-shot evaluation of every encoder under every body. Results append to one
# jsonl per encoder so the matrix assembles from the recorded rows.
ARMS=ticl_zeroshot,ticl_prior_uniform,ticl_prior_gauss,ticl_prior_mixed
log "### evaluating the matrix: arms=$ARMS"
cd imbreg || exit 1
for DS in ptbxl_enc_e3 ptbxl_enc_e4 ptbxl_enc_e5 ptbxl_l1_enc ptbxl_tabpfn_enc; do
    log "### $DS"
    python3 -u run.py --dataset "$DS" --protocol dir --tag dir_matrix \
        --arms "$ARMS" --seeds 0,1,2,3,4 || { log "$DS eval FAILED"; FAILED=1; }
done

log "### completeness check"
FILES=""
for DS in ptbxl_enc_e3 ptbxl_enc_e4 ptbxl_enc_e5 ptbxl_l1_enc ptbxl_tabpfn_enc; do
    FILES="$FILES results/${DS}_dir_matrix.jsonl"
done
python3 check_cells.py $FILES --arms "$ARMS" --seeds 0,1,2,3,4 || FAILED=1
cd ..

if [ "$FAILED" -eq 0 ]; then
    touch .encmatrix_done; log "ENCODER_MATRIX_COMPLETE"
else
    log "ENCODER_MATRIX_INCOMPLETE -- sentinel not written"
fi
