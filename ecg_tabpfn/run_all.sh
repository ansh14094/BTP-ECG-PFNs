#!/usr/bin/env bash
# End-to-end PTB-XL experiment: encoder -> embeddings -> arms -> report.
#
# Idempotent and resumable at every stage, so it is safe to invoke at @reboot or by
# hand while another copy is running:
#   * flock      - a second invocation exits instead of colliding
#   * stage skip - each stage detects its own completed output and returns
#   * run.py     - skips (arm, seed) cells already recorded
# Nothing computes while the machine is off; this only resumes on next boot.
#
# Launch:    setsid nohup ./run_all.sh < /dev/null >> logs/run_all.log 2>&1 & disown
# Watch:     tail -f logs/run_all.log
# Done when: grep -q PTBXL_PIPELINE_COMPLETE logs/run_all.log

set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs ckpt results

exec 7> .runall.lock
flock -n 7 || { echo "[$(date -Is)] another run_all holds the lock; exiting"; exit 0; }

ENC_LOG="logs/enc_main.log"
ENC_CKPT="ckpt/enc_main.pt"
STEPS="${STEPS:-3000}"
SEEDS="${SEEDS:-0,1,2,3,4}"

say() { echo "[$(date -Is)] $*"; }

# ---- stage 1: encoder trained through frozen TabPFN -------------------------
if grep -q "ENCODER_TRAINING_COMPLETE" "$ENC_LOG" 2>/dev/null; then
    say "stage 1: already complete"
elif pgrep -f "train_encoder.py" > /dev/null; then
    say "stage 1: already running elsewhere; waiting for it"
    while pgrep -f "train_encoder.py" > /dev/null; do sleep 30; done
    say "stage 1: the running job finished"
else
    # On a reboot-resume, continue from the best checkpoint rather than restarting
    # from the original L1 weights and discarding the work already done.
    WARM=""
    [ -f "$ENC_CKPT" ] && WARM="--warm-start $ENC_CKPT" && say "stage 1: resuming from $ENC_CKPT"
    say "stage 1: launching encoder training"
    python3 -u train_encoder.py --steps "$STEPS" --n-ctx 2048 --n-qry 512 \
        --lr 1e-4 --eval-every 200 --patience 8 --ctx-sampling uniform \
        --tag enc_main $WARM >> "$ENC_LOG" 2>&1 \
        || say "stage 1 FAILED"
fi

if [ ! -f "$ENC_CKPT" ]; then
    say "stage 1 produced no checkpoint; cannot continue"
    exit 1
fi

# ---- stage 2: embeddings ----------------------------------------------------
if [ ! -f results/emb_baseline.npz ]; then
    say "stage 2: extracting control (L1 backbone) embeddings"
    python3 -u extract_embeddings.py --baseline-ckpt \
        "../checkpoint_full/1000_timesteps_resnet1d_wang_fds_full_adam_l1_0.001_32/ckpt.best.pth.tar" \
        --out results/emb_baseline.npz || say "stage 2 (control) FAILED"
else
    say "stage 2: control embeddings already present"
fi

say "stage 2: extracting TabPFN-trained encoder embeddings"
python3 -u extract_embeddings.py --ckpt "$ENC_CKPT" \
    --out results/emb_enc_main.npz || { say "stage 2 FAILED"; exit 1; }

# ---- stage 3: arms, both protocols, both encoders ---------------------------
say "stage 3: running arms"
SEEDS="$SEEDS" ./run_ecg.sh || say "stage 3 reported failures (continuing to report)"

# ---- stage 4: report --------------------------------------------------------
say "stage 4: generating report"
python3 -u make_ecg_report.py || say "stage 4 FAILED"

say "PTBXL_PIPELINE_COMPLETE"
touch .pipeline_done
