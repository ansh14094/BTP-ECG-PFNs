#!/usr/bin/env bash
# Waits for the Timestral embeddings and the calibration run, then starts the
# Timestral-encoder arms. Keeps the GPU to one training job at a time.
set -u
cd "$(dirname "$0")" || exit 1
log() { echo "[$(date -Is)] $*"; }

log "waiting for timestral embeddings..."
until [ -f ecg_tabpfn/results/emb_timestral_pca.npz ]; do
    pgrep -f "timestral_encoder.py" > /dev/null || { log "extractor exited without output -- aborting"; exit 1; }
    sleep 30
done
log "embeddings present"

log "waiting for calibration_experiment.py to finish..."
while pgrep -f "calibration_experiment.py" > /dev/null; do sleep 60; done
log "calibration finished"

log "starting timestral-encoder arms"
exec ./run_timestral_encoder.sh
