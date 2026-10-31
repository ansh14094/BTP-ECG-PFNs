#!/usr/bin/env bash
# Start the prior-level run only once the TabICL gap-fill has released the GPU.
# Serialising these is not optional: running them together already destroyed one
# full attempt (both arms OOMed).
set -u
cd "$(dirname "$0")" || exit 1
echo "[$(date -Is)] waiting for run_ticl_gaps to finish..."
while pgrep -f run_ticl_gaps > /dev/null; do sleep 120; done
echo "[$(date -Is)] gap-fill done; GPU should be clear"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tail -1
cd tabicl_prior || exit 1
exec ./run_both.sh 3000
