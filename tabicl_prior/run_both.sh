#!/usr/bin/env bash
# Experiment 6: continued pre-training of TabICL on its own prior, Gaussian vs
# uniform row ordering. STRICTLY SEQUENTIAL -- one GPU, one training job.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .tsicl.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .tsicl_done ] && { echo "[$(date -Is)] already complete"; exit 0; }
STEPS="${1:-3000}"

# This job must own the GPU. A concurrent run already cost one full attempt:
# both arms OOMed because a TabICL fine-tuning sweep held memory alongside them.
# `nice` does NOT help here -- it is CPU priority and has no effect on GPU memory.
while pgrep -f "run\.py --dataset" > /dev/null || pgrep -f "run_ticl_gaps" > /dev/null; do
    echo "[$(date -Is)] waiting for the GPU to clear before starting..."
    sleep 120
done

FAILED=0
for M in uniform gauss; do
    if [ -f "runs/$M/.done" ]; then echo "[$(date -Is)] $M already done"; continue; fi
    echo "[$(date -Is)] ===== arm $M ($STEPS steps) ====="
    mkdir -p "runs/$M"
    if ./run_arm.sh "$M" "$STEPS" "runs/$M"; then
        touch "runs/$M/.done"
    else
        echo "[$(date -Is)] arm $M FAILED"
        FAILED=1
    fi
done

# Only claim completion if both arms actually succeeded. The previous version
# touched this unconditionally and reported TSICL_COMPLETE after both arms died.
if [ "$FAILED" -eq 0 ]; then
    touch .tsicl_done
    echo "[$(date -Is)] TSICL_COMPLETE"
else
    echo "[$(date -Is)] TSICL_INCOMPLETE -- at least one arm failed; sentinel NOT written"
fi
