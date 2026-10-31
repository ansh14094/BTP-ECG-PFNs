#!/usr/bin/env bash
# Resume everything still outstanding for Experiment 2 (AgeDB + TabICL).
# Safe to run at @reboot or by hand: each stage holds a lock and skips completed work.
#
# GPU discipline: at most the two AgeDB encoders run together (~60 GB of the 95 GB
# card). Everything after that is strictly sequential, because a TabICL fine-tune on
# the PTB-XL embeddings alone wants ~30 GB and running these concurrently caused
# CUDA OOM cascades earlier.
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p agedb_tabpfn/logs
say() { echo "[$(date -Is)] $*"; }

[ -f .exp2_done ] && { say "experiment 2 already complete"; exit 0; }

# Relaunch any AgeDB encoder that has not finished. Resumes from its own best
# checkpoint rather than restarting from ImageNet weights.
for OBJ in tabpfn l1; do
    LOG="agedb_tabpfn/logs/enc_${OBJ}.log"
    grep -q "ENCODER_TRAINING_COMPLETE" "$LOG" 2>/dev/null && { say "encoder ${OBJ}: done"; continue; }
    pgrep -f "train_encoder.py --objective ${OBJ}" > /dev/null && { say "encoder ${OBJ}: running"; continue; }
    say "encoder ${OBJ}: launching"
    ( cd agedb_tabpfn && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        setsid nohup python3 -u train_encoder.py --objective "$OBJ" --steps 3000 \
          --n-ctx 1024 --n-qry 256 --lr 1e-4 --eval-every 250 --patience 8 \
          --tag "enc_${OBJ}" < /dev/null >> "logs/enc_${OBJ}.log" 2>&1 & )
done

# The orchestrator waits for the encoders, then runs every remaining stage in order.
if pgrep -f "run_experiment2.sh" > /dev/null; then
    say "orchestrator already running"
else
    say "launching orchestrator"
    setsid nohup ./run_experiment2.sh < /dev/null >> logs_exp2.log 2>&1 &
fi
say "resume_exp2 dispatched"
