#!/usr/bin/env bash
# One ordered chain for everything still outstanding, so the jobs cannot race each
# other for the shared GPU. Cheapest and most load-bearing first.
#
#   1. Task C   clean-split re-evaluation      (~1 h)  unblocks Tables 2 and 3
#   2. Task B1  target-balanced prior arm      (~20 m) Table 1a
#   3. Task A   ResNet baselines, 10 runs      (long)  Table 2 rows 1-2
#
# B1 moved ahead of A on 2026-11-10. Its 3000-step prior checkpoint has landed, so
# it is no longer blocked, and it is ~20 minutes against Task A's many hours. If the
# chain dies again, the cheap deliverable is already banked.
#
# Each stage has its own lock and sentinel, so this is safe to re-run: finished
# stages return immediately.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .runall.lock
flock -n 9 || { echo "[$(date -Is)] run_all already active"; exit 0; }
log() { echo "[$(date -Is)] $*"; }

log "===== stage 1: Task C (clean split) ====="
./run_taskC_clean.sh 2>&1 | sed 's/^/[C] /'

log "===== stage 2: Task B1 evaluation ====="
./run_taskB1_eval.sh 2>&1 | sed 's/^/[B1] /'

log "===== stage 3: Task A (ResNet baselines) ====="
./run_task_a.sh 2>&1 | sed 's/^/[A] /'

log "RUN_ALL_FINISHED"
