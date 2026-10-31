#!/usr/bin/env bash
# Relaunch every dataset's run. Safe to invoke at any time, including @reboot.
#
# Nothing computes while the machine is off. This only restarts the work on the next
# boot. Correctness rests on two properties:
#   * run_one.sh takes a per-dataset flock, so this cannot collide with a run that is
#     already going (it exits immediately for datasets already in flight)
#   * run.py skips (arm, seed) cells already recorded, so a restart costs at most the
#     single cell that was in flight when the power went
#
# Called from cron via an absolute path, because cron's environment is minimal.

set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs results

for d in abalone communities wine_white; do
    if [ -f ".done_${d}" ]; then
        echo "[$(date -Is)] ${d}: already complete, skipping"
        continue
    fi
    echo "[$(date -Is)] ${d}: launching"
    setsid nohup ./run_one.sh "$d" < /dev/null >> "logs/${d}.log" 2>&1 &
done
wait 2>/dev/null || true
echo "[$(date -Is)] resume_all dispatched"
