# Shared GPU mutex for this project. Source it, then call gpu_lock.
#
# Why not `pgrep -f <script name>`: that pattern matches ANY process whose command
# line contains the string, including the waiting shell itself and any interactive
# diagnostic command that happens to mention the script. It cost 14 hours on
# 2026-10-30, when an AgeDB re-run sat waiting on a `pgrep` that was matching its
# own waiter. A lock file has no such ambiguity: a holder either exists or does not.
#
# flock here is BLOCKING, so a queued job sleeps in the kernel until the GPU frees
# instead of polling. The lock is released when the holding shell exits, including
# when it is killed, so a crashed job cannot wedge the queue.
#
# Note this guards OUR jobs only. The workstation's GPU is shared with another user,
# whose processes this cannot see; it bounds our own concurrency to one, which is
# what repeatedly went wrong.
GPU_LOCK_FILE="${GPU_LOCK_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.gpu.lock}"

gpu_lock() {
    exec 8> "$GPU_LOCK_FILE" || return 1
    if ! flock -n 8; then
        echo "[$(date -Is)] GPU held by another job in this project; queuing..."
        flock 8
    fi
    echo "[$(date -Is)] GPU lock acquired (pid $$)"
}
