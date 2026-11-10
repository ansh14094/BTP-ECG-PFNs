#!/usr/bin/env bash
# Task A: ResNet baselines for PTB-XL age, under the unified protocol.
#
# Two arms, five seeds each:
#   vanilla       plain L1, no imbalance method
#   ranksim_conr  L1 + RankSim ranking regulariser + ConR contrastive loss
#
# Hyperparameters are copied from the original run that produced
# checkpoint_full/.../ckpt.best.pth.tar (adam, l1, lr 1e-3, batch 32, 90 epochs,
# schedule 60/80) so the new vanilla runs are comparable to it; that run reported
# test L1 7.5485, which is the number to sanity-check seed 0 against.
#
# FDS stays OFF in both arms. The model exposes its pooled encoding through
# --return_features now, so RankSim and ConR no longer drag FDS in with them.
#
# Evaluation uses the OFFICIAL PTB-XL test split, never the DIR resample: every
# encoder in this study was fitted on the official train split, so a DIR test set
# is ~80% records the encoder has already seen (results/STATUS.md).
#
# STRICTLY SEQUENTIAL behind the shared GPU mutex -- the workstation GPU is shared
# with another user and is already heavily loaded.
set -u
cd "$(dirname "$0")" || exit 1
exec 9> .taska.lock
flock -n 9 || { echo "[$(date -Is)] another run holds the lock"; exit 0; }
[ -f .taska_done ] && { echo "[$(date -Is)] already complete"; exit 0; }

. ./gpu_lock.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# train.py imports tensorboard_logger, whose bundled *_pb2.py files were generated
# against protobuf 3.x and cannot be constructed under the protobuf 7.x installed
# here -- the import dies with "Descriptors cannot be created directly" before a
# single epoch runs. This killed all 10 runs instantly on 2026-11-10.
#
# This is protobuf's own documented workaround: pure-Python parsing accepts the old
# generated code. It is slower, but tensorboard_logger only writes a few scalars per
# epoch, so the cost is nil. Preferred over downgrading protobuf system-wide, which
# would risk breaking tabicl/tabpfn for every other experiment and for the other user
# on this workstation.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Pin to the RTX PRO 6000 (95 GB) and hide the T400 (3.6 GB). train.py wraps the
# model in DataParallel over every visible device, so without this every batch is
# split across both cards and the tiny one sets the pace -- and can OOM outright.
#
# The two tools disagree on numbering: nvidia-smi orders by PCI bus (T400=0,
# RTX PRO=1) while torch defaults to fastest-first (RTX PRO=0). CUDA_VISIBLE_DEVICES
# is read in PCI order, so PCI_BUS_ID is forced here to make "1" unambiguous.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
log() { echo "[$(date -Is)] $*"; }
FAILED=0
EPOCHS="${TASKA_EPOCHS:-90}"
STORE=checkpoint_taskA

train_one() {  # arm, seed, extra flags...
    local ARM="$1" SEED="$2"; shift 2
    # Locate the run directory by glob rather than reconstructing it. train.py
    # composes the name itself, and two details are easy to get wrong: it appends
    # --store_name to the model name, giving a DOUBLE underscore
    # ("..._wang_fds__vanilla_..."), and for --ranking/--conr it also embeds every
    # regulariser hyperparameter ("..._ranking_0.1_0.1_conr_1.0_0.1_0.01_...").
    # Hardcoding the name meant all 10 runs on 2026-11-10 trained successfully and
    # were then reported as "no checkpoint", because the path never matched.
    local NAME
    NAME=$(cd "$STORE" 2>/dev/null && ls -d 1000_timesteps_resnet1d_wang_fds__${ARM}*_seed${SEED} 2>/dev/null | head -1)
    if [ -n "$NAME" ] && [ -f "$STORE/$NAME/ckpt.best.pth.tar" ]; then
        log "### $ARM seed $SEED already trained; skipping"
    else
        log "### training $ARM seed $SEED ($EPOCHS epochs)"
        python3 -u train.py --dataset 1000_timesteps --data_dir ./data \
            --model resnet1d_wang_fds --store_root "$STORE" --store_name "_${ARM}" \
            --optimizer adam --loss l1 --lr 0.001 --batch_size 32 \
            --epoch "$EPOCHS" --schedule 60 80 --workers 4 --seed "$SEED" "$@" \
            || { log "$ARM seed $SEED TRAIN FAILED"; FAILED=1; return 1; }
        NAME=$(cd "$STORE" 2>/dev/null && ls -d 1000_timesteps_resnet1d_wang_fds__${ARM}*_seed${SEED} 2>/dev/null | head -1)
    fi
    [ -n "$NAME" ] && [ -f "$STORE/$NAME/ckpt.best.pth.tar" ] || { log "$ARM seed $SEED no checkpoint"; FAILED=1; return 1; }
    python3 -u eval/predict_resnet.py --ckpt "$STORE/$NAME/ckpt.best.pth.tar" \
        --arm "$ARM" --seed "$SEED" --tag taskA \
        || { log "$ARM seed $SEED PREDICT FAILED"; FAILED=1; }
}

gpu_lock
for S in 0 1 2 3 4; do train_one vanilla "$S"; done
for S in 0 1 2 3 4; do train_one ranksim_conr "$S" --ranking --conr; done

if [ "$FAILED" -eq 0 ]; then
    touch .taska_done; log "TASK_A_COMPLETE"
else
    log "TASK_A_INCOMPLETE -- sentinel not written"
fi
