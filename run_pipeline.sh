#!/bin/bash
# Full pipeline: train the ResNet1D age regressor, then run the TabPFN
# feature-extraction experiment on its best checkpoint.
#
# Safe to run repeatedly and from @reboot: it takes a lock, skips stages that
# already finished, and resumes training from the last epoch checkpoint.
set -uo pipefail
cd "/mnt/storage/Aryan LBP/BTP" || exit 1

DONE_MARKER=".pipeline_done"
EPOCHS=90
STORE_ROOT=checkpoint_full
STORE_NAME=full
RUN_DIR="${STORE_ROOT}/1000_timesteps_resnet1d_wang_fds_${STORE_NAME}_adam_l1_0.001_32"
CKPT_LAST="${RUN_DIR}/ckpt.pth.tar"
CKPT_BEST="${RUN_DIR}/ckpt.best.pth.tar"
TRAIN_DONE="${RUN_DIR}/.train_done"
TABPFN_RESULT="tabpfn_results/tabpfn_preds_test.csv"

if [ -f "$DONE_MARKER" ]; then
    echo "[$(date)] Pipeline already completed ($DONE_MARKER present); nothing to do."
    exit 0
fi

# Single instance only, so an @reboot run cannot collide with a manual one.
exec 9>".pipeline.lock"
if ! flock -n 9; then
    echo "[$(date)] Another instance holds the lock; exiting."
    exit 0
fi

if [ -f /home/aryanbtp/.config/tabpfn/token.env ]; then
    set -a; . /home/aryanbtp/.config/tabpfn/token.env; set +a
fi

# Physical GPU order is not stable across reboots, so select the big card by name.
GPU_IDX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | grep -i "RTX PRO 6000" | cut -d, -f1 | tr -d ' ')
if [ -z "$GPU_IDX" ]; then
    echo "[$(date)] Could not find the RTX PRO 6000; aborting rather than falling back to the 4GB T400"
    echo "PIPELINE_FAILED"
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDX"
echo "[$(date)] Using GPU index $GPU_IDX"

# ---------------------------------------------------------------- STEP 1
if [ -f "$TRAIN_DONE" ]; then
    echo "[$(date)] STEP 1 already done, skipping training."
else
    RESUME_ARG=""
    if [ -f "$CKPT_LAST" ]; then
        RESUME_ARG="--resume $CKPT_LAST"
        echo "[$(date)] STEP 1: resuming training from $CKPT_LAST"
    else
        # train.py prompts interactively when its run dir exists without a
        # checkpoint to resume, which would hang a detached run.
        rm -rf "$RUN_DIR"
        echo "[$(date)] STEP 1: training from scratch ($EPOCHS epochs)"
    fi

    # shellcheck disable=SC2086
    python3 train.py --data_dir ./data --dataset 1000_timesteps --epoch "$EPOCHS" \
        --batch_size 32 --workers 4 --store_root "$STORE_ROOT" --store_name "$STORE_NAME" \
        $RESUME_ARG
    if [ $? -ne 0 ]; then
        echo "[$(date)] STEP 1 FAILED"
        echo "PIPELINE_FAILED"
        exit 1
    fi
    touch "$TRAIN_DONE"
    echo "[$(date)] STEP 1 DONE"
fi

if [ ! -f "$CKPT_BEST" ]; then
    echo "[$(date)] Expected checkpoint missing: $CKPT_BEST"
    echo "PIPELINE_FAILED"
    exit 1
fi

# ---------------------------------------------------------------- STEP 2
if [ -f "$TABPFN_RESULT" ]; then
    echo "[$(date)] STEP 2 already done, skipping TabPFN."
else
    echo "[$(date)] STEP 2: TabPFN feature-extraction experiment"
    python3 tabpfn_feature_experiment.py --resume "$CKPT_BEST" --model resnet1d_wang_fds \
        --data_dir ./data --dataset 1000_timesteps
    if [ $? -ne 0 ]; then
        echo "[$(date)] STEP 2 FAILED"
        echo "PIPELINE_FAILED"
        exit 1
    fi
    echo "[$(date)] STEP 2 DONE"
fi

touch "$DONE_MARKER"
echo "[$(date)] PIPELINE_COMPLETE"
