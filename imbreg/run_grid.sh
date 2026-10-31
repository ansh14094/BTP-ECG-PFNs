#!/usr/bin/env bash
# Experiment driver for the TabPFN-2.5 imbalanced-regression study.
#
# Survives session disconnect and an overnight power-off:
#   * flock      - a @reboot run cannot collide with a manual launch
#   * idempotent - run.py skips (arm, seed) cells already in its results file, so a
#                  restart resumes instead of recomputing
#   * sentinel   - writes .grid_done and logs GRID_COMPLETE when every stage is done
#
# Launch detached:  setsid nohup ./run_grid.sh < /dev/null >> logs/grid.log 2>&1 & disown
# Watch:            tail -f logs/grid.log
# Done when:        grep -q GRID_COMPLETE logs/grid.log

set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs results

exec 9> .grid.lock
if ! flock -n 9; then
    echo "[$(date -Is)] another grid run holds the lock; exiting"
    exit 0
fi

SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
EPOCHS="${EPOCHS:-60}"
PY=python3

stage() {
    local tag="$1"; shift
    echo "[$(date -Is)] === stage ${tag} ==="
    $PY run.py --dataset abalone --seeds "$SEEDS" --epochs "$EPOCHS" --tag "$tag" "$@" \
        || echo "[$(date -Is)] stage ${tag} FAILED (continuing)"
}

# --- reference arms ----------------------------------------------------------
stage baseline   --arms lgbm,zeroshot

# Inference-only remedies, i.e. the closest prior work plus the cheapest possible
# way to exploit a balanced validation set. zeroshot_recal is adversarial on
# purpose: a model that regresses toward the majority region yields compressed
# predictions, and an affine/isotonic correction undoes much of that for free. If it
# matches the fine-tuned arms, the fine-tuning is not what is doing the work.
stage prior_work --arms zeroshot_recal,zeroshot_recal_iso,ctx_balanced,imcontext

# --- learning-rate sweep, applied to BOTH fine-tuning arms -------------------
# Standard (uniform-context) fine-tuning turned out to be a no-op at lr=1e-5 on this
# dataset: early stopping kept reverting to the initial weights. Reporting that as
# "the fine-tuning baseline" would be a rigged comparison, so both arms get the same
# sweep and each is later reported at its own val-selected learning rate.
for LR in 1e-5 3e-5 1e-4 3e-4; do
    stage "lr${LR}" --arms ft_uniform,ft_gauss --lr "${LR}"
done

# --- ablations, each changing exactly one knob of the biased sampler ---------
ABL_LR="${ABL_LR:-1e-4}"
# Where the over-represented region sits: always the data's mode instead of a fresh
# random quantile per episode.
stage abl_center --arms ft_gauss --lr "$ABL_LR" --center-mode mode
# Query distribution: over-sample rare target regions in the loss.
stage abl_query  --arms ft_gauss --lr "$ABL_LR" --query-mode balanced
# Severity of the induced imbalance: a narrower Gaussian discards more of the tail.
stage abl_sigma  --arms ft_gauss --lr "$ABL_LR" --sigma-min 0.10 --sigma-max 0.40
# How aggressively the context is thinned. The uniform arm is re-run at the same
# ctx_keep_frac, otherwise this would confound weighting with context size.
stage abl_keep   --arms ft_uniform,ft_gauss --lr "$ABL_LR" --ctx-keep-frac 0.25

# --- natural test distribution ------------------------------------------------
# The headline protocol uses a DIR-balanced test set, which by construction rewards
# giving up majority-region accuracy for tail accuracy. A plain random split shows
# what that trade costs when the test set keeps the dataset's natural skew.
# Reporting only the balanced number would overstate the method.
echo "[$(date -Is)] === natural-split protocol ==="
for SPEC in "nat_baseline lgbm,zeroshot" "nat_main ft_uniform,ft_gauss"; do
    set -- $SPEC
    $PY run.py --dataset abalone --seeds "$SEEDS" --epochs "$EPOCHS" \
        --protocol natural --tag "$1" --arms "$2" --lr "$ABL_LR" \
        || echo "[$(date -Is)] stage $1 FAILED (continuing)"
done

echo "[$(date -Is)] GRID_COMPLETE"
touch .grid_done
