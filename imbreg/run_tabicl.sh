#!/usr/bin/env bash
# Does the Gaussian-context idea transfer to a DIFFERENT in-context model?
#
# Same intervention, same datasets, same protocol as the TabPFN study, but with
# TabICL v2 as the predictor. If ft_gauss beats ft_uniform here too, the effect is a
# property of in-context learning rather than of TabPFN specifically -- which is the
# question worth answering and the one a reviewer will ask.
#
# Both fine-tuning arms get the same learning-rate sweep and are reported at their own
# validation-selected rate, as in the TabPFN runs.
#
# Usage: ./run_tabicl.sh <dataset>     (one dataset per process; they are independent)
set -u
cd "$(dirname "$0")" || exit 1
DS="${1:?dataset}"; SEEDS="${SEEDS:-0,1,2,3,4}"; EPOCHS="${EPOCHS:-30}"
LRS="${LRS:-1e-5 3e-5 1e-4}"
PROTO="${PROTO:-dir}"; PRE=""; [ "$PROTO" = "dir" ] && [ "$DS" != "${DS#ptbxl}" ] && PRE="dir_"

exec 9> ".ticl_${DS}.lock"
flock -n 9 || { echo "[$(date -Is)] ${DS}: lock held"; exit 0; }
[ -f ".done_ticl_${DS}" ] && { echo "[$(date -Is)] ${DS}: done"; exit 0; }

echo "[$(date -Is)] ########## TabICL ${DS} start ##########"
python3 run.py --dataset "$DS" --protocol "$PROTO" --seeds "$SEEDS" \
    --tag "${PRE}ticl_base" --arms ticl_zeroshot \
    || echo "[$(date -Is)] ${DS}/ticl_base FAILED"
for LR in $LRS; do
    echo "[$(date -Is)] ${DS}: lr=${LR}"
    python3 run.py --dataset "$DS" --protocol "$PROTO" --seeds "$SEEDS" \
        --epochs "$EPOCHS" --tag "${PRE}ticl_lr${LR}" \
        --arms ticl_ft_uniform,ticl_ft_gauss --lr "$LR" \
        || echo "[$(date -Is)] ${DS}/ticl_lr${LR} FAILED"
done
echo "[$(date -Is)] ########## TabICL ${DS} CELL_COMPLETE ##########"
touch ".done_ticl_${DS}"
