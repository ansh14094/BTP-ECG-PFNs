#!/usr/bin/env bash
# One (encoder, protocol) combination of the PTB-XL grid.
#
# The four combinations write to disjoint result files and never interact, so running
# them concurrently is safe and uses the GPU far better than a sequential driver
# (each needs about 3 GB of 98 GB). Measured sequentially the full grid was ~7.5 h,
# because a fine-tune cell on 7566 rows x 128 features takes ~146 s and 176 remain.
#
# Usage: ./run_cell.sh <dataset> <protocol>
# Resumable: run.py skips (arm, seed) cells already recorded.

set -u
cd "$(dirname "$0")/../imbreg" || exit 1
DS="${1:?dataset}"; PROTO="${2:?protocol}"
SEEDS="${SEEDS:-0,1,2,3,4}"; EPOCHS="${EPOCHS:-30}"; LRS="${LRS:-1e-5 3e-5 1e-4}"
PRE=""; [ "$PROTO" = "dir" ] && PRE="dir_"

exec 9> ".ecg_${DS}_${PROTO}.lock"
flock -n 9 || { echo "[$(date -Is)] ${DS}/${PROTO}: lock held; exiting"; exit 0; }
[ -f "../ecg_tabpfn/.done_${DS}_${PROTO}" ] && { echo "[$(date -Is)] ${DS}/${PROTO}: done"; exit 0; }

echo "[$(date -Is)] ########## ${DS} [${PROTO}]: start ##########"
python3 run.py --dataset "$DS" --protocol "$PROTO" --seeds "$SEEDS" \
    --tag "${PRE}baseline" --arms lgbm,zeroshot,zeroshot_recal,zeroshot_recal_iso \
    || echo "[$(date -Is)] ${DS}/${PROTO}/baseline FAILED"
for LR in $LRS; do
    echo "[$(date -Is)] ${DS} [${PROTO}]: lr=${LR}"
    python3 run.py --dataset "$DS" --protocol "$PROTO" --seeds "$SEEDS" \
        --epochs "$EPOCHS" --tag "${PRE}lr${LR}" \
        --arms ft_uniform,ft_gauss,ft_gauss_recal --lr "$LR" \
        || echo "[$(date -Is)] ${DS}/${PROTO}/lr${LR} FAILED"
done
echo "[$(date -Is)] ########## ${DS} [${PROTO}]: CELL_COMPLETE ##########"
touch "../ecg_tabpfn/.done_${DS}_${PROTO}"
