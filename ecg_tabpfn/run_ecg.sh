#!/usr/bin/env bash
# Stage 3: run the imbalanced-regression arms on PTB-XL encoder embeddings.
#
# Two encoders, identical downstream treatment:
#   ptbxl_l1_enc      existing backbone trained on an L1 age loss        (control)
#   ptbxl_tabpfn_enc  encoder trained end-to-end through a frozen TabPFN-2.5
# Encoder-vs-encoder at the `zeroshot` arm isolates stage 1; ft_gauss-vs-ft_uniform
# within an encoder isolates stage 2.
#
# TWO PROTOCOLS, because neither alone answers the question:
#
#   fixed  PTB-XL's official train/val/test partition. Directly comparable with this
#          repo's earlier numbers (ResNet1d L1 test MAE 7.549; frozen backbone +
#          TabPFN 7.441). But its test set has only SIX few-shot points out of 954,
#          so it cannot measure tail performance -- which is what the method targets.
#
#   dir    Same data, val/test rebalanced over 1-year age bins (Yang et al., ICML
#          2021), giving 179 few-shot test points instead of 6. This is where the
#          tail claim can actually be evaluated. It is the same convention the
#          AgeDB-DIR work in this repo uses.
#
# zeroshot_recal_iso is included from the start: on all three tabular datasets it beat
# Gaussian-context fine-tuning on every tail metric, so omitting it would only
# postpone the same finding.
#
# Detached: setsid nohup ./run_ecg.sh < /dev/null >> logs/ecg_arms.log 2>&1 & disown

set -u
cd "$(dirname "$0")/../imbreg" || exit 1
mkdir -p ../ecg_tabpfn/logs results

exec 9> .ecg.lock
flock -n 9 || { echo "[$(date -Is)] another ecg run holds the lock"; exit 0; }

DATASETS="${DATASETS:-ptbxl_l1_enc ptbxl_tabpfn_enc}"
SEEDS="${SEEDS:-0,1,2,3,4}"
EPOCHS="${EPOCHS:-30}"
LRS="${LRS:-1e-5 3e-5 1e-4}"
BASE_ARMS="lgbm,zeroshot,zeroshot_recal,zeroshot_recal_iso"
FT_ARMS="ft_uniform,ft_gauss,ft_gauss_recal"

run_protocol() {               # $1 = protocol, $2 = tag prefix
    local proto="$1" pre="$2" DS="$3"
    python3 run.py --dataset "$DS" --protocol "$proto" --seeds "$SEEDS" \
        --tag "${pre}baseline" --arms "$BASE_ARMS" \
        || echo "[$(date -Is)] ${DS}/${proto}/baseline FAILED (continuing)"
    for LR in $LRS; do
        echo "[$(date -Is)] ${DS} [${proto}]: lr=${LR}"
        python3 run.py --dataset "$DS" --protocol "$proto" --seeds "$SEEDS" \
            --epochs "$EPOCHS" --tag "${pre}lr${LR}" --arms "$FT_ARMS" --lr "$LR" \
            || echo "[$(date -Is)] ${DS}/${proto}/lr${LR} FAILED (continuing)"
    done
}

for DS in $DATASETS; do
    [ -f ".done_${DS}" ] && { echo "[$(date -Is)] ${DS}: already complete"; continue; }
    echo "[$(date -Is)] ########## ${DS}: start ##########"
    run_protocol fixed ""     "$DS"
    run_protocol dir   "dir_" "$DS"
    echo "[$(date -Is)] ########## ${DS}: DATASET_COMPLETE ##########"
    touch ".done_${DS}"
done
echo "[$(date -Is)] ECG_ARMS_COMPLETE"
touch ../ecg_tabpfn/.arms_done
