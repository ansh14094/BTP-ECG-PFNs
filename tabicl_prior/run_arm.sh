#!/usr/bin/env bash
# Timing trial: scaled-down continued pre-training, to measure seconds/step.
# Architecture flags are copied verbatim from scripts/train_v2_reg_stage3.sh and
# MUST match the released checkpoint. Only data size and step count are scaled.
set -u
cd "$(dirname "$0")" || exit 1
CKPT=$(ls ~/.cache/huggingface/hub/models--jingang--TabICL/snapshots/*/tabicl-regressor-v2-*.ckpt | head -1)
MODE="${1:-gauss}"; STEPS="${2:-3000}"; OUT="${3:-runs/$MODE}"
mkdir -p "$OUT"
export PYTHONPATH="$(pwd)"
python3 -u train_gauss_prior.py \
    --reorder_mode "$MODE" --reorder_seed 0 \
    --wandb_log False --device cuda --dtype float32 \
    --np_seed 43 --torch_seed 43 \
    --max_steps "$STEPS" --batch_size 8 --micro_batch_size 1 \
    --lr 2e-5 --muon True --beta1 0.9 --weight_decay 0.01 --use_cautious_wd False \
    --scheduler cosine_with_restarts --warmup_proportion 0.01 \
    --cosine_num_cycles 1 --cosine_amplitude_decay 1 --cosine_lr_end 1e-7 \
    --gradient_clipping 1.0 \
    --regression_method quantile --num_quantiles 999 \
    --prior_type graph_scm --prior_device cpu --n_jobs 8 \
    --batch_size_per_gp 1 --min_features 1 --max_features 100 \
    --min_seq_len 400 --max_seq_len 3000 --log_seq_len True \
    --min_train_size 0.79 --max_train_size 0.81 --seq_len_per_gp True \
    --graph_noise False --filter_unpredictable_graphs True \
    --filter_unpredictable_datasets True --allow_act_warping False \
    --min_n_nodes 2 --max_n_nodes 32 --cauchy_dag_offset 0.0 \
    --embed_dim 128 --col_num_blocks 3 --col_nhead 8 --col_num_inds 128 \
    --col_affine False --col_feature_group same --col_feature_group_size 3 \
    --col_target_aware True --col_ssmax True \
    --row_num_blocks 3 --row_nhead 8 --row_num_cls 4 \
    --row_rope_base 100000 --row_rope_interleaved False \
    --icl_num_blocks 12 --icl_nhead 8 --icl_ssmax True \
    --ssmax_type qassmax-mlp-elementwise --ff_factor 2 --norm_first True \
    --zero_init False --use_flash_attn3 False --norm_type layernorm_nobias \
    --recompute True \
    --checkpoint_dir "$OUT" --checkpoint_path "$CKPT" --only_load_model True \
    --save_temp_every 250 --save_perm_every 1000
