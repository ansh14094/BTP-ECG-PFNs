#!/usr/bin/env bash
# Relaunch all four PTB-XL grid cells in parallel, then write the report when every
# cell has finished. Safe to call at @reboot or by hand: each cell takes its own
# flock and skips work already recorded.
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs
for DS in ptbxl_l1_enc ptbxl_tabpfn_enc; do
  for P in fixed dir; do
    [ -f ".done_${DS}_${P}" ] && continue
    # Bound threads per cell: four concurrent cells otherwise oversubscribe the
    # 48-core machine roughly 2x and slow every fine-tune cell by ~80%.
    OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 RUN_N_THREADS=10 LGBM_N_JOBS=8 \
      setsid nohup ./run_cell.sh "$DS" "$P" < /dev/null >> "logs/${DS}_${P}.log" 2>&1 &
  done
done
# Wait for all four sentinels, then generate the report.
setsid nohup bash -c '
  cd "'"$(pwd)"'"
  until [ -f .done_ptbxl_l1_enc_fixed ] && [ -f .done_ptbxl_l1_enc_dir ] \
     && [ -f .done_ptbxl_tabpfn_enc_fixed ] && [ -f .done_ptbxl_tabpfn_enc_dir ]; do
    sleep 60
  done
  python3 -u make_ecg_report.py >> logs/run_all.log 2>&1
  echo "[$(date -Is)] PTBXL_PIPELINE_COMPLETE" >> logs/run_all.log
  touch .pipeline_done
' < /dev/null > /dev/null 2>&1 &
echo "[$(date -Is)] dispatched parallel cells + report watcher"
