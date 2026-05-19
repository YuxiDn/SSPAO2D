#!/usr/bin/env bash
set -u

cd /data/yxdeng993/code/SSPAO2D || exit 1
source /data/yxdeng993/miniconda3/bin/activate rcan

log="outputs/sfenet2d_full_enc128_bs4_val400/screen_train.log"
{
  echo "===== $(date -Is) screen launch pid=$$ ====="
  echo "cwd=$(pwd)"
  echo "CUDA_VISIBLE_DEVICES=1,5"
} >> "$log"

CUDA_VISIBLE_DEVICES=1,5 torchrun --standalone --nproc_per_node=2 \
  scripts/train_supervised.py \
  -c configs/supervised_sfenet2d_full.json \
  -o outputs/sfenet2d_full_enc128_bs4_val400 \
  --resume outputs/sfenet2d_full_enc128_bs4_val400/last.pt \
  --resume-optimizer \
  --continue-epoch-numbers \
  --append-metrics >> "$log" 2>&1

status=$?
echo "===== $(date -Is) screen train exited status=$status =====" >> "$log"
exit "$status"
