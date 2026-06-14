#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python3 scripts/train/sft/collect_sft_data.py \
  --mode oracle \
  --output-dir experiments/sft_collect_airline \
  --env-name airline \
  --task-split test \
  --start-index 0 \
  --end-index 50

python3 scripts/train/sft/train_sft_lora.py \
  --config configs/train/sft/sft_airline_lora_4x4090.yaml
