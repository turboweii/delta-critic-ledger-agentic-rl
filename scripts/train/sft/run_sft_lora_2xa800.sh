#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

torchrun --standalone --nproc_per_node=2 scripts/train/sft/train_sft_lora.py \
  --config configs/train/sft/sft_airline_lora_2xa800_80g.yaml
