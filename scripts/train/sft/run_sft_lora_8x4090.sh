#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

torchrun --standalone --nproc_per_node=8 scripts/train/sft/train_sft_lora.py \
  --config configs/train/sft/sft_airline_lora_8x4090.yaml
