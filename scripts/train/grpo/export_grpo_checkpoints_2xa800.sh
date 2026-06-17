#!/bin/bash
set -euo pipefail

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-experiments/delta_ledger_grpo_2xa800/checkpoints}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/delta_ledger_grpo_2xa800}
BASE_MODEL=${BASE_MODEL:-experiments/sft_lora_merged}
STEPS=${STEPS:-50,100,150,200,300}

CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
BASE_MODEL="${BASE_MODEL}" \
STEPS="${STEPS}" \
bash scripts/train/grpo/export_grpo_checkpoints.sh
