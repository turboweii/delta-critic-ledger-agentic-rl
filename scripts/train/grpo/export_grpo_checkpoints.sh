#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${ROOT_DIR}"

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-experiments/delta_ledger_grpo_8x4090/checkpoints}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/delta_ledger_grpo_8x4090}
BASE_MODEL=${BASE_MODEL:-experiments/sft_lora_merged}
STEPS=${STEPS:-50,100,150,200,300}
STRICT=${STRICT:-1}

IFS=',' read -ra step_values <<< "${STEPS}"
for step in "${step_values[@]}"; do
  actor_dir="${CHECKPOINT_ROOT}/global_step_${step}/actor"
  if [[ ! -d "${actor_dir}/lora_adapter" ]]; then
    if [[ "${STRICT}" == "1" ]]; then
      echo "Missing checkpoint: ${actor_dir}/lora_adapter" >&2
      exit 1
    fi
    echo "Skipping missing checkpoint: ${actor_dir}" >&2
    continue
  fi
  python3 scripts/train/grpo/export_grpo_checkpoint.py \
    --base-model "${BASE_MODEL}" \
    --checkpoint "${actor_dir}" \
    --output "${OUTPUT_ROOT}/hf_step_${step}"
done
