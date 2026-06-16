#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${ROOT_DIR}"

VARIANTS=${VARIANTS:-terminal_only,delta_only,delta_ledger,delta_ledger_adaptive}
STEPS=${STEPS:-300}
DRY_RUN=${DRY_RUN:-0}
ASSISTANT_GPUS=${ASSISTANT_GPUS:-0}
OUT_ROOT=${OUT_ROOT:-outputs/ablation_checkpoint_eval}

mkdir -p "${OUT_ROOT}/configs"

IFS=',' read -ra selected_variants <<< "${VARIANTS}"
IFS=',' read -ra selected_steps <<< "${STEPS}"

for variant in "${selected_variants[@]}"; do
  cfg="${OUT_ROOT}/configs/checkpoints_${variant}.yaml"
  {
    echo "experiment_name: ablation_checkpoint_eval/${variant}"
    echo "base_config: configs/eval/eval_airline_delta_grpo_8x4090_32b_user.yaml"
    echo "checkpoints:"
    for step in "${selected_steps[@]}"; do
      echo "  - step: ${step}"
      echo "    model_name: experiments/ablation_grpo_8x4090/${variant}/hf_step_${step}"
      echo "    served_model_name: delta-assistant-7b-${variant}-step${step}"
    done
  } > "${cfg}"

  echo "[eval-ablation:${variant}] config=${cfg}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    python3 scripts/eval/eval_checkpoints.py \
      --config "${cfg}" \
      --assistant-gpus "${ASSISTANT_GPUS}" \
      --dry-run
  else
    python3 scripts/eval/eval_checkpoints.py \
      --config "${cfg}" \
      --assistant-gpus "${ASSISTANT_GPUS}"
  fi
done

python3 scripts/eval/summarize_ablation_results.py \
  --root "${OUT_ROOT}" \
  --output "${OUT_ROOT}/summary.csv"
