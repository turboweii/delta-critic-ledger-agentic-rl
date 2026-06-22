#!/bin/bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-dcl-agentic-rl}
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  source "${CONDA_PROFILE:-${HOME}/miniconda/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV}"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export PYTHONPATH="$(pwd)/src:${TAU_BENCH_PATH:-$(pwd)/../tau-bench}:${VERL_PATH:-$(pwd)/../verl}:${PYTHONPATH:-}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
VERL_CONFIG_PATH=${VERL_CONFIG_PATH:-$(pwd)/../verl/verl/trainer/config}

mkdir -p experiments/delta_ledger_grpo_4x3090_smoke \
  outputs/grpo_delta_traces_4x3090_smoke

python scripts/setup/patch_verl_vllm_compat.py

python -m verl.trainer.main_ppo \
  --config-path="$(pwd)/configs/train/grpo" \
  --config-name=delta_ledger_grpo_4x3090_smoke \
  "hydra.searchpath=[file://${VERL_CONFIG_PATH}]" \
  "$@"
