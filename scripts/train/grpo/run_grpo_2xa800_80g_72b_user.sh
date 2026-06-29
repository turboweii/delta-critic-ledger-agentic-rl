#!/bin/bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-dcl-agentic-rl}
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  activated=0
  for profile in \
    "${CONDA_PROFILE:-}" \
    "${HOME}/miniconda/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -n "${profile}" && -f "${profile}" ]]; then
      source "${profile}"
      conda activate "${CONDA_ENV}"
      activated=1
      break
    fi
  done
  [[ "${activated}" == "1" ]] || {
    echo "Cannot activate ${CONDA_ENV}; activate it before running or set CONDA_PROFILE." >&2
    exit 1
  }
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TAU_BENCH_PATH=${TAU_BENCH_PATH:-$(pwd)/../tau-bench}
if [[ -z "${VERL_PATH:-}" ]]; then
  if [[ -d "$(pwd)/../verl/verl" ]]; then
    VERL_PATH="$(pwd)/../verl"
  elif [[ -d "/home/turbo/llm/agentic-grpo/verl/verl" ]]; then
    VERL_PATH="/home/turbo/llm/agentic-grpo/verl"
  else
    VERL_PATH="$(pwd)/../verl"
  fi
fi
export VERL_PATH
export PYTHONPATH="$(pwd)/src:${TAU_BENCH_PATH}:${VERL_PATH}:${PYTHONPATH:-}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://localhost:8001/v1}
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://localhost:8001/v1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

mkdir -p experiments/grpo_2xa800

python scripts/setup/patch_verl_vllm_compat.py

python -m verl.trainer.main_ppo \
  --config-path="$(pwd)/configs/train/grpo" \
  --config-name=grpo_2xa800_80g_72b_user \
  2>&1 | tee experiments/grpo_2xa800/train.log
