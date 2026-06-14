#!/bin/bash
set -euo pipefail

source "${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-dcl-agentic-rl}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
TAU_BENCH_PATH=${TAU_BENCH_PATH:-$(pwd)/../tau-bench}
VERL_PATH=${VERL_PATH:-$(pwd)/../verl}
export PYTHONPATH="$(pwd)/src:${TAU_BENCH_PATH}:${VERL_PATH}:${PYTHONPATH:-}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

mkdir -p experiments/delta_ledger_grpo_8x4090 outputs/grpo_delta_traces

python -m verl.trainer.main_ppo \
  --config-path="$(pwd)/configs/train/grpo" \
  --config-name=delta_ledger_grpo_8x4090_32b_user
