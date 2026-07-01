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
  if [[ -d "$(pwd)/verl/verl" ]]; then
    VERL_PATH="$(pwd)/verl"
  elif [[ -d "$(pwd)/../verl/verl" ]]; then
    VERL_PATH="$(pwd)/../verl"
  elif [[ -d "/home/turbo/llm/agentic-grpo/verl/verl" ]]; then
    VERL_PATH="/home/turbo/llm/agentic-grpo/verl"
  else
    VERL_PATH="$(pwd)/verl"
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

# Optional GRPO enhancements. Defaults keep ordinary GRPO exactly on the vanilla path.
export ADAPTIVE_KL_ENABLED=${ADAPTIVE_KL_ENABLED:-false}
export ADAPTIVE_KL_TARGET=${ADAPTIVE_KL_TARGET:-0.1}
export ADAPTIVE_KL_HORIZON=${ADAPTIVE_KL_HORIZON:-10000}
export B_NDSR_ENABLED=${B_NDSR_ENABLED:-false}
export JASS_ENABLED=${JASS_ENABLED:-false}
export LLM_JUDGE_ENABLED=${LLM_JUDGE_ENABLED:-false}
export JASS_JUDGE_MODEL=${JASS_JUDGE_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}
export JASS_JUDGE_BASE_URL=${JASS_JUDGE_BASE_URL:-http://localhost:8001/v1}
export LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}
export LLM_JUDGE_BASE_URL=${LLM_JUDGE_BASE_URL:-http://localhost:8001/v1}

# Seen-task curriculum: sample only seen tasks, with a scheduled covered/uncovered mix.
export GRPO_SEEN_CURRICULUM_ENABLED=${GRPO_SEEN_CURRICULUM_ENABLED:-true}
export GRPO_SEEN_CURRICULUM_SPLIT_JSON=${GRPO_SEEN_CURRICULUM_SPLIT_JSON:-experiments/sft_collect_airline/split.json}
export GRPO_SEEN_CURRICULUM=${GRPO_SEEN_CURRICULUM:-40:0.85,100:0.60,180:0.40,300:0.25}
export GRPO_SEEN_CURRICULUM_SEED=${GRPO_SEEN_CURRICULUM_SEED:-20260701}
export LLM_JUDGE_ALPHA=${LLM_JUDGE_ALPHA:-0.2}
export B_NDSR_ROOT_MIN_SAMPLES=${B_NDSR_ROOT_MIN_SAMPLES:-4}
export B_NDSR_ROOT_MAX_SAMPLES=${B_NDSR_ROOT_MAX_SAMPLES:-8}
export B_NDSR_ROOT_INCREMENT=${B_NDSR_ROOT_INCREMENT:-2}
export B_NDSR_TOTAL_BUDGET_PER_TASK=${B_NDSR_TOTAL_BUDGET_PER_TASK:-12}
export B_NDSR_SUFFIX_MIN_SAMPLES=${B_NDSR_SUFFIX_MIN_SAMPLES:-4}

EXTRA_ARGS=()
if [[ "${ADAPTIVE_KL_ENABLED,,}" == "true" || "${ADAPTIVE_KL_ENABLED}" == "1" ]]; then
  EXTRA_ARGS+=(algorithm.kl_ctrl.type=adaptive)
  EXTRA_ARGS+=(algorithm.kl_ctrl.target_kl="${ADAPTIVE_KL_TARGET}")
  EXTRA_ARGS+=(algorithm.kl_ctrl.horizon="${ADAPTIVE_KL_HORIZON}")
else
  EXTRA_ARGS+=(algorithm.kl_ctrl.type=fixed)
fi

if [[ "${B_NDSR_ENABLED,,}" == "true" || "${B_NDSR_ENABLED}" == "1" ]]; then
  EXTRA_ARGS+=(actor_rollout_ref.rollout.n=1)
  EXTRA_ARGS+=(actor_rollout_ref.actor.ppo_mini_batch_size=2)
  EXTRA_ARGS+=(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1)
  EXTRA_ARGS+=(actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1)
  EXTRA_ARGS+=(trainer.experiment_name=grpo_2xa800_80g_72b_user_b_ndsr)
fi

mkdir -p experiments/grpo_2xa800

python scripts/setup/patch_verl_vllm_compat.py

python -m verl.trainer.main_ppo \
  --config-path="$(pwd)/configs/train/grpo" \
  --config-name=grpo_2xa800_80g_72b_user \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee experiments/grpo_2xa800/train.log
