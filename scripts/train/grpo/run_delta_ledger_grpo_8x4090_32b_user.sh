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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
TAU_BENCH_PATH=${TAU_BENCH_PATH:-$(pwd)/../tau-bench}
VERL_PATH=${VERL_PATH:-$(pwd)/../verl}
export PYTHONPATH="$(pwd)/src:${TAU_BENCH_PATH}:${VERL_PATH}:${PYTHONPATH:-}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

mkdir -p experiments/delta_ledger_grpo_8x4090 outputs/grpo_delta_traces

ADAPTIVE_GRPO_CONTROL=${ADAPTIVE_GRPO_CONTROL:-1}
ADAPTIVE_OVERRIDES=()
if [[ "${ADAPTIVE_GRPO_CONTROL}" == "1" ]]; then
  python3 scripts/train/grpo/adaptive_kl_entropy.py \
    --trace-dir outputs/grpo_delta_traces \
    --config configs/train/grpo/adaptive_kl_entropy.yaml \
    --format summary
  mapfile -t ADAPTIVE_OVERRIDES < <(
    python3 scripts/train/grpo/adaptive_kl_entropy.py \
      --trace-dir outputs/grpo_delta_traces \
      --config configs/train/grpo/adaptive_kl_entropy.yaml \
      --format hydra-lines
  )
fi

python -m verl.trainer.main_ppo \
  --config-path="$(pwd)/configs/train/grpo" \
  --config-name=delta_ledger_grpo_8x4090_32b_user \
  "${ADAPTIVE_OVERRIDES[@]}"
