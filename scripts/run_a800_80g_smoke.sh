#!/bin/bash
set -euo pipefail

# Minimal pipeline validation on one A800 80GB. The production target remains 8x4090.
# Default data is deterministic oracle bootstrap data; real 32B teacher collection is optional.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

STAGE=${1:-all}
GPU=${GPU:-0}
CONDA_ENV=${CONDA_ENV:-dcl-agentic-rl}
MODEL_7B=${MODEL_7B:-../models/Qwen2.5-7B-Instruct}
MODEL_32B_AWQ=${MODEL_32B_AWQ:-../models/Qwen2.5-32B-Instruct-AWQ}
MODEL_32B_NAME=${MODEL_32B_NAME:-Qwen/Qwen2.5-32B-Instruct-AWQ}
TAU_BENCH_PATH=${TAU_BENCH_PATH:-$(pwd)/../tau-bench}
VERL_PATH=${VERL_PATH:-$(pwd)/../verl}
LOG_DIR=${LOG_DIR:-outputs/a800_80g_smoke/logs}

SERVER_32B_PID=""
ASSISTANT_PID=""

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="$(pwd)/src:${TAU_BENCH_PATH}:${VERL_PATH}:${PYTHONPATH:-}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

activate_env() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    return
  fi
  for profile in \
    "${CONDA_PROFILE:-}" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -n "${profile}" && -f "${profile}" ]]; then
      source "${profile}"
      conda activate "${CONDA_ENV}"
      return
    fi
  done
  echo "Cannot activate conda environment ${CONDA_ENV}. Activate it before running." >&2
  exit 1
}

cleanup_servers() {
  for pid in "${ASSISTANT_PID}" "${SERVER_32B_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      for _ in {1..30}; do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" 2>/dev/null || true
      fi
      wait "${pid}" 2>/dev/null || true
    fi
  done
  ASSISTANT_PID=""
  SERVER_32B_PID=""
}
trap cleanup_servers EXIT INT TERM

wait_for_server() {
  local name=$1
  local url=$2
  local pid=$3
  local attempts=${SERVER_WAIT_ATTEMPTS:-120}
  for ((i = 1; i <= attempts; i++)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited during startup; inspect ${LOG_DIR}/${name}.log" >&2
      return 1
    fi
    if python3 - "${url}" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
urllib.request.urlopen(sys.argv[1].rstrip("/") + "/models", timeout=3).read()
PY
    then
      echo "${name} ready at ${url}"
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for ${name}; inspect ${LOG_DIR}/${name}.log" >&2
  return 1
}

start_32b_server() {
  local gpu_mem_util=${1:-0.50}
  local max_num_seqs=${2:-2}
  local max_model_len=${3:-4096}
  cleanup_servers
  CUDA_DEVICES="${GPU}" MODEL_PATH="${MODEL_32B_AWQ}" \
    SERVED_MODEL_NAME="${MODEL_32B_NAME}" PORT=8001 TP_SIZE=1 \
    GPU_MEM_UTIL="${gpu_mem_util}" MAX_MODEL_LEN="${max_model_len}" \
    MAX_NUM_SEQS="${max_num_seqs}" \
    bash scripts/vllm_server/start_teacher_32b_awq_8x4090.sh \
    >"${LOG_DIR}/model32b.log" 2>&1 &
  SERVER_32B_PID=$!
  wait_for_server model32b http://localhost:8001/v1 "${SERVER_32B_PID}"
}

start_assistant_server() {
  local model_path=${1:-$(pwd)/experiments/sft_lora_a800_smoke_merged}
  local served_name=${2:-delta-assistant-7b-a800-smoke}
  CUDA_DEVICES="${GPU}" MODEL_PATH="${model_path}" \
    SERVED_MODEL_NAME="${served_name}" PORT=8000 TP_SIZE=1 \
    GPU_MEM_UTIL=${ASSISTANT_GPU_MEM_UTIL:-0.28} MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 \
    bash scripts/vllm_server/start_assistant_7b.sh \
    >"${LOG_DIR}/assistant.log" 2>&1 &
  ASSISTANT_PID=$!
  wait_for_server assistant http://localhost:8000/v1 "${ASSISTANT_PID}"
}

preflight() {
  activate_env
  [[ -f "${MODEL_7B}/config.json" ]] || { echo "Missing 7B model: ${MODEL_7B}" >&2; exit 1; }
  [[ -f "${MODEL_32B_AWQ}/config.json" ]] || { echo "Missing 32B AWQ model: ${MODEL_32B_AWQ}" >&2; exit 1; }
  [[ -d "${TAU_BENCH_PATH}/tau_bench" ]] || { echo "Missing tau-bench: ${TAU_BENCH_PATH}" >&2; exit 1; }
  python3 - <<'PY'
from importlib.metadata import version

import torch
import pandas
import pyarrow
import tau_bench
import verl
import vllm
assert torch.cuda.is_available(), "CUDA is not available"
memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert memory_gb >= 75, f"Expected an 80GB GPU, found {memory_gb:.1f}GB"
assert torch.__version__.startswith("2.7.0"), f"Expected torch 2.7.0, found {torch.__version__}"
assert vllm.__version__ == "0.9.2", f"Expected vLLM 0.9.2, found {vllm.__version__}"
assert version("verl").startswith("0.6.1"), f"Expected veRL 0.6.1, found {version('verl')}"
print(f"GPU: {torch.cuda.get_device_name(0)} ({memory_gb:.1f}GB)")
print(f"vLLM: {vllm.__version__}")
print(f"veRL: {version('verl')}")
PY
  python3 scripts/train/grpo/gen_tool_config.py --tau-bench-path "${TAU_BENCH_PATH}"
  python3 scripts/test/interface_audit.py \
    --grpo-config configs/train/grpo/delta_ledger_grpo_1xa800_80g_smoke.yaml \
    --expected-gpus 1
  python3 scripts/run_tests.py
}

build_grpo_data() {
  [[ -s experiments/sft_collect_airline_a800_smoke/split.json ]] || {
    echo "Missing real SFT split. Run the collect stage first." >&2
    exit 1
  }
  python3 scripts/train/grpo/build_grpo_parquet.py \
    --train-task-ids-from experiments/sft_collect_airline_a800_smoke/split.json \
    --num-total-tasks 4 \
    --output-train experiments/grpo_airline_a800_smoke/train.parquet \
    --output-val experiments/grpo_airline_a800_smoke/val.parquet
}

prepare_smoke_data() {
  activate_env
  echo "[A800 smoke] Building deterministic tau-bench oracle bootstrap data"
  python3 scripts/train/sft/collect_sft_data.py \
    --mode oracle \
    --output-dir experiments/sft_collect_airline_a800_smoke \
    --task-ids 0,1,2,3 \
    --holdout-size 1 \
    --num-workers 1 \
    --overwrite
  [[ -s experiments/sft_collect_airline_a800_smoke/train.jsonl ]] || {
    echo "Oracle bootstrap produced no SFT rows; inspect summary.json." >&2
    exit 1
  }
  build_grpo_data
}

collect_real_sft() {
  activate_env
  echo "[A800 smoke] Real 32B teacher rollout with a real 32B user simulator"
  echo "[A800 smoke] One local 32B-AWQ endpoint serves both roles sequentially"
  start_32b_server 0.50 2 8192
  python3 scripts/train/sft/collect_sft_data.py \
    --mode teacher_rollout \
    --output-dir experiments/sft_collect_airline_a800_real_check \
    --task-ids 0 \
    --use-user-sim \
    --user-model "${MODEL_32B_NAME}" \
    --user-provider openai \
    --user-base-url http://localhost:8001/v1 \
    --teacher-model "${MODEL_32B_NAME}" \
    --teacher-base-url http://localhost:8001/v1 \
    --best-of-n 1 \
    --temperatures 0.0 \
    --max-turns 8 \
    --teacher-max-tokens 384 \
    --num-workers 1 \
    --holdout-size 0 \
    --overwrite
  cleanup_servers
  [[ -s experiments/sft_collect_airline_a800_real_check/train.jsonl ]] || {
    echo "No successful teacher trajectories were collected; inspect summary.json." >&2
    exit 1
  }
}

run_sft() {
  activate_env
  [[ -s experiments/sft_collect_airline_a800_smoke/train.jsonl ]] || prepare_smoke_data
  echo "[A800 smoke] One-epoch 7B LoRA SFT on deterministic bootstrap trajectories"
  python3 scripts/train/sft/train_sft_lora.py \
    --config configs/train/sft/sft_airline_lora_1xa800_80g_smoke.yaml \
    --model-path "${MODEL_7B}"
}

eval_sft() {
  activate_env
  [[ -f experiments/sft_lora_a800_smoke_merged/config.json ]] || run_sft
  echo "[A800 smoke] Two-task real tau-bench evaluation of the SFT policy"
  start_32b_server 0.32 2 4096
  start_assistant_server
  python3 scripts/eval/run_policy_eval.py \
    --config configs/eval/eval_airline_sft_1xa800_80g_smoke.yaml
  cleanup_servers
}

run_grpo() {
  activate_env
  [[ -f experiments/sft_lora_a800_smoke_merged/config.json ]] || run_sft
  [[ -s experiments/grpo_airline_a800_smoke/train.parquet ]] || build_grpo_data
  echo "[A800 smoke] Ten-step real multi-turn Delta/Ledger GRPO with final validation"
  start_32b_server ${GRPO_USER_GPU_MEM_UTIL:-0.28} 1 4096
  mkdir -p experiments/delta_ledger_grpo_1xa800_80g_smoke outputs/grpo_delta_traces
  python -m verl.trainer.main_ppo \
    --config-path="$(pwd)/configs/train/grpo" \
    --config-name=delta_ledger_grpo_1xa800_80g_smoke
  cleanup_servers
  python3 scripts/train/grpo/export_grpo_checkpoint.py \
    --base-model experiments/sft_lora_a800_smoke_merged \
    --checkpoint experiments/delta_ledger_grpo_1xa800_80g_smoke/checkpoints/global_step_10/actor \
    --output experiments/delta_ledger_grpo_1xa800_80g_smoke/hf_step_10
}

eval_grpo() {
  activate_env
  [[ -f experiments/delta_ledger_grpo_1xa800_80g_smoke/hf_step_10/config.json ]] || run_grpo
  echo "[A800 smoke] Loading the exported GRPO checkpoint for final tau-bench evaluation"
  start_32b_server 0.32 2 4096
  start_assistant_server \
    "$(pwd)/experiments/delta_ledger_grpo_1xa800_80g_smoke/hf_step_10" \
    delta-assistant-7b-a800-grpo-step10
  python3 scripts/eval/run_policy_eval.py \
    --config configs/eval/eval_airline_grpo_1xa800_80g_smoke.yaml
  cleanup_servers
}

case "${STAGE}" in
  all)
    preflight
    prepare_smoke_data
    run_sft
    eval_sft
    run_grpo
    eval_grpo
    ;;
  preflight) preflight ;;
  prepare) prepare_smoke_data ;;
  collect-real) collect_real_sft ;;
  sft) run_sft ;;
  eval-sft) eval_sft ;;
  grpo) run_grpo ;;
  eval-grpo) eval_grpo ;;
  *)
    echo "Usage: $0 [all|preflight|prepare|collect-real|sft|eval-sft|grpo|eval-grpo]" >&2
    exit 2
    ;;
esac
