#!/bin/bash
set -euo pipefail

# Additive single-GPU validation path. Production training remains 8x4090.
# Usage: bash scripts/run_a800_80g_smoke.sh [all|prepare|collect-real|sft|grpo]

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

STAGE=${1:-all}
GPU=${GPU:-0}
CONDA_ENV=${CONDA_ENV:-dcl-agentic-rl}
MODEL_7B=${MODEL_7B:-../models/Qwen2.5-7B-Instruct}
MODEL_32B_AWQ=${MODEL_32B_AWQ:-../models/Qwen2.5-32B-Instruct-AWQ}
TAU_BENCH_PATH=${TAU_BENCH_PATH:-$(pwd)/../tau-bench}
VERL_PATH=${VERL_PATH:-$(pwd)/../verl}
UPSTREAM_PROJECT=${UPSTREAM_PROJECT:-}
LOG_DIR=${LOG_DIR:-outputs/a800_80g_smoke/logs}

USER_PID=""
TEACHER_PID=""

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="$(pwd)/src:${TAU_BENCH_PATH}:${VERL_PATH}:${PYTHONPATH:-}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

if [[ -f "${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}" ]]; then
  source "${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV}"
fi

if [[ -z "${UPSTREAM_PROJECT}" ]]; then
  for candidate in \
    "$(pwd)/../agentic-grpo-longhorizon" \
    "$(pwd)/../agentic-grpo-longhorizon-main/agentic-grpo-longhorizon"; do
    if [[ -f "${candidate}/scripts/train/grpo/build_grpo_parquet.py" ]]; then
      UPSTREAM_PROJECT="${candidate}"
      break
    fi
  done
fi

cleanup_servers() {
  for pid in "${TEACHER_PID}" "${USER_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup_servers EXIT INT TERM

wait_for_server() {
  local name=$1
  local url=$2
  local pid=$3
  local attempts=${SERVER_WAIT_ATTEMPTS:-120}
  for ((i = 1; i <= attempts; i++)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} server exited during startup; inspect ${LOG_DIR}/${name}.log" >&2
      return 1
    fi
    if python3 - "${url}" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
urllib.request.urlopen(sys.argv[1].rstrip("/") + "/models", timeout=3).read()
PY
    then
      echo "${name} server is ready at ${url}"
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for ${name}; inspect ${LOG_DIR}/${name}.log" >&2
  return 1
}

start_user() {
  CUDA_DEVICES="${GPU}" MODEL_PATH="${MODEL_32B_AWQ}" \
    SERVED_MODEL_NAME=delta-user-32b-awq PORT=8001 TP_SIZE=1 \
    GPU_MEM_UTIL=${USER_GPU_MEM_UTIL:-0.26} MAX_MODEL_LEN=${USER_MAX_MODEL_LEN:-4096} \
    MAX_NUM_SEQS=${USER_MAX_NUM_SEQS:-1} \
    bash scripts/vllm_server/start_user_32b_awq_8x4090.sh \
    >"${LOG_DIR}/user.log" 2>&1 &
  USER_PID=$!
  wait_for_server user http://localhost:8001/v1 "${USER_PID}"
}

start_teacher() {
  CUDA_DEVICES="${GPU}" MODEL_PATH="${MODEL_32B_AWQ}" \
    SERVED_MODEL_NAME=delta-teacher-32b-awq PORT=8002 TP_SIZE=1 \
    GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.26} MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-4096} \
    MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-1} \
    bash scripts/vllm_server/start_teacher_32b_awq_8x4090.sh \
    >"${LOG_DIR}/teacher.log" 2>&1 &
  TEACHER_PID=$!
  wait_for_server teacher http://localhost:8002/v1 "${TEACHER_PID}"
}

prepare_data() {
  echo "[A800 smoke] Building deterministic oracle SFT data"
  python3 scripts/train/sft/collect_sft_data.py \
    --mode oracle \
    --output-dir experiments/sft_collect_airline_a800_smoke \
    --task-ids 0,1,2,3,4,5 \
    --no-use-user-sim \
    --holdout-size 1 \
    --overwrite

  echo "[A800 smoke] Building the small GRPO parquet set"
  if [[ -z "${UPSTREAM_PROJECT}" ]]; then
    echo "Cannot find upstream build_grpo_parquet.py; set UPSTREAM_PROJECT." >&2
    return 1
  fi
  python3 "${UPSTREAM_PROJECT}/scripts/train/grpo/build_grpo_parquet.py" \
    --seen-task-ids 0,1,2,3 \
    --num-total-tasks 6 \
    --output-train experiments/grpo_airline_a800_smoke/train.parquet \
    --output-val experiments/grpo_airline_a800_smoke/val.parquet
}

collect_real() {
  echo "[A800 smoke] Starting two quantized 32B services for a tiny real collection"
  start_user
  start_teacher
  python3 scripts/train/sft/collect_sft_data.py \
    --mode teacher_rollout \
    --output-dir experiments/sft_collect_airline_a800_smoke_real \
    --task-ids 0,1 \
    --best-of-n 2 \
    --temperatures 0.0,0.7 \
    --max-turns 8 \
    --teacher-max-tokens 384 \
    --num-workers 1 \
    --holdout-size 0 \
    --overwrite
}

run_sft() {
  [[ -s experiments/sft_collect_airline_a800_smoke/train.jsonl ]] || prepare_data
  echo "[A800 smoke] Running one-epoch 7B LoRA SFT"
  python3 scripts/train/sft/train_sft_lora.py \
    --config configs/train/sft/sft_airline_lora_1xa800_80g_smoke.yaml
}

run_grpo() {
  [[ -d experiments/sft_lora_a800_smoke_merged ]] || run_sft
  [[ -s experiments/grpo_airline_a800_smoke/train.parquet ]] || prepare_data
  cleanup_servers
  USER_PID=""
  TEACHER_PID=""
  echo "[A800 smoke] Starting the 32B-AWQ user simulator beside single-GPU GRPO"
  start_user
  mkdir -p experiments/delta_ledger_grpo_1xa800_80g_smoke outputs/grpo_delta_traces
  python -m verl.trainer.main_ppo \
    --config-path="$(pwd)/configs/train/grpo" \
    --config-name=delta_ledger_grpo_1xa800_80g_smoke
}

case "${STAGE}" in
  all)
    prepare_data
    run_sft
    run_grpo
    ;;
  prepare) prepare_data ;;
  collect-real) collect_real ;;
  sft) run_sft ;;
  grpo) run_grpo ;;
  *)
    echo "Usage: $0 [all|prepare|collect-real|sft|grpo]" >&2
    exit 2
    ;;
esac
