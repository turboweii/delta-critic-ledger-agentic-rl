#!/bin/bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"../models/Qwen2.5-7B-Instruct"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"delta-assistant-7b"}
PORT=${PORT:-8000}
CUDA_DEVICES=${CUDA_DEVICES:-0}
TP_SIZE=${TP_SIZE:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.78}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export VLLM_USE_V1=${VLLM_USE_V1:-1}

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code

