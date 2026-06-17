#!/bin/bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"../models/Qwen2.5-32B-Instruct-AWQ"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen2.5-32B-Instruct-AWQ"}
PORT=${PORT:-8001}
CUDA_DEVICES=${CUDA_DEVICES:-1}
TP_SIZE=${TP_SIZE:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.70}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export VLLM_USE_V1=${VLLM_USE_V1:-1}

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code
