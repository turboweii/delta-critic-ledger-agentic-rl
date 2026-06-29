#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}
TEACHER_BASE_URL=${TEACHER_BASE_URL:-http://localhost:8002/v1}
USER_MODEL=${USER_MODEL:-openai/Qwen/Qwen2.5-72B-Instruct-AWQ}  # user sim aligned with reference (72B)
USER_BASE_URL=${USER_BASE_URL:-http://localhost:8001/v1}

python3 scripts/train/sft/collect_sft_data.py \
  --mode teacher_rollout \
  --output-dir experiments/sft_collect_airline \
  --env-name airline \
  --task-split test \
  --start-index 0 \
  --end-index 50 \
  --use-user-sim \
  --user-model "${USER_MODEL}" \
  --user-provider openai \
  --user-base-url "${USER_BASE_URL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --teacher-base-url "${TEACHER_BASE_URL}" \
  --best-of-n "${BEST_OF_N:-16}" \
  --temperatures "${TEACHER_TEMPERATURES:-0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.5,0.8,0.8,0.8,0.8,1.0,1.0,1.0,1.0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --holdout-size 10

[[ -s experiments/sft_collect_airline/train.jsonl ]] || {
  echo "No successful teacher trajectories were collected; inspect experiments/sft_collect_airline/summary.json." >&2
  exit 1
}

bash scripts/train/grpo/prepare_grpo_data.sh
