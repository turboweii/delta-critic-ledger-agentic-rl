#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python3 scripts/train/sft/collect_sft_data.py \
  --mode teacher_rollout \
  --output-dir experiments/sft_collect_airline \
  --env-name airline \
  --task-split test \
  --start-index 0 \
  --end-index 50 \
  --use-user-sim \
  --user-model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --user-provider openai \
  --user-base-url http://localhost:8001/v1 \
  --teacher-model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --teacher-base-url http://localhost:8002/v1 \
  --best-of-n "${BEST_OF_N:-6}" \
  --temperatures "${TEACHER_TEMPERATURES:-0.0,0.0,0.5,0.5,0.8,1.0}" \
  --num-workers "${NUM_WORKERS:-1}" \
  --holdout-size 10

[[ -s experiments/sft_collect_airline/train.jsonl ]] || {
  echo "No successful teacher trajectories were collected; inspect experiments/sft_collect_airline/summary.json." >&2
  exit 1
}

bash scripts/train/grpo/prepare_grpo_data.sh
