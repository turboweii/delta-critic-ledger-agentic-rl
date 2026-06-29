#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://localhost:8001/v1}
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://localhost:8001/v1}

python3 scripts/eval/run_policy_eval.py \
  --config configs/eval/eval_airline_grpo_2xa800_72b_user.yaml \
  "$@"
