#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:$(pwd)/../agentic-grpo-longhorizon-main/tau-bench:${PYTHONPATH:-}"
python3 scripts/eval/run_policy_eval.py \
  --config configs/eval/eval_airline_delta_grpo_8x4090_32b_user.yaml \
  "$@"

