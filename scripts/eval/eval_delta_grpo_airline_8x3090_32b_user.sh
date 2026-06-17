#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${TAU_BENCH_PATH:-$(pwd)/../tau-bench}:${PYTHONPATH:-}"
python3 scripts/eval/run_policy_eval.py \
  --config configs/eval/eval_airline_delta_grpo_8x3090_32b_user.yaml \
  "$@"
