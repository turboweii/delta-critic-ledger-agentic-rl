#!/bin/bash
set -euo pipefail

python3 scripts/eval/eval_checkpoints.py \
  --config configs/eval/checkpoints_delta_grpo.yaml \
  "$@"

