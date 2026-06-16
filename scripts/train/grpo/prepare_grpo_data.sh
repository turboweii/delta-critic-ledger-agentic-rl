#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${ROOT_DIR}"

SPLIT_FILE=${SPLIT_FILE:-experiments/sft_collect_airline/split.json}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/grpo_airline}

[[ -s "${SPLIT_FILE}" ]] || {
  echo "Missing SFT split file: ${SPLIT_FILE}" >&2
  exit 1
}

python3 scripts/train/grpo/build_grpo_parquet.py \
  --train-task-ids-from "${SPLIT_FILE}" \
  --val-task-ids-from "${SPLIT_FILE}" \
  --output-train "${OUTPUT_DIR}/train.parquet" \
  --output-val "${OUTPUT_DIR}/val.parquet"
