#!/bin/bash
set -euo pipefail

OUT_DIR=${OUT_DIR:-outputs/memory_profile}
INTERVAL=${INTERVAL:-5}
DURATION=${DURATION:-300}
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/nvidia_smi_$(date +%Y%m%d_%H%M%S).csv"

echo "timestamp,index,name,memory.used,memory.total,utilization.gpu" > "${OUT_FILE}"
END=$((SECONDS + DURATION))
while [ "${SECONDS}" -lt "${END}" ]; do
  nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits >> "${OUT_FILE}"
  sleep "${INTERVAL}"
done

echo "Wrote ${OUT_FILE}"

