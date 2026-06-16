#!/bin/bash
set -euo pipefail

# A/B profile the GRPO memory impact of:
# - rollout.calculate_log_probs + rollout_correction.bypass_mode
# - actor/model fused kernels
#
# Run this only after GRPO data exists and the 32B user simulator is online.

CONFIG_PATH=${CONFIG_PATH:-$(pwd)/configs/train/grpo}
CONFIG_NAME=${CONFIG_NAME:-delta_ledger_grpo_1xa800_80g_smoke}
OUT_DIR=${OUT_DIR:-outputs/memory_profile_ab}
STEPS=${STEPS:-3}
INTERVAL=${INTERVAL:-1}
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "${OUT_DIR}"

profile_run() {
  local label=$1
  shift
  local csv="${OUT_DIR}/${label}_nvidia_smi.csv"
  local log="${OUT_DIR}/${label}_train.log"

  echo "timestamp,index,name,memory.used,memory.total,utilization.gpu" > "${csv}"
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits >> "${csv}" || true
      sleep "${INTERVAL}"
    done
  ) &
  local profiler_pid=$!

  set +e
  "${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name="${CONFIG_NAME}" \
    trainer.total_training_steps="${STEPS}" \
    trainer.experiment_name="memory_ab_${label}" \
    trainer.default_local_dir="${OUT_DIR}/checkpoints_${label}" \
    "$@" \
    >"${log}" 2>&1
  local status=$?
  set -e

  kill "${profiler_pid}" 2>/dev/null || true
  wait "${profiler_pid}" 2>/dev/null || true

  if [[ "${status}" != "0" ]]; then
    echo "Run ${label} failed; inspect ${log}" >&2
    return "${status}"
  fi
  echo "Wrote ${csv}"
}

echo "[optimized] bypass_mode + rollout logprobs + fused kernels"
profile_run optimized \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  algorithm.rollout_correction.bypass_mode=true \
  actor_rollout_ref.actor.use_fused_kernels=true \
  actor_rollout_ref.model.use_fused_kernels=true

echo "[control] recompute old logprobs through FSDP actor; fused kernels disabled"
profile_run control \
  actor_rollout_ref.rollout.calculate_log_probs=false \
  algorithm.rollout_correction.bypass_mode=false \
  actor_rollout_ref.actor.use_fused_kernels=false \
  actor_rollout_ref.model.use_fused_kernels=false

"${PYTHON_BIN}" scripts/test/summarize_memory_profile.py \
  --optimized "${OUT_DIR}/optimized_nvidia_smi.csv" \
  --control "${OUT_DIR}/control_nvidia_smi.csv" \
  | tee "${OUT_DIR}/summary.csv"

echo "Summary: ${OUT_DIR}/summary.csv"
