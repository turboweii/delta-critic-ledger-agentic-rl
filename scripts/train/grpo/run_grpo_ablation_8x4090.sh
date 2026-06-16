#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${ROOT_DIR}"

CONDA_ENV=${CONDA_ENV:-dcl-agentic-rl}
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  activated=0
  for profile in \
    "${CONDA_PROFILE:-}" \
    "${HOME}/miniconda/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -n "${profile}" && -f "${profile}" ]]; then
      source "${profile}"
      conda activate "${CONDA_ENV}"
      activated=1
      break
    fi
  done
  [[ "${activated}" == "1" ]] || {
    echo "Cannot activate ${CONDA_ENV}; activate it before running or set CONDA_PROFILE." >&2
    exit 1
  }
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
TAU_BENCH_PATH=${TAU_BENCH_PATH:-$(pwd)/../tau-bench}
VERL_PATH=${VERL_PATH:-$(pwd)/../verl}
export PYTHONPATH="$(pwd)/src:${TAU_BENCH_PATH}:${VERL_PATH}:${PYTHONPATH:-}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

DRY_RUN=${DRY_RUN:-0}
TOTAL_STEPS=${TOTAL_STEPS:-300}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-50}
STEPS=${STEPS:-50,100,150,200,300}
VARIANTS=${VARIANTS:-terminal_only,delta_only,delta_ledger,delta_ledger_adaptive}

run_variant() {
  local name=$1
  local interaction_config=$2
  local adaptive=$3
  local exp_root="experiments/ablation_grpo_8x4090/${name}"
  local trace_dir="outputs/grpo_ablation_traces/${name}"
  local checkpoint_dir="${exp_root}/checkpoints"

  mkdir -p "${exp_root}" "${trace_dir}"

  local overrides=(
    "trainer.total_training_steps=${TOTAL_STEPS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.experiment_name=ablation_${name}"
    "trainer.default_local_dir=${checkpoint_dir}"
    "actor_rollout_ref.rollout.multi_turn.interaction_config_path=${interaction_config}"
    "algorithm.rollout_correction.bypass_mode=true"
    "actor_rollout_ref.rollout.calculate_log_probs=true"
    "actor_rollout_ref.actor.use_fused_kernels=true"
    "actor_rollout_ref.model.use_fused_kernels=true"
  )

  if [[ "${adaptive}" == "1" ]]; then
    mapfile -t adaptive_overrides < <(
      python3 scripts/train/grpo/adaptive_kl_entropy.py \
        --trace-dir "${trace_dir}" \
        --config configs/train/grpo/adaptive_kl_entropy.yaml \
        --format hydra-lines
    )
    overrides+=("${adaptive_overrides[@]}")
    overrides+=("actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/agent_loop/tau_bench_agent_loop.yaml")
  else
    overrides+=("actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/agent_loop/tau_bench_agent_loop_no_adaptive.yaml")
  fi

  echo "[ablation:${name}] interaction=${interaction_config} adaptive=${adaptive}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'python -m verl.trainer.main_ppo --config-path=%q --config-name=delta_ledger_grpo_8x4090_32b_user' "$(pwd)/configs/train/grpo"
    printf ' %q' "${overrides[@]}"
    printf '\n'
    return
  fi

  export DCL_ADAPTIVE_ENTROPY="${adaptive}"
  python -m verl.trainer.main_ppo \
    --config-path="$(pwd)/configs/train/grpo" \
    --config-name=delta_ledger_grpo_8x4090_32b_user \
    "${overrides[@]}"

  CHECKPOINT_ROOT="${checkpoint_dir}" \
    OUTPUT_ROOT="${exp_root}" \
    BASE_MODEL=experiments/sft_lora_merged \
    STEPS="${STEPS}" \
    STRICT=0 \
    bash scripts/train/grpo/export_grpo_checkpoints.sh
}

IFS=',' read -ra selected <<< "${VARIANTS}"
for variant in "${selected[@]}"; do
  case "${variant}" in
    terminal_only) run_variant terminal_only configs/interaction_config/tau_bench_airline_terminal_only.yaml 0 ;;
    delta_only) run_variant delta_only configs/interaction_config/tau_bench_airline_delta_only.yaml 0 ;;
    delta_ledger) run_variant delta_ledger configs/interaction_config/tau_bench_airline_delta_ledger_ablation.yaml 0 ;;
    delta_ledger_adaptive) run_variant delta_ledger_adaptive configs/interaction_config/tau_bench_airline_delta_ledger_adaptive.yaml 1 ;;
    *)
      echo "Unknown ablation variant: ${variant}" >&2
      exit 2
      ;;
  esac
done
