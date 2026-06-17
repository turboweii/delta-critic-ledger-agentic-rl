# Delta-Critic Ledger Agentic RL

Standalone project prototype for state-transition credit assignment on tau-bench-style
long-horizon tool agents.

This project is intentionally independent from `agentic-grpo-longhorizon`. The default
training and evaluation pipeline uses real tau-bench airline teacher rollouts,
tool schemas, vLLM endpoints, and veRL multi-turn rollouts. Mock airline scripts are
kept only for local unit tests and fast sanity checks.

## Idea

Long-horizon tool agents often receive only a final binary reward. Delta-Critic
decomposes that terminal signal into state-transition credit:

```text
delta_t = Phi(state_after_t, target_state) - Phi(state_before_t, target_state)
```

Evidence Ledger then labels whether write actions are grounded in tool-observed
evidence, exposing parameter grounding and premature-write failures.

The GRPO path also includes a conservative Adaptive KL / Entropy Controller.
It uses the same Evidence Ledger traces to detect stalled exploration,
bad groundedness, or high task progress:

- online rollout entropy is adjusted inside the custom veRL agent loop by
  changing `temperature` and `top_p` before assistant generations;
- trace-based KL overrides are produced by
  `scripts/train/grpo/adaptive_kl_entropy.py` for the next GRPO run/segment;
- the controller is bounded by `configs/train/grpo/adaptive_kl_entropy.yaml`
  and can be disabled with `ADAPTIVE_GRPO_CONTROL=0`.

## Project Layout

```text
config/        Backward-compatible single demo config
configs/       Reward and experiment configs for ablations
docs/          Project plan, method notes, experiment design
experiments/   Experiment manifests and future run records
outputs/       Generated traces and summaries
scripts/       Runnable entrypoints
src/           Reusable implementation modules
```

## Local Smoke Tests

```bash
python3 scripts/run_tests.py
python3 scripts/run_demo.py --config config/demo_airline.json
python3 scripts/run_ablation.py --experiment configs/experiments/airline_mini.json
python3 scripts/analyze_outputs.py --summary outputs/airline_mini/summary.json
python3 scripts/eval/analyze_failure_modes.py --runs outputs/airline_mini_yaml/runs.jsonl
python3 scripts/test/stress_context_isolation.py --concurrency 32
```

These local smoke tests use the lightweight fixture environment only to validate
Delta-Critic and Evidence Ledger behavior without GPUs. The default training and
data pipeline below uses real tau-bench.

## Server Pipeline For 2xA800 80GB + 32B User

For the 2xA800 80GB server, use the dedicated runbook:

```bash
docs/server_runbook_2xa800_80g.md
```

The short version is:

```bash
# GPU 1: 32B-AWQ user simulator, also reused as teacher during collection.
CUDA_DEVICES=1 bash scripts/vllm_server/start_user_32b_awq_2xa800.sh

# SFT data + SFT training.
bash scripts/train/sft/collect_sft_teacher_2xa800.sh
# Stop the 32B vLLM before this step; 2-GPU SFT uses both A800s.
bash scripts/train/sft/run_sft_lora_2xa800.sh

# Restart the 32B user simulator on GPU 1 before GRPO.
CUDA_DEVICES=1 bash scripts/vllm_server/start_user_32b_awq_2xa800.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train/grpo/run_delta_ledger_grpo_2xa800_80g_32b_user.sh

# Export merged HF checkpoints before GRPO evaluation.
bash scripts/train/grpo/export_grpo_checkpoints_2xa800.sh
```

The main configs are:

- `configs/models/2xa800_80g_qwen.yaml`
- `configs/train/sft/sft_airline_lora_2xa800_80g.yaml`
- `configs/train/grpo/delta_ledger_grpo_2xa800_80g_32b_user.yaml`

## Server Pipeline For 8x4090 + 32B User

Default main model plan is in `configs/models/8x4090_qwen.yaml`:

- Assistant/policy LLM: `Qwen/Qwen2.5-7B-Instruct`
- User simulator LLM: `Qwen/Qwen2.5-32B-Instruct-AWQ`
- 8x4090 training mode: LoRA SFT and GRPO/RFT with 7B policy, stronger 32B user simulator

Runnable server stages:

```bash
# 0. Install server dependencies, then install tau-bench/veRL following their repos
pip install -r requirements-server.txt

# 1. Generate veRL tool config with tau-bench schemas
python3 scripts/train/grpo/gen_tool_config.py
python3 scripts/test/check_prompt_budget.py \
  --config configs/train/grpo/delta_ledger_grpo_8x4090_32b_user.yaml \
  --model-path ../models/Qwen2.5-7B-Instruct \
  --tau-bench-path ../tau-bench

# 2. Start 32B-AWQ teacher policy and 32B-AWQ user simulator for SFT data
bash scripts/vllm_server/start_teacher_32b_awq_8x4090.sh
bash scripts/vllm_server/start_user_32b_awq_8x4090.sh

# 3. Collect real tau-bench teacher-rollout SFT data and build GRPO parquet
bash scripts/train/sft/collect_sft_teacher_8x4090.sh

# 4. Stop teacher/user vLLM servers, then train and merge LoRA policy
bash scripts/train/sft/run_sft_lora_8x4090.sh

# 5. Start assistant endpoint from experiments/sft_lora_merged for eval/data collection
MODEL_PATH=experiments/sft_lora_merged \
SERVED_MODEL_NAME=delta-assistant-7b-sft \
PORT=8000 \
bash scripts/vllm_server/start_assistant_7b.sh

# 6. Collect real tau-bench policy rollouts with Delta/Ledger traces
python3 scripts/data/collect_tau_rollouts.py \
  --config configs/eval/eval_airline_sft_8x4090_32b_user.yaml \
  --output-dir experiments/data_airline_delta

# 7. Delta-Critic + Evidence Ledger GRPO
bash scripts/train/grpo/run_delta_ledger_grpo_8x4090_32b_user.sh

# Optional: inspect the adaptive controller decision from recent Ledger traces
python3 scripts/train/grpo/adaptive_kl_entropy.py \
  --trace-dir outputs/grpo_delta_traces \
  --format summary

# 8. Export veRL LoRA checkpoints to standalone HF models
bash scripts/train/grpo/export_grpo_checkpoints.sh

# 9. Evaluation (keep the 32B user simulator online on port 8001)
bash scripts/eval/eval_sft_airline_8x4090_32b_user.sh
bash scripts/eval/eval_delta_grpo_airline_8x4090_32b_user.sh
bash scripts/eval/eval_checkpoints_delta_grpo.sh
```

Use `--dry-run` on SFT/eval scripts to validate paths without GPUs.

The veRL integration entrypoint is:

```text
delta_critic_ledger.verl_integration.interaction.DeltaTauBenchInteraction
```

It computes:

```text
terminal_reward + beta_delta * state_delta + beta_evidence * ledger_bonus
```

For low-resource fallback, keep using `configs/models/4x4090_qwen.yaml` and
`scripts/train/grpo/run_delta_ledger_grpo_4x4090.sh`.

The demo writes:

- `outputs/demo_delta_trace.jsonl`
- `outputs/demo_ledger_trace.jsonl`
- `outputs/demo_summary.json`

The ablation runner writes:

- `outputs/airline_mini/runs.jsonl`
- `outputs/airline_mini/summary.json`

## Current Ablations

Reward configs:

- `terminal_only`
- `delta_only`
- `ledger_only`
- `delta_ledger`

Experiment configs:

- `airline_mini`
- `airline_grounding_stress`
- `airline_state_delta_stress`

## Independence

The runtime imports only this repository's `src/` package plus independently
installed tau-bench and veRL v0.6.1. It does not import files, configs, or modules
from another agentic-RL project.
