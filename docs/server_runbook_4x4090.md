# 4x4090 Server Runbook

## Model Choices

Fallback runnable setup:

- Assistant/policy: `Qwen/Qwen2.5-7B-Instruct`
- User simulator: `Qwen/Qwen2.5-7B-Instruct`

This is intentionally smaller than the main 8x4090 setup. For final experiments,
prefer the 8x4090 + 32B-AWQ user runbook.

GPU plan:

- GPU 0: assistant vLLM server for eval.
- GPU 1: user simulator vLLM server.
- GPU 0-3: LoRA SFT / GRPO training when servers are stopped.

## Setup

Install dependencies similar to the reference project:

```bash
pip install torch transformers peft datasets pyyaml vllm
pip install -e ../agentic-grpo-longhorizon-main/tau-bench
pip install -e ../agentic-grpo-longhorizon-main/verl
```

Generate veRL tool config:

```bash
python3 scripts/train/grpo/gen_tool_config.py
```

For memory profiling during training:

```bash
DURATION=600 INTERVAL=5 bash scripts/test/profile_memory.sh
```

## SFT Warmup

```bash
bash scripts/train/sft/run_sft_lora_4x4090.sh
```

For a config check without training:

```bash
python3 scripts/train/sft/collect_sft_data.py
python3 scripts/train/sft/train_sft_lora.py --dry-run
```

## vLLM Servers

Run in two terminals:

```bash
bash scripts/vllm_server/start_assistant_7b.sh
bash scripts/vllm_server/start_user_7b.sh
```

Health check:

```bash
python3 scripts/vllm_server/check_servers.py
```

## Delta-Ledger GRPO

Stop eval vLLM servers first if you need all 4 GPUs for training.

```bash
python3 scripts/train/grpo/gen_tool_config.py
bash scripts/train/grpo/run_delta_ledger_grpo_4x4090.sh
```

If OOM happens, switch config name in the run script to:

```text
delta_ledger_grpo_4x4090_lowmem
```

After low-memory smoke tests are stable, try:

```text
delta_ledger_grpo_4x4090_fast
```

## Evaluation Utilities

```bash
bash scripts/eval/eval_checkpoints_delta_grpo.sh --dry-run
python3 scripts/eval/build_per_task_summary.py \
  --runs outputs/airline_mini_yaml/runs.jsonl \
  --output outputs/airline_mini_yaml/per_task_summary.csv
python3 scripts/eval/analyze_failure_modes.py \
  --runs outputs/airline_mini_yaml/runs.jsonl \
  --output outputs/airline_mini_yaml/failure_modes.json
```

The key project-specific runtime class is:

```text
delta_critic_ledger.verl_integration.interaction.DeltaTauBenchInteraction
```

It computes:

```text
terminal_reward + beta_delta * state_delta + beta_evidence * ledger_bonus
```
