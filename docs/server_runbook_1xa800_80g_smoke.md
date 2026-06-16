# 1xA800 80GB Pipeline Smoke Test

This is a reduced validation of the same pipeline used by the 8x4090 experiment.
It is not a replacement training configuration.

## What Is Preserved

- Real tau-bench airline environment and tool schemas.
- Deterministic tau-bench oracle bootstrap data followed by 7B LoRA SFT.
- Real 32B-AWQ user simulation during evaluation and GRPO.
- Delta-Critic + Evidence Ledger reward inside veRL multi-turn GRPO.
- Real SFT policy evaluation and GRPO validation.

The optional `collect-real` stage checks a single real 32B teacher trajectory,
but it is deliberately excluded from the default smoke chain.

The `prepare` stage uses the task instruction directly and does not start or call
the 32B user simulator.

## Required Layout

```text
/workspace/
  delta-critic-ledger-agentic-rl/
  tau-bench/
  verl/
  models/
    Qwen2.5-7B-Instruct/
    Qwen2.5-32B-Instruct-AWQ/
```

No `agentic-grpo-longhorizon` checkout is required.

## Environment

```bash
conda activate dcl-agentic-rl
cd /workspace/delta-critic-ledger-agentic-rl

git -C /workspace/verl fetch --tags
git -C /workspace/verl checkout v0.6.1
python -m pip install -r requirements-server.txt
python -m pip install -e /workspace/tau-bench
python -m pip install -e /workspace/verl
```

Install FlashAttention after PyTorch when it is not already available:

```bash
MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation
```

## Run In Stages

```bash
bash scripts/run_a800_80g_smoke.sh preflight
bash scripts/run_a800_80g_smoke.sh prepare
bash scripts/run_a800_80g_smoke.sh sft
bash scripts/run_a800_80g_smoke.sh eval-sft
bash scripts/run_a800_80g_smoke.sh grpo
bash scripts/run_a800_80g_smoke.sh eval-grpo
```

The `grpo` stage uses the same Adaptive KL / Entropy Controller as the 8x4090
run. It validates that Ledger traces, online sampling control, and bounded Hydra
overrides are wired correctly. Disable it for a plain baseline smoke run with:

```bash
ADAPTIVE_GRPO_CONTROL=0 bash scripts/run_a800_80g_smoke.sh grpo
```

To separately verify real teacher collection, run:

```bash
bash scripts/run_a800_80g_smoke.sh collect-real
```

This loads one 32B-AWQ vLLM endpoint and writes to
`experiments/sft_collect_airline_a800_real_check/`; it does not replace the
deterministic data used by the default smoke SFT.

After the stages work individually, the complete command is:

```bash
bash scripts/run_a800_80g_smoke.sh all
```

Monitor memory in a second terminal:

```bash
watch -n 1 nvidia-smi
```

## Outputs

```text
experiments/sft_collect_airline_a800_smoke/
experiments/sft_lora_a800_smoke_merged/
outputs/eval_sft_1xa800_80g_smoke/
experiments/delta_ledger_grpo_1xa800_80g_smoke/checkpoints/
experiments/delta_ledger_grpo_1xa800_80g_smoke/hf_step_10/
outputs/eval_grpo_1xa800_80g_smoke/
outputs/grpo_delta_traces/
```

The production 8x4090 configurations and output directories are separate and
remain the primary experiment path.
