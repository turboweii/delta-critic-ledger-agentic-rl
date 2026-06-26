# Long-Horizon Agentic GRPO

Outcome-only GRPO for tau-bench airline long-horizon tool agents, with two
credit / anti-hack add-ons: **hard process constraints** (Leg 1) and
**divergence-based per-turn credit** (Leg 2). See `docs/method.md`.

Training and evaluation use real tau-bench airline rollouts, tool schemas, vLLM
endpoints, and veRL multi-turn rollouts.

## Idea

The reward is terminal task success only. Finer process signal is kept OUT of
the optimized reward (anything inside it gets optimized and hacked) and routed
through two non-hackable channels:

- **Leg 1 — hard constraint**: mechanically-checkable process errors (placeholder
  args, schema violations, loops, max-turn stall, ungrounded writes) reject a
  rollout — its reward is forced to the failure band even if the environment
  judged it a success, and it is excluded from the GRPO group baseline. Binary,
  non-tradeable, so not hackable.
- **Leg 2 — divergence credit**: within a GRPO group, turns where the rollouts
  split (divergence turns = real decision points) get higher credit; shared
  turns get less. Read off trajectory structure, never learned.

A conservative Adaptive KL/Entropy Controller is also available on the rollout
side (disable with `ADAPTIVE_GRPO_CONTROL=0`).

## Project Layout

    configs/       2xA800 train/eval configs plus shared reward/tool configs
    docs/          runbooks and method notes
    experiments/   Experiment manifests and run records
    outputs/       Generated traces and summaries
    scripts/       Runnable entrypoints
    src/           Reusable implementation modules

## Local Smoke Tests

    python3 scripts/run_tests.py
    python3 scripts/test/test_legs.py
    python3 scripts/test/stress_context_isolation.py --concurrency 32

Pure-Python logic tests for Leg 1 / Leg 2 and the rollout integration — no GPUs
needed.

## Server Pipeline For 2xA800 80GB

Use one of the two dedicated runbooks:

        docs/server_runbook_2xa800_72b_teacher_32b_user.md

Main 2xA800 entrypoints:

    # User simulator on GPU 1.
    CUDA_DEVICES=1 bash scripts/vllm_server/start_user_32b_awq_2xa800.sh
    # Optional 72B teacher on GPU 0 for SFT collection.
    CUDA_DEVICES=0 bash scripts/vllm_server/start_teacher_72b_awq_2xa800.sh
    # SFT data collection (72B-teacher / 32B-user setup).
    bash scripts/train/sft/collect_sft_teacher_72b_user_32b_2xa800.sh
    # Stop vLLM servers before SFT; 2-GPU SFT uses both A800s.
    bash scripts/train/sft/run_sft_lora_2xa800.sh
    # Restart the 32B user simulator on GPU 1 before GRPO.
    CUDA_DEVICES=1 bash scripts/vllm_server/start_user_32b_awq_2xa800.sh
    CUDA_VISIBLE_DEVICES=0 bash scripts/train/grpo/run_long_horizon_grpo_2xa800_80g_32b_user.sh
    # Export merged HF checkpoints before evaluation.
    bash scripts/train/grpo/export_grpo_checkpoints_2xa800.sh
    # Evaluation.
    bash scripts/eval/eval_sft_airline_2xa800_32b_user.sh

Main configs:

- configs/models/2xa800_80g_qwen_72b_teacher.yaml
- configs/train/sft/sft_airline_lora_2xa800_80g.yaml
- configs/train/grpo/long_horizon_grpo_2xa800_80g_32b_user.yaml
- configs/eval/eval_airline_sft_2xa800_32b_user.yaml

The veRL integration entrypoint is:

    delta_critic_ledger.verl_integration.interaction.LongHorizonTauBenchInteraction

It computes terminal-outcome reward plus the Leg-1 constraint gate (violated
rollouts forced to the failure band). Leg-2 divergence credit is consumed by the
patched advantage estimator (`grpo_long_horizon`, injected by
`scripts/setup/patch_verl_long_horizon_grpo.py`).

## Independence

The runtime imports only this repository's `src` package plus independently
installed tau-bench and veRL. It does not import files, configs, or modules from
another agentic-RL project.
