# Delta-Critic Ledger Agentic RL

Standalone project prototype for state-transition credit assignment on tau-bench-style long-horizon tool agents.

This repository is now trimmed for the 2xA800 workflow. Training and evaluation use real tau-bench airline teacher rollouts, tool schemas, vLLM endpoints, and veRL multi-turn rollouts. Mock airline scripts are kept only for local unit tests and fast sanity checks.

## Idea

Long-horizon tool agents often receive only a final binary reward. Delta-Critic decomposes that terminal signal into state-transition credit:

    delta_t = Phi(state_after_t, target_state) - Phi(state_before_t, target_state)

Evidence Ledger labels whether write actions are grounded in tool-observed evidence, exposing parameter grounding and premature-write failures.

The GRPO path also includes a conservative Adaptive KL / Entropy Controller. It uses Evidence Ledger traces to detect stalled exploration, bad groundedness, or high task progress. It can be disabled with ADAPTIVE_GRPO_CONTROL=0.

## Project Layout

    config/        Backward-compatible single demo config
    configs/       2xA800 train/eval configs plus shared reward/tool configs
    docs/          2xA800 runbooks and method notes
    experiments/   Experiment manifests and run records
    outputs/       Generated traces and summaries
    scripts/       Runnable entrypoints
    src/           Reusable implementation modules

## Local Smoke Tests

    python3 scripts/run_tests.py
    python3 scripts/run_demo.py --config config/demo_airline.json
    python3 scripts/run_ablation.py --experiment configs/experiments/airline_mini.json
    python3 scripts/analyze_outputs.py --summary outputs/airline_mini/summary.json
    python3 scripts/eval/analyze_failure_modes.py --runs outputs/airline_mini_yaml/runs.jsonl
    python3 scripts/test/stress_context_isolation.py --concurrency 32

These local smoke tests use the lightweight fixture environment only to validate Delta-Critic and Evidence Ledger behavior without GPUs. The default training and data pipeline below uses real tau-bench.

## Server Pipeline For 2xA800 80GB

Use one of the two dedicated runbooks:

    docs/server_runbook_2xa800_80g.md
    docs/server_runbook_2xa800_72b_teacher_32b_user.md

Main 2xA800 entrypoints:

    # User simulator on GPU 1.
    CUDA_DEVICES=1 bash scripts/vllm_server/start_user_32b_awq_2xa800.sh

    # Optional 72B teacher on GPU 0 for SFT collection.
    CUDA_DEVICES=0 bash scripts/vllm_server/start_teacher_72b_awq_2xa800.sh

    # SFT data collection. Use the 72B-teacher variant by default.
    bash scripts/train/sft/collect_sft_teacher_72b_user_32b_2xa800.sh

    # Alternative lighter 32B-teacher collection.
    bash scripts/train/sft/collect_sft_teacher_2xa800.sh

    # Stop vLLM servers before SFT; 2-GPU SFT uses both A800s.
    bash scripts/train/sft/run_sft_lora_2xa800.sh

    # Restart the 32B user simulator on GPU 1 before GRPO.
    CUDA_DEVICES=1 bash scripts/vllm_server/start_user_32b_awq_2xa800.sh
    CUDA_VISIBLE_DEVICES=0 bash scripts/train/grpo/run_delta_ledger_grpo_2xa800_80g_32b_user.sh

    # Check whether Delta/Ledger dense reward is empirically aligned.
    python3 scripts/train/grpo/calibrate_delta_ledger_reward.py --wandb

    # Export merged HF checkpoints before GRPO evaluation.
    bash scripts/train/grpo/export_grpo_checkpoints_2xa800.sh

    # Evaluation.
    bash scripts/eval/eval_sft_airline_2xa800_32b_user.sh
    bash scripts/eval/eval_delta_grpo_airline_2xa800_32b_user.sh
    bash scripts/eval/eval_checkpoints_delta_grpo.sh

Main configs:

- configs/models/2xa800_80g_qwen.yaml
- configs/models/2xa800_80g_qwen_72b_teacher.yaml
- configs/train/sft/sft_airline_lora_2xa800_80g.yaml
- configs/train/grpo/delta_ledger_grpo_2xa800_80g_32b_user.yaml
- configs/eval/eval_airline_sft_2xa800_32b_user.yaml
- configs/eval/eval_airline_delta_grpo_2xa800_32b_user.yaml

The veRL integration entrypoint is:

    delta_critic_ledger.verl_integration.interaction.DeltaTauBenchInteraction

It computes a clipped conservative score:

    terminal_reward + beta_delta * clipped_state_delta + beta_evidence * clipped_ledger_bonus

The default clipping keeps dense shaping from overpowering the terminal tau-bench reward:

    delta in [-1, 1], evidence in [-2, 1], final score in [-0.2, 1.4]

Reward calibration check:

    python3 scripts/train/grpo/calibrate_delta_ledger_reward.py --wandb

This reports point-biserial correlations between Delta/Ledger features and task success. If positive delta or grounded writes are not empirically aligned with success, reduce their weights before running long GRPO jobs.

## Current Ablations

Reward configs:

- terminal_only
- delta_only
- ledger_only
- delta_ledger

Experiment configs:

- airline_mini
- airline_grounding_stress
- airline_state_delta_stress

## Independence

The runtime imports only this repository's src package plus independently installed tau-bench and veRL. It does not import files, configs, or modules from another agentic-RL project.
