# Tau-Bench SFT + GRPO

This project contains one clean pipeline:

1. collect tau-bench SFT data
2. train SFT
3. evaluate SFT
4. train ordinary veRL GRPO
5. evaluate GRPO

## Main 2xA800 Entrypoints

```bash
CUDA_DEVICES=1 bash scripts/vllm_server/start_user_72b_awq_2xa800.sh
CUDA_DEVICES=0 bash scripts/vllm_server/start_teacher_72b_awq_2xa800.sh
bash scripts/train/sft/collect_sft_teacher_72b_user_72b_2xa800.sh
bash scripts/train/sft/run_sft_lora_2xa800.sh
bash scripts/eval/eval_sft_airline_2xa800_72b_user.sh
CUDA_DEVICES=1 bash scripts/vllm_server/start_user_72b_awq_2xa800.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train/grpo/run_grpo_2xa800_80g_72b_user.sh
bash scripts/train/grpo/export_grpo_checkpoints_2xa800.sh
bash scripts/eval/eval_grpo_airline_2xa800_72b_user.sh
```

## Main Configs

- configs/models/2xa800_80g_qwen_72b_teacher.yaml
- configs/train/sft/sft_airline_lora_2xa800_80g.yaml
- configs/eval/eval_airline_sft_2xa800_72b_user.yaml
- configs/train/grpo/grpo_2xa800_80g_72b_user.yaml
- configs/eval/eval_airline_grpo_2xa800_72b_user.yaml
- configs/interaction_config/tau_bench_airline.yaml

GRPO uses veRL native `algorithm.adv_estimator: grpo`.
