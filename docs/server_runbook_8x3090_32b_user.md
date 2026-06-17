# 8x3090 Server Runbook: 7B Policy + 32B-AWQ User

This follows the 8x4090 pipeline, but uses more conservative vLLM and GRPO
settings for RTX 3090 24GB cards.

## Layout

- GPUs 0-5: 7B policy GRPO training.
- GPUs 6-7: 32B-AWQ user simulator during GRPO/evaluation.
- SFT can use all 8 GPUs; stop vLLM servers before SFT.

## SFT Data Collection

Start teacher and user simulator servers in separate tmux panes:

```bash
CUDA_DEVICES=4,5 PORT=8002 bash scripts/vllm_server/start_teacher_32b_awq_8x3090.sh
CUDA_DEVICES=6,7 PORT=8001 bash scripts/vllm_server/start_user_32b_awq_8x3090.sh
```

Collect trajectories and build GRPO parquet data:

```bash
bash scripts/train/sft/collect_sft_teacher_8x3090.sh
```

If the two 32B servers are too tight or slow, use one server sequentially:

```bash
CUDA_DEVICES=6,7 PORT=8001 bash scripts/vllm_server/start_user_32b_awq_8x3090.sh
python3 scripts/train/sft/collect_sft_data.py \
  --mode teacher_rollout \
  --output-dir experiments/sft_collect_airline \
  --env-name airline \
  --task-split test \
  --start-index 0 \
  --end-index 50 \
  --use-user-sim \
  --user-model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --user-provider openai \
  --user-base-url http://localhost:8001/v1 \
  --teacher-model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --teacher-base-url http://localhost:8001/v1 \
  --best-of-n 2 \
  --temperatures 0.0,0.5 \
  --num-workers 1 \
  --holdout-size 10
bash scripts/train/grpo/prepare_grpo_data.sh
```

## SFT

Stop all vLLM servers, then:

```bash
bash scripts/train/sft/run_sft_lora_8x3090.sh
```

The merged model is written to `experiments/sft_lora_merged`.

## SFT Evaluation

```bash
CUDA_DEVICES=6,7 PORT=8001 bash scripts/vllm_server/start_user_32b_awq_8x3090.sh
CUDA_DEVICES=0 MODEL_PATH=experiments/sft_lora_merged \
  SERVED_MODEL_NAME=delta-assistant-7b-sft \
  bash scripts/vllm_server/start_assistant_7b.sh
bash scripts/eval/eval_sft_airline_8x3090_32b_user.sh
```

## GRPO

Stop the assistant server on GPU 0. Keep the user simulator on GPUs 6-7.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
bash scripts/train/grpo/run_delta_ledger_grpo_8x3090_32b_user.sh
```

If this OOMs, reduce `rollout.max_num_seqs` from 6 to 4 in
`configs/train/grpo/delta_ledger_grpo_8x3090_32b_user.yaml`.

## Export and Evaluate GRPO

```bash
bash scripts/train/grpo/export_grpo_checkpoints_8x3090.sh
CUDA_DEVICES=0 MODEL_PATH=experiments/delta_ledger_grpo_8x3090/hf_step_300 \
  SERVED_MODEL_NAME=delta-assistant-7b-grpo \
  bash scripts/vllm_server/start_assistant_7b.sh
bash scripts/eval/eval_delta_grpo_airline_8x3090_32b_user.sh
```
