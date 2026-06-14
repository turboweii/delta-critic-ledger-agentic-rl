# 8x4090 Server Runbook: 7B Assistant + 32B-AWQ User

## Model Plan

- Assistant/policy: `Qwen/Qwen2.5-7B-Instruct`
- User simulator: `Qwen/Qwen2.5-32B-Instruct-AWQ`

GPU plan for evaluation:

- GPU 0: assistant 7B vLLM
- GPU 1-3: user 32B-AWQ vLLM, TP=3
- GPU 4-7: free for parallel jobs or checkpoint prep

GPU plan for training:

- SFT: GPU 0-7 after stopping teacher/user servers.
- GRPO: GPU 0-5 for veRL; GPU 6-7 keep the 32B-AWQ user simulator online.

## Smoke Tests

```bash
python3 scripts/run_tests.py
python3 scripts/test/stress_context_isolation.py --concurrency 32
python3 scripts/train/grpo/gen_tool_config.py
```

## SFT

Start both 32B-AWQ services first. The default SFT collector uses real tau-bench
teacher rollouts: 32B-AWQ acts as the teacher policy, another 32B-AWQ endpoint
acts as the user simulator, and only successful clean trajectories enter SFT.

Teacher policy:

```bash
CUDA_DEVICES=0,1 \
MODEL_PATH=../models/Qwen2.5-32B-Instruct-AWQ \
SERVED_MODEL_NAME=delta-teacher-32b-awq \
PORT=8002 \
TP_SIZE=2 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=4 \
bash scripts/vllm_server/start_teacher_32b_awq_8x4090.sh
```

User simulator:

```bash
CUDA_DEVICES=2,3 \
MODEL_PATH=../models/Qwen2.5-32B-Instruct-AWQ \
SERVED_MODEL_NAME=delta-user-32b-awq \
PORT=8001 \
TP_SIZE=2 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=4 \
bash scripts/vllm_server/start_user_32b_awq_8x4090.sh
```

Check:

```bash
python3 scripts/vllm_server/check_servers.py \
  --teacher http://localhost:8002/v1 \
  --user http://localhost:8001/v1
```

```bash
bash scripts/train/sft/collect_sft_teacher_8x4090.sh
```

After collection finishes, stop the teacher and user vLLM servers to release GPU
memory. Then run LoRA SFT:

```bash
bash scripts/train/sft/run_sft_lora_8x4090.sh
```

Dry run:

```bash
python3 scripts/train/sft/collect_sft_data.py \
  --mode oracle \
  --env-name airline \
  --task-split test \
  --start-index 0 \
  --end-index 50
python3 scripts/train/sft/train_sft_lora.py \
  --config configs/train/sft/sft_airline_lora_8x4090.yaml \
  --dry-run
```

## vLLM Servers

Assistant:

```bash
CUDA_DEVICES=0 \
MODEL_PATH=../models/Qwen2.5-7B-Instruct \
SERVED_MODEL_NAME=delta-assistant-7b \
PORT=8000 \
bash scripts/vllm_server/start_assistant_7b.sh
```

User simulator:

```bash
CUDA_DEVICES=1,2 \
MODEL_PATH=../models/Qwen2.5-32B-Instruct-AWQ \
SERVED_MODEL_NAME=delta-user-32b-awq \
PORT=8001 \
TP_SIZE=2 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=4 \
bash scripts/vllm_server/start_user_32b_awq_8x4090.sh
```

Check:

```bash
python3 scripts/vllm_server/check_servers.py \
  --assistant http://localhost:8000/v1 \
  --user http://localhost:8001/v1
```

## GRPO

Before GRPO, stop the assistant eval server and restart the user simulator on GPUs 6-7:

```bash
CUDA_DEVICES=6,7 \
MODEL_PATH=../models/Qwen2.5-32B-Instruct-AWQ \
SERVED_MODEL_NAME=delta-user-32b-awq \
PORT=8001 \
TP_SIZE=2 \
bash scripts/vllm_server/start_user_32b_awq_8x4090.sh
```

Optional pre-GRPO real rollout dataset:

```bash
python3 scripts/data/collect_tau_rollouts.py \
  --config configs/eval/eval_airline_sft_8x4090_32b_user.yaml \
  --output-dir experiments/data_airline_delta
```

```bash
bash scripts/train/grpo/run_delta_ledger_grpo_8x4090_32b_user.sh
```

Memory profile:

```bash
DURATION=600 INTERVAL=5 bash scripts/test/profile_memory.sh
```

## Eval

```bash
bash scripts/eval/eval_sft_airline_8x4090_32b_user.sh
bash scripts/eval/eval_delta_grpo_airline_8x4090_32b_user.sh
```

Use `--dry-run` first to validate output paths:

```bash
bash scripts/eval/eval_sft_airline_8x4090_32b_user.sh --dry-run
```
