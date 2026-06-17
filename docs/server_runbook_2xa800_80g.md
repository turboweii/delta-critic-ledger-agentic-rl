# 2xA800 80GB Server Runbook: 7B Policy + 32B-AWQ User

This runbook adapts the original 8x4090 production path to a 2xA800 80GB
server. The layout is:

- GPU 0: 7B policy SFT/GRPO training and assistant vLLM during evaluation.
- GPU 1: 32B-AWQ user simulator vLLM. During SFT data collection, the same
  endpoint also serves the teacher role.

## 1. Preflight

```bash
cd delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENAI_API_KEY=dummy
```

Expected local models:

```text
../models/Qwen2.5-7B-Instruct
../models/Qwen2.5-32B-Instruct-AWQ
```

The model/GPU plan is recorded in:

```bash
configs/models/2xa800_80g_qwen.yaml
```

## 2. Start the 32B-AWQ User/Teacher Server

Run this in a long-lived tmux pane:

```bash
CUDA_DEVICES=1 \
PORT=8001 \
TP_SIZE=1 \
GPU_MEM_UTIL=0.70 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=4 \
bash scripts/vllm_server/start_user_32b_awq_2xa800.sh
```

During SFT collection, both teacher and user simulator requests go to
`http://localhost:8001/v1`.

## 3. Collect SFT Data

```bash
bash scripts/train/sft/collect_sft_teacher_2xa800.sh
```

Defaults are conservative for one 32B endpoint:

- `BEST_OF_N=4`
- `NUM_WORKERS=1`
- `TEACHER_TEMPERATURES=0.0,0.0,0.5,0.8`

For a faster but heavier collection:

```bash
BEST_OF_N=6 NUM_WORKERS=2 bash scripts/train/sft/collect_sft_teacher_2xa800.sh
```

This also prepares GRPO parquet data under `experiments/grpo_airline/`.

## 4. Run 2-GPU SFT

Stop the 32B-AWQ vLLM from step 2 before SFT, because SFT uses both A800s.

```bash
bash scripts/train/sft/run_sft_lora_2xa800.sh
```

This uses:

```bash
configs/train/sft/sft_airline_lora_2xa800_80g.yaml
```

The merged model is written to:

```bash
experiments/sft_lora_merged
```

## 5. Evaluate SFT

Restart the 32B-AWQ user server on GPU 1:

```bash
CUDA_DEVICES=1 bash scripts/vllm_server/start_user_32b_awq_2xa800.sh
```

Start the assistant server on GPU 0:

```bash
CUDA_DEVICES=0 \
MODEL_PATH=experiments/sft_lora_merged \
SERVED_MODEL_NAME=delta-assistant-7b-sft \
PORT=8000 \
TP_SIZE=1 \
GPU_MEM_UTIL=0.70 \
MAX_MODEL_LEN=16384 \
bash scripts/vllm_server/start_assistant_7b.sh
```

Keep the 32B user server from step 2 running, then:

```bash
bash scripts/eval/eval_sft_airline_2xa800_32b_user.sh
```

## 6. Run GRPO

Stop the assistant vLLM if it is still running on GPU 0. Keep the 32B user
simulator on GPU 1 running.

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train/grpo/run_delta_ledger_grpo_2xa800_80g_32b_user.sh
```

This uses:

```bash
configs/train/grpo/delta_ledger_grpo_2xa800_80g_32b_user.yaml
```

The key memory changes versus the 8x4090 config are:

- `trainer.n_gpus_per_node: 1`
- `rollout.tensor_model_parallel_size: 1`
- `data.train_batch_size: 2`
- `actor.ppo_mini_batch_size: 2`
- `rollout.max_num_seqs: 4`
- `rollout.gpu_memory_utilization: 0.32`

If GPU 0 OOMs, first reduce:

```yaml
actor_rollout_ref:
  rollout:
    max_num_seqs: 2
    gpu_memory_utilization: 0.28
data:
  train_batch_size: 1
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 1
```

## 7. Export and Evaluate GRPO Checkpoint

After training, export the checkpoint you want to test, for example step 300:

```bash
bash scripts/train/grpo/export_grpo_checkpoints_2xa800.sh
```

Start the assistant server:

```bash
CUDA_DEVICES=0 \
MODEL_PATH=experiments/delta_ledger_grpo_2xa800/hf_step_300 \
SERVED_MODEL_NAME=delta-assistant-7b-grpo \
PORT=8000 \
TP_SIZE=1 \
GPU_MEM_UTIL=0.70 \
MAX_MODEL_LEN=16384 \
bash scripts/vllm_server/start_assistant_7b.sh
```

Keep the 32B user server on GPU 1 running, then:

```bash
bash scripts/eval/eval_delta_grpo_airline_2xa800_32b_user.sh
```
