# 2xA800 80GB: 72B Teacher, 32B User, 7B Policy

The GPU allocation changes by stage:

- SFT collection: GPU 0 runs the 72B-AWQ teacher; GPU 1 runs the 72B-AWQ user.
- SFT training: both GPUs train the 7B LoRA; stop both vLLM servers first.
- GRPO: GPU 0 trains and rolls out the 7B policy; GPU 1 runs the 72B-AWQ user.

Expected local models:

```text
../models/Qwen2.5-7B-Instruct
../models/Qwen2.5-72B-Instruct-AWQ
../models/Qwen2.5-72B-Instruct-AWQ
```

The complete model allocation is recorded in:

```text
configs/models/2xa800_80g_qwen_72b_teacher.yaml
```

## 1. Collect SFT Data

Start the 72B teacher on GPU 0:

```bash
CUDA_DEVICES=0 \
PORT=8002 \
TP_SIZE=1 \
GPU_MEM_UTIL=0.88 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=2 \
bash scripts/vllm_server/start_teacher_72b_awq_2xa800.sh
```

Start the 72B user simulator on GPU 1:

```bash
CUDA_DEVICES=1 \
PORT=8001 \
TP_SIZE=1 \
GPU_MEM_UTIL=0.70 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=4 \
bash scripts/vllm_server/start_user_72b_awq_2xa800.sh
```

Verify both services:

```bash
curl http://localhost:8002/v1/models
curl http://localhost:8001/v1/models
```

Collect successful teacher trajectories and prepare GRPO parquet data:

```bash
BEST_OF_N=4 \
NUM_WORKERS=1 \
bash scripts/train/sft/collect_sft_teacher_72b_user_72b_2xa800.sh
```

Keep `NUM_WORKERS=1` initially. The 72B endpoint is expected to be the
collection bottleneck.

## 2. Train the 7B SFT Model

Stop both vLLM servers so both A800s are free, then run:

```bash
bash scripts/train/sft/run_sft_lora_2xa800.sh
```

The merged model is written to:

```text
experiments/sft_lora_merged
```

## 3. Evaluate SFT

Start only the 72B user simulator on GPU 1:

```bash
CUDA_DEVICES=1 bash scripts/vllm_server/start_user_72b_awq_2xa800.sh
```

Start the SFT assistant on GPU 0:

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

Run evaluation:

```bash
bash scripts/eval/eval_sft_airline_2xa800_72b_user.sh
```

## 4. Run GRPO with the 32B User

Stop the assistant server on GPU 0. Keep the 72B user server on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train/grpo/run_grpo_2xa800_80g_72b_user.sh
```

The 72B teacher is not loaded during GRPO. GRPO
experiments should use the same SFT checkpoint, 72B user endpoint, data split,
and evaluation configuration. After GRPO, use the normal evaluation flow. The GRPO reward is the terminal tau-bench outcome reward.

## 5. Export and Evaluate GRPO

```bash
bash scripts/train/grpo/export_grpo_checkpoints_2xa800.sh
```

Then follow the checkpoint-serving and evaluation commands in
`docs/server_runbook_2xa800_80g.md`.


