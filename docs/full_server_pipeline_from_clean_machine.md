# Clean Server Pipeline: Delta-Critic + Evidence Ledger Agentic RL

本文档描述如何在一台干净的 8 x RTX 4090 服务器上，从零开始跑通整个项目。

默认项目路径假设为：

```bash
/home/ubuntu/delta-critic-ledger-agentic-rl
```

默认目录结构建议：

```text
/home/ubuntu/
  delta-critic-ledger-agentic-rl/
  tau-bench/
  verl/
  models/
    Qwen2.5-7B-Instruct/
    Qwen2.5-32B-Instruct-AWQ/
```

## 0. 基础检查

先确认驱动和 GPU 正常：

```bash
nvidia-smi
```

应能看到 8 张 4090。若看不到 GPU，需要先安装 NVIDIA driver，这一步不属于项目代码流程。

## 1. 创建 Python 环境

推荐 Python 3.10：

```bash
conda create -n dcl-agentic-rl python=3.10 -y
conda activate dcl-agentic-rl

python -m pip install --upgrade pip setuptools wheel
```

## 2. 准备代码

把本项目放到：

```bash
/home/ubuntu/delta-critic-ledger-agentic-rl
```

进入项目：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
```

安装项目依赖：

```bash
pip install -r requirements-server.txt
```

## 3. 安装 tau-bench 和 veRL

在 `/home/ubuntu` 下准备 tau-bench 和 veRL：

```bash
cd /home/ubuntu
git clone <TAU_BENCH_REPO_URL> tau-bench
git clone <VERL_REPO_URL> verl
```

安装：

```bash
conda activate dcl-agentic-rl

pip install -e /home/ubuntu/tau-bench
pip install -e /home/ubuntu/verl
```

设置 `PYTHONPATH`：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH
```

建议把这行写进 `~/.bashrc` 或每个 tmux window 里手动执行。

## 4. 下载模型

安装 Hugging Face 工具：

```bash
pip install huggingface_hub
```

创建模型目录：

```bash
mkdir -p /home/ubuntu/models
```

下载 7B assistant base：

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /home/ubuntu/models/Qwen2.5-7B-Instruct
```

下载 32B-AWQ teacher/user：

```bash
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ \
  --local-dir /home/ubuntu/models/Qwen2.5-32B-Instruct-AWQ
```

项目默认配置使用相对路径：

```text
../models/Qwen2.5-7B-Instruct
../models/Qwen2.5-32B-Instruct-AWQ
```

所以项目放在 `/home/ubuntu/delta-critic-ledger-agentic-rl` 时，上面的模型路径是匹配的。

## 5. 项目接口检查

回到项目目录：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH
```

运行本地检查：

```bash
python3 scripts/test/interface_audit.py
python3 scripts/run_tests.py
python3 scripts/train/grpo/gen_tool_config.py
```

这三步必须先通过。

其中：

```bash
python3 scripts/train/grpo/gen_tool_config.py
```

会生成或刷新：

```text
configs/tool_config/tau_bench_airline_tools.yaml
```

## 6. 使用 tmux

强烈建议使用 tmux。这个项目有多个长进程：

```text
teacher 32B vLLM
user 32B vLLM
assistant 7B vLLM
SFT data collection
SFT training
GRPO training
evaluation
memory monitor
```

创建 SFT session：

```bash
tmux new -s dcl_sft
```

常用命令：

```bash
tmux ls
tmux attach -t dcl_sft
tmux new-window -n <name>
tmux kill-window -t <name>
tmux kill-session -t dcl_sft
```

在每个 tmux window 里都建议先执行：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH
```

## 7. SFT 数据采集

### 7.1 SFT 在做什么

SFT 现在采用 teacher rollout distillation：

```text
teacher policy: Qwen2.5-32B-Instruct-AWQ
user simulator: Qwen2.5-32B-Instruct-AWQ
environment: tau-bench airline
sampling: 50 tasks x best_of_8
filter: 只保留 success=True 且未 contaminated 的轨迹
holdout: 10 个 task 不进 SFT train
student: Qwen2.5-7B-Instruct LoRA
```

SFT 不是直接 replay oracle actions 给 7B，而是让 32B teacher 真实和 tau-bench 环境交互，采成功轨迹蒸馏 7B。

### 7.2 启动 teacher 32B

新 tmux window：

```bash
tmux new-window -n teacher32b
```

执行：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH

CUDA_DEVICES=0,1 \
MODEL_PATH=../models/Qwen2.5-32B-Instruct-AWQ \
SERVED_MODEL_NAME=delta-teacher-32b-awq \
PORT=8002 \
TP_SIZE=2 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=4 \
bash scripts/vllm_server/start_teacher_32b_awq_8x4090.sh
```

### 7.3 启动 user simulator 32B

新 tmux window：

```bash
tmux new-window -n user32b
```

执行：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH

CUDA_DEVICES=2,3 \
MODEL_PATH=../models/Qwen2.5-32B-Instruct-AWQ \
SERVED_MODEL_NAME=delta-user-32b-awq \
PORT=8001 \
TP_SIZE=2 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=4 \
bash scripts/vllm_server/start_user_32b_awq_8x4090.sh
```

### 7.4 检查 teacher/user 服务

新 tmux window：

```bash
tmux new-window -n collect_sft
```

执行：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH

python3 scripts/vllm_server/check_servers.py \
  --teacher http://localhost:8002/v1 \
  --user http://localhost:8001/v1
```

### 7.5 采集 SFT 数据

```bash
bash scripts/train/sft/collect_sft_teacher_8x4090.sh
```

输出目录：

```text
experiments/sft_collect_airline/
```

关键文件：

```text
task_XXXX.jsonl
task_XXXX.meta.json
task_XXXX_contaminated.jsonl
train.jsonl
eval.jsonl
holdout_train.jsonl
split.json
summary.json
collect_config.json
```

查看统计：

```bash
cat experiments/sft_collect_airline/summary.json
```

重点看：

```text
num_tasks_with_success
task_coverage_rate
total_success_trajectories
total_contaminated_trajectories
n_train_trajectories
n_holdout_trajectories
```

如果 `n_train_trajectories` 很少，可以提高 `best_of_n`，或降低并发压力，或检查 teacher tool-call 解析是否正常。

### 7.6 停掉 teacher/user vLLM

SFT 训练要用 8 张卡，所以采集结束后要释放显存。

可以在 teacher/user window 里按 `Ctrl+C`，或者：

```bash
tmux kill-window -t teacher32b
tmux kill-window -t user32b
```

确认显存释放：

```bash
nvidia-smi
```

## 8. SFT 训练

### 8.1 SFT 训练做什么

SFT 训练 7B assistant LoRA。

读取：

```text
experiments/sft_collect_airline/train.jsonl
experiments/sft_collect_airline/eval.jsonl
```

训练 loss 使用 assistant-only masking：

```text
system/user/tool observation -> labels = -100
assistant natural language -> labels = token id
assistant tool_calls -> labels = token id
```

也就是说，模型只学习 assistant 应该如何：

```text
思考
调用工具
生成工具参数
根据工具结果继续对话
最终回复用户
```

### 8.2 运行 SFT

新 tmux window：

```bash
tmux new-window -n sft_train
```

执行：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/train/sft/run_sft_lora_8x4090.sh
```

输出：

```text
experiments/sft_lora_8x4090/
experiments/sft_lora_merged/
```

其中：

```text
experiments/sft_lora_merged
```

是后面 GRPO 的初始 policy。

## 9. SFT Evaluation

### 9.1 启动 assistant 7B SFT

新 tmux session：

```bash
tmux new -s dcl_eval
tmux rename-window assistant7b
```

执行：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH

CUDA_DEVICES=0 \
MODEL_PATH=experiments/sft_lora_merged \
SERVED_MODEL_NAME=delta-assistant-7b-sft \
PORT=8000 \
bash scripts/vllm_server/start_assistant_7b.sh
```

### 9.2 启动 user 32B

新 window：

```bash
tmux new-window -n user32b_eval
```

执行：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH

CUDA_DEVICES=1,2 \
MODEL_PATH=../models/Qwen2.5-32B-Instruct-AWQ \
SERVED_MODEL_NAME=delta-user-32b-awq \
PORT=8001 \
TP_SIZE=2 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=4 \
bash scripts/vllm_server/start_user_32b_awq_8x4090.sh
```

检查：

```bash
python3 scripts/vllm_server/check_servers.py \
  --assistant http://localhost:8000/v1 \
  --user http://localhost:8001/v1
```

### 9.3 跑 SFT eval

新 window：

```bash
tmux new-window -n eval_sft
```

执行：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH

bash scripts/eval/eval_sft_airline_8x4090_32b_user.sh
```

输出：

```text
outputs/eval_sft_airline_8x4090_32b_user/eval_report.json
```

核心指标：

```text
pass_at_1
pass_at_4
success_rate
error_rate
avg_tool_calls
per_task
```

## 10. 收集真实 rollout trace

该步骤用于分析 SFT policy 的行为和 Delta/Evidence trace。

```bash
python3 scripts/data/collect_tau_rollouts.py \
  --config configs/eval/eval_airline_sft_8x4090_32b_user.yaml \
  --output-dir experiments/data_airline_delta
```

输出：

```text
experiments/data_airline_delta/rollouts.jsonl
experiments/data_airline_delta/summary.json
```

每条 rollout 包含：

```text
messages
terminal_reward
combined_reward
delta_trace
ledger_trace
success/error
```

## 11. Delta-Critic 模块

Delta-Critic 解决长链路 credit assignment。

每个 tau-bench task 初始化时：

```text
s0 = env.data 初始数据库状态
target_actions = env.task.actions 去掉 respond
s* = 在 copy 出来的 target_data 上 replay target_actions
D_goal = diff(s0, s*)
```

每个 agent 工具调用前后：

```text
Phi(s) = D_goal 中已经和 s* 一致的字段比例
delta_t = Phi(s_after) - Phi(s_before)
```

含义：

```text
正确写操作 -> delta_t > 0
无效 read -> delta_t = 0
错误写操作 / 破坏已有正确字段 -> delta_t < 0
```

注意：Delta-Critic 使用 tau-bench 的 ground-truth `task.actions` 构造目标状态，因此它是 tau-bench-style environment 下的 state-delta reward shaping，不是无 oracle 的通用 reward。

## 12. Evidence Ledger 模块

Evidence Ledger 解决参数 grounding 和过早写操作。

它维护：

```text
user_id
reservation_id
payment_id
flight_number
date
passenger
constraints
```

每个实体记录来源工具和来源 turn。

写工具调用前检查：

```text
book_reservation
cancel_reservation
update_reservation_flights
update_reservation_baggages
update_reservation_passengers
send_certificate
```

分类：

```text
evidence_grounded_write
ungrounded_write
conflicting_write
not_write
```

它提供：

```text
训练信号: evidence bonus / bad write penalty
诊断指标: ungrounded_write_rate, conflicting_write_rate, grounded_write_rate
```

## 13. GRPO 训练

### 13.1 GRPO 在做什么

GRPO 从 SFT merged policy 开始：

```text
initial policy = experiments/sft_lora_merged
```

环境：

```text
assistant policy: Qwen2.5-7B SFT
user simulator: Qwen2.5-32B-AWQ
tools: real tau-bench airline tools
state: real tau-bench env.data
```

reward：

```text
R = terminal_reward
  + beta_delta * sum(delta_t)
  + beta_evidence * evidence_bonus
```

默认：

```text
beta_delta = 0.3
beta_evidence = 0.1
```

GRPO 不只优化最终成功，也优化：

```text
每步数据库状态是否朝目标推进
写操作参数是否来自证据
是否过早写
是否破坏已有正确状态
```

### 13.2 调整 GRPO GPU 分配

GRPO rollout 期间 user simulator 必须在线。关闭 assistant 7B eval server，重新把
user 32B-AWQ 放到 GPU 6-7；GPU 0-5 留给 veRL GRPO。

```bash
tmux kill-window -t assistant7b
tmux kill-window -t user32b_eval
```

重新启动 user simulator：

```bash
CUDA_DEVICES=6,7 \
MODEL_PATH=../models/Qwen2.5-32B-Instruct-AWQ \
SERVED_MODEL_NAME=delta-user-32b-awq \
PORT=8001 \
TP_SIZE=2 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=4 \
bash scripts/vllm_server/start_user_32b_awq_8x4090.sh
```

确认 GPU 0-5 已释放、GPU 6-7 运行 user simulator：

```bash
nvidia-smi
```

### 13.3 启动 GRPO

新 session：

```bash
tmux new -s dcl_grpo
```

window `grpo`：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
bash scripts/train/grpo/run_delta_ledger_grpo_8x4090_32b_user.sh
```

GRPO 配置：

```text
configs/train/grpo/delta_ledger_grpo_8x4090_32b_user.yaml
```

关键配置：

```text
rollout.name = vllm
multi_turn.enable = true
interaction_config_path = configs/interaction_config/tau_bench_airline_delta_ledger.yaml
tool_config_path = configs/tool_config/tau_bench_airline_tools.yaml
calculate_log_probs = true
algorithm.rollout_correction.bypass_mode = true
n_gpus_per_node = 8
```

输出：

```text
experiments/delta_ledger_grpo_8x4090/checkpoints
```

### 13.4 监控显存

新 window：

```bash
tmux new-window -n monitor
```

执行：

```bash
watch -n 2 nvidia-smi
```

或：

```bash
DURATION=3600 INTERVAL=5 bash scripts/test/profile_memory.sh
```

## 14. GRPO Evaluation

### 14.1 启动 GRPO checkpoint assistant

假设评估 step 300：

```bash
tmux new -s dcl_grpo_eval
tmux rename-window assistant_grpo
```

执行：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH

CUDA_DEVICES=0 \
MODEL_PATH=experiments/delta_ledger_grpo_8x4090/hf_step_300 \
SERVED_MODEL_NAME=delta-assistant-7b-grpo \
PORT=8000 \
bash scripts/vllm_server/start_assistant_7b.sh
```

### 14.2 启动 user 32B

```bash
tmux new-window -n user32b_grpo_eval
```

执行：

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH

CUDA_DEVICES=1,2 \
MODEL_PATH=../models/Qwen2.5-32B-Instruct-AWQ \
SERVED_MODEL_NAME=delta-user-32b-awq \
PORT=8001 \
TP_SIZE=2 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=4 \
bash scripts/vllm_server/start_user_32b_awq_8x4090.sh
```

检查：

```bash
python3 scripts/vllm_server/check_servers.py \
  --assistant http://localhost:8000/v1 \
  --user http://localhost:8001/v1
```

### 14.3 跑 GRPO eval

```bash
bash scripts/eval/eval_delta_grpo_airline_8x4090_32b_user.sh
bash scripts/eval/eval_checkpoints_delta_grpo.sh
```

输出：

```text
outputs/eval_delta_grpo_airline_8x4090_32b_user/eval_report.json
outputs/delta_ledger_checkpoint_eval/
```

核心指标：

```text
pass_at_1
pass_at_4
success_rate
error_rate
avg_tool_calls
wrong/no-op write rate
ungrounded_write_rate
conflicting_write_rate
grounded_write_rate
avg_delta_reward_sum
avg_evidence_bonus_sum
```

## 15. 推荐消融实验

至少跑四组：

```text
terminal only
terminal + Delta-Critic
terminal + Evidence Ledger
terminal + Delta-Critic + Evidence Ledger
```

对应 reward configs：

```text
configs/reward/terminal_only.yaml
configs/reward/delta_only.yaml
configs/reward/ledger_only.yaml
configs/reward/delta_ledger.yaml
```

展示重点：

```text
pass_at_1 / pass_at_4 是否提升
ungrounded_write_rate 是否下降
wrong/no-op write rate 是否下降
delta_reward_sum 与成功率是否相关
Evidence-grounded write 的成功率是否更高
```

## 16. 常见问题

### 16.1 vLLM 启动 OOM

降低：

```text
MAX_MODEL_LEN
MAX_NUM_SEQS
GPU_MEM_UTIL
```

例如：

```bash
MAX_MODEL_LEN=8192 MAX_NUM_SEQS=2 bash scripts/vllm_server/start_teacher_32b_awq_8x4090.sh
```

### 16.2 SFT 采集成功轨迹太少

检查：

```bash
cat experiments/sft_collect_airline/summary.json
```

可尝试：

```text
best_of_n 从 8 增加到 12 或 16
num_workers 从 2 降到 1
确认 teacher endpoint 开了 --enable-auto-tool-choice 和 --tool-call-parser hermes
确认 user simulator 正常返回，不是空回复或格式错误
```

### 16.3 GRPO OOM

优先改：

```text
rollout.max_num_seqs
rollout.max_num_batched_tokens
ref.log_prob_micro_batch_size_per_gpu
actor.ppo_micro_batch_size_per_gpu
```

先跑低资源 smoke，再拉长训练。

### 16.4 Delta-Critic 没有 goal fields

检查 tau-bench task 是否有 `task.actions`。

Delta-Critic 依赖：

```text
s* = replay env.task.actions 后的 target data
```

如果 `task.actions` 为空或 replay 失败，Delta-Critic 就无法提供有效状态差分信号。

## 17. 最短命令版

```bash
cd /home/ubuntu/delta-critic-ledger-agentic-rl
conda activate dcl-agentic-rl
export PYTHONPATH=$PWD/src:/home/ubuntu/tau-bench:/home/ubuntu/verl:$PYTHONPATH

python3 scripts/test/interface_audit.py
python3 scripts/run_tests.py
python3 scripts/train/grpo/gen_tool_config.py

# SFT data collection: start teacher/user in tmux windows first
bash scripts/vllm_server/start_teacher_32b_awq_8x4090.sh
bash scripts/vllm_server/start_user_32b_awq_8x4090.sh
bash scripts/train/sft/collect_sft_teacher_8x4090.sh

# Stop teacher/user vLLM, then train SFT
bash scripts/train/sft/run_sft_lora_8x4090.sh

# Start assistant/user, then eval SFT
bash scripts/vllm_server/start_assistant_7b.sh
bash scripts/vllm_server/start_user_32b_awq_8x4090.sh
bash scripts/eval/eval_sft_airline_8x4090_32b_user.sh

# Stop assistant eval server; keep/restart user simulator on GPUs 6-7, then train GRPO on GPUs 0-5
bash scripts/train/grpo/run_delta_ledger_grpo_8x4090_32b_user.sh

# Restart assistant/user, then eval GRPO
bash scripts/eval/eval_delta_grpo_airline_8x4090_32b_user.sh
bash scripts/eval/eval_checkpoints_delta_grpo.sh
```
