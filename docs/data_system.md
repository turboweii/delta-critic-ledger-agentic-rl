# Data System

SFT collection command for the 2xA800 setup (72B teacher, 72B user):

    bash scripts/train/sft/collect_sft_teacher_72b_user_72b_2xa800.sh

SFT collector outputs:

- experiments/sft_collect_airline/task_XXXX.jsonl
- experiments/sft_collect_airline/task_XXXX.meta.json
- experiments/sft_collect_airline/task_XXXX_contaminated.jsonl
- experiments/sft_collect_airline/train.jsonl
- experiments/sft_collect_airline/eval.jsonl
- experiments/sft_collect_airline/holdout_train.jsonl
- experiments/sft_collect_airline/split.json
- experiments/sft_collect_airline/summary.json

The 2xA800 SFT setup runs 50 airline tasks, keeps only successful clean teacher trajectories, and reserves 10 tasks as holdout.

Default real tau-bench rollout command:

    python3 scripts/data/collect_tau_rollouts.py       --config configs/eval/eval_airline_sft_2xa800_72b_user.yaml       --output-dir experiments/data_airline_delta

Outputs:

- experiments/data_airline_delta/rollouts.jsonl
- experiments/data_airline_delta/summary.json

Each rollout row contains raw OpenAI-style messages, terminal reward, combined Delta/Ledger reward, traces, and success/error metadata.
