# Data System

Default real SFT collection command:

```bash
bash scripts/train/sft/collect_sft_teacher_8x4090.sh
```

SFT collector outputs:

- `experiments/sft_collect_airline/task_XXXX.jsonl`
- `experiments/sft_collect_airline/task_XXXX.meta.json`
- `experiments/sft_collect_airline/task_XXXX_contaminated.jsonl`
- `experiments/sft_collect_airline/train.jsonl`
- `experiments/sft_collect_airline/eval.jsonl`
- `experiments/sft_collect_airline/holdout_train.jsonl`
- `experiments/sft_collect_airline/split.json`
- `experiments/sft_collect_airline/summary.json`

The default 8x4090 SFT setup runs 50 airline tasks with `best_of_n=8`, keeps only
successful clean teacher trajectories, and reserves 10 tasks as holdout.

Default real tau-bench rollout command:

```bash
python3 scripts/data/collect_tau_rollouts.py \
  --config configs/eval/eval_airline_sft_8x4090_32b_user.yaml \
  --output-dir experiments/data_airline_delta
```

Outputs:

- `experiments/data_airline_delta/rollouts.jsonl`
- `experiments/data_airline_delta/summary.json`

Each rollout row contains:

- raw OpenAI-style messages
- terminal reward and combined Delta/Ledger reward
- `delta_trace`
- `ledger_trace`
- success/error metadata

Mock rollout collection remains available only as a local unit-test fixture:

```bash
python3 scripts/data/collect_mock_rollouts.py
```
