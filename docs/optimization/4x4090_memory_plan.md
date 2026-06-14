# 4x4090 Memory Plan

## Profiles

- `delta_ledger_grpo_4x4090_lowmem.yaml`: first server smoke test.
- `delta_ledger_grpo_4x4090.yaml`: default short run.
- `delta_ledger_grpo_4x4090_fast.yaml`: use after memory is stable.

## Initial Settings

| profile | group size | max model len | response len | train batch |
| --- | ---: | ---: | ---: | ---: |
| lowmem | 2 | 8192 | 4096 | 2 |
| default | 4 | 16384 | 8192 | 4 |
| fast | 4 | 16384 | 8192 | 4 |

## Debug Order

1. Run lowmem for 10-20 steps.
2. If rollout OOMs, reduce `rollout.max_num_seqs` first.
3. If ref logprob OOMs, reduce `ref.log_prob_micro_batch_size_per_gpu`.
4. If actor update OOMs, reduce `ppo_mini_batch_size`.
5. Only increase context length after stable checkpoint/eval works.

## Runtime Memory Guards

- Keep full tool observations out of training state; store only short previews.
- `delta_critic.max_trace_steps` limits retained per-trajectory trace entries.
- Use `scripts/test/profile_memory.sh` during every first run on a new server.

