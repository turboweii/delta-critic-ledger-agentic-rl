# Method: Long-Horizon GRPO + Hard Constraints + Divergence Credit

This project trains long-horizon tool agents (tau-bench airline) with
**outcome-only GRPO** plus two add-ons that improve credit assignment **without**
introducing hackable shaped rewards.

## Core principle

The reward stays **terminal outcome only** (task success 0/1). Any finer process
signal is kept OUT of the optimized reward, because anything inside the reward
gets optimized and gets hacked. Process information is routed through two
non-hackable channels instead — a hard constraint and a fixed credit structure.

## Leg 1 — process correctness as a HARD constraint

Mechanically-checkable process errors reject a rollout, even if the environment
judged it a success:

- placeholder write args (`<reservation_id>`, `todo`, `xxx`)
- tool args that violate their schema
- loops (repeated tool+args / repeated same error / pure-think loop)
- max-turn stall
- **ungrounded writes**: a write tool whose entity-reference params
  (`reservation_id` / `payment_id` / `user_id` / `flight_number`) were never
  returned by any prior tool — the agent invented them.

```
constrained_reward = success AND NOT violated
```

A violated rollout is forced into the failure band (reward masking in
`reward_state.compute_long_horizon_components`) and, on the advantage side,
excluded from the GRPO group baseline (`constraint_violated` in the patched
advantage estimator). This is a **binary, non-tradeable** constraint: there is
no gradient toward "partially satisfying" it, so it is not hackable. A
curriculum (`warmup_steps` + `set_training_step`) relaxes it during warmup so a
naive early policy can still learn.

Grounding is **schema-driven**
(`grounded_write_verifier.entity_keys_from_schema`): which params are entity
references is read from the tool schema's descriptions, so it generalizes across
tool sets (tau-bench airline / retail / telecom, or other tool agents), not just
airline naming conventions.

## Leg 2 — per-turn credit from trajectory divergence

GRPO flattens one outcome advantage over every token. Leg 2 redistributes it:
within a GRPO group (G rollouts of the same prompt), a turn is a **divergence
turn** if the rollouts do NOT all take the same action there — that is where the
outcome actually depends on the choice. Divergence turns get higher credit
weight; shared turns (everyone does the same read/think) get less.

Weights are read off the trajectory **structure** (`divergence.py`), never
learned, never in the reward, so this channel is not hackable either. Total
credit is preserved — only its distribution across turns changes.

## Why this is not DPO / not a process reward model

- Reward = outcome only. No learned per-step scorer feeds the reward (that would
  be a PRM and would be hacked).
- No critic / GAE (hard to train stably on long horizons).
- DPO is a different paradigm (offline preference); this stays online GRPO.

## What is NOT yet wired (remaining integration work)

The algorithm and the advantage-consumption side are done and unit-tested. The
remaining work is the **rollout → DataProto** plumbing on the veRL side
(`constraint_violated` and `turn_token_weights` batch fields, token→turn
alignment, same-group divergence in the trainer) and true reject-resample
(re-generate to refill the batch). These need a live veRL rollout to validate.

See `src/delta_critic_ledger/long_horizon/` for the implementations and
`scripts/test/test_legs.py` for the tests.
