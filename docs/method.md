# Method: Delta-Critic + Evidence Ledger

## Delta-Critic

Tau-bench-style tasks expose target actions. Instead of asking an LLM judge whether
an intermediate step is good, Delta-Critic replays target actions on an oracle copy
of the environment database and obtains a target database state.

The goal field set is:

```text
D_goal = leaf_diff(initial_database, oracle_target_database)
```

Each tool action is then scored by marginal progress:

```text
Phi(s) = matched_goal_fields(s, oracle_target) / |D_goal|
delta_t = Phi(s_after_t) - Phi(s_before_t)
```

This turns sparse terminal reward into interpretable state-transition credit.

## Evidence Ledger

Evidence Ledger tracks entities observed through tools:

- `user_id`
- `reservation_id`
- `payment_id`
- `flight_number`
- `date`

Before write tools, it labels parameters as:

- `evidence_grounded_write`
- `ungrounded_write`
- `conflicting_write`
- `not_write`

This diagnoses whether a model is using tool evidence or guessing parameters.

## Why It Is Not DPO

DPO compares complete responses or trajectories. This project instead creates a
per-tool-call credit signal from environment state transitions and evidence flow.
The signal could be used by GRPO, PPO, RFT, or offline analysis; the core method is
reward decomposition, not preference optimization.

