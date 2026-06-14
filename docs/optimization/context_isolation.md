# Context Isolation

Tau-bench multi-turn rollout must not share environment state across trajectories.

Implemented guards:

- `CURRENT_TAU_ENV` and `CURRENT_TAU_STATE` are context variables.
- each trajectory state stores `instance_id`, `trace_id`, `env_id`, and `state_id`.
- tool execution raises if the current env does not match `state.env_id`.
- interaction raises when `instance_id` is unknown or mismatched.
- trace retention is capped by `delta_critic.max_trace_steps`.

Local stress test:

```bash
python3 scripts/test/stress_context_isolation.py --concurrency 32
```

The test creates concurrent mock trajectories, each mutating a different
reservation, and asserts that every trajectory keeps independent env/state/delta
signals.

