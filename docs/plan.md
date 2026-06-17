# Project Plan: Delta-Critic + Evidence Ledger

## Summary

Build a compact but complete agentic RL research project around two modules:

1. **Delta-Critic**: assigns credit to each tool action by measuring its marginal
   contribution to the target database state.
2. **Evidence Ledger**: tracks entities observed from tool results and diagnoses
   whether write actions are grounded, conflicting, or premature.

The goal is to address tau-bench-like failures that are not solved by ordinary
terminal reward: sparse credit assignment, parameter grounding errors, state
tracking failures, and wrong/no-op writes.

## Method

Delta-Critic avoids heuristic process rewards. Given an initial database `s0` and
target actions, it replays target actions on an oracle copy to produce `s*`. It
then computes a structural diff `D_goal = diff(s0, s*)`. During rollout, each
tool call receives:

```text
Phi(s) = fraction of D_goal fields already equal to s*
delta_t = Phi(s_after_t) - Phi(s_before_t)
```

Evidence Ledger extracts entities such as `reservation_id`, `user_id`,
`payment_id`, `flight_number`, and `date` from tool parameters and observations.
Before a write tool is scored, it checks whether key parameters have appeared in
previous evidence.

## Training Signal

The online GRPO integration uses:

```text
R = terminal_reward
  + beta_delta * sum(delta_t)
  + beta_evidence * sum(evidence_bonus_t)
```

Default demo values:

- `beta_delta = 0.3`
- `beta_evidence = 0.1`

## Implementation Scope

This standalone implementation includes:

- A mock airline environment with read/write tools for CPU unit tests.
- Delta-Critic goal diff and per-step reward.
- Evidence Ledger entity extraction and grounding diagnosis.
- Demo scripts that generate delta and ledger traces.
- Unit tests that do not require tau-bench or external packages.
- Real tau-bench airline teacher-rollout SFT data collection.
- 7B LoRA SFT and veRL multi-turn GRPO integration.
- An 8x4090 production configuration with a 32B-AWQ user simulator.
- A 2xA800 80GB production-style configuration that keeps the 32B-AWQ user
  simulator on GPU 1 while GPU 0 runs 7B policy GRPO.
