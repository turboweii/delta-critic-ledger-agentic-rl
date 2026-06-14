# Experiment Design

## Ablations

The project includes four reward configs:

- `terminal_only`: original sparse outcome reward.
- `delta_only`: terminal reward + state-delta credit.
- `ledger_only`: terminal reward + evidence grounding bonus.
- `delta_ledger`: combined method.

## Mock Task Suite

The standalone task suite has three airline-style tasks:

- `cancel_reservation`: target state is a cancelled reservation.
- `baggage_update`: target state updates baggage fields and payment history.
- `flight_update`: target state updates cabin and flight segments.

Each task has several trajectory variants:

- `successful`: reads evidence and writes correctly.
- `wrong_write`: uses a wrong/conflicting identifier.
- `premature_write`: writes before collecting evidence.
- `noop_then_success`: performs no-op reads before success.

## Metrics

The ablation runner computes:

- success rate
- average combined reward
- average delta reward
- average evidence bonus
- conflicting write rate
- ungrounded write rate
- grounded write rate
- average first positive delta step

