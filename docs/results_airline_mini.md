# Airline Mini Results

Command:

```bash
python3 scripts/run_ablation.py --experiment configs/experiments/airline_mini.json --output-dir outputs/airline_mini
python3 scripts/analyze_outputs.py --summary outputs/airline_mini/summary.json
```

## Summary

| reward | success_rate | avg_combined_reward | avg_delta | grounded | conflicting | ungrounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| delta_ledger | 0.667 | 0.850 | 0.667 | 0.333 | 0.333 | 0.333 |
| delta_only | 0.667 | 0.867 | 0.667 | 0.333 | 0.333 | 0.333 |
| ledger_only | 0.667 | 0.650 | 0.667 | 0.333 | 0.333 | 0.333 |
| terminal_only | 0.667 | 0.667 | 0.667 | 0.333 | 0.333 | 0.333 |

## Interpretation

This deterministic mini suite is not meant to prove model improvement. It verifies
that the project can generate the signals needed for agentic RL:

- successful writes produce positive state-delta credit;
- wrong identifiers are labeled as conflicting writes;
- premature writes are separated from evidence-grounded writes;
- reward configs produce different combined rewards over the same trajectories.

The next step is to replace `MockAirlineTools` with real tau-bench `env.step`
calls and feed the combined reward into GRPO/RFT training.

