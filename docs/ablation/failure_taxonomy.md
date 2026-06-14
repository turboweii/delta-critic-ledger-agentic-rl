# Failure Taxonomy

Delta-Critic and Evidence Ledger traces support automatic failure labels:

- `wrong_write`: a write used a conflicting grounded entity.
- `ungrounded_write`: a write used a parameter without prior evidence.
- `state_regression`: a tool call moved goal fields away from the oracle target.
- `no_positive_delta`: the trajectory never made state progress.
- `late_positive_delta`: the first positive delta happened after many turns.
- `missing_write`: no write action was attempted.
- `tool_loop`: the same tool+parameters repeated at least three times.
- `success_clean`: success without detected issues.
- `success_with_issues`: final success despite diagnosed risky actions.

These labels are intended for evaluation and analysis, not for hard environment
intervention.

