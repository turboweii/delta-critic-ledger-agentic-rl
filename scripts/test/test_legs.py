#!/usr/bin/env python3
"""Local smoke tests for Leg 1 (constraint gate) and Leg 2 (per-turn credit).

Pure Python — no torch / veRL / vllm needed. Run from project root:
    PYTHONPATH=src python scripts/test/test_legs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger import (  # noqa: E402
    ConstraintGateConfig,
    apply_constraint_gate,
    distribute_advantage_over_turns,
    evaluate_constraint,
    per_turn_credit_weights,
)
from delta_critic_ledger.long_horizon import ProcessFeatures  # noqa: E402
from delta_critic_ledger.long_horizon.advantage import (  # noqa: E402
    decision_flags_from_tool_calls,
    turn_token_weights_for_rollout,
)
from delta_critic_ledger.long_horizon.constraint_gate import enforce_prob  # noqa: E402
from delta_critic_ledger.verl_integration.reward_state import (  # noqa: E402
    compute_long_horizon_components,
    init_long_horizon_state,
)


def _feat(**kw) -> ProcessFeatures:
    return ProcessFeatures(**kw)


# ---- Leg 1: constraint gate ----
def test_clean_success_passes():
    v = evaluate_constraint(_feat(tool_calls=5, write_tool_calls=2))
    assert not v.violated
    g = apply_constraint_gate(1.0, v)
    assert g["constrained_reward"] == 1.0 and not g["rejected"]


def test_placeholder_rejects_even_success():
    v = evaluate_constraint(_feat(placeholder_args=1, tool_calls=3))
    assert v.violated and "placeholder_args" in v.reasons
    g = apply_constraint_gate(1.0, v)
    assert g["rejected"] and g["constrained_reward"] == 0.0
    assert g["would_have_succeeded"] is True


def test_loop_schema_maxturn_trip():
    for kw, expect in [
        ({"loop_detected": 1}, "loop_detected"),
        ({"schema_violations": 1}, "schema_violation"),
        ({"max_turn_hit": 1}, "max_turn_stall"),
    ]:
        v = evaluate_constraint(_feat(**kw))
        assert v.violated and expect in v.reasons


def test_disabled_gate_never_rejects():
    cfg = ConstraintGateConfig(enabled=False)
    assert not evaluate_constraint(_feat(placeholder_args=5), cfg).violated


def test_curriculum():
    assert enforce_prob(0, 100) == 0.0
    assert enforce_prob(50, 100) == 0.5
    assert enforce_prob(200, 100) == 1.0
    assert enforce_prob(5, 0) == 1.0


def test_reward_state_rejects_violated_success():
    state = {"total_reward": 1.0, "long_horizon_state": init_long_horizon_state({})}
    state["long_horizon_state"]["feature_tracker"].features.placeholder_args = 1
    state["long_horizon_state"]["feature_tracker"].features.tool_calls = 3
    comp = compute_long_horizon_components(state)
    assert comp["rejected"] is True and comp["score"] == 0.0


def test_reward_state_keeps_clean_success():
    state = {"total_reward": 1.0, "long_horizon_state": init_long_horizon_state({})}
    comp = compute_long_horizon_components(state)
    assert comp["rejected"] is False and comp["score"] >= 1.0


# ---- Leg 2: per-turn credit ----
def test_uniform_weights():
    w = per_turn_credit_weights(4)
    assert len(w) == 4 and abs(sum(w) - 1.0) < 1e-9


def test_decision_weights():
    w = per_turn_credit_weights(4, scheme="decision", decision_turn_mask=[False, True, False, True])
    assert abs(sum(w) - 1.0) < 1e-9 and w[1] > w[0]


def test_distribute_preserves_total():
    per_turn = distribute_advantage_over_turns(0.73, 5)
    assert abs(sum(per_turn) - 0.73) < 1e-9


def test_turn_token_weights_helper():
    w = turn_token_weights_for_rollout([False, True, False], [5, 4, 3])
    assert len(w) == 12 and w[5:9] == [2.0] * 4 and w[:5] == [1.0] * 5


def test_normalize_scale_invariant_decision_upweighted():
    # mirror the patched estimator: norm_w = w / mean_w over policy tokens
    mask = [1, 1, 0, 1, 1, 1, 0, 1, 1]
    weights = [1, 1, 0, 2, 2, 2, 0, 1, 1]
    sum_m = sum(mask)
    mean_w = sum(w * m for w, m in zip(weights, mask)) / sum_m
    norm = [w / mean_w for w in weights]
    assert abs(sum(n * m for n, m in zip(norm, mask)) / sum_m - 1.0) < 1e-9
    assert norm[3] > norm[0]
    adv = [0.5 * n * m for n, m in zip(norm, mask)]
    assert adv[2] == 0.0 and adv[6] == 0.0


def test_decision_flags_bridge():
    flags = decision_flags_from_tool_calls(
        [["get_user_details"], ["update_reservation_flights", "think"], ["think"]]
    )
    assert flags == [False, True, False]


def test_grounded_write_verifier():
    from delta_critic_ledger.long_horizon.grounded_write_verifier import evaluate_grounded_write

    r = evaluate_grounded_write({"reservation_id": "ABC123", "cabin": "business"}, ["Reservation ABC123 confirmed"])
    assert r.grounded and r.checked_keys == ("reservation_id",)  # cabin skipped (choice field)
    r2 = evaluate_grounded_write({"reservation_id": "ABC123", "payment_id": "credit_card_999"}, ["Reservation ABC123"])
    assert not r2.grounded and r2.ungrounded_keys == ("payment_id",)


def test_reward_state_grounding_reject():
    from delta_critic_ledger.verl_integration.reward_state import compute_long_horizon_components, init_long_horizon_state

    state = {
        "total_reward": 1.0,
        "action_history": [
            {"tool": "get_user_details", "parameters": {"user_id": "u1"}, "observation_preview": "user u1, res ABC123"},
            {"tool": "update_reservation_flights", "parameters": {"reservation_id": "ABC123", "payment_id": "credit_card_999"}, "observation_preview": "updated"},
        ],
        "long_horizon_state": init_long_horizon_state({}),
    }
    comp = compute_long_horizon_components(state)
    assert comp["rejected"] and comp["score"] == 0.0
    assert "ungrounded_write" in comp["constraint_violation"]["reasons"]


def test_divergence_detection():
    from delta_critic_ledger.long_horizon.divergence import (
        divergence_flags_for_group,
        divergence_turn_weights,
        first_divergence_turn,
    )

    seqs = [
        [("get_user_details", "{}"), ("cancel_reservation", '{"r":"A"}')],
        [("get_user_details", "{}"), ("update_reservation_flights", '{"r":"A"}')],
        [("get_user_details", "{}"), ("cancel_reservation", '{"r":"A"}')],
    ]
    assert divergence_flags_for_group(seqs)[0] == [False, True]
    assert divergence_turn_weights(seqs)[0] == [1.0, 2.0]
    assert first_divergence_turn(seqs) == 1


def test_advantage_reject_excludes_violated():
    import statistics

    # pure-python mirror of the patched estimator's reject: violated rollouts
    # get advantage 0 and are excluded from the group baseline.
    scores = [1.0, 1.0, 0.0, 0.0]
    violated = [True, False, False, False]
    valid = [scores[i] for i in range(4) if not violated[i]]
    assert abs(statistics.mean(valid) - 1 / 3) < 1e-9  # baseline from valid subset, not 0.5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
