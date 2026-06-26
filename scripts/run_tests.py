#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.long_horizon import (  # noqa: E402
    LoopGuard,
    ProcessFeatureTracker,
    RewardEnvelope,
    RewardEnvelopeConfig,
    balanced_aggregation_loss,
    normalize_advantages,
    sequence_mean_loss,
    should_update_group,
    smooth_advantages,
    token_mean_loss,
)
from delta_critic_ledger.long_horizon.advantage import dynamic_clip_range  # noqa: E402
from delta_critic_ledger.verl_integration.context import make_initial_state  # noqa: E402
from delta_critic_ledger.verl_integration.reward_state import (  # noqa: E402
    compute_long_horizon_components,
    init_long_horizon_state,
    record_final_response,
    record_tool_transition,
)
from delta_critic_ledger.verl_integration.tools import execute_tau_tool_action  # noqa: E402
from delta_critic_ledger.schemas import Action  # noqa: E402
from delta_critic_ledger.adaptive_control import (  # noqa: E402
    AdaptiveEntropyController,
    AdaptiveKlEntropyController,
    summarize_rollout_state,
)


FEATURE_KEYS = [
    "placeholder_count",
    "duplicate_tool_args_count",
    "tool_error_count",
    "repeat_same_error_count",
    "recovery_after_error_count",
    "read_tool_diversity",
    "write_before_read_count",
    "early_transfer",
    "early_final_respond",
    "trajectory_length",
    "schema_violation_count",
    "empty_reasoning_count",
    "think_loop_count",
]


def test_reward_envelope_keeps_success_above_failure() -> None:
    envelope = RewardEnvelope(RewardEnvelopeConfig(dense_beta=1.0, success_max=1.2, failure_max=0.4))
    failed = envelope.compute(terminal_reward=0.0, process_score=10.0)
    success = envelope.compute(terminal_reward=1.0, process_score=-10.0)
    assert failed["score"] == 0.4
    assert success["score"] == 1.0
    assert failed["score"] < success["score"]


def test_process_features_are_generic() -> None:
    tracker = ProcessFeatureTracker()
    tracker.record_tool("update_reservation_flights", {"reservation_id": "<reservation_id>"}, "Error: missing flight")
    tracker.record_tool("update_reservation_flights", {"reservation_id": "<reservation_id>"}, "Error: missing flight")
    tracker.record_tool("search_direct_flight", {"origin": "SFO", "destination": "JFK"}, "Found flights")
    features = tracker.to_dict()
    assert features["tool_calls"] == 3
    assert features["placeholder_args"] == 2
    assert features["repeated_tool_args"] == 1
    assert features["repeated_same_errors"] == 1
    assert features["recovered_after_error"] == 1
    assert features["read_tool_calls"] == 1


def test_process_features_use_stable_trace_feature_names() -> None:
    tracker = ProcessFeatureTracker()
    tracker.record_tool("update_reservation_flights", {"reservation_id": "<reservation_id>"}, "Error: missing flight")
    tracker.record_empty_reasoning()
    features = tracker.to_dict()
    for key in FEATURE_KEYS:
        assert key in features
    assert features["placeholder_count"] == 1
    assert features["tool_error_count"] == 1
    assert features["write_before_read_count"] == 1
    assert features["empty_reasoning_count"] == 1


def test_online_process_features_record_final_transfer_and_empty_reasoning() -> None:
    state = make_initial_state(task_id=0)
    state["long_horizon_state"] = init_long_horizon_state({})
    record_final_response(state, "")
    record_tool_transition(state, "transfer_to_human_agents", {"summary": "help"}, "Transfer successful")
    record_tool_transition(state, "think", {"thought": ""}, "ok")
    features = compute_long_horizon_components(state)["process_features"]
    assert features["early_final_respond"] == 1
    assert features["early_transfer"] == 1
    assert features["empty_reasoning_count"] == 2


def test_loop_guard_stops_repeated_tool_args() -> None:
    guard = LoopGuard.from_config({"max_repeated_tool_args": 2})
    tracker = ProcessFeatureTracker()
    assert not guard.observe("get_user_details", {"user_id": "u1"}, "ok", tracker).should_stop
    assert not guard.observe("get_user_details", {"user_id": "u1"}, "ok", tracker).should_stop
    decision = guard.observe("get_user_details", {"user_id": "u1"}, "ok", tracker)
    assert decision.should_stop is True
    assert decision.reason == "repeated_tool_args"
    assert tracker.to_dict()["loop_detected"] == 1


def test_long_horizon_reward_state_records_loop_and_features() -> None:
    state = make_initial_state(task_id=0)
    state["long_horizon_state"] = init_long_horizon_state({"loop_guard": {"max_repeated_tool_args": 1}})
    record_tool_transition(state, "get_user_details", {"user_id": "u1"}, "ok")
    record_tool_transition(state, "get_user_details", {"user_id": "u1"}, "ok")
    components = compute_long_horizon_components(state)
    assert components["reward_mode"] == "long_horizon_terminal_envelope"
    assert components["score"] <= 0.4
    assert components["process_features"]["loop_detected"] == 1
    assert components["loop_stop_reason"] == "repeated_tool_args"


def test_advantage_smoothing_and_zero_variance() -> None:
    adv, stats = normalize_advantages([0.0, 0.0, 0.0])
    assert adv == [0.0, 0.0, 0.0]
    assert stats.std == 0.0
    smooth, stats = smooth_advantages([0.0, 1.0], running_mean=0.5, running_std=0.5, blend=0.5)
    assert smooth[0] < 0.0 and smooth[1] > 0.0
    assert stats.count == 2


def test_dynamic_clip_range_adjusts_width() -> None:
    widened = dynamic_clip_range(base_low=0.8, base_high=1.2, clip_ratio=0.5, target_clip_ratio=0.2)
    narrowed = dynamic_clip_range(base_low=0.8, base_high=1.2, clip_ratio=0.01, target_clip_ratio=0.2)
    assert widened.high - widened.low > narrowed.high - narrowed.low


def test_group_filter_skips_zero_variance_and_loops() -> None:
    assert should_update_group([0.0, 0.0, 0.0]).should_update is False
    assert should_update_group([0.0, 1.0, 0.0]).should_update is True
    decision = should_update_group([0.0, 1.0, 0.0], loop_rate=0.8, max_loop_rate=0.5)
    assert decision.should_update is False
    assert decision.reason == "loop_rate_too_high"


def test_balanced_aggregation_reduces_sign_length_coupling() -> None:
    token_losses = [[1.0], [1.0] * 100]
    advantages = [1.0, -1.0]
    token_loss = token_mean_loss(token_losses, advantages)
    seq_loss = sequence_mean_loss(token_losses, advantages)
    balanced = balanced_aggregation_loss(token_losses, advantages)
    assert token_loss < -0.9
    assert seq_loss == 0.0
    assert balanced == 0.0


def test_terminal_tool_does_not_leak_oracle_state() -> None:
    class TransferTool:
        @staticmethod
        def invoke(data, **kwargs):
            return "Transfer successful"

    class RewardResult:
        reward = 0.0

    class FakeEnv:
        def __init__(self):
            self.data = {"status": "agent_state"}
            self.actions = []
            self.tools_map = {"transfer_to_human_agents": TransferTool()}
            self.terminate_tools = ["transfer_to_human_agents"]

        def calculate_reward(self):
            self.data = {"status": "reward_reset_sideeffect"}
            self.actions.append(Action("reward_replay_action", {}))
            return RewardResult()

    env = FakeEnv()
    action = Action("transfer_to_human_agents", {"summary": "help"})
    observation, reward, done, after = execute_tau_tool_action(env, action)
    assert observation == "Transfer successful"
    assert reward == 0.0 and done is True
    assert after == {"status": "agent_state"}
    assert env.data == {"status": "agent_state"}
    assert [item.name for item in env.actions] == ["transfer_to_human_agents"]


def test_adaptive_controllers_use_generic_process_features() -> None:
    controller = AdaptiveEntropyController.from_config({"enabled": True})
    base = {"temperature": 0.7, "top_p": 0.9}
    bad = make_initial_state(task_id=0)
    bad["process_features"] = {"tool_calls": 3, "tool_error_rate": 0.5, "repeat_tool_args_rate": 0.0}
    params, decision = controller.adjust_sampling_params(base, bad)
    assert decision.mode == "repair"
    assert params["temperature"] < base["temperature"]

    stalled = summarize_rollout_state({"process_features": {"tool_calls": 3}})
    decision = AdaptiveKlEntropyController().recommend(stalled)
    assert decision.mode == "explore"
    assert decision.actor_kl_loss_coef < 0.01


def main() -> None:
    test_reward_envelope_keeps_success_above_failure()
    test_process_features_are_generic()
    test_process_features_use_stable_trace_feature_names()
    test_online_process_features_record_final_transfer_and_empty_reasoning()
    test_loop_guard_stops_repeated_tool_args()
    test_long_horizon_reward_state_records_loop_and_features()
    test_advantage_smoothing_and_zero_variance()
    test_dynamic_clip_range_adjusts_width()
    test_group_filter_skips_zero_variance_and_loops()
    test_balanced_aggregation_reduces_sign_length_coupling()
    test_terminal_tool_does_not_leak_oracle_state()
    test_adaptive_controllers_use_generic_process_features()
    print("All project tests passed.")


if __name__ == "__main__":
    main()
