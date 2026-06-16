#!/usr/bin/env python3
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.adaptive_control import (  # noqa: E402
    AdaptiveEntropyController,
    AdaptiveKlEntropyController,
    summarize_rollout_state,
)
from delta_critic_ledger import Action, DeltaCritic, DeltaLedgerRunner, EvidenceLedger
from delta_critic_ledger.mock_airline import MockAirlineTools, make_demo_data
from delta_critic_ledger.verl_integration.context import make_initial_state
from delta_critic_ledger.verl_integration.reward_state import record_tool_transition
from delta_critic_ledger.verl_integration.tools import execute_tau_tool_action


def test_delta_critic_cancel() -> None:
    tools = MockAirlineTools()
    data = make_demo_data()
    critic = DeltaCritic.from_target_actions(
        data,
        [Action("cancel_reservation", {"reservation_id": "Z7GOZK"})],
        tools,
    )
    assert "reservations.Z7GOZK.status" in critic.goal_field_names()
    assert critic.phi(data) == 0.0

    after = copy.deepcopy(data)
    tools(after, Action("cancel_reservation", {"reservation_id": "Z7GOZK"}))
    step = critic.score_step(0, "cancel_reservation", data, after)
    assert step.delta_reward == 1.0
    assert step.changed_goal_fields == ["reservations.Z7GOZK.status"]


def test_wrong_write_no_progress() -> None:
    tools = MockAirlineTools()
    data = make_demo_data()
    critic = DeltaCritic.from_target_actions(data, [Action("cancel_reservation", {"reservation_id": "Z7GOZK"})], tools)
    after = copy.deepcopy(data)
    tools(after, Action("cancel_reservation", {"reservation_id": "BAD999"}))
    step = critic.score_step(0, "cancel_reservation", data, after)
    assert step.delta_reward == 0.0
    assert not step.changed_goal_fields


def test_evidence_ledger_grounding() -> None:
    tools = MockAirlineTools()
    data = make_demo_data()
    ledger = EvidenceLedger(seed_entities={"user_id": ["olivia_gonzalez_2305"]})

    ungrounded = ledger.check_write(0, "cancel_reservation", {"reservation_id": "Z7GOZK"})
    assert ungrounded.write_grounding_status == "ungrounded_write"

    obs = tools(data, Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}))
    ledger.update_from_tool(1, "get_user_details", {"user_id": "olivia_gonzalez_2305"}, obs)

    grounded = ledger.check_write(2, "cancel_reservation", {"reservation_id": "Z7GOZK"})
    assert grounded.write_grounding_status == "evidence_grounded_write"

    conflict = ledger.check_write(3, "cancel_reservation", {"reservation_id": "BAD999"})
    assert conflict.write_grounding_status == "conflicting_write"


def test_runner_end_to_end() -> None:
    tools = MockAirlineTools()
    runner = DeltaLedgerRunner(
        task_id=1,
        initial_data=make_demo_data(),
        target_actions=[Action("cancel_reservation", {"reservation_id": "Z7GOZK"})],
        execute_tool=tools,
        seed_entities={"user_id": ["olivia_gonzalez_2305"]},
    )
    report = runner.run([
        Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}),
        Action("cancel_reservation", {"reservation_id": "BAD999"}),
        Action("get_reservation_details", {"reservation_id": "Z7GOZK"}),
        Action("cancel_reservation", {"reservation_id": "Z7GOZK"}),
    ])
    assert report.final_success is True
    assert report.terminal_reward == 1.0
    assert report.delta_reward_sum == 1.0
    assert any(s.ledger.write_grounding_status == "conflicting_write" for s in report.steps)
    assert any(s.ledger.write_grounding_status == "evidence_grounded_write" for s in report.steps)


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
            self.data = {"status": "oracle_target"}
            self.actions.append(Action("oracle_write", {}))
            return RewardResult()

    env = FakeEnv()
    action = Action("transfer_to_human_agents", {"summary": "help"})
    observation, reward, done, after = execute_tau_tool_action(env, action)
    assert observation == "Transfer successful"
    assert reward == 0.0 and done is True
    assert after == {"status": "agent_state"}
    assert env.data == {"status": "agent_state"}
    assert [item.name for item in env.actions] == ["transfer_to_human_agents"]


def test_grounded_noop_cannot_farm_evidence_reward() -> None:
    tools = MockAirlineTools()
    before = make_demo_data()
    critic = DeltaCritic.from_target_actions(
        before,
        [Action("cancel_reservation", {"reservation_id": "Z7GOZK"})],
        tools,
    )
    state = make_initial_state(task_id=0)
    state["delta_reward_state"] = {
        "critic": critic,
        "ledger": EvidenceLedger(seed_entities={"reservation_id": ["Z7GOZK"]}),
        "beta_delta": 0.3,
        "beta_evidence": 0.1,
    }
    record_tool_transition(
        state,
        "cancel_reservation",
        {"reservation_id": "Z7GOZK"},
        "Already cancelled",
        before,
        copy.deepcopy(before),
    )
    assert state["delta_reward_sum"] == 0.0
    assert state["evidence_bonus_sum"] == 0.0


def test_recovered_goal_field_cannot_repeat_evidence_reward() -> None:
    tools = MockAirlineTools()
    initial = make_demo_data()
    cancelled = copy.deepcopy(initial)
    tools(cancelled, Action("cancel_reservation", {"reservation_id": "Z7GOZK"}))
    critic = DeltaCritic(initial, cancelled)
    state = make_initial_state(task_id=0)
    state["delta_reward_state"] = {
        "critic": critic,
        "ledger": EvidenceLedger(seed_entities={"reservation_id": ["Z7GOZK"]}),
        "beta_delta": 0.3,
        "beta_evidence": 0.1,
    }
    params = {"reservation_id": "Z7GOZK"}
    record_tool_transition(state, "cancel_reservation", params, "Cancelled", initial, cancelled)
    record_tool_transition(state, "cancel_reservation", params, "Regressed", cancelled, initial)
    record_tool_transition(state, "cancel_reservation", params, "Cancelled again", initial, cancelled)
    assert state["delta_reward_sum"] == 1.0
    assert state["evidence_bonus_sum"] == 1.0


def test_adaptive_entropy_controller_modes() -> None:
    controller = AdaptiveEntropyController.from_config({"enabled": True})
    assert AdaptiveEntropyController.from_config(None).config.enabled is True
    assert AdaptiveEntropyController.from_config({"enabled": False}).config.enabled is False
    base = {"temperature": 0.7, "top_p": 0.9}

    stalled = make_initial_state(task_id=0)
    stalled["num_tool_calls"] = 2
    params, decision = controller.adjust_sampling_params(base, stalled)
    assert decision.mode == "explore"
    assert params["temperature"] > base["temperature"]

    progressed = make_initial_state(task_id=0)
    progressed["delta_reward_state"] = {"goal_fields": ["a", "b"]}
    progressed["num_tool_calls"] = 2
    progressed["delta_steps"] = [
        {"delta_reward": 1.0, "changed_goal_fields": ["a"]},
        {"delta_reward": 1.0, "changed_goal_fields": ["b"]},
    ]
    params, decision = controller.adjust_sampling_params(base, progressed)
    assert decision.mode == "consolidate"
    assert params["temperature"] < base["temperature"]

    bad = make_initial_state(task_id=0)
    bad["num_tool_calls"] = 2
    bad["ledger_steps"] = [
        {"write_grounding_status": "conflicting_write"},
        {"write_grounding_status": "evidence_grounded_write"},
    ]
    params, decision = controller.adjust_sampling_params(base, bad)
    assert decision.mode == "repair"
    assert params["temperature"] < base["temperature"]


def test_adaptive_kl_controller_recommendations() -> None:
    controller = AdaptiveKlEntropyController()

    stalled = summarize_rollout_state({"num_tool_calls": 3, "delta_steps": [], "ledger_steps": []})
    decision = controller.recommend(stalled)
    assert decision.mode == "explore"
    assert decision.actor_kl_loss_coef < 0.01
    assert decision.temperature > 0.7

    bad = summarize_rollout_state({
        "num_tool_calls": 3,
        "ledger_steps": [{"write_grounding_status": "ungrounded_write"}],
    })
    decision = controller.recommend(bad)
    assert decision.mode == "repair"
    assert decision.actor_kl_loss_coef > 0.01
    assert decision.temperature < 0.7


def main() -> None:
    test_delta_critic_cancel()
    test_wrong_write_no_progress()
    test_evidence_ledger_grounding()
    test_runner_end_to_end()
    test_terminal_tool_does_not_leak_oracle_state()
    test_grounded_noop_cannot_farm_evidence_reward()
    test_recovered_goal_field_cannot_repeat_evidence_reward()
    test_adaptive_entropy_controller_modes()
    test_adaptive_kl_controller_recommendations()
    print("All project tests passed.")


if __name__ == "__main__":
    main()
