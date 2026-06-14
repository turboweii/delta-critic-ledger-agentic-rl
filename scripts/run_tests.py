#!/usr/bin/env python3
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger import Action, DeltaCritic, DeltaLedgerRunner, EvidenceLedger
from delta_critic_ledger.mock_airline import MockAirlineTools, make_demo_data


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


def main() -> None:
    test_delta_critic_cancel()
    test_wrong_write_no_progress()
    test_evidence_ledger_grounding()
    test_runner_end_to_end()
    print("All project tests passed.")


if __name__ == "__main__":
    main()

