#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger import Action
from delta_critic_ledger.mock_airline import MockAirlineTools, make_demo_data
from delta_critic_ledger.verl_integration.context import CURRENT_TAU_ENV, CURRENT_TAU_STATE, make_initial_state
from delta_critic_ledger.verl_integration.reward_state import init_delta_reward_state, record_tool_transition


class MockEnv:
    def __init__(self, reservation_id: str):
        self.data = make_demo_data()
        original = self.data["reservations"].pop("Z7GOZK")
        original["reservation_id"] = reservation_id
        self.data["reservations"][reservation_id] = original
        self.data["users"]["olivia_gonzalez_2305"]["reservations"] = [reservation_id]
        self.tools = MockAirlineTools()
        self.tools_map = {
            "cancel_reservation": type("CancelTool", (), {"invoke": staticmethod(lambda data, **kw: self.tools.cancel_reservation(data, **kw))}),
            "get_user_details": type("UserTool", (), {"invoke": staticmethod(lambda data, **kw: self.tools.get_user_details(data, **kw))}),
            "get_reservation_details": type("ReservationTool", (), {"invoke": staticmethod(lambda data, **kw: self.tools.get_reservation_details(data, **kw))}),
        }
        self.task = type("Task", (), {
            "user_id": "olivia_gonzalez_2305",
            "actions": [Action("cancel_reservation", {"reservation_id": reservation_id})],
        })()


async def run_one(idx: int) -> tuple[str, float, list[str]]:
    reservation_id = f"R{idx:05d}"[-6:]
    env = MockEnv(reservation_id)
    instance_id = f"instance-{idx}"
    state = make_initial_state(idx, instance_id=instance_id, env_id=id(env))
    state["state_id"] = id(state)
    state["delta_reward_state"] = init_delta_reward_state(env, beta_delta=0.3, beta_evidence=0.1)

    CURRENT_TAU_ENV.set(env)
    CURRENT_TAU_STATE.set(state)
    await asyncio.sleep(0)

    before = copy.deepcopy(env.data)
    obs = env.tools.cancel_reservation(env.data, reservation_id)
    after = copy.deepcopy(env.data)
    record_tool_transition(state, "cancel_reservation", {"reservation_id": reservation_id}, obs, before, after)
    state["num_tool_calls"] += 1

    current_env = CURRENT_TAU_ENV.get()
    current_state = CURRENT_TAU_STATE.get()
    assert current_env is env
    assert current_state is state
    assert state["instance_id"] == instance_id
    assert state["env_id"] == id(env)
    assert state["delta_reward_sum"] == 1.0
    assert env.data["reservations"][reservation_id]["status"] == "cancelled"
    return reservation_id, state["delta_reward_sum"], [step.tool for step in state["delta_steps"]]


async def main_async(concurrency: int) -> None:
    results = await asyncio.gather(*(run_one(i) for i in range(concurrency)))
    reservation_ids = [r[0] for r in results]
    assert len(set(reservation_ids)) == concurrency
    assert all(delta == 1.0 for _, delta, _ in results)
    print(f"Context isolation stress passed: {concurrency} concurrent trajectories")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    asyncio.run(main_async(args.concurrency))


if __name__ == "__main__":
    main()

