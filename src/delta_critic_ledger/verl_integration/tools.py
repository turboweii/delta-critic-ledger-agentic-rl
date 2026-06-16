from __future__ import annotations

import copy
import json
from typing import Any, Optional
from uuid import uuid4

from delta_critic_ledger.verl_integration.context import CURRENT_TAU_ENV, CURRENT_TAU_STATE
from delta_critic_ledger.verl_integration.reward_state import record_tool_transition

try:
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
except Exception:  # pragma: no cover - only absent in standalone local tests
    BaseTool = object
    OpenAIFunctionToolSchema = object

    class ToolResponse:  # type: ignore
        def __init__(self, text: str = ""):
            self.text = text

try:
    from verl.utils.rollout_trace import rollout_trace_op
except Exception:  # pragma: no cover - only absent in standalone local tests
    def rollout_trace_op(func):
        return func


def execute_tau_tool_action(env: Any, action: Any) -> tuple[str, float, bool, dict[str, Any]]:
    """Execute a tau-bench tool without leaking oracle reward state into the rollout."""
    is_terminal = action.name in getattr(env, "terminate_tools", [])
    if not is_terminal:
        step_res = env.step(action)
        return (
            str(getattr(step_res, "observation", "")),
            float(getattr(step_res, "reward", 0.0)),
            bool(getattr(step_res, "done", False)),
            copy.deepcopy(env.data),
        )

    # tau-bench calculate_reward() resets env.data and replays oracle actions.
    # Preserve the actual post-tool state and remove those oracle actions afterward.
    env.actions.append(action)
    try:
        observation = str(env.tools_map[action.name].invoke(data=env.data, **action.kwargs))
    except Exception as exc:
        observation = f"Error: {exc}"
    post_tool_data = copy.deepcopy(env.data)
    action_count = len(env.actions)
    try:
        reward_result = env.calculate_reward()
        reward = float(getattr(reward_result, "reward", 0.0))
    finally:
        env.data = copy.deepcopy(post_tool_data)
        del env.actions[action_count:]
    return observation, reward, True, post_tool_data


class TauBenchDeltaToolBase(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        env = CURRENT_TAU_ENV.get()
        state = CURRENT_TAU_STATE.get()
        if env is None or state is None:
            raise RuntimeError("TauBench env/state missing from context.")
        if state.get("env_id") is not None and state.get("env_id") != id(env):
            raise RuntimeError(
                f"Context isolation violation: state.env_id={state.get('env_id')} current_env_id={id(env)}"
            )

        from tau_bench.types import Action

        before = copy.deepcopy(env.data)
        action = Action(name=self.name, kwargs=parameters)
        try:
            obs, inc_reward, is_done, after = execute_tau_tool_action(env, action)
        except Exception as exc:
            after = copy.deepcopy(env.data)
            obs = f"Error: {type(exc).__name__}: {exc}"
            record_tool_transition(state, self.name, parameters, obs, before, after)
            state["num_tool_calls"] += 1
            state["action_history"].append({
                "tool": self.name,
                "parameters": parameters,
                "observation_preview": obs[:500],
                "is_error": True,
                "instance_id": state.get("instance_id"),
                "trace_id": state.get("trace_id"),
            })
            return ToolResponse(text=obs), 0.0, {
                "inc_reward": 0.0,
                "done": False,
                "tool": self.name,
                "error": type(exc).__name__,
                "instance_id": state.get("instance_id"),
                "trace_id": state.get("trace_id"),
                "env_id": state.get("env_id"),
            }
        record_tool_transition(state, self.name, parameters, obs, before, after)

        state["total_reward"] += inc_reward
        state["num_tool_calls"] += 1
        state["done"] = state["done"] or is_done
        state["action_history"].append({
            "tool": self.name,
            "parameters": parameters,
            "observation_preview": obs[:500],
            "is_error": obs.startswith("Error:"),
            "instance_id": state.get("instance_id"),
            "trace_id": state.get("trace_id"),
        })
        return ToolResponse(text=obs), 0.0, {
            "inc_reward": inc_reward,
            "done": is_done,
            "tool": self.name,
            "instance_id": state.get("instance_id"),
            "trace_id": state.get("trace_id"),
            "env_id": state.get("env_id"),
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass


class TauBench_book_reservation_Tool(TauBenchDeltaToolBase): pass
class TauBench_calculate_Tool(TauBenchDeltaToolBase): pass
class TauBench_cancel_reservation_Tool(TauBenchDeltaToolBase): pass
class TauBench_get_reservation_details_Tool(TauBenchDeltaToolBase): pass
class TauBench_get_user_details_Tool(TauBenchDeltaToolBase): pass
class TauBench_list_all_airports_Tool(TauBenchDeltaToolBase): pass
class TauBench_search_direct_flight_Tool(TauBenchDeltaToolBase): pass
class TauBench_search_onestop_flight_Tool(TauBenchDeltaToolBase): pass
class TauBench_send_certificate_Tool(TauBenchDeltaToolBase): pass
class TauBench_think_Tool(TauBenchDeltaToolBase): pass
class TauBench_transfer_to_human_agents_Tool(TauBenchDeltaToolBase): pass
class TauBench_update_reservation_baggages_Tool(TauBenchDeltaToolBase): pass
class TauBench_update_reservation_flights_Tool(TauBenchDeltaToolBase): pass
class TauBench_update_reservation_passengers_Tool(TauBenchDeltaToolBase): pass
