from __future__ import annotations

from typing import Any


def init_tau_bench_state(config: dict[str, Any] | None = None) -> dict[str, Any]:
    del config
    return {}


def ensure_tau_bench_state(state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    del config
    return state.setdefault("tau_bench_state", {})


def record_tool_transition(
    state: dict[str, Any],
    tool: str,
    parameters: dict[str, Any],
    observation: str,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    *,
    schema_violation: bool = False,
) -> dict[str, Any]:
    del state, tool, parameters, observation, before, after, schema_violation
    return {"loop_stop": False, "loop_stop_reason": ""}


def record_final_response(state: dict[str, Any], content: str) -> None:
    del state, content


def record_user_turn(state: dict[str, Any]) -> None:
    del state


def mark_max_turn_hit(state: dict[str, Any]) -> None:
    state["done"] = True


def compute_tau_bench_components(state: dict[str, Any]) -> dict[str, Any]:
    score = float(state.get("total_reward", 0.0) or 0.0)
    return {
        "score": score,
        "terminal_reward": score,
        "dense_reward": 0.0,
        "reward_mode": "terminal_outcome",
        "process_features": {},
        "loop_stop_reason": "",
        "success": score > 0.0,
    }


def compute_tau_bench_reward(state: dict[str, Any]) -> float:
    return float(compute_tau_bench_components(state)["score"])

