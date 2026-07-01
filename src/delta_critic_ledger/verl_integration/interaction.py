from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from delta_critic_ledger.verl_integration.context import CURRENT_TAU_ENV, CURRENT_TAU_STATE, make_initial_state
from delta_critic_ledger.verl_integration.reward_state import (
    compute_tau_bench_components,
    ensure_tau_bench_state,
    init_tau_bench_state,
    mark_max_turn_hit,
    record_final_response,
    record_user_turn,
)
from delta_critic_ledger.tau_compat import create_tau_env
from delta_critic_ledger.training import b_ndsr

try:
    from verl.interactions.base import BaseInteraction
except Exception:  # pragma: no cover
    class BaseInteraction:  # type: ignore
        def __init__(self, config: dict):
            self.config = config

logger = logging.getLogger(__name__)


def _load_tool_schemas(path: str) -> dict:
    """Load tool schemas when a veRL tool config is supplied."""
    try:
        import yaml

        data = yaml.safe_load(open(path))
        schemas: dict = {}
        for t in (data or {}).get("tools", []):
            fn = (t.get("tool_schema") or {}).get("function") or {}
            name = fn.get("name")
            if name:
                schemas[name] = fn.get("parameters") or {}
        return schemas
    except Exception:
        return {}




def _replay_b_ndsr_actions(env: Any, state: dict[str, Any], replay_actions: list[dict[str, Any]]) -> None:
    if not replay_actions:
        return
    if not b_ndsr.is_enabled():
        raise RuntimeError("Received B-NDSR replay actions while B_NDSR_ENABLED is false.")

    from tau_bench.types import Action, RESPOND_ACTION_NAME

    for replay_action in replay_actions:
        if state.get("done", False):
            break
        kind = replay_action.get("kind")
        if kind == "tool":
            action = Action(
                name=replay_action["tool_name"],
                kwargs=dict(replay_action.get("parameters") or {}),
            )
            state["num_tool_calls"] += 1
        elif kind == "respond":
            action = Action(
                name=RESPOND_ACTION_NAME,
                kwargs={"content": replay_action.get("content", "")},
            )
            record_final_response(state, replay_action.get("content", "") or "")
            state["num_user_turns"] += 1
            record_user_turn(state)
        else:
            raise ValueError(f"Unknown B-NDSR replay action kind: {kind}")

        step_res = env.step(action)
        state["total_reward"] += float(getattr(step_res, "reward", 0.0))
        if bool(getattr(step_res, "done", False)):
            state["done"] = True


class TauBenchInteraction(BaseInteraction):
    """veRL interaction for tau-bench with ordinary terminal outcome reward."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.env_name = config.get("env_name", "airline")
        self.user_strategy = config.get("user_strategy", "llm")
        self.user_model = config.get("user_model", "Qwen/Qwen2.5-7B-Instruct")
        self.user_provider = config.get("user_provider", "openai")
        self.user_base_url = config.get("user_base_url", "http://localhost:8001/v1")
        self.task_split = config.get("task_split", "test")
        self.max_turns = int(config.get("max_turns", 24))
        self.interaction_config = dict(config.get("interaction", {}) or {})
        self.trace_dir = self.interaction_config.get("trace_dir")
        tool_config_path = config.get("tool_config_path")
        if tool_config_path:
            self.interaction_config["tool_schemas"] = _load_tool_schemas(tool_config_path)
        self._instance_dict: dict[str, dict] = {}

    async def start_interaction(self, instance_id: Optional[str] = None, task_id: int = 0, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        from tau_bench.envs import get_env

        env = create_tau_env(
            get_env,
            env_name=self.env_name,
            user_strategy=self.user_strategy,
            user_model=self.user_model,
            user_provider=self.user_provider,
            user_api_base=self.user_base_url,
            task_split=self.task_split,
            task_index=int(task_id),
        )
        reset_result = env.reset(task_index=int(task_id))
        initial_user_response = str(getattr(reset_result, "observation", reset_result))
        state = make_initial_state(int(task_id), instance_id=instance_id, env_id=id(env))
        state["state_id"] = id(state)
        state["tau_bench_state"] = init_tau_bench_state(self.interaction_config)
        state["initial_user_response"] = initial_user_response
        replay_actions = kwargs.get("b_ndsr_replay_actions") or []
        if replay_actions:
            _replay_b_ndsr_actions(env, state, replay_actions)
        CURRENT_TAU_ENV.set(env)
        CURRENT_TAU_STATE.set(state)
        self._instance_dict[instance_id] = {
            "env": env,
            "state": state,
            "task_id": int(task_id),
            "env_id": id(env),
            "initial_user_response": initial_user_response,
        }
        return instance_id

    async def get_initial_response(self, instance_id: str) -> str:
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            raise RuntimeError(f"Unknown instance_id={instance_id}")
        return str(entry["initial_user_response"])

    def is_done(self, instance_id: str) -> bool:
        entry = self._instance_dict.get(instance_id)
        return bool(entry and entry["state"].get("done"))

    def get_controller_state(self, instance_id: str) -> dict[str, Any] | None:
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            return None
        return entry["state"]

    async def generate_response(self, instance_id: str, messages: list[dict[str, Any]], **kwargs) -> tuple[bool, str, float, dict[str, Any]]:
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            raise RuntimeError(f"Unknown instance_id={instance_id}; start_interaction was not called or was finalized.")
        env = entry["env"]
        state = entry["state"]
        if state.get("instance_id") != instance_id or state.get("env_id") != id(env):
            raise RuntimeError(
                f"Interaction state mismatch: instance={instance_id}, "
                f"state.instance={state.get('instance_id')}, env_id={id(env)}, state.env_id={state.get('env_id')}"
            )
        CURRENT_TAU_ENV.set(env)
        CURRENT_TAU_STATE.set(state)
        ensure_tau_bench_state(state, self.interaction_config)

        assistant_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_content = msg.get("content", "") or ""
                break

        from tau_bench.types import Action, RESPOND_ACTION_NAME

        record_final_response(state, assistant_content)
        try:
            step_res = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": assistant_content}))
        except Exception as exc:
            state["done"] = True
            components = compute_tau_bench_components(state)
            final_score = float(components["score"])
            return True, "", final_score, self._score_info(state, components, error=f"{type(exc).__name__}: {exc}")

        inc_reward = float(getattr(step_res, "reward", 0.0))
        is_done = bool(getattr(step_res, "done", False))
        state["total_reward"] += inc_reward
        state["num_user_turns"] += 1
        record_user_turn(state)
        total_turns = state["num_user_turns"] + state["num_tool_calls"]

        if total_turns >= self.max_turns:
            mark_max_turn_hit(state)

        if is_done or state.get("done") or total_turns >= self.max_turns:
            state["done"] = True
            components = compute_tau_bench_components(state)
            final_score = float(components["score"])
            return True, "", final_score, self._score_info(state, components)

        return False, str(getattr(step_res, "observation", "")), 0.0, {
            "task_id": state["task_id"],
            "turn": total_turns,
            "instance_id": state.get("instance_id"),
            "trace_id": state.get("trace_id"),
        }

    async def calculate_score(self, instance_id: str, **kwargs) -> dict:
        entry = self._instance_dict.get(instance_id)
        if not entry:
            return {"score": 0.0, "outcome_score": 0.0}
        state = entry["state"]
        components = compute_tau_bench_components(state)
        return {
            "score": components["score"],
            "outcome_score": components["terminal_reward"],
            "dense_reward": components["dense_reward"],
            "reward_components": components,
            "process_features": components.get("process_features", {}),
            "loop_stop_reason": components.get("loop_stop_reason", ""),
            "instance_id": state.get("instance_id"),
            "trace_id": state.get("trace_id"),
        }

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        entry = self._instance_dict.pop(instance_id, None)
        if entry is None or not self.trace_dir:
            return
        state = entry["state"]
        trace_id = state.get("trace_id") or instance_id
        out_dir = Path(self.trace_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        components = compute_tau_bench_components(state)
        payload = {
            "instance_id": instance_id,
            "trace_id": trace_id,
            "task_id": state.get("task_id"),
            "done": state.get("done"),
            "outcome_reward": components["terminal_reward"],
            "combined_reward": components["score"],
            "dense_reward": components["dense_reward"],
            "reward_components": components,
            "process_features": components.get("process_features", {}),
            "loop_stop_reason": components.get("loop_stop_reason", ""),
            "num_tool_calls": state.get("num_tool_calls", 0),
            "num_user_turns": state.get("num_user_turns", 0),
            "trace_truncated": state.get("trace_truncated", False),
            "action_history": state.get("action_history", []),
        }
        (out_dir / f"{trace_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _score_info(self, state: dict[str, Any], components: dict[str, Any], error: str | None = None) -> dict[str, Any]:
        info = {
            "reward_mode": components["reward_mode"],
            "reward_components": components,
            "outcome_reward": components["terminal_reward"],
            "dense_reward": components["dense_reward"],
            "process_features": components.get("process_features", {}),
            "loop_stop_reason": components.get("loop_stop_reason", ""),
            "num_tool_calls": state["num_tool_calls"],
            "task_id": state["task_id"],
            "instance_id": state.get("instance_id"),
            "trace_id": state.get("trace_id"),
            "env_id": state.get("env_id"),
            "state_id": state.get("state_id"),
        }
        if error:
            info["error"] = error
        return info


# Backward-compatible class name for existing veRL configs.
DeltaTauBenchInteraction = TauBenchInteraction



