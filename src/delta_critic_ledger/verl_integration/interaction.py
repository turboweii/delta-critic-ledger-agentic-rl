from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from delta_critic_ledger.verl_integration.context import CURRENT_TAU_ENV, CURRENT_TAU_STATE, make_initial_state
from delta_critic_ledger.verl_integration.reward_state import compute_delta_ledger_reward, init_delta_reward_state
from delta_critic_ledger.tau_compat import create_tau_env

try:
    from verl.interactions.base import BaseInteraction
except Exception:  # pragma: no cover
    class BaseInteraction:  # type: ignore
        def __init__(self, config: dict):
            self.config = config

logger = logging.getLogger(__name__)


class DeltaTauBenchInteraction(BaseInteraction):
    """veRL interaction that computes terminal + Delta-Critic + Evidence Ledger reward."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.env_name = config.get("env_name", "airline")
        self.user_strategy = config.get("user_strategy", "llm")
        self.user_model = config.get("user_model", "Qwen/Qwen2.5-7B-Instruct")
        self.user_provider = config.get("user_provider", "openai")
        self.user_base_url = config.get("user_base_url", "http://localhost:8001/v1")
        self.task_split = config.get("task_split", "test")
        self.max_turns = int(config.get("max_turns", 24))
        delta_cfg = config.get("delta_critic", {})
        self.beta_delta = float(delta_cfg.get("beta_delta", 0.3))
        self.beta_evidence = float(delta_cfg.get("beta_evidence", 0.1))
        self.include_paths = list(delta_cfg.get("include_paths", []))
        self.exclude_paths = list(delta_cfg.get("exclude_paths", []))
        self.max_trace_steps = int(delta_cfg.get("max_trace_steps", 128))
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
        state["max_trace_steps"] = self.max_trace_steps
        state["delta_reward_state"] = init_delta_reward_state(
            env,
            self.beta_delta,
            self.beta_evidence,
            include_paths=self.include_paths,
            exclude_paths=self.exclude_paths,
        )
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

        assistant_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_content = msg.get("content", "") or ""
                break

        from tau_bench.types import Action, RESPOND_ACTION_NAME

        try:
            step_res = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": assistant_content}))
        except Exception as exc:
            state["done"] = True
            final_score = compute_delta_ledger_reward(state)
            return True, "", final_score, {
                "reward_mode": "delta_ledger",
                "outcome_reward": 1.0 if state["total_reward"] >= 1.0 else 0.0,
                "delta_reward_sum": state.get("delta_reward_sum", 0.0),
                "evidence_bonus_sum": state.get("evidence_bonus_sum", 0.0),
                "num_tool_calls": state["num_tool_calls"],
                "task_id": state["task_id"],
                "instance_id": state.get("instance_id"),
                "trace_id": state.get("trace_id"),
                "env_id": state.get("env_id"),
                "state_id": state.get("state_id"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        inc_reward = float(getattr(step_res, "reward", 0.0))
        is_done = bool(getattr(step_res, "done", False))
        state["total_reward"] += inc_reward
        state["num_user_turns"] += 1
        total_turns = state["num_user_turns"] + state["num_tool_calls"]

        if is_done or total_turns >= self.max_turns:
            state["done"] = True
            final_score = compute_delta_ledger_reward(state)
            return True, "", final_score, {
                "reward_mode": "delta_ledger",
                "outcome_reward": 1.0 if state["total_reward"] >= 1.0 else 0.0,
                "delta_reward_sum": state.get("delta_reward_sum", 0.0),
                "evidence_bonus_sum": state.get("evidence_bonus_sum", 0.0),
                "num_tool_calls": state["num_tool_calls"],
                "task_id": state["task_id"],
                "instance_id": state.get("instance_id"),
                "trace_id": state.get("trace_id"),
                "env_id": state.get("env_id"),
                "state_id": state.get("state_id"),
            }

        return False, str(getattr(step_res, "observation", "")), 0.0, {
            "task_id": state["task_id"],
            "turn": total_turns,
            "instance_id": state.get("instance_id"),
            "trace_id": state.get("trace_id"),
        }

    async def calculate_score(self, instance_id: str, **kwargs) -> dict:
        entry = self._instance_dict.get(instance_id)
        if not entry:
            return {"score": 0.0, "outcome_score": 0.0, "delta_score": 0.0, "evidence_score": 0.0}
        state = entry["state"]
        outcome = 1.0 if state.get("total_reward", 0.0) >= 1.0 else 0.0
        score = compute_delta_ledger_reward(state)
        return {
            "score": score,
            "outcome_score": outcome,
            "delta_score": state.get("delta_reward_sum", 0.0),
            "evidence_score": state.get("evidence_bonus_sum", 0.0),
            "instance_id": state.get("instance_id"),
            "trace_id": state.get("trace_id"),
        }

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
