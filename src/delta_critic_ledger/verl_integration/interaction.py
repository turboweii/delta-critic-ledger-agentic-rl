from __future__ import annotations

import logging
import json
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from delta_critic_ledger.adaptive_control import summarize_rollout_state
from delta_critic_ledger.verl_integration.context import CURRENT_TAU_ENV, CURRENT_TAU_STATE, make_initial_state
from delta_critic_ledger.verl_integration.reward_state import compute_delta_ledger_components, init_delta_reward_state
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
        self.delta_clip_min = delta_cfg.get("delta_clip_min", -1.0)
        self.delta_clip_max = delta_cfg.get("delta_clip_max", 1.0)
        self.evidence_clip_min = delta_cfg.get("evidence_clip_min", -2.0)
        self.evidence_clip_max = delta_cfg.get("evidence_clip_max", 1.0)
        self.score_clip_min = delta_cfg.get("score_clip_min", -0.2)
        self.score_clip_max = delta_cfg.get("score_clip_max", 1.4)
        self.trace_dir = delta_cfg.get("trace_dir")
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
            delta_clip_min=self.delta_clip_min,
            delta_clip_max=self.delta_clip_max,
            evidence_clip_min=self.evidence_clip_min,
            evidence_clip_max=self.evidence_clip_max,
            score_clip_min=self.score_clip_min,
            score_clip_max=self.score_clip_max,
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
            components = compute_delta_ledger_components(state)
            final_score = float(components["score"])
            return True, "", final_score, {
                "reward_mode": "delta_ledger",
                "reward_components": components,
                "outcome_reward": components["outcome_reward"],
                "delta_reward_sum": components["delta_reward_sum"],
                "raw_delta_reward_sum": components["raw_delta_reward_sum"],
                "evidence_bonus_sum": components["evidence_bonus_sum"],
                "raw_evidence_bonus_sum": components["raw_evidence_bonus_sum"],
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
            components = compute_delta_ledger_components(state)
            final_score = float(components["score"])
            return True, "", final_score, {
                "reward_mode": "delta_ledger",
                "reward_components": components,
                "outcome_reward": components["outcome_reward"],
                "delta_reward_sum": components["delta_reward_sum"],
                "raw_delta_reward_sum": components["raw_delta_reward_sum"],
                "evidence_bonus_sum": components["evidence_bonus_sum"],
                "raw_evidence_bonus_sum": components["raw_evidence_bonus_sum"],
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
        components = compute_delta_ledger_components(state)
        return {
            "score": components["score"],
            "outcome_score": components["outcome_reward"],
            "delta_score": components["delta_reward_sum"],
            "raw_delta_score": components["raw_delta_reward_sum"],
            "evidence_score": components["evidence_bonus_sum"],
            "raw_evidence_score": components["raw_evidence_bonus_sum"],
            "dense_reward": components["dense_reward"],
            "reward_components": components,
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

        def encode(value: Any) -> Any:
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, dict):
                return {str(key): encode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [encode(item) for item in value]
            return value

        components = compute_delta_ledger_components(state)
        payload = {
            "instance_id": instance_id,
            "trace_id": trace_id,
            "task_id": state.get("task_id"),
            "done": state.get("done"),
            "outcome_reward": components["outcome_reward"],
            "combined_reward": components["score"],
            "dense_reward": components["dense_reward"],
            "delta_reward_sum": components["delta_reward_sum"],
            "raw_delta_reward_sum": components["raw_delta_reward_sum"],
            "evidence_bonus_sum": components["evidence_bonus_sum"],
            "raw_evidence_bonus_sum": components["raw_evidence_bonus_sum"],
            "reward_components": encode(components),
            "num_tool_calls": state.get("num_tool_calls", 0),
            "num_user_turns": state.get("num_user_turns", 0),
            "trace_truncated": state.get("trace_truncated", False),
            "action_history": encode(state.get("action_history", [])),
            "delta_steps": encode(state.get("delta_steps", [])),
            "ledger_steps": encode(state.get("ledger_steps", [])),
            "adaptive_progress": encode(summarize_rollout_state(state)),
        }
        (out_dir / f"{trace_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
