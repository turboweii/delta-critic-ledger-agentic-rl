from __future__ import annotations

import contextvars
import uuid
from typing import Any, Optional

CURRENT_TAU_ENV: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("current_tau_env", default=None)
CURRENT_TAU_STATE: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("current_tau_state", default=None)


def make_initial_state(task_id: int, instance_id: str | None = None, env_id: int | None = None) -> dict:
    trace_id = str(uuid.uuid4())
    return {
        "instance_id": instance_id,
        "trace_id": trace_id,
        "task_id": int(task_id),
        "env_id": env_id,
        "state_id": None,
        "total_reward": 0.0,
        "num_tool_calls": 0,
        "num_user_turns": 0,
        "done": False,
        "action_history": [],
        "delta_steps": [],
        "ledger_steps": [],
        "delta_reward_sum": 0.0,
        "evidence_bonus_sum": 0.0,
    }
