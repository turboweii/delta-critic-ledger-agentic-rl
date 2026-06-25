from __future__ import annotations

import copy
from typing import Any

from delta_critic_ledger import Action
from delta_critic_ledger.delta_critic import DeltaCritic
from delta_critic_ledger.evidence_ledger import EvidenceLedger


def _clip(value: float, low: float | None = None, high: float | None = None) -> float:
    if low is not None:
        value = max(float(low), value)
    if high is not None:
        value = min(float(high), value)
    return value


def tau_action_to_local(action: Any) -> Action:
    kwargs = getattr(action, "kwargs", None)
    if kwargs is None and hasattr(action, "arguments"):
        kwargs = getattr(action, "arguments")
    if kwargs is None and isinstance(action, dict):
        kwargs = action.get("kwargs") or action.get("arguments") or {}
        name = action.get("name")
    else:
        name = getattr(action, "name")
    return Action(name=name, kwargs=dict(kwargs or {}))


def execute_tau_tool_on_data(env: Any, data: dict[str, Any], action: Action) -> str:
    if action.name not in env.tools_map:
        return f"Error: unknown target action {action.name}"
    try:
        return str(env.tools_map[action.name].invoke(data=data, **action.kwargs))
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


def init_delta_reward_state(
    env: Any,
    beta_delta: float,
    beta_evidence: float,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    delta_clip_min: float | None = -1.0,
    delta_clip_max: float | None = 1.0,
    evidence_clip_min: float | None = -2.0,
    evidence_clip_max: float | None = 1.0,
    score_clip_min: float | None = -0.2,
    score_clip_max: float | None = 1.4,
    success_floor: float | None = 1.0,
) -> dict:
    initial_data = copy.deepcopy(env.data)
    target_actions: list[Action] = []
    for raw_action in env.task.actions:
        action = tau_action_to_local(raw_action)
        if action.name != "respond":
            target_actions.append(action)

    def executor(data: dict[str, Any], action: Action) -> str:
        return execute_tau_tool_on_data(env, data, action)

    critic = DeltaCritic.from_target_actions(
        initial_data,
        target_actions,
        executor,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
    )
    ledger = EvidenceLedger(seed_entities={"user_id": [getattr(env.task, "user_id", "")]})
    return {
        "critic": critic,
        "ledger": ledger,
        "beta_delta": float(beta_delta),
        "beta_evidence": float(beta_evidence),
        "goal_fields": critic.goal_field_names(),
        "include_paths": include_paths or [],
        "exclude_paths": exclude_paths or [],
        "delta_clip_min": delta_clip_min,
        "delta_clip_max": delta_clip_max,
        "evidence_clip_min": evidence_clip_min,
        "evidence_clip_max": evidence_clip_max,
        "score_clip_min": score_clip_min,
        "score_clip_max": score_clip_max,
        "success_floor": success_floor,
    }


def record_tool_transition(state: dict, tool: str, parameters: dict, observation: str, before: dict, after: dict) -> None:
    reward_state = state.get("delta_reward_state")
    if not reward_state:
        return
    step_idx = int(state.get("num_tool_calls", 0))
    ledger = reward_state["ledger"]
    critic = reward_state["critic"]

    ledger_step = ledger.check_write(step_idx, tool, parameters)
    delta_step = critic.score_step(step_idx, tool, before, after)
    ledger.update_from_tool(step_idx, tool, parameters, observation)
    ledger_step.known_entities = ledger.snapshot()

    evidence_bonus = ledger.evidence_bonus(ledger_step)
    if evidence_bonus > 0.0:
        rewarded_fields = set(state.setdefault("evidence_rewarded_goal_fields", []))
        new_goal_fields = set(delta_step.changed_goal_fields) - rewarded_fields
        if delta_step.delta_reward <= 0.0 or not new_goal_fields:
            # No-ops and regress/recover loops must not farm positive evidence reward.
            evidence_bonus = 0.0
        else:
            rewarded_fields.update(new_goal_fields)
            state["evidence_rewarded_goal_fields"] = sorted(rewarded_fields)
    max_trace_steps = int(state.get("max_trace_steps", 128))
    if len(state["delta_steps"]) < max_trace_steps:
        state["delta_steps"].append(delta_step)
        state["ledger_steps"].append(ledger_step)
    else:
        state["trace_truncated"] = True
    state["delta_reward_sum"] += delta_step.delta_reward
    state["evidence_bonus_sum"] += evidence_bonus


def compute_delta_ledger_components(state: dict) -> dict[str, float | bool]:
    outcome = 1.0 if state.get("total_reward", 0.0) >= 1.0 else 0.0
    reward_state = state.get("delta_reward_state") or {}
    beta_delta = float(reward_state.get("beta_delta", 0.3))
    beta_evidence = float(reward_state.get("beta_evidence", 0.1))
    raw_delta = float(state.get("delta_reward_sum", 0.0) or 0.0)
    raw_evidence = float(state.get("evidence_bonus_sum", 0.0) or 0.0)
    clipped_delta = _clip(
        raw_delta,
        reward_state.get("delta_clip_min", -1.0),
        reward_state.get("delta_clip_max", 1.0),
    )
    clipped_evidence = _clip(
        raw_evidence,
        reward_state.get("evidence_clip_min", -2.0),
        reward_state.get("evidence_clip_max", 1.0),
    )
    dense_reward = beta_delta * clipped_delta + beta_evidence * clipped_evidence
    unclipped_score = outcome + dense_reward
    score = _clip(
        unclipped_score,
        reward_state.get("score_clip_min", -0.2),
        reward_state.get("score_clip_max", 1.4),
    )
    success_floor = reward_state.get("success_floor", 1.0)
    if outcome >= 1.0 and success_floor is not None:
        score = max(score, float(success_floor))
    return {
        "score": score,
        "outcome_reward": outcome,
        "dense_reward": dense_reward,
        "raw_delta_reward_sum": raw_delta,
        "delta_reward_sum": clipped_delta,
        "raw_evidence_bonus_sum": raw_evidence,
        "evidence_bonus_sum": clipped_evidence,
        "beta_delta": beta_delta,
        "beta_evidence": beta_evidence,
        "unclipped_score": unclipped_score,
        "score_was_clipped": score != unclipped_score,
    }


def compute_delta_ledger_reward(state: dict) -> float:
    return float(compute_delta_ledger_components(state)["score"])
