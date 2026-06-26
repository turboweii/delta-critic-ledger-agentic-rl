from __future__ import annotations

from typing import Any

from delta_critic_ledger.long_horizon import LoopGuard, ProcessFeatureTracker, RewardEnvelope
from delta_critic_ledger.long_horizon.constraint_gate import (
    ConstraintGateConfig,
    apply_constraint_gate,
    evaluate_constraint,
)
from delta_critic_ledger.long_horizon.grounded_write_verifier import (
    count_ungrounded_writes,
    entity_keys_from_schema,
)


EMPTY_REASONING_TEXT = {"", "none", "null", "n/a"}

# Training step for the constraint-gate curriculum. The trainer calls
# set_training_step(step) each iteration (server-side hook); locally it stays 0,
# which (with warmup_steps=0 default) means the gate is always on for tests.
_TRAIN_STEP: dict = {"value": 0}


def set_training_step(step: int) -> None:
    """Hook for the trainer to publish the current global step (curriculum)."""
    _TRAIN_STEP["value"] = int(step)


def _curriculum_enforce(gate_cfg: ConstraintGateConfig) -> bool:
    """Whether the constraint gate is enforced this step.

    ``warmup_steps > 0`` relaxes the gate for the first ``warmup_steps`` steps
    (a naive early policy that violates everything can still learn); afterwards
    fully on. With ``warmup_steps <= 0`` (default) it is always on.
    """
    if not gate_cfg.enabled:
        return False
    if gate_cfg.warmup_steps <= 0:
        return True
    return _TRAIN_STEP["value"] >= gate_cfg.warmup_steps


def init_long_horizon_state(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    gate_cfg = config.get("constraint_gate") or {}
    known_gate = {f for f in ConstraintGateConfig.__dataclass_fields__}
    gate_kwargs = {k: v for k, v in dict(gate_cfg).items() if k in known_gate}
    # Schema-driven grounding (domain-agnostic): precompute entity keys per tool
    # from the OpenAI function schemas, so grounding keys off schema descriptions
    # ("such as 'ZFA04Y'"), not airline-specific naming conventions.
    tool_schemas = config.get("tool_schemas") or {}
    tool_entity_keys = {
        tool: entity_keys_from_schema(sch) for tool, sch in tool_schemas.items()
    }
    schema_provider = (lambda tool: tool_entity_keys.get(tool, set())) if tool_entity_keys else None
    return {
        "feature_tracker": ProcessFeatureTracker(),
        "loop_guard": LoopGuard.from_config(config.get("loop_guard")),
        "reward_envelope": RewardEnvelope.from_config(config.get("reward_envelope")),
        "constraint_gate": ConstraintGateConfig(**gate_kwargs),
        "schema_provider": schema_provider,
        "loop_stop_reason": "",
    }


def ensure_long_horizon_state(state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    if "long_horizon_state" not in state:
        state["long_horizon_state"] = init_long_horizon_state(config)
    return state["long_horizon_state"]


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
    del before, after
    lh_state = ensure_long_horizon_state(state)
    tracker: ProcessFeatureTracker = lh_state["feature_tracker"]
    tracker.record_tool(tool, parameters, observation, schema_violation=schema_violation)
    tool_name = (tool or "").lower()
    if "transfer_to_human" in tool_name and (tracker.features.read_tool_calls == 0 or tracker.features.tool_calls <= 2):
        tracker.mark_premature_transfer()
    if tool_name.endswith("think") or tool_name == "think":
        text = _extract_reasoning_text(parameters)
        if _is_empty_reasoning(text):
            tracker.record_empty_reasoning()
    decision = lh_state["loop_guard"].observe(tool, parameters, observation, tracker=tracker)
    if decision.should_stop:
        state["done"] = True
        lh_state["loop_stop_reason"] = decision.reason
    return {"loop_stop": decision.should_stop, "loop_stop_reason": decision.reason}


def _extract_reasoning_text(parameters: dict[str, Any]) -> str:
    for key in ("thought", "thinking", "reasoning", "content", "text", "summary"):
        value = parameters.get(key)
        if value is not None:
            return str(value)
    if not parameters:
        return ""
    return " ".join(str(value) for value in parameters.values() if value is not None)


def _is_empty_reasoning(text: str) -> bool:
    stripped = (text or "").strip().lower()
    return stripped in EMPTY_REASONING_TEXT or len(stripped) < 3


def record_final_response(state: dict[str, Any], content: str) -> None:
    """Record generic process features for a direct final response."""
    lh_state = ensure_long_horizon_state(state)
    tracker: ProcessFeatureTracker = lh_state["feature_tracker"]
    if _is_empty_reasoning(content):
        tracker.record_empty_reasoning()
    if tracker.features.read_tool_calls == 0 and tracker.features.tool_calls == 0:
        tracker.mark_premature_final()


def record_user_turn(state: dict[str, Any]) -> None:
    lh_state = ensure_long_horizon_state(state)
    lh_state["feature_tracker"].record_user_turn()


def mark_max_turn_hit(state: dict[str, Any]) -> None:
    lh_state = ensure_long_horizon_state(state)
    lh_state["feature_tracker"].mark_max_turn_hit()


def compute_long_horizon_components(state: dict[str, Any]) -> dict[str, Any]:
    lh_state = ensure_long_horizon_state(state)
    tracker: ProcessFeatureTracker = lh_state["feature_tracker"]
    envelope: RewardEnvelope = lh_state["reward_envelope"]
    features = tracker.to_dict()
    features["ungrounded_write_count"] = count_ungrounded_writes(
        state.get("action_history", []),
        schema_provider=lh_state.get("schema_provider"),
        initial_context=state.get("initial_user_response"),
    )
    components = envelope.compute(
        terminal_reward=float(state.get("total_reward", 0.0) or 0.0),
        process_score=0.0,
    )

    # Leg 1 — hard constraint gate: mechanically-checkable process errors
    # (placeholder / schema / loop / max-turn stall) reject a rollout, forcing
    # its reward into the failure band even if the env judged it a success.
    # This is a binary constraint, not a shaped reward, so it is not hackable.
    gate_cfg: ConstraintGateConfig = lh_state.get("constraint_gate") or ConstraintGateConfig()
    violation = evaluate_constraint(features, gate_cfg)
    raw_terminal = float(
        components.get("raw_terminal_reward", components.get("terminal_reward", 0.0))
    )
    gated = apply_constraint_gate(raw_terminal, violation, enforce=_curriculum_enforce(gate_cfg))
    if gated["rejected"]:
        components["score"] = 0.0
        components["success"] = False
    components["rejected"] = gated["rejected"]
    components["would_have_succeeded"] = gated["would_have_succeeded"]
    components["constraint_violation"] = violation.to_dict()

    components.update({
        "reward_mode": "long_horizon_terminal_envelope",
        "process_features": features,
        "loop_stop_reason": lh_state.get("loop_stop_reason", ""),
    })
    return components


def compute_long_horizon_reward(state: dict[str, Any]) -> float:
    return float(compute_long_horizon_components(state)["score"])

