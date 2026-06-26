"""Leg 1 core — grounded-write verifier.

A write tool call is *grounded* if its parameter values appeared in prior tool
observations (the agent read them from the environment rather than invented
them). An ungrounded write — booking with a reservation_id no tool ever
returned, paying with a payment_id pulled from thin air — is the canonical
agentic failure mode, it is mechanically checkable (string match of param
values against prior observations), and it is consumed here as a HARD
CONSTRAINT (see constraint_gate.py), never as a shaped reward — so it is not a
hackable surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

_PLACEHOLDER_TOKENS = {
    "", "none", "null", "n/a", "todo", "tbd", "unknown",
    "placeholder", "xxx", "example", "test", "foo", "bar",
}


@dataclass(frozen=True)
class GroundingResult:
    grounded: bool
    ungrounded_keys: tuple[str, ...] = ()
    checked_keys: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        return "ungrounded:" + ",".join(self.ungrounded_keys) if self.ungrounded_keys else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "ungrounded_keys": list(self.ungrounded_keys),
            "checked_keys": list(self.checked_keys),
        }


def _flatten_values(node: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    """Yield (key_path, string_value) for every scalar in a nested params tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield from _flatten_values(v, key)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _flatten_values(v, key)
    elif isinstance(node, bool):
        yield (prefix, "true" if node else "false")
    elif isinstance(node, (int, float)):
        yield (prefix, str(node))
    elif isinstance(node, str):
        yield (prefix, node)


def _is_checkable(value: str, min_value_len: int) -> bool:
    v = (value or "").strip()
    if len(v) < min_value_len:
        return False
    if v.lower() in _PLACEHOLDER_TOKENS:
        return False
    return True


def _is_entity_key(key: str) -> bool:
    """Heuristic: is this parameter an entity reference that must be grounded?

    ID / number / code fields (reservation_id, payment_id, user_id,
    flight_number) must come from the environment; choice fields (cabin,
    flight_type, origin, insurance) are the agent's own selection and are NOT
    checked — requiring them to be "read" would be a false positive.
    """
    leaf = key.lower().split(".")[-1].split("[")[-1].rstrip("]")
    return (
        leaf.endswith("_id")
        or leaf.endswith("_number")
        or leaf.endswith("_code")
        or "number" in leaf
    )


def entity_keys_from_schema(parameters_schema: dict[str, Any]) -> set[str]:
    """Extract entity-reference param leaf names from an OpenAI function schema.

    A param is an entity reference (must be grounded in prior observations) when
    its description references a concrete environment-issued value — "stored in
    user profile", an explicit identifier, or an id/number. Choice params (enum
    like cabin, or IATA code) do NOT match. Recurses into nested objects and
    array items so entity params inside e.g. ``flights[].flight_number`` or
    ``payment_methods[].payment_id`` are also caught. Returns leaf names.
    """
    entity: set[str] = set()

    def visit(node: dict[str, Any]) -> None:
        props = (node or {}).get("properties", {}) or {}
        for name, spec in props.items():
            spec = spec or {}
            desc = str(spec.get("description", "")).lower()
            is_entity = (
                "identifier" in desc
                or "stored in" in desc
                or ("id" in desc and spec.get("type") == "string" and "enum" not in spec)
                or ("number" in desc and spec.get("type") == "string" and "enum" not in spec)
            )
            if is_entity:
                entity.add(name)
            if spec.get("type") == "object":
                visit(spec)
            elif spec.get("type") == "array":
                items = spec.get("items") or {}
                if items.get("type") == "object":
                    visit(items)

    visit(parameters_schema)
    return entity


def is_value_in_observations(value: str, observations: Iterable[str]) -> bool:
    needle = value.strip()
    if not needle:
        return True
    return any(needle in str(obs) for obs in observations)


def evaluate_grounded_write(
    parameters: dict[str, Any],
    prior_observations: Iterable[str],
    *,
    min_value_len: int = 3,
    entity_only: bool = True,
    entity_keys: set[str] | None = None,
) -> GroundingResult:
    """Check entity-reference param values appear in prior observations.

    Which params are "entity references" is decided one of two ways:
      * ``entity_keys`` (preferred, schema-driven): only leaf params in this set
        are checked. Build it with :func:`entity_keys_from_schema` so grounding
        is domain-agnostic (any tool schema, not just airline naming).
      * fallback (``entity_only=True``): the ``_is_entity_key`` naming heuristic.

    Args:
        parameters: write tool kwargs (nested dict/list allowed).
        prior_observations: observation texts seen BEFORE this write.
        min_value_len: values shorter than this are not checked.
        entity_only: if True (and no ``entity_keys``), only check entity fields by name.
        entity_keys: if given, only check these leaf param names (schema-driven).

    Returns:
        GroundingResult indicating which param values were invented.
    """
    observations = [str(o) for o in prior_observations]
    checked: list[str] = []
    ungrounded: list[str] = []
    for key, raw_value in _flatten_values(parameters):
        if not _is_checkable(raw_value, min_value_len):
            continue
        leaf = key.split(".")[-1].split("[")[-1].rstrip("]")
        if entity_keys is not None:
            if leaf not in entity_keys:
                continue
        elif entity_only and not _is_entity_key(key):
            continue
        checked.append(key)
        if not is_value_in_observations(raw_value, observations):
            ungrounded.append(key)
    return GroundingResult(
        grounded=len(ungrounded) == 0,
        ungrounded_keys=tuple(ungrounded),
        checked_keys=tuple(checked),
    )


def count_ungrounded_writes(
    action_history: list[dict[str, Any]],
    *,
    is_write_fn=None,
    min_value_len: int = 3,
    schema_provider=None,
    initial_context: str | None = None,
) -> int:
    """Count write actions whose params are not grounded in prior observations.

    ``action_history`` items carry ``tool``, ``parameters`` and an observation
    field (``observation_preview`` or ``observation``) — the shape produced by
    the rollout integration. The observation field MUST be stored in full (not
    truncated): grounding does substring match of entity IDs against it, and
    truncating long observations (e.g. ``get_user_details`` returns several KB)
    drops IDs near the end and causes false ungrounded verdicts.

    ``initial_context`` is prepended to the prior context (e.g. the initial user
    message / task instruction). IDs that the task or user provides directly
    (like ``user_id``, which the agent is told rather than reads from a tool)
    must count as grounded — otherwise every write that carries a task-provided
    user_id would be falsely rejected.

    Args:
        schema_provider: optional ``tool_name -> set(entity_param_names)``. If
            given, grounding is schema-driven (domain-agnostic); otherwise falls
            back to the ``_is_entity_key`` naming heuristic.
        initial_context: optional text (initial user message / task instruction)
            prepended to prior observations so task/user-provided IDs are grounded.
    """
    from .process_features import is_write_tool

    write_fn = is_write_fn or is_write_tool
    count = 0
    prior_obs: list[str] = [str(initial_context)] if initial_context else []
    for action in action_history or []:
        tool = str(action.get("tool", ""))
        params = action.get("parameters") or {}
        if write_fn(tool) and params:
            entity_keys = schema_provider(tool) if schema_provider else None
            result = evaluate_grounded_write(
                params, prior_obs, min_value_len=min_value_len, entity_keys=entity_keys,
            )
            if not result.grounded:
                count += 1
        obs = action.get("observation_preview") or action.get("observation") or ""
        if obs:
            prior_obs.append(str(obs))
    return count
