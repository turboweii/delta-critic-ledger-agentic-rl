from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .schemas import LedgerStep

READ_TOOLS = {
    "get_user_details",
    "get_reservation_details",
    "search_direct_flight",
    "search_onestop_flight",
    "list_all_airports",
    "calculate",
}
WRITE_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "update_reservation_passengers",
    "send_certificate",
}
WRITE_REQUIRED_EVIDENCE = {
    "book_reservation": {"user_id", "flight_number", "date", "payment_id"},
    "cancel_reservation": {"reservation_id"},
    "update_reservation_flights": {"reservation_id", "flight_number", "date", "payment_id"},
    "update_reservation_baggages": {"reservation_id", "payment_id"},
    "update_reservation_passengers": {"reservation_id"},
    "send_certificate": {"user_id"},
}
ENTITY_REGEX = {
    "reservation_id": re.compile(r"\b[A-Z0-9]{6}\b"),
    "payment_id": re.compile(r"\b(?:credit_card|gift_card|certificate)_[0-9]+\b"),
    "user_id": re.compile(r"\b(?!credit_card_|gift_card_|certificate_)[a-z]+_[a-z]+_[0-9]+\b"),
    "flight_number": re.compile(r"\b[A-Z]{3}[0-9]{3}\b"),
    "date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
}


@dataclass
class Evidence:
    entity_type: str
    value: str
    source_tool: str
    source_step: int
    source_path: str


def path_to_str(path: tuple[Any, ...]) -> str:
    parts = []
    for item in path:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            if parts:
                parts.append(".")
            parts.append(str(item))
    return "".join(parts)


def walk_json(value: Any, prefix: tuple[Any, ...] = ()) -> Iterable[tuple[tuple[Any, ...], Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk_json(item, prefix + (key,))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from walk_json(item, prefix + (idx,))
    else:
        yield prefix, value


def infer_entity_type(field_name: str, value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    lower = field_name.lower()
    if ENTITY_REGEX["payment_id"].fullmatch(value):
        return "payment_id"
    if "reservation_id" in lower:
        return "reservation_id"
    if "user_id" in lower:
        return "user_id"
    if "payment_id" in lower:
        return "payment_id"
    if "flight_number" in lower:
        return "flight_number"
    if "date" in lower and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return "date"
    for entity_type, pattern in ENTITY_REGEX.items():
        if pattern.fullmatch(value):
            return entity_type
    return None


def parse_observation(observation: str) -> Any:
    try:
        return json.loads(observation)
    except Exception:
        return observation


def values_by_entity_type(params: Any) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path, leaf in walk_json(params):
        if not path:
            continue
        entity_type = infer_entity_type(str(path[-1]), leaf)
        if entity_type:
            out.setdefault(entity_type, set()).add(str(leaf))
    return out


def extract_entities_from_json(value: Any) -> list[tuple[str, str, str]]:
    found = []
    for path, leaf in walk_json(value):
        if not path:
            continue
        entity_type = infer_entity_type(str(path[-1]), leaf)
        if entity_type:
            found.append((entity_type, str(leaf), path_to_str(path)))
    return found


def extract_entities_from_text(text: str) -> list[tuple[str, str, str]]:
    found = []
    for entity_type, pattern in ENTITY_REGEX.items():
        for match in pattern.findall(text or ""):
            found.append((entity_type, match, "$text"))
    return found


class EvidenceLedger:
    """Tracks tool-observed entities and labels whether writes are grounded."""

    def __init__(self, seed_entities: Optional[dict[str, Iterable[str]]] = None):
        self._evidence: dict[str, dict[str, list[Evidence]]] = {}
        for entity_type, values in (seed_entities or {}).items():
            for value in values:
                self.add(entity_type, str(value), "task_seed", -1, "$seed")

    def add(self, entity_type: str, value: str, source_tool: str, source_step: int, source_path: str) -> None:
        evidence = Evidence(entity_type, value, source_tool, source_step, source_path)
        self._evidence.setdefault(entity_type, {}).setdefault(value, []).append(evidence)

    def update_from_tool(self, step_idx: int, tool: str, parameters: dict[str, Any], observation: str) -> None:
        if tool in READ_TOOLS:
            for entity_type, values in values_by_entity_type(parameters).items():
                for value in values:
                    self.add(entity_type, value, tool, step_idx, "$parameters")

        parsed = parse_observation(observation)
        entities = extract_entities_from_json(parsed)
        if isinstance(parsed, str):
            entities.extend(extract_entities_from_text(parsed))
        for entity_type, value, path in entities:
            self.add(entity_type, value, tool, step_idx, path)

    def check_write(self, step_idx: int, tool: str, parameters: dict[str, Any]) -> LedgerStep:
        if tool not in WRITE_TOOLS:
            return LedgerStep(step_idx=step_idx, tool=tool, write_grounding_status="not_write", known_entities=self.snapshot())

        required = WRITE_REQUIRED_EVIDENCE.get(tool, set())
        values = values_by_entity_type(parameters)
        missing_fields, conflicting_fields = [], []
        used: dict[str, list[str]] = {}

        for entity_type in sorted(required):
            param_values = values.get(entity_type, set())
            if not param_values:
                missing_fields.append(entity_type)
                continue
            known_values = set(self._evidence.get(entity_type, {}))
            grounded = sorted(param_values & known_values)
            if grounded:
                used[entity_type] = grounded
            elif known_values:
                conflicting_fields.append(entity_type)
            else:
                missing_fields.append(entity_type)

        if missing_fields:
            status = "ungrounded_write"
        elif conflicting_fields:
            status = "conflicting_write"
        else:
            status = "evidence_grounded_write"

        return LedgerStep(
            step_idx=step_idx,
            tool=tool,
            write_grounding_status=status,
            missing_evidence_fields=missing_fields,
            conflicting_fields=conflicting_fields,
            used_evidence=used,
            known_entities=self.snapshot(),
        )

    def evidence_bonus(self, step: LedgerStep) -> float:
        if step.write_grounding_status == "evidence_grounded_write":
            return 1.0
        if step.write_grounding_status == "conflicting_write":
            return -1.0
        if step.write_grounding_status == "ungrounded_write":
            return -0.5
        return 0.0

    def snapshot(self) -> dict[str, list[str]]:
        return {entity_type: sorted(values) for entity_type, values in sorted(self._evidence.items())}

