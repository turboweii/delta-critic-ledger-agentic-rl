from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


PLACEHOLDER_RE = re.compile(
    r"(<[^>]+>|\{[^{}]+\}|\b(todo|tbd|unknown|none|null|n/a|placeholder|xxx)\b)",
    re.IGNORECASE,
)
READ_TOOL_HINTS = ("get_", "list_", "search_", "find_", "lookup_", "retrieve_", "read")
WRITE_TOOL_HINTS = ("book_", "cancel_", "update_", "send_", "create_", "delete_", "transfer")


def canonical_args(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters or {}, sort_keys=True, ensure_ascii=True, default=str)


def is_error_observation(observation: str) -> bool:
    text = (observation or "").strip().lower()
    return text.startswith("error:") or "traceback" in text or "exception" in text


def has_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return bool(PLACEHOLDER_RE.search(value.strip()))
    if isinstance(value, dict):
        return any(has_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(has_placeholder(item) for item in value)
    return False


def is_read_tool(tool: str) -> bool:
    name = tool.lower()
    return name.startswith(READ_TOOL_HINTS) or any(hint in name for hint in ("search", "lookup", "retrieve"))


def is_write_tool(tool: str) -> bool:
    name = tool.lower()
    return name.startswith(WRITE_TOOL_HINTS)


@dataclass
class ProcessFeatures:
    tool_calls: int = 0
    user_turns: int = 0
    repeated_tool_args: int = 0
    tool_errors: int = 0
    repeated_same_errors: int = 0
    recovered_after_error: int = 0
    placeholder_args: int = 0
    read_before_write_violations: int = 0
    read_tool_calls: int = 0
    write_tool_calls: int = 0
    unique_read_tools: int = 0
    premature_final: int = 0
    premature_transfer: int = 0
    pure_think_streak_max: int = 0
    loop_detected: int = 0
    schema_violations: int = 0
    max_turn_hit: int = 0
    empty_reasoning_count: int = 0

    @property
    def trajectory_length(self) -> int:
        return self.tool_calls + self.user_turns

    def rates(self) -> dict[str, float]:
        calls = max(1, self.tool_calls)
        reads = max(1, self.read_tool_calls)
        return {
            "tool_error_rate": self.tool_errors / calls,
            "repeat_tool_args_rate": self.repeated_tool_args / calls,
            "repeat_same_error_rate": self.repeated_same_errors / calls,
            "placeholder_arg_rate": self.placeholder_args / calls,
            "read_before_write_violation_rate": self.read_before_write_violations / calls,
            "read_tool_diversity": self.unique_read_tools / reads,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(self.rates())
        data.update({
            "placeholder_count": self.placeholder_args,
            "duplicate_tool_args_count": self.repeated_tool_args,
            "tool_error_count": self.tool_errors,
            "repeat_same_error_count": self.repeated_same_errors,
            "recovery_after_error_count": self.recovered_after_error,
            "write_before_read_count": self.read_before_write_violations,
            "early_transfer": self.premature_transfer,
            "early_final_respond": self.premature_final,
            "trajectory_length": self.trajectory_length,
            "schema_violation_count": self.schema_violations,
            "think_loop_count": self.pure_think_streak_max,
        })
        return data


@dataclass
class ProcessFeatureTracker:
    features: ProcessFeatures = field(default_factory=ProcessFeatures)
    seen_tool_args: set[tuple[str, str]] = field(default_factory=set)
    seen_read_tools: set[str] = field(default_factory=set)
    has_read_context: bool = False
    previous_error_signature: tuple[str, str] | None = None
    pure_think_streak: int = 0

    def record_tool(
        self,
        tool: str,
        parameters: dict[str, Any],
        observation: str,
        *,
        schema_violation: bool = False,
    ) -> None:
        self.features.tool_calls += 1
        args_key = canonical_args(parameters)
        signature = (tool, args_key)
        repeated = signature in self.seen_tool_args
        if repeated:
            self.features.repeated_tool_args += 1
        self.seen_tool_args.add(signature)

        if has_placeholder(parameters):
            self.features.placeholder_args += 1

        if schema_violation:
            self.features.schema_violations += 1

        read = is_read_tool(tool)
        write = is_write_tool(tool)
        if read:
            self.features.read_tool_calls += 1
            self.seen_read_tools.add(tool)
            self.features.unique_read_tools = len(self.seen_read_tools)
            self.has_read_context = True
        if write:
            self.features.write_tool_calls += 1
            if not self.has_read_context:
                self.features.read_before_write_violations += 1

        error = is_error_observation(observation)
        if error:
            self.features.tool_errors += 1
            error_signature = (tool, observation.strip()[:160])
            if self.previous_error_signature == error_signature:
                self.features.repeated_same_errors += 1
            self.previous_error_signature = error_signature
        elif self.previous_error_signature is not None:
            self.features.recovered_after_error += 1
            self.previous_error_signature = None

        if tool.lower().endswith("think") or tool.lower() == "think":
            self.pure_think_streak += 1
        else:
            self.pure_think_streak = 0
        self.features.pure_think_streak_max = max(self.features.pure_think_streak_max, self.pure_think_streak)

    def record_user_turn(self) -> None:
        self.features.user_turns += 1

    def mark_premature_final(self) -> None:
        self.features.premature_final = 1

    def mark_premature_transfer(self) -> None:
        self.features.premature_transfer = 1

    def record_empty_reasoning(self) -> None:
        self.features.empty_reasoning_count += 1

    def mark_loop(self) -> None:
        self.features.loop_detected = 1

    def mark_max_turn_hit(self) -> None:
        self.features.max_turn_hit = 1

    def to_dict(self) -> dict[str, Any]:
        return self.features.to_dict()
