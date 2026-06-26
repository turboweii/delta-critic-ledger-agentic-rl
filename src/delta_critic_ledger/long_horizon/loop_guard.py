from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .process_features import ProcessFeatureTracker, canonical_args, is_error_observation


@dataclass(frozen=True)
class LoopGuardConfig:
    enabled: bool = True
    max_repeated_tool_args: int = 3
    max_repeated_same_errors: int = 2
    max_pure_think_streak: int = 4


@dataclass(frozen=True)
class LoopGuardDecision:
    should_stop: bool
    reason: str = ""


class LoopGuard:
    def __init__(self, config: LoopGuardConfig | None = None):
        self.config = config or LoopGuardConfig()
        self._tool_arg_counts: dict[tuple[str, str], int] = {}
        self._error_counts: dict[tuple[str, str], int] = {}
        self._pure_think_streak = 0

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "LoopGuard":
        if not config:
            return cls()
        known = {field for field in LoopGuardConfig.__dataclass_fields__}
        values = {key: value for key, value in dict(config).items() if key in known}
        return cls(LoopGuardConfig(**values))

    def observe(
        self,
        tool: str,
        parameters: dict[str, Any],
        observation: str,
        tracker: ProcessFeatureTracker | None = None,
    ) -> LoopGuardDecision:
        if not self.config.enabled:
            return LoopGuardDecision(False)

        args_key = canonical_args(parameters)
        signature = (tool, args_key)
        self._tool_arg_counts[signature] = self._tool_arg_counts.get(signature, 0) + 1
        if self._tool_arg_counts[signature] > self.config.max_repeated_tool_args:
            if tracker is not None:
                tracker.mark_loop()
            return LoopGuardDecision(True, "repeated_tool_args")

        if is_error_observation(observation):
            error_signature = (tool, observation.strip()[:160])
            self._error_counts[error_signature] = self._error_counts.get(error_signature, 0) + 1
            if self._error_counts[error_signature] > self.config.max_repeated_same_errors:
                if tracker is not None:
                    tracker.mark_loop()
                return LoopGuardDecision(True, "repeated_same_error")

        if tool.lower().endswith("think") or tool.lower() == "think":
            self._pure_think_streak += 1
        else:
            self._pure_think_streak = 0
        if self._pure_think_streak > self.config.max_pure_think_streak:
            if tracker is not None:
                tracker.mark_loop()
            return LoopGuardDecision(True, "pure_think_loop")

        return LoopGuardDecision(False)
