from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _clip(value: float, low: float | None, high: float | None) -> float:
    if low is not None:
        value = max(float(low), value)
    if high is not None:
        value = min(float(high), value)
    return value


@dataclass(frozen=True)
class RewardEnvelopeConfig:
    success_min: float = 1.0
    success_max: float = 1.2
    failure_min: float = -0.2
    failure_max: float = 0.4
    dense_beta: float = 0.0


class RewardEnvelope:
    def __init__(self, config: RewardEnvelopeConfig | None = None):
        self.config = config or RewardEnvelopeConfig()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "RewardEnvelope":
        if not config:
            return cls()
        known = {field for field in RewardEnvelopeConfig.__dataclass_fields__}
        values = {key: value for key, value in dict(config).items() if key in known}
        return cls(RewardEnvelopeConfig(**values))

    def compute(self, terminal_reward: float, process_score: float = 0.0) -> dict[str, float | bool]:
        success = float(terminal_reward) >= 1.0
        dense = self.config.dense_beta * float(process_score)
        raw = (1.0 if success else 0.0) + dense
        if success:
            score = _clip(raw, self.config.success_min, self.config.success_max)
        else:
            score = _clip(raw, self.config.failure_min, self.config.failure_max)
        return {
            "score": score,
            "terminal_reward": 1.0 if success else 0.0,
            "raw_terminal_reward": float(terminal_reward),
            "dense_reward": dense,
            "process_score": float(process_score),
            "raw_score": raw,
            "success": success,
            "score_was_clipped": score != raw,
        }
