"""Reference/test utilities for long-horizon GRPO.

Online training uses the tensor implementation injected into veRL by
``scripts/setup/patch_verl_long_horizon_grpo.py``. Keep this file aligned
with that patch, but do not treat imports from here as proof that the
trainer is using the feature.
"""

from __future__ import annotations


import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GroupFilterDecision:
    should_update: bool
    reason: str
    mean_reward: float
    reward_std: float
    success_rate: float
    weight: float


def _stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / len(values)
    return mean, math.sqrt(variance)


def should_update_group(
    rewards: Iterable[float],
    *,
    min_std: float = 1e-4,
    min_success_rate: float = 0.0,
    max_success_rate: float = 1.0,
    loop_rate: float = 0.0,
    max_loop_rate: float = 1.0,
) -> GroupFilterDecision:
    values = [float(item) for item in rewards]
    mean, std = _stats(values)
    if not values:
        return GroupFilterDecision(False, "empty_group", mean, std, 0.0, 0.0)

    success_rate = sum(1 for item in values if item >= 1.0) / len(values)
    if std < float(min_std):
        return GroupFilterDecision(False, "zero_reward_variance", mean, std, success_rate, 0.0)
    if success_rate < float(min_success_rate):
        return GroupFilterDecision(False, "below_success_band", mean, std, success_rate, 0.0)
    if success_rate > float(max_success_rate):
        return GroupFilterDecision(False, "above_success_band", mean, std, success_rate, 0.0)
    if float(loop_rate) > float(max_loop_rate):
        return GroupFilterDecision(False, "loop_rate_too_high", mean, std, success_rate, 0.0)

    # Middle success rates are more useful for group-relative learning.
    middle = 1.0 - abs(success_rate - 0.5) * 2.0
    weight = max(0.1, min(1.0, 0.5 + 0.5 * middle))
    return GroupFilterDecision(True, "ok", mean, std, success_rate, weight)
