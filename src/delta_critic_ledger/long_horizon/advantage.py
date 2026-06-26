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
class AdvantageStats:
    mean: float
    std: float
    count: int


@dataclass(frozen=True)
class ClipRange:
    low: float
    high: float


def normalize_advantages(rewards: Iterable[float], eps: float = 1e-6) -> tuple[list[float], AdvantageStats]:
    values = [float(item) for item in rewards]
    if not values:
        return [], AdvantageStats(0.0, 0.0, 0)
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / len(values)
    std = math.sqrt(variance)
    if std < eps:
        return [0.0 for _ in values], AdvantageStats(mean, std, len(values))
    return [(item - mean) / (std + eps) for item in values], AdvantageStats(mean, std, len(values))


def smooth_advantages(
    rewards: Iterable[float],
    *,
    running_mean: float = 0.0,
    running_std: float = 1.0,
    blend: float = 0.0,
    eps: float = 1e-6,
    clip: float | None = 5.0,
) -> tuple[list[float], AdvantageStats]:
    """Return group-local GRPO advantages; optional global blend is ablation-only."""
    values = [float(item) for item in rewards]
    local_advantages, stats = normalize_advantages(values, eps=eps)
    denom = max(float(running_std), eps)
    out: list[float] = []
    for reward, local_advantage in zip(values, local_advantages):
        global_advantage = (reward - float(running_mean)) / denom
        advantage = (1.0 - float(blend)) * local_advantage + float(blend) * global_advantage
        if clip is not None:
            advantage = max(-float(clip), min(float(clip), advantage))
        out.append(advantage)
    return out, stats


def dynamic_clip_range(
    *,
    base_low: float = 0.8,
    base_high: float = 1.2,
    clip_ratio: float = 0.0,
    target_clip_ratio: float = 0.05,
    min_width: float = 0.05,
    max_width: float = 0.4,
) -> ClipRange:
    """Adjust PPO/GRPO ratio clipping from observed clipping pressure."""
    center = (float(base_low) + float(base_high)) / 2.0
    width = (float(base_high) - float(base_low)) / 2.0
    if float(clip_ratio) > float(target_clip_ratio):
        width *= 1.15
    elif float(clip_ratio) < float(target_clip_ratio) * 0.5:
        width *= 0.9
    width = max(float(min_width), min(float(max_width), width))
    return ClipRange(low=center - width, high=center + width)


# ---------------------------------------------------------------------------
# Leg 2 — per-turn credit assignment via FIXED structural weights.
#
# GRPO flattens one trajectory-level outcome scalar over every token, which is
# why long-horizon credit is so coarse. Instead of learning a per-step scorer
# (which would enter the reward and get hacked — see constraint_gate.py), we
# redistribute the outcome credit across turns with a fixed, non-learned rule.
# Total credit is preserved (sum of per-turn credit == outcome advantage), so
# this changes *where* credit lands, not *how much* there is.
#
# Anti-hack: the weights are a structural rule, never learned and never inside
# the optimized objective, so no gradient reaches them. The agent cannot shape
# the weight curve; it can only improve the policy within each turn.
# ---------------------------------------------------------------------------


def per_turn_credit_weights(
    num_turns: int,
    *,
    scheme: str = "uniform",
    decision_turn_mask: list[bool] | None = None,
) -> list[float]:
    """Fixed per-turn credit weights, normalized to sum to 1.

    Args:
        num_turns: number of turns in the trajectory.
        scheme: ``"uniform"`` gives every turn an equal share; ``"decision"``
            up-weights turns flagged as commitment points (write / terminal).
        decision_turn_mask: length-``num_turns`` booleans marking decision
            turns. Used when ``scheme="decision"``.

    Returns:
        Normalized weights summing to 1 (empty list if ``num_turns <= 0``).
    """
    if num_turns <= 0:
        return []
    if scheme == "decision" and decision_turn_mask is not None:
        raw = [2.0 if decision_turn_mask[i] else 1.0 for i in range(num_turns)]
    else:
        raw = [1.0] * num_turns
    total = sum(raw)
    return [w / total for w in raw] if total > 0 else raw


def distribute_advantage_over_turns(
    outcome_advantage: float,
    num_turns: int,
    *,
    scheme: str = "uniform",
    decision_turn_mask: list[bool] | None = None,
) -> list[float]:
    """Redistribute a trajectory-level outcome advantage across its turns.

    Per-turn credit signal: instead of smearing ``outcome_advantage`` equally
    over every token, each turn receives ``outcome_advantage * w_t`` so decision
    turns carry more credit. Sum-preserving by construction.

    Args:
        outcome_advantage: the trajectory-level advantage (e.g. GRPO group-relative).
        num_turns: number of turns in the trajectory.
        scheme / decision_turn_mask: see :func:`per_turn_credit_weights`.

    Returns:
        A length-``num_turns`` list of per-turn advantages summing to
        ``outcome_advantage``. The caller broadcasts each value over that turn's
        tokens before feeding into the policy loss.
    """
    weights = per_turn_credit_weights(
        num_turns, scheme=scheme, decision_turn_mask=decision_turn_mask
    )
    return [float(outcome_advantage) * w for w in weights]


def turn_token_weights_for_rollout(
    turn_is_decision: list[bool],
    tokens_per_turn: list[int],
    *,
    decision_weight: float = 2.0,
    base_weight: float = 1.0,
) -> list[float]:
    """Per-token credit weight for one rollout (Leg 2 helper, plain-python mirror).

    Decision turns (write / commit) get ``decision_weight``, every other turn
    gets ``base_weight``, broadcast to every token in that turn. This is the
    list form of what the patched advantage estimator consumes as a tensor; the
    estimator normalizes the mean weight over policy tokens to 1.0 so only the
    *distribution* of credit changes, not the scale.

    Args:
        turn_is_decision: per-turn flag (True for write/commit turns).
        tokens_per_turn: token count in each turn (must sum to response length).
        decision_weight / base_weight: fixed structural weights (never learned).

    Returns:
        Flat per-token weight list, length ``sum(tokens_per_turn)``.
    """
    if len(turn_is_decision) != len(tokens_per_turn):
        raise ValueError("turn_is_decision and tokens_per_turn must align")
    weights: list[float] = []
    for is_dec, n in zip(turn_is_decision, tokens_per_turn):
        w = decision_weight if is_dec else base_weight
        weights.extend([float(w)] * int(n))
    return weights


def decision_flags_from_tool_calls(tool_calls_per_turn: list[list[str]]) -> list[bool]:
    """Per-turn decision flag for Leg 2 (action -> decision bridge).

    A turn is a *decision turn* if it issued any write/commit tool call. Fixed
    structural rule (never learned); feeds ``turn_is_decision`` into
    :func:`turn_token_weights_for_rollout`.

    Args:
        tool_calls_per_turn: tool names emitted in each assistant turn.

    Returns:
        Per-turn booleans.
    """
    from .process_features import is_write_tool

    return [any(is_write_tool(str(t)) for t in tools) for tools in tool_calls_per_turn]
