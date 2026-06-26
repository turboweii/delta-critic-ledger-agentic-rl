"""Leg 1 — process correctness as a HARD constraint, not a shaped reward.

Motivation
----------
Long-horizon tool agents only see a terminal 0/1 reward. The classic fix is to
synthesize a denser signal from process features (placeholder args, grounding,
read-before-write, ...). But the moment that signal enters the reward it is
*optimized*, and therefore *hacked*: the agent learns to inflate read-tool
diversity, manufacture recoveries, pad trajectory length, etc. Any hand-tuned process scorer that
feeds the reward ends up needing a pile of magic weights and accumulates
"removed because caused anti-incentive" fixes — the signature of a proxy that
has drifted from the true goal.

This module inverts the role. The reward stays terminal-outcome only. The same
mechanically-checkable process errors become a **binary, non-tradeable
constraint**: a rollout that violates it is *rejected* — its reward is forced to
0 even when the environment judged it a success::

    constrained_reward = success AND NOT violated

Why this is anti-hack
---------------------
A reward is continuous: the agent can trade a little proxy signal for a little
reward. A constraint is discrete: there is no gradient toward "partially
satisfying" it, and being clean (e.g. no placeholder write args) is itself a
real task requirement, not a proxy. So pushing toward the constraint pushes
toward the true task, not toward an exploitable surrogate.

A curriculum (`enforce_prob`) relaxes the gate during warmup so a fresh policy
that violates everything can still learn before the gate bites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConstraintGateConfig:
    """Which mechanically-checkable errors trigger a reject."""

    enabled: bool = True
    reject_placeholder: bool = True          # write called with placeholder args
    reject_schema_violation: bool = True     # tool args violated their schema
    reject_loop: bool = True                 # repeated tool+args / same error / pure-think loop
    reject_max_turn_stall: bool = True       # hit max turns without resolving
    reject_premature_terminal: bool = False  # final/transfer before any write (opt-in)
    reject_ungrounded_write: bool = True     # write param value not seen in prior observations

    # Tolerances: how many of each before the gate trips. 0 = zero-tolerance.
    max_placeholder_args: int = 0
    max_schema_violations: int = 0
    max_repeated_tool_args: int = 0
    max_repeated_same_errors: int = 0
    max_ungrounded_writes: int = 0

    # Curriculum: gate is only enforced with probability ramping 0 -> 1 over
    # `warmup_steps`. Lets a naive early policy survive before the gate bites.
    warmup_steps: int = 0


@dataclass(frozen=True)
class ConstraintViolation:
    """Result of evaluating one rollout against the gate."""

    violated: bool
    reasons: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        return ";".join(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {"violated": self.violated, "reasons": list(self.reasons), "reason": self.reason}


def _feat(features: Any, name: str, default: float = 0.0) -> float:
    """Read a field from a ProcessFeatures dataclass or a dict."""
    if isinstance(features, dict):
        return float(features.get(name, default) or 0.0)
    return float(getattr(features, name, default) or 0.0)


def evaluate_constraint(
    features: Any,
    config: ConstraintGateConfig | None = None,
) -> ConstraintViolation:
    """Aggregate a rollout's process features into a single violation verdict.

    Args:
        features: a ``ProcessFeatures`` instance or its ``to_dict()`` output.
        config: gate config. ``None`` uses the default.

    Returns:
        A :class:`ConstraintViolation`.
    """
    cfg = config or ConstraintGateConfig()
    if not cfg.enabled:
        return ConstraintViolation(False)

    reasons: list[str] = []
    if cfg.reject_placeholder and _feat(features, "placeholder_args") > cfg.max_placeholder_args:
        reasons.append("placeholder_args")
    if cfg.reject_schema_violation and _feat(features, "schema_violations") > cfg.max_schema_violations:
        reasons.append("schema_violation")
    if cfg.reject_loop:
        if _feat(features, "loop_detected") >= 1.0:
            reasons.append("loop_detected")
        elif _feat(features, "repeated_tool_args") > cfg.max_repeated_tool_args and cfg.max_repeated_tool_args >= 0:
            # only trip on excess repeats when an explicit non-zero tolerance is set;
            # otherwise loop_detected (set by LoopGuard) is the canonical signal.
            if cfg.max_repeated_tool_args > 0:
                reasons.append("repeated_tool_args")
    if cfg.reject_max_turn_stall and _feat(features, "max_turn_hit") >= 1.0:
        reasons.append("max_turn_stall")
    if cfg.reject_premature_terminal and (
        _feat(features, "premature_final") >= 1.0 or _feat(features, "premature_transfer") >= 1.0
    ):
        reasons.append("premature_terminal")
    if cfg.reject_ungrounded_write and _feat(features, "ungrounded_write_count") > cfg.max_ungrounded_writes:
        reasons.append("ungrounded_write")

    return ConstraintViolation(violated=bool(reasons), reasons=tuple(reasons))


def enforce_prob(step: int, warmup_steps: int) -> float:
    """Probability the gate is active at ``step`` under a linear curriculum.

    ``warmup_steps <= 0`` means the gate is always active. The caller decides
    deterministically (e.g. ``enforce = uniform() < enforce_prob(...)``) so the
    gate itself stays pure / testable.
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(step) / float(warmup_steps)))


def apply_constraint_gate(
    terminal_reward: float,
    violation: ConstraintViolation,
    *,
    enforce: bool = True,
    success_threshold: float = 1.0,
) -> dict[str, Any]:
    """Compute the constrained reward.

    ``constrained_reward = success AND NOT (violated AND enforce)``. A violated
    rollout that would have been a success is forced to 0 — i.e. it enters the
    GRPO group as a *failure* sample, which is exactly the contrast signal we
    want, with no continuous proxy to hack.

    Returns a dict with the constrained reward plus diagnostics so the trainer
    can log reject rate / would-have-succeeded rate.
    """
    success = float(terminal_reward) >= float(success_threshold)
    rejected = bool(enforce and violation.violated)
    constrained = (1.0 if success else 0.0) if not rejected else 0.0
    return {
        "constrained_reward": constrained,
        "rejected": rejected,
        "would_have_succeeded": success,
        "success": success and not rejected,
        "terminal_reward": float(terminal_reward),
        **violation.to_dict(),
    }
