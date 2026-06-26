"""Leg 2 core — divergence-based per-turn credit.

For a GRPO group (G rollouts of the same prompt), a turn is a *divergence turn*
if the rollouts do NOT all take the same action at that turn position — i.e. it
is where trajectories split and the outcome actually depends on the choice.
Credit should concentrate on divergence turns (the real decision points), not
on the shared read/think prefix where every rollout does the same thing.

Pure-python and testable. The rollout side feeds each group's action sequences
here, gets back per-turn divergence flags, and turns them into
``turn_token_weights`` (via ``turn_token_weights_for_rollout``) that the patched
advantage estimator consumes. Anti-hack: divergence is read off the trajectory
*structure*, never learned, never inside the optimized reward.
"""
from __future__ import annotations

import json
from typing import Sequence

ActionStep = tuple[str, str]  # (tool, canonical_params_str)


def canonical_action_seq(action_history: Sequence[dict]) -> list[ActionStep]:
    """Build a comparable action sequence from an action_history list.

    Reduces each step to (tool, canonical_params) so structurally identical
    calls compare equal. Think steps are kept — a pure-think loop is itself a
    useful divergence signal.
    """
    seq: list[ActionStep] = []
    for a in action_history or []:
        tool = str(a.get("tool", ""))
        params = a.get("parameters") or {}
        seq.append((tool, json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)))
    return seq


def divergence_flags_for_group(
    action_seqs: Sequence[Sequence[ActionStep]],
) -> list[list[bool]]:
    """Per-turn divergence flag for each rollout in a GRPO group.

    Turn position ``t`` is a divergence turn if the rollouts that reach ``t`` do
    not all share the same action there. Returns one bool list per rollout,
    aligned to that rollout's own length.

    Args:
        action_seqs: one action sequence per rollout in the group (same prompt).

    Returns:
        Per-rollout list of divergence flags.
    """
    g = len(action_seqs)
    if g <= 1:
        return [[False] * len(s) for s in action_seqs]
    max_len = max((len(s) for s in action_seqs), default=0)
    flags: list[list[bool]] = [[] for _ in range(g)]
    for pos in range(max_len):
        actions = [action_seqs[i][pos] for i in range(g) if pos < len(action_seqs[i])]
        agree = len(set(actions)) <= 1  # all same (or only one rollout reaches here)
        for i in range(g):
            if pos < len(action_seqs[i]):
                flags[i].append(not agree)
    return flags


def divergence_turn_weights(
    action_seqs: Sequence[Sequence[ActionStep]],
    *,
    divergence_weight: float = 2.0,
    base_weight: float = 1.0,
) -> list[list[float]]:
    """Per-turn credit weight for each rollout, from the group's divergence.

    Divergence turns get ``divergence_weight``, shared turns get ``base_weight``.
    Feed each rollout's weight list together with its ``tokens_per_turn`` into
    ``turn_token_weights_for_rollout`` to produce the per-token weights the
    patched advantage estimator consumes.
    """
    flags = divergence_flags_for_group(action_seqs)
    return [
        [divergence_weight if f else base_weight for f in rollout_flags]
        for rollout_flags in flags
    ]


def first_divergence_turn(action_seqs: Sequence[Sequence[ActionStep]]) -> int | None:
    """First turn position where the group splits, or None if they never diverge."""
    g = len(action_seqs)
    if g <= 1:
        return None
    max_len = max((len(s) for s in action_seqs), default=0)
    for pos in range(max_len):
        actions = [action_seqs[i][pos] for i in range(g) if pos < len(action_seqs[i])]
        if len(set(actions)) > 1:
            return pos
    return None
