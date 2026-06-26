"""Reference/test utilities for long-horizon GRPO.

Online training uses the tensor implementation injected into veRL by
``scripts/setup/patch_verl_long_horizon_grpo.py``. Keep this file aligned
with that patch, but do not treat imports from here as proof that the
trainer is using the feature.
"""

from __future__ import annotations


from typing import Iterable, Sequence


def _as_floats(values: Iterable[float]) -> list[float]:
    return [float(item) for item in values]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def token_mean_loss(token_losses: Sequence[Sequence[float]], advantages: Sequence[float]) -> float:
    """Aggregate every valid token equally."""
    numerator = 0.0
    denominator = 0
    for losses, advantage in zip(token_losses, advantages):
        values = _as_floats(losses)
        numerator += float(advantage) * sum(values)
        denominator += len(values)
    return numerator / max(1, denominator)


def sequence_mean_loss(token_losses: Sequence[Sequence[float]], advantages: Sequence[float]) -> float:
    """Aggregate each response equally after taking its token mean."""
    seq_terms = []
    for losses, advantage in zip(token_losses, advantages):
        values = _as_floats(losses)
        if values:
            seq_terms.append(float(advantage) * _mean(values))
    return _mean(seq_terms)


def balanced_aggregation_loss(token_losses: Sequence[Sequence[float]], advantages: Sequence[float]) -> float:
    """Balanced Aggregation style loss for GRPO.

    Positive-advantage and negative-advantage responses are aggregated as
    separate token means, then combined by response-count weights. This reduces
    sign-length coupling where one side dominates only because it is longer.
    """
    pos_tokens: list[float] = []
    neg_tokens: list[float] = []
    pos_count = 0
    neg_count = 0
    zero_terms: list[float] = []

    for losses, advantage in zip(token_losses, advantages):
        values = _as_floats(losses)
        if not values:
            continue
        adv = float(advantage)
        weighted = [adv * item for item in values]
        if adv > 0.0:
            pos_tokens.extend(weighted)
            pos_count += 1
        elif adv < 0.0:
            neg_tokens.extend(weighted)
            neg_count += 1
        else:
            zero_terms.append(0.0)

    total_count = pos_count + neg_count + len(zero_terms)
    if total_count == 0:
        return 0.0

    loss = 0.0
    if pos_count:
        loss += (pos_count / total_count) * _mean(pos_tokens)
    if neg_count:
        loss += (neg_count / total_count) * _mean(neg_tokens)
    return loss
