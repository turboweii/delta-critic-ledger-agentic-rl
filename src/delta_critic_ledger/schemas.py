from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    name: str
    kwargs: dict[str, Any]


@dataclass
class DeltaStep:
    step_idx: int
    tool: str
    phi_before: float
    phi_after: float
    delta_reward: float
    changed_goal_fields: list[str] = field(default_factory=list)
    regressed_goal_fields: list[str] = field(default_factory=list)


@dataclass
class LedgerStep:
    step_idx: int
    tool: str
    write_grounding_status: str
    missing_evidence_fields: list[str] = field(default_factory=list)
    conflicting_fields: list[str] = field(default_factory=list)
    used_evidence: dict[str, list[str]] = field(default_factory=dict)
    known_entities: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class CombinedStep:
    step_idx: int
    tool: str
    parameters: dict[str, Any]
    observation: str
    delta: DeltaStep
    ledger: LedgerStep


@dataclass
class TrajectoryReport:
    task_id: int
    final_success: bool
    terminal_reward: float
    delta_reward_sum: float
    evidence_bonus_sum: float
    combined_reward: float
    goal_fields: list[str]
    steps: list[CombinedStep]

