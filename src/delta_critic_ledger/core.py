from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Optional

from .delta_critic import DeltaCritic, GoalField, path_to_str, value_at, values_equal
from .evidence_ledger import EvidenceLedger
from .schemas import Action, CombinedStep, TrajectoryReport


def flatten_goal_projection(data: dict[str, Any], fields: list[GoalField]) -> dict[str, Any]:
    return {path_to_str(field.path): value_at(data, field.path) for field in fields}


class DeltaLedgerRunner:
    """Runs a trajectory through a tool executor and emits delta/ledger traces."""

    def __init__(
        self,
        task_id: int,
        initial_data: dict[str, Any],
        target_actions: list[Action],
        execute_tool: Callable[[dict[str, Any], Action], str],
        seed_entities: Optional[dict[str, Iterable[str]]] = None,
        beta_delta: float = 0.3,
        beta_evidence: float = 0.1,
        include_paths: Optional[Iterable[str]] = None,
        exclude_paths: Optional[Iterable[str]] = None,
    ):
        self.task_id = task_id
        self.initial_data = copy.deepcopy(initial_data)
        self.execute_tool = execute_tool
        self.delta_critic = DeltaCritic.from_target_actions(
            initial_data,
            target_actions,
            execute_tool,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
        )
        self.ledger = EvidenceLedger(seed_entities=seed_entities)
        self.beta_delta = beta_delta
        self.beta_evidence = beta_evidence

    def run(self, trajectory: list[Action]) -> TrajectoryReport:
        data = copy.deepcopy(self.initial_data)
        steps: list[CombinedStep] = []
        delta_sum = 0.0
        evidence_sum = 0.0

        for idx, action in enumerate(trajectory):
            ledger_step = self.ledger.check_write(idx, action.name, action.kwargs)
            before = copy.deepcopy(data)
            observation = self.execute_tool(data, action)
            after = copy.deepcopy(data)
            delta_step = self.delta_critic.score_step(idx, action.name, before, after)

            self.ledger.update_from_tool(idx, action.name, action.kwargs, observation)
            ledger_step.known_entities = self.ledger.snapshot()

            delta_sum += delta_step.delta_reward
            evidence_sum += self.ledger.evidence_bonus(ledger_step)
            steps.append(
                CombinedStep(
                    step_idx=idx,
                    tool=action.name,
                    parameters=action.kwargs,
                    observation=observation,
                    delta=delta_step,
                    ledger=ledger_step,
                )
            )

        final_success = values_equal(
            flatten_goal_projection(data, self.delta_critic.goal_fields),
            flatten_goal_projection(self.delta_critic.target_data, self.delta_critic.goal_fields),
        )
        terminal_reward = 1.0 if final_success else 0.0
        combined_reward = terminal_reward + self.beta_delta * delta_sum + self.beta_evidence * evidence_sum
        return TrajectoryReport(
            task_id=self.task_id,
            final_success=final_success,
            terminal_reward=terminal_reward,
            delta_reward_sum=round(delta_sum, 6),
            evidence_bonus_sum=round(evidence_sum, 6),
            combined_reward=round(combined_reward, 6),
            goal_fields=self.delta_critic.goal_field_names(),
            steps=steps,
        )


def report_to_dict(report: TrajectoryReport) -> dict[str, Any]:
    return asdict(report)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
