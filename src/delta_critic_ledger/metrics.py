from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any

from .schemas import TrajectoryReport
from .failure_taxonomy import classify_failure


def first_positive_delta_step(report: TrajectoryReport) -> int | None:
    for step in report.steps:
        if step.delta.delta_reward > 0:
            return step.step_idx
    return None


def summarize_reports(reports: list[TrajectoryReport]) -> dict[str, Any]:
    if not reports:
        return {}

    total_writes = 0
    grounded = 0
    conflicting = 0
    ungrounded = 0
    first_positive = []

    for report in reports:
        pos = first_positive_delta_step(report)
        if pos is not None:
            first_positive.append(pos)
        for step in report.steps:
            status = step.ledger.write_grounding_status
            if status in {"evidence_grounded_write", "conflicting_write", "ungrounded_write"}:
                total_writes += 1
            if status == "evidence_grounded_write":
                grounded += 1
            elif status == "conflicting_write":
                conflicting += 1
            elif status == "ungrounded_write":
                ungrounded += 1

    return {
        "num_runs": len(reports),
        "success_rate": mean(1.0 if report.final_success else 0.0 for report in reports),
        "avg_terminal_reward": mean(report.terminal_reward for report in reports),
        "avg_combined_reward": mean(report.combined_reward for report in reports),
        "avg_delta_reward_sum": mean(report.delta_reward_sum for report in reports),
        "avg_evidence_bonus_sum": mean(report.evidence_bonus_sum for report in reports),
        "total_write_actions": total_writes,
        "grounded_write_rate": grounded / total_writes if total_writes else 0.0,
        "conflicting_write_rate": conflicting / total_writes if total_writes else 0.0,
        "ungrounded_write_rate": ungrounded / total_writes if total_writes else 0.0,
        "avg_first_positive_delta_step": mean(first_positive) if first_positive else None,
    }


def compact_report(report: TrajectoryReport, reward_name: str, task_id: str, trajectory_id: str) -> dict[str, Any]:
    return {
        "reward_name": reward_name,
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "final_success": report.final_success,
        "terminal_reward": report.terminal_reward,
        "delta_reward_sum": report.delta_reward_sum,
        "evidence_bonus_sum": report.evidence_bonus_sum,
        "combined_reward": report.combined_reward,
        "goal_fields": report.goal_fields,
        "failure_modes": classify_failure(report),
        "steps": [asdict(step) for step in report.steps],
    }
