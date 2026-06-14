from __future__ import annotations

from collections import Counter
from typing import Any

from .schemas import TrajectoryReport


def first_positive_delta_step(report: TrajectoryReport) -> int | None:
    for step in report.steps:
        if step.delta.delta_reward > 0:
            return step.step_idx
    return None


def classify_failure(report: TrajectoryReport, late_threshold: int = 6) -> list[str]:
    labels: list[str] = []
    statuses = [step.ledger.write_grounding_status for step in report.steps]
    deltas = [step.delta.delta_reward for step in report.steps]

    if any(status == "conflicting_write" for status in statuses):
        labels.append("wrong_write")
    if any(status == "ungrounded_write" for status in statuses):
        labels.append("ungrounded_write")
    if any(delta < 0 for delta in deltas):
        labels.append("state_regression")
    if not any(delta > 0 for delta in deltas):
        labels.append("no_positive_delta")
    else:
        first_pos = first_positive_delta_step(report)
        if first_pos is not None and first_pos >= late_threshold:
            labels.append("late_positive_delta")

    write_statuses = {"evidence_grounded_write", "conflicting_write", "ungrounded_write"}
    has_write = any(status in write_statuses for status in statuses)
    if not has_write:
        labels.append("missing_write")

    repeated_tools = Counter((step.tool, str(sorted(step.parameters.items()))) for step in report.steps)
    if any(count >= 3 for count in repeated_tools.values()):
        labels.append("tool_loop")

    if report.final_success and not labels:
        labels.append("success_clean")
    elif report.final_success:
        labels.append("success_with_issues")
    return labels


def summarize_failure_modes(reports: list[TrajectoryReport]) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    for report in reports:
        counter.update(classify_failure(report))
    return {
        "num_reports": len(reports),
        "failure_modes": dict(sorted(counter.items())),
    }
