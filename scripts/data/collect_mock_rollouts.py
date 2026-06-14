#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger import DeltaLedgerRunner
from delta_critic_ledger.config import ensure_dir
from delta_critic_ledger.core import report_to_dict
from delta_critic_ledger.metrics import summarize_reports
from delta_critic_ledger.mock_airline import MockAirlineTools
from delta_critic_ledger.mock_tasks import get_task_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "data_airline_delta_mock"))
    args = parser.parse_args()

    out = ensure_dir(args.output_dir)
    tasks = get_task_registry()
    tools = MockAirlineTools()
    reports = []

    with open(out / "raw_trajectories.jsonl", "w", encoding="utf-8") as raw_f, \
        open(out / "delta_trace.jsonl", "w", encoding="utf-8") as delta_f, \
        open(out / "ledger_trace.jsonl", "w", encoding="utf-8") as ledger_f:
        for task_id, task in tasks.items():
            for trajectory_id, trajectory in task.trajectories.items():
                runner = DeltaLedgerRunner(
                    task_id=hash(task_id) % 100000,
                    initial_data=task.initial_data,
                    target_actions=task.target_actions,
                    execute_tool=tools,
                    seed_entities=task.seed_entities,
                )
                report = runner.run(trajectory)
                reports.append(report)
                raw_f.write(json.dumps({
                    "task_id": task_id,
                    "trajectory_id": trajectory_id,
                    "actions": [asdict(action) for action in trajectory],
                    "final_success": report.final_success,
                }, ensure_ascii=False) + "\n")
                for step in report.steps:
                    delta_f.write(json.dumps({
                        "task_id": task_id,
                        "trajectory_id": trajectory_id,
                        **asdict(step.delta),
                    }, ensure_ascii=False) + "\n")
                    ledger_f.write(json.dumps({
                        "task_id": task_id,
                        "trajectory_id": trajectory_id,
                        **asdict(step.ledger),
                    }, ensure_ascii=False) + "\n")

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summarize_reports(reports), f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote mock rollout dataset to {out}")


if __name__ == "__main__":
    main()

