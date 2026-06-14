#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger import DeltaLedgerRunner
from delta_critic_ledger.config import load_config
from delta_critic_ledger.metrics import compact_report, summarize_reports
from delta_critic_ledger.mock_airline import MockAirlineTools
from delta_critic_ledger.mock_tasks import get_task_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=str(ROOT / "configs" / "experiments" / "airline_mini.yaml"))
    parser.add_argument("--reward-dir", default=str(ROOT / "configs" / "reward"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "airline_mini"))
    args = parser.parse_args()

    experiment = load_config(Path(args.experiment))
    reward_paths = sorted(Path(args.reward_dir).glob("*.yaml"))
    tasks = get_task_registry()
    tools = MockAirlineTools()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = {"experiment": experiment["name"], "rewards": {}}
    for reward_path in reward_paths:
        reward_cfg = load_config(reward_path)
        reports = []
        reward_name = reward_cfg["name"]
        for task_id in experiment["task_ids"]:
            task = tasks[task_id]
            for trajectory_id in experiment["trajectory_ids"]:
                if trajectory_id not in task.trajectories:
                    continue
                runner = DeltaLedgerRunner(
                    task_id=hash(task_id) % 100000,
                    initial_data=task.initial_data,
                    target_actions=task.target_actions,
                    execute_tool=tools,
                    seed_entities=task.seed_entities,
                    beta_delta=float(reward_cfg["beta_delta"]),
                    beta_evidence=float(reward_cfg["beta_evidence"]),
                )
                report = runner.run(task.trajectories[trajectory_id])
                reports.append(report)
                all_rows.append(compact_report(report, reward_name, task_id, trajectory_id))
        summary["rewards"][reward_name] = summarize_reports(reports)

    with open(output_dir / "runs.jsonl", "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
