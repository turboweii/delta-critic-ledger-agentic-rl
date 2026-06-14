#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger import Action, DeltaLedgerRunner
from delta_critic_ledger.core import report_to_dict, write_json
from delta_critic_ledger.mock_airline import MockAirlineTools, make_demo_data


def load_actions(items: list[dict]) -> list[Action]:
    return [Action(name=item["name"], kwargs=item.get("kwargs", {})) for item in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "demo_airline.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    runner = DeltaLedgerRunner(
        task_id=int(config["task_id"]),
        initial_data=make_demo_data(),
        target_actions=load_actions(config["target_actions"]),
        execute_tool=MockAirlineTools(),
        seed_entities=config.get("seed_entities", {}),
        beta_delta=float(config.get("beta_delta", 0.3)),
        beta_evidence=float(config.get("beta_evidence", 0.1)),
    )
    report = runner.run(load_actions(config["trajectory"]))
    payload = report_to_dict(report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(str(output_dir / "demo_summary.json"), payload)

    with open(output_dir / "demo_delta_trace.jsonl", "w", encoding="utf-8") as f:
        for step in report.steps:
            f.write(json.dumps(asdict(step.delta), ensure_ascii=False) + "\n")

    with open(output_dir / "demo_ledger_trace.jsonl", "w", encoding="utf-8") as f:
        for step in report.steps:
            f.write(json.dumps(asdict(step.ledger), ensure_ascii=False) + "\n")

    print(json.dumps({
        "final_success": report.final_success,
        "combined_reward": report.combined_reward,
        "delta_reward_sum": report.delta_reward_sum,
        "evidence_bonus_sum": report.evidence_bonus_sum,
        "output_dir": str(output_dir),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

