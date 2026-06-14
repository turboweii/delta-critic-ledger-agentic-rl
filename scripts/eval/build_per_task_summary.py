#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-report", default="")
    parser.add_argument("--runs", default="")
    parser.add_argument("--output", default="outputs/per_task_summary.csv")
    args = parser.parse_args()

    rows = []
    if args.eval_report:
        with open(args.eval_report, "r", encoding="utf-8") as f:
            report = json.load(f)
        grouped = defaultdict(list)
        for item in report.get("results", []):
            grouped[item["task_id"]].append(item)
        for task_id, items in sorted(grouped.items()):
            rows.append({
                "task_id": task_id,
                "num_samples": len(items),
                "success_count": sum(1 for x in items if x.get("success")),
                "pass^1": sum(1 for x in items if x.get("success")) / len(items),
                "avg_tool_calls": sum(x.get("num_tool_calls", 0) for x in items) / len(items),
                "error_count": sum(1 for x in items if x.get("error")),
            })
    elif args.runs:
        grouped = defaultdict(list)
        with open(args.runs, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                grouped[item["task_id"]].append(item)
        for task_id, items in sorted(grouped.items()):
            rows.append({
                "task_id": task_id,
                "num_samples": len(items),
                "success_count": sum(1 for x in items if x.get("final_success")),
                "pass^1": sum(1 for x in items if x.get("final_success")) / len(items),
                "avg_delta_reward": sum(x.get("delta_reward_sum", 0.0) for x in items) / len(items),
                "avg_evidence_bonus": sum(x.get("evidence_bonus_sum", 0.0) for x in items) / len(items),
            })
    else:
        raise SystemExit("Provide --eval-report or --runs")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["task_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

