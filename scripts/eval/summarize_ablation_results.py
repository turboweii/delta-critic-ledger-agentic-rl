#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"step_(\d+)$")


def read_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real GRPO ablation eval reports.")
    parser.add_argument("--root", default="outputs/ablation_checkpoint_eval")
    parser.add_argument("--output", default="outputs/ablation_checkpoint_eval/summary.csv")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for report_path in sorted(root.glob("*/step_*/eval_report.json")):
        variant = report_path.parents[1].name
        step_dir = report_path.parent.name
        match = STEP_RE.match(step_dir)
        step = int(match.group(1)) if match else -1
        report = read_report(report_path)
        row = {
            "variant": variant,
            "step": step,
            "success_rate": report.get("success_rate", 0.0),
            "pass_at_1": report.get("pass_at_1", 0.0),
            "error_rate": report.get("error_rate", 0.0),
            "avg_tool_calls": report.get("avg_tool_calls", 0.0),
            "num_samples": report.get("num_samples", 0),
            "num_tasks": report.get("num_tasks", 0),
        }
        by_split = report.get("by_split") or {}
        for split_name, metrics in by_split.items():
            row[f"{split_name}_success_rate"] = metrics.get("success_rate", 0.0)
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["variant", "step", "success_rate", "pass_at_1", "seen_success_rate", "unseen_success_rate"]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
