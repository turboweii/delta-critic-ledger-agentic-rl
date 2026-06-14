#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs/airline_mini_yaml/runs.jsonl")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    counter: Counter[str] = Counter()
    rows = []
    with open(args.runs, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
            counter.update(row.get("failure_modes", []))

    summary = {
        "num_runs": len(rows),
        "failure_modes": dict(sorted(counter.items())),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

