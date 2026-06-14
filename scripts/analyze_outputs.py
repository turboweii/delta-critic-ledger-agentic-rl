#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/airline_mini/summary.json")
    args = parser.parse_args()
    path = Path(args.summary)
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print(f"Experiment: {summary['experiment']}")
    print("reward,success_rate,avg_combined_reward,delta,grounded,conflicting,ungrounded")
    for reward, metrics in sorted(summary["rewards"].items()):
        print(
            f"{reward},"
            f"{metrics['success_rate']:.3f},"
            f"{metrics['avg_combined_reward']:.3f},"
            f"{metrics['avg_delta_reward_sum']:.3f},"
            f"{metrics['grounded_write_rate']:.3f},"
            f"{metrics['conflicting_write_rate']:.3f},"
            f"{metrics['ungrounded_write_rate']:.3f}"
        )


if __name__ == "__main__":
    main()

