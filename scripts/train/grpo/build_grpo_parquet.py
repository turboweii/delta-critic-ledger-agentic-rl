#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = (
    "# Current Date Context\n"
    "The current date is 2024-05-15 (Wednesday). "
    "When users mention dates without specifying the year, assume 2024."
)
INTERACTION_NAME = "tau_bench_airline"


def parse_ids(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one task ID is required.")
    return values


def build_row(task_id: int, index: int, split: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}],
        "extra_info": {
            "index": index,
            "task_id": task_id,
            "split": split,
            "interaction_kwargs": {
                "name": INTERACTION_NAME,
                "task_id": task_id,
            },
        },
        "data_source": INTERACTION_NAME,
        "reward_model": {"ground_truth": ""},
        "ability": INTERACTION_NAME,
    }


def write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tau-bench GRPO parquet datasets.")
    train_group = parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument("--train-task-ids", help="Comma-separated training task IDs.")
    train_group.add_argument("--train-task-ids-from", help="JSON file containing seen_task_ids.")
    parser.add_argument("--val-task-ids", default="", help="Comma-separated validation task IDs; defaults to train IDs.")
    parser.add_argument("--num-total-tasks", type=int, default=None, help="Use task IDs 0..N-1 for validation.")
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-val", required=True)
    args = parser.parse_args()

    if args.train_task_ids_from:
        metadata = json.loads(Path(args.train_task_ids_from).read_text(encoding="utf-8"))
        train_ids = [int(value) for value in metadata.get("seen_task_ids", [])]
        if not train_ids:
            raise ValueError(f"No seen_task_ids found in {args.train_task_ids_from}")
    else:
        train_ids = parse_ids(args.train_task_ids)
    if args.val_task_ids:
        val_ids = parse_ids(args.val_task_ids)
    elif args.num_total_tasks is not None:
        val_ids = list(range(args.num_total_tasks))
    else:
        val_ids = train_ids
    train_set = set(train_ids)

    train_rows = [build_row(task_id, index, "seen") for index, task_id in enumerate(train_ids)]
    val_rows = [
        build_row(task_id, index, "seen" if task_id in train_set else "unseen")
        for index, task_id in enumerate(val_ids)
    ]

    write_parquet(train_rows, Path(args.output_train))
    write_parquet(val_rows, Path(args.output_val))
    print(f"train rows: {len(train_rows)} -> {args.output_train}")
    print(f"val rows: {len(val_rows)} -> {args.output_val}")


if __name__ == "__main__":
    main()
