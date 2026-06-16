#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.prompts import DATE_CONTEXT, tau_system_prompt
from delta_critic_ledger.tau_compat import create_tau_env

DATA_SOURCE = "tau_bench_airline"
INTERACTION_NAME = "tau_bench_airline_delta_ledger"


def parse_ids(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one task ID is required.")
    return values


def build_row(task_id: int, index: int, split: str, system_prompt: str = DATE_CONTEXT) -> dict:
    return {
        "prompt": [{"role": "system", "content": system_prompt}],
        "extra_info": {
            "index": index,
            "task_id": task_id,
            "split": split,
            "interaction_kwargs": {
                "name": INTERACTION_NAME,
                "task_id": task_id,
            },
        },
        "data_source": DATA_SOURCE,
        "reward_model": {"ground_truth": ""},
        "ability": DATA_SOURCE,
    }


def write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def load_tau_system_prompt(env_name: str, task_split: str, tau_bench_path: str | None) -> str:
    if tau_bench_path:
        sys.path.insert(0, tau_bench_path)
    from tau_bench.envs import get_env

    env = create_tau_env(
        get_env,
        env_name=env_name,
        user_strategy="human",
        user_model="unused",
        user_provider="openai",
        task_split=task_split,
        task_index=0,
    )
    return tau_system_prompt(env.wiki)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tau-bench GRPO parquet datasets.")
    train_group = parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument("--train-task-ids", help="Comma-separated training task IDs.")
    train_group.add_argument("--train-task-ids-from", help="JSON file containing seen_task_ids.")
    parser.add_argument("--val-task-ids", default="", help="Comma-separated validation task IDs.")
    parser.add_argument("--val-task-ids-from", help="JSON file containing unseen_task_ids.")
    parser.add_argument("--num-total-tasks", type=int, default=None, help="Use task IDs 0..N-1 for validation.")
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-val", required=True)
    parser.add_argument("--env-name", default="airline")
    parser.add_argument("--task-split", default="test")
    parser.add_argument("--tau-bench-path", default=None)
    args = parser.parse_args()

    if args.train_task_ids_from:
        metadata = json.loads(Path(args.train_task_ids_from).read_text(encoding="utf-8"))
        train_ids = [int(value) for value in metadata.get("seen_task_ids", [])]
        if not train_ids:
            raise ValueError(f"No seen_task_ids found in {args.train_task_ids_from}")
    else:
        train_ids = parse_ids(args.train_task_ids)
    val_modes = sum(bool(value) for value in (args.val_task_ids, args.val_task_ids_from, args.num_total_tasks))
    if val_modes > 1:
        raise ValueError("Choose only one validation source.")
    if args.val_task_ids:
        val_ids = parse_ids(args.val_task_ids)
    elif args.val_task_ids_from:
        val_metadata = json.loads(Path(args.val_task_ids_from).read_text(encoding="utf-8"))
        val_ids = [int(value) for value in val_metadata.get("unseen_task_ids", [])]
        if not val_ids:
            raise ValueError(f"No unseen_task_ids found in {args.val_task_ids_from}")
    elif args.num_total_tasks is not None:
        val_ids = list(range(args.num_total_tasks))
    else:
        val_ids = train_ids
    train_set = set(train_ids)
    system_prompt = load_tau_system_prompt(args.env_name, args.task_split, args.tau_bench_path)

    train_rows = [build_row(task_id, index, "seen", system_prompt) for index, task_id in enumerate(train_ids)]
    val_rows = [
        build_row(task_id, index, "seen" if task_id in train_set else "unseen", system_prompt)
        for index, task_id in enumerate(val_ids)
    ]

    write_parquet(train_rows, Path(args.output_train))
    write_parquet(val_rows, Path(args.output_val))
    print(f"train rows: {len(train_rows)} -> {args.output_train}")
    print(f"val rows: {len(val_rows)} -> {args.output_val}")


if __name__ == "__main__":
    main()
