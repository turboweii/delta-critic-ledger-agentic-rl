#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.config import load_config
from delta_critic_ledger.prompts import tau_system_prompt
from delta_critic_ledger.tau_compat import create_tau_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the real tau-bench prompt against the GRPO token budget.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tau-bench-path", default=None)
    args = parser.parse_args()

    if args.tau_bench_path:
        sys.path.insert(0, args.tau_bench_path)
    from tau_bench.envs import get_env
    from transformers import AutoTokenizer

    cfg = load_config(args.config)
    env = create_tau_env(
        get_env,
        env_name="airline",
        user_strategy="human",
        user_model="unused",
        user_provider="openai",
        task_split="test",
        task_index=0,
    )
    tool_cfg = yaml.safe_load((ROOT / cfg["data"]["tool_config_path"]).read_text(encoding="utf-8"))
    tools = [item["tool_schema"] for item in tool_cfg["tools"]]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": tau_system_prompt(env.wiki)},
            {"role": "user", "content": "Please help me with my airline reservation."},
        ],
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
    )
    budget = int(cfg["data"]["max_prompt_length"])
    count = len(prompt_ids)
    if count > budget:
        raise SystemExit(f"Prompt budget exceeded: real prompt uses {count} tokens, configured limit is {budget}.")
    print(f"prompt budget: ok ({count}/{budget} tokens)")


if __name__ == "__main__":
    main()
