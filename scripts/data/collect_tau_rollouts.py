#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.config import ensure_dir, load_config
from delta_critic_ledger.evaluation import OpenAICompatPolicy, parse_tool_arguments
from delta_critic_ledger.verl_integration.context import make_initial_state
from delta_critic_ledger.verl_integration.reward_state import (
    compute_delta_ledger_reward,
    init_delta_reward_state,
    record_tool_transition,
)


def add_tau_bench_path(path: str | None) -> None:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(ROOT.parent / "tau-bench")
    for candidate in candidates:
        if (candidate / "tau_bench").exists():
            sys.path.insert(0, str(candidate))
            return
    raise RuntimeError("Cannot find tau-bench. Pass --tau-bench-path.")


def run_rollout(env: Any, policy: OpenAICompatPolicy, task_id: int, max_turns: int, reward_cfg: dict[str, Any]) -> dict[str, Any]:
    from tau_bench.types import Action, RESPOND_ACTION_NAME

    reset = env.reset(task_index=task_id)
    policy.set_tools(env.tools_info)
    state = make_initial_state(task_id, instance_id=f"collect_{task_id}", env_id=id(env))
    state["max_trace_steps"] = int(reward_cfg.get("max_trace_steps", 128))
    state["delta_reward_state"] = init_delta_reward_state(
        env,
        beta_delta=float(reward_cfg.get("beta_delta", 0.3)),
        beta_evidence=float(reward_cfg.get("beta_evidence", 0.1)),
        include_paths=reward_cfg.get("include_paths", []),
        exclude_paths=reward_cfg.get("exclude_paths", []),
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": env.wiki},
        {"role": "user", "content": str(reset.observation)},
    ]
    reward = 0.0
    error = None

    for turn in range(max_turns):
        try:
            assistant = policy(messages)
            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []
            if tool_calls:
                for tc in tool_calls[:1]:
                    name = tc["function"]["name"]
                    args = parse_tool_arguments(tc["function"].get("arguments"))
                    before = copy.deepcopy(env.data)
                    step = env.step(Action(name=name, kwargs=args))
                    after = copy.deepcopy(env.data)
                    record_tool_transition(state, name, args, str(step.observation), before, after)
                    state["num_tool_calls"] += 1
                    state["total_reward"] += float(step.reward)
                    reward = float(step.reward)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{state['num_tool_calls']}"),
                        "name": name,
                        "content": str(step.observation),
                    })
                    if step.done:
                        state["done"] = True
                        break
            else:
                step = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": assistant.get("content", "")}))
                state["num_user_turns"] += 1
                state["total_reward"] += float(step.reward)
                reward = float(step.reward)
                if step.done:
                    state["done"] = True
                    break
                messages.append({"role": "user", "content": str(step.observation)})
            if state.get("done"):
                break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break

    return {
        "task_id": task_id,
        "success": reward >= 1.0,
        "terminal_reward": reward,
        "combined_reward": compute_delta_ledger_reward(state),
        "num_tool_calls": state["num_tool_calls"],
        "num_user_turns": state["num_user_turns"],
        "error": error,
        "messages": messages,
        "delta_trace": [asdict(step) for step in state.get("delta_steps", [])],
        "ledger_trace": [asdict(step) for step in state.get("ledger_steps", [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real tau-bench policy rollouts with Delta/Ledger traces.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval" / "eval_airline_sft_8x4090_32b_user.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "data_airline_delta"))
    parser.add_argument("--tau-bench-path", default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    add_tau_bench_path(args.tau_bench_path)
    try:
        from tau_bench.envs import get_env
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency while importing tau-bench: {missing}. "
            "Install server dependencies with `pip install -r requirements-server.txt` "
            "and install tau-bench in editable mode before collecting real rollouts."
        ) from exc

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    policy_cfg = cfg["policy"]
    reward_cfg = cfg.get("reward", {})
    out = ensure_dir(args.output_dir)
    runs_path = out / "rollouts.jsonl"

    num_tasks = int(args.num_tasks or env_cfg.get("num_tasks", 50))
    num_samples = int(args.num_samples or env_cfg.get("num_samples_per_task", 1))
    policy = OpenAICompatPolicy(
        model_name=policy_cfg["served_model_name"],
        base_url=policy_cfg["base_url"],
        temperature=float(policy_cfg.get("temperature", 0.7)),
        top_p=float(policy_cfg.get("top_p", 0.9)),
    )

    rows = []
    with open(runs_path, "w", encoding="utf-8") as f:
        for task_id in range(num_tasks):
            for sample_id in range(num_samples):
                from delta_critic_ledger.tau_compat import create_tau_env

                env = create_tau_env(
                    get_env,
                    env_name=env_cfg["env_name"],
                    user_strategy="llm",
                    user_model=env_cfg["user_model"],
                    user_provider=env_cfg["user_provider"],
                    user_api_base=env_cfg["user_base_url"],
                    task_split=env_cfg["task_split"],
                    task_index=task_id,
                )
                row = run_rollout(env, policy, task_id, int(env_cfg.get("max_turns", 30)), reward_cfg)
                row["sample_id"] = sample_id
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"task={task_id} sample={sample_id} success={row['success']} tools={row['num_tool_calls']} error={row['error']}")

    summary = {
        "source": "tau_bench_policy_rollouts",
        "num_samples": len(rows),
        "num_tasks": len({r["task_id"] for r in rows}),
        "success_rate": sum(1 for r in rows if r["success"]) / len(rows) if rows else 0.0,
        "pass_at_1": sum(1 for r in rows if r["success"]) / len(rows) if rows else 0.0,
        "avg_tool_calls": sum(r["num_tool_calls"] for r in rows) / len(rows) if rows else 0.0,
        "rollouts": str(runs_path),
        "config": cfg,
    }
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    max_samples = max((len(items) for items in by_task.values()), default=0)
    summary[f"pass_at_{max_samples}"] = (
        sum(1 for items in by_task.values() if any(item["success"] for item in items)) / len(by_task)
        if by_task else 0.0
    )
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
