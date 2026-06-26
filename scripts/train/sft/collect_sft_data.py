#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.config import ensure_dir
from delta_critic_ledger.evaluation import OpenAICompatPolicy
from delta_critic_ledger.prompts import tau_system_prompt
from delta_critic_ledger.tau_compat import create_tau_env

def add_tau_bench_path(path: str | None) -> None:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(ROOT.parent / "tau-bench")
    for candidate in candidates:
        if (candidate / "tau_bench").exists():
            sys.path.insert(0, str(candidate))
            return
    raise RuntimeError("Cannot find tau-bench. Pass --tau-bench-path or place tau-bench next to this project.")


def parse_task_ids(raw: str | None, start: int, end: int) -> list[int]:
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return list(range(start, end))


def parse_temperatures(raw: str, best_of_n: int) -> list[float]:
    temps = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(temps) == 1:
        temps = temps * best_of_n
    if len(temps) != best_of_n:
        raise ValueError(f"--temperatures length must be 1 or best_of_n={best_of_n}, got {len(temps)}")
    return temps


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    return json.loads(str(raw) or "{}")


def action_name(action: Any) -> str:
    if isinstance(action, dict):
        return action["name"]
    return getattr(action, "name")


def action_kwargs(action: Any) -> dict[str, Any]:
    kwargs = getattr(action, "kwargs", None)
    if kwargs is None and isinstance(action, dict):
        kwargs = action.get("kwargs") or action.get("arguments") or {}
    return dict(kwargs or {})


def build_env(args: argparse.Namespace, task_id: int):
    from tau_bench.envs import get_env

    user_strategy = args.user_strategy if args.use_user_sim else "human"
    return create_tau_env(
        get_env,
        env_name=args.env_name,
        user_strategy=user_strategy,
        user_model=args.user_model,
        user_provider=args.user_provider,
        user_api_base=args.user_base_url,
        task_split=args.task_split,
        task_index=task_id,
    )


def assistant_tool_call(action: Any, call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": action_name(action),
                    "arguments": json.dumps(action_kwargs(action), ensure_ascii=False),
                },
            }
        ],
    }


def collect_oracle_example(args: argparse.Namespace, task_id: int) -> dict[str, Any]:
    from tau_bench.types import RESPOND_ACTION_NAME

    env = build_env(args, task_id)
    task = env.task
    messages: list[dict[str, Any]] = [{"role": "system", "content": tau_system_prompt(env.wiki)}]
    if args.use_user_sim:
        reset = env.reset(task_index=task_id)
        messages.append({"role": "user", "content": str(reset.observation)})
    else:
        messages.append({"role": "user", "content": task.instruction})

    tool_idx = 0
    for action in task.actions:
        name = action_name(action)
        kwargs = action_kwargs(action)
        if name == RESPOND_ACTION_NAME:
            messages.append({"role": "assistant", "content": kwargs.get("content", "")})
            continue
        call_id = f"oracle_{task_id}_{tool_idx}"
        messages.append(assistant_tool_call(action, call_id))
        try:
            observation = env.tools_map[name].invoke(data=env.data, **kwargs)
        except Exception as exc:
            observation = f"Error: {exc}"
        messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": str(observation)})
        tool_idx += 1

    if messages[-1].get("role") != "assistant":
        final = "The requested change has been completed."
        if task.outputs:
            final += " " + " ".join(str(x) for x in task.outputs)
        messages.append({"role": "assistant", "content": final})
    return {
        "task_id": task_id,
        "sample_idx": 0,
        "temperature": 0.0,
        "success": True,
        "reward": 1.0,
        "num_turns": len([m for m in messages if m["role"] == "assistant"]),
        "num_tool_calls": len([m for m in messages if m["role"] == "tool"]),
        "messages": messages,
        "source": "tau_bench_oracle_actions",
    }


def run_teacher_rollout(args: argparse.Namespace, task_id: int, sample_idx: int, temperature: float) -> dict[str, Any]:
    from tau_bench.types import Action, RESPOND_ACTION_NAME

    env = build_env(args, task_id)
    reset = env.reset(task_index=task_id)
    policy = OpenAICompatPolicy(
        model_name=args.teacher_model,
        base_url=args.teacher_base_url,
        temperature=temperature,
        top_p=args.teacher_top_p,
        max_tokens=args.teacher_max_tokens,
        max_context_chars=args.contamination_char_limit,
    )
    policy.set_tools(env.tools_info)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": tau_system_prompt(env.wiki)},
        {"role": "user", "content": str(reset.observation)},
    ]
    reward = 0.0
    error = None
    num_tool_calls = 0
    contaminated_from_turn = None
    terminated = False

    for turn in range(args.max_turns):
        try:
            assistant = policy(messages)
            if policy.was_truncated and contaminated_from_turn is None:
                contaminated_from_turn = turn
            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []
            if tool_calls:
                for tc in tool_calls[:1]:
                    name = tc["function"]["name"]
                    kwargs = parse_tool_arguments(tc["function"].get("arguments"))
                    step = env.step(Action(name=name, kwargs=kwargs))
                    num_tool_calls += 1
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{num_tool_calls}"),
                        "name": name,
                        "content": str(step.observation),
                    })
                    reward = float(step.reward)
                    if step.done:
                        terminated = True
                        break
            else:
                step = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": assistant.get("content", "")}))
                reward = float(step.reward)
                if step.done:
                    break
                messages.append({"role": "user", "content": str(step.observation)})
            if terminated:
                break
            if reward >= 1.0:
                break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break

    return {
        "task_id": task_id,
        "sample_idx": sample_idx,
        "temperature": temperature,
        "success": reward >= 1.0,
        "reward": reward,
        "num_turns": len([m for m in messages if m["role"] == "assistant"]),
        "num_tool_calls": num_tool_calls,
        "error": error,
        "was_contaminated_from_turn": contaminated_from_turn,
        "messages": messages,
        "source": "teacher_rollout_distillation",
    }


def collect_one_task(args: argparse.Namespace, task_id: int, out: Path, temperatures: list[float]) -> dict[str, Any]:
    out_file = out / f"task_{task_id:04d}.jsonl"
    contaminated_file = out / f"task_{task_id:04d}_contaminated.jsonl"
    meta_file = out / f"task_{task_id:04d}.meta.json"
    if meta_file.exists() and not args.overwrite:
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    successes = []
    contaminated = []
    attempts = []
    if args.mode == "oracle":
        row = collect_oracle_example(args, task_id)
        successes.append(row)
        attempts.append({k: row[k] for k in ("sample_idx", "temperature", "success", "reward", "num_turns", "num_tool_calls")})
    else:
        for sample_idx, temperature in enumerate(temperatures):
            row = run_teacher_rollout(args, task_id, sample_idx, temperature)
            attempts.append({
                "sample_idx": sample_idx,
                "temperature": temperature,
                "success": row["success"],
                "reward": row["reward"],
                "num_turns": row["num_turns"],
                "num_tool_calls": row["num_tool_calls"],
                "error": row["error"],
                "was_contaminated": row["was_contaminated_from_turn"] is not None,
            })
            if row["was_contaminated_from_turn"] is not None:
                contaminated.append(row)
            elif row["success"]:
                successes.append(row)

    with open(out_file, "w", encoding="utf-8") as f:
        for row in successes:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(contaminated_file, "w", encoding="utf-8") as f:
        for row in contaminated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "task_id": task_id,
        "mode": args.mode,
        "best_of_n": args.best_of_n if args.mode == "teacher_rollout" else 1,
        "num_successes": len(successes),
        "num_contaminated": len(contaminated),
        "any_success": bool(successes),
        "attempts": attempts,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return meta


def build_split(task_ids: list[int], holdout_size: int) -> tuple[list[int], list[int]]:
    if holdout_size <= 0 or len(task_ids) <= holdout_size:
        return list(task_ids), []
    stride = len(task_ids) / holdout_size
    holdout = sorted({task_ids[int(i * stride)] for i in range(holdout_size)})
    i = -1
    while len(holdout) < holdout_size:
        cand = task_ids[i]
        if cand not in holdout:
            holdout.append(cand)
        i -= 1
    holdout = sorted(holdout[:holdout_size])
    seen = [task_id for task_id in task_ids if task_id not in holdout]
    return seen, holdout


def merge_success_files(out: Path, task_ids: list[int], seen: list[int], holdout: list[int]) -> tuple[int, int]:
    n_train = 0
    n_holdout = 0
    with open(out / "train.jsonl", "w", encoding="utf-8") as train_f, open(out / "eval.jsonl", "w", encoding="utf-8") as eval_f, open(out / "holdout_train.jsonl", "w", encoding="utf-8") as hold_f:
        for task_id in task_ids:
            path = out / f"task_{task_id:04d}.jsonl"
            if not path.exists():
                continue
            target = train_f if task_id in seen else hold_f
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    target.write(line)
                    if task_id in seen:
                        n_train += 1
                    else:
                        eval_f.write(line)
                        n_holdout += 1
    return n_train, n_holdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect tau-bench SFT data via teacher rollout distillation.")
    parser.add_argument("--mode", choices=["teacher_rollout", "oracle"], default="teacher_rollout")
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "sft_collect_airline"))
    parser.add_argument("--tau-bench-path", default=None)
    parser.add_argument("--env-name", default="airline")
    parser.add_argument("--task-split", default="test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=50)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--use-user-sim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--user-strategy", default="llm")
    parser.add_argument("--user-model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--user-provider", default="openai")
    parser.add_argument("--user-base-url", default="http://localhost:8001/v1")
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    parser.add_argument("--teacher-base-url", default="http://localhost:8002/v1")
    parser.add_argument("--teacher-top-p", type=float, default=0.9)
    parser.add_argument("--teacher-max-tokens", type=int, default=768)
    parser.add_argument("--best-of-n", type=int, default=8)
    parser.add_argument("--temperatures", default="0.0,0.0,0.5,0.5,0.8,0.8,1.0,1.0")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--holdout-size", type=int, default=10)
    parser.add_argument("--contamination-char-limit", type=int, default=35000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    add_tau_bench_path(args.tau_bench_path)
    out = ensure_dir(args.output_dir)
    task_ids = parse_task_ids(args.task_ids, args.start_index, args.end_index)
    temperatures = parse_temperatures(args.temperatures, args.best_of_n)

    try:
        metas = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(collect_one_task, args, task_id, out, temperatures): task_id for task_id in task_ids}
            for future in as_completed(futures):
                meta = future.result()
                metas.append(meta)
                print(f"task={meta['task_id']} successes={meta['num_successes']} contaminated={meta['num_contaminated']}")
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency while importing tau-bench: {missing}. "
            "Install server dependencies with `pip install -r requirements-server.txt` and install tau-bench in editable mode."
        ) from exc

    metas.sort(key=lambda row: row["task_id"])
    seen, holdout = build_split(task_ids, args.holdout_size)
    n_train, n_holdout = merge_success_files(out, task_ids, seen, holdout)
    summary = {
        "source": args.mode,
        "env_name": args.env_name,
        "task_split": args.task_split,
        "teacher_model": args.teacher_model if args.mode == "teacher_rollout" else None,
        "user_sim_model": args.user_model if args.use_user_sim else None,
        "best_of_n": args.best_of_n if args.mode == "teacher_rollout" else 1,
        "num_tasks_attempted": len(metas),
        "num_tasks_with_success": sum(1 for row in metas if row["any_success"]),
        "task_coverage_rate": sum(1 for row in metas if row["any_success"]) / len(metas) if metas else 0.0,
        "total_success_trajectories": sum(row["num_successes"] for row in metas),
        "total_contaminated_trajectories": sum(row["num_contaminated"] for row in metas),
        "seen_task_ids": seen,
        "unseen_task_ids": holdout,
        "n_train_trajectories": n_train,
        "n_holdout_trajectories": n_holdout,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump({"seen_task_ids": seen, "unseen_task_ids": holdout, "total_tasks": len(task_ids)}, f, indent=2)
        f.write("\n")
    with open(out / "collect_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"output_dir": str(out), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
