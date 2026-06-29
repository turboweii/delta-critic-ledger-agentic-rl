#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.config import ensure_dir, load_config
from delta_critic_ledger.evaluation import OpenAICompatPolicy, run_single_task, write_eval_report


def add_tau_bench_path() -> None:
    candidates = []
    if os.environ.get("TAU_BENCH_PATH"):
        candidates.append(Path(os.environ["TAU_BENCH_PATH"]))
    candidates.append(ROOT.parent / "tau-bench")
    for candidate in candidates:
        if (candidate / "tau_bench").is_dir():
            sys.path.insert(0, str(candidate))
            return


def log_eval_to_wandb(report: dict, cfg: dict, config_path: str) -> None:
    wandb_cfg = cfg.get("wandb", {})
    project = wandb_cfg.get("project", "delta-critic-ledger-agentic-rl")
    run_name = wandb_cfg.get("run_name") or Path(cfg["output"]["dir"]).name
    group = wandb_cfg.get("group", "eval")
    job_type = wandb_cfg.get("job_type", "eval")

    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("wandb is required for eval logging. Install it in the training environment.") from exc

    run = wandb.init(
        project=project,
        name=run_name,
        group=group,
        job_type=job_type,
        config={"eval_config_path": config_path, **cfg},
    )
    metrics = {
        "eval/success_rate": report.get("success_rate"),
        "eval/pass_at_1": report.get("pass_at_1"),
        "eval/error_rate": report.get("error_rate"),
        "eval/num_samples": report.get("num_samples"),
        "eval/num_tasks": report.get("num_tasks"),
        "eval/avg_tool_calls": report.get("avg_tool_calls"),
    }
    for split_name, split in (report.get("by_split") or {}).items():
        for key, value in split.items():
            metrics[f"eval/{split_name}/{key}"] = value
    wandb.log(metrics)

    per_task = report.get("per_task") or []
    if per_task:
        columns = sorted({key for row in per_task for key in row.keys()})
        table = wandb.Table(columns=columns)
        for row in per_task:
            table.add_data(*[row.get(column) for column in columns])
        wandb.log({"eval/per_task": table})

    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval" / "eval_airline_sft_2xa800_72b_user.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(ROOT / cfg["output"]["dir"])

    if args.dry_run:
        manifest = {
            "status": "dry_run",
            "note": "Starts real eval only when tau-bench, vLLM servers, and policy endpoint are available.",
            "config": cfg,
        }
        with open(out / "eval_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote dry-run eval manifest to {out}")
        return

    add_tau_bench_path()
    from tau_bench.envs import get_env

    env_cfg = cfg["env"]
    policy_cfg = cfg["policy"]
    results = []
    num_tasks = int(env_cfg.get("num_tasks", 50))
    num_samples = int(env_cfg.get("num_samples_per_task", 4))
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
            policy = OpenAICompatPolicy(
                model_name=policy_cfg["served_model_name"],
                base_url=policy_cfg["base_url"],
                temperature=float(policy_cfg.get("temperature", 0.7)),
                top_p=float(policy_cfg.get("top_p", 0.9)),
                max_tokens=int(policy_cfg.get("max_tokens", 1024)),
                max_context_chars=policy_cfg.get("max_context_chars"),
            )
            result = run_single_task(env, policy, task_idx=task_id, max_turns=int(env_cfg.get("max_turns", 30)))
            result.sample_id = sample_id
            results.append(result)
            print(
                f"task={task_id} sample={sample_id} success={result.success} "
                f"reward={result.reward} tools={result.num_tool_calls} error={result.error}"
            )
    report = write_eval_report(results, out, cfg)
    log_eval_to_wandb(report, cfg, args.config)
    print(
        json.dumps(
            {
                "success_rate": report["success_rate"],
                "error_rate": report["error_rate"],
                "num_samples": report["num_samples"],
                "output": str(out / "eval_report.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
