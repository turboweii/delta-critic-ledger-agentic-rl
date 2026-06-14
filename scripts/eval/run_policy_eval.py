#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.config import ensure_dir, load_config
from delta_critic_ledger.evaluation import OpenAICompatPolicy, run_single_task, write_eval_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval" / "eval_airline_sft_4x4090.yaml"))
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

    tau_bench_path = ROOT.parent / "agentic-grpo-longhorizon-main" / "tau-bench"
    sys.path.insert(0, str(tau_bench_path))
    from tau_bench.envs import get_env

    env_cfg = cfg["env"]
    policy_cfg = cfg["policy"]
    results = []
    num_tasks = int(env_cfg.get("num_tasks", 50))
    num_samples = int(env_cfg.get("num_samples_per_task", 4))
    for task_id in range(num_tasks):
        for sample_id in range(num_samples):
            env = get_env(
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
            )
            result = run_single_task(env, policy, task_idx=task_id, max_turns=int(env_cfg.get("max_turns", 30)))
            result.sample_id = sample_id
            results.append(result)
            print(
                f"task={task_id} sample={sample_id} success={result.success} "
                f"reward={result.reward} tools={result.num_tool_calls} error={result.error}"
            )
    report = write_eval_report(results, out, cfg)
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
