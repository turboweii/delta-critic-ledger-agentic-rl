#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from delta_critic_ledger.tau_compat import create_tau_env


AIRLINE_TOOLS = [
    "book_reservation",
    "calculate",
    "cancel_reservation",
    "get_reservation_details",
    "get_user_details",
    "list_all_airports",
    "search_direct_flight",
    "search_onestop_flight",
    "send_certificate",
    "think",
    "transfer_to_human_agents",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "configs" / "tool_config" / "tau_bench_airline_tools.yaml"))
    parser.add_argument("--tau-bench-path", default=None)
    args = parser.parse_args()

    candidates = []
    if args.tau_bench_path:
        candidates.append(Path(args.tau_bench_path))
    candidates.append(ROOT.parent / "tau-bench")
    for candidate in candidates:
        if (candidate / "tau_bench").is_dir():
            sys.path.insert(0, str(candidate))
            break

    out = Path(args.output)
    try:
        from tau_bench.envs import get_env

        env = create_tau_env(
            get_env,
            env_name="airline",
            user_strategy="human",
            user_model="unused",
            user_provider="openai",
            task_split="test",
            task_index=0,
        )
        schemas = {tool["function"]["name"]: tool for tool in env.tools_info}
    except Exception as exc:
        print(f"[WARN] Could not import tau-bench tools: {exc}")
        if not out.exists():
            raise SystemExit(
                "tau-bench schema generation failed and no committed tool config exists. "
                "Pass --tau-bench-path /path/to/tau-bench."
            ) from exc
        existing = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
        existing_tools = existing.get("tools", [])
        schemas = {
            item["tool_schema"]["function"]["name"]: item["tool_schema"]
            for item in existing_tools
            if item.get("tool_schema", {}).get("function", {}).get("name")
        }
        missing = sorted(set(AIRLINE_TOOLS) - set(schemas))
        if missing:
            raise SystemExit(
                "Existing tool config is incomplete after tau-bench import failed; "
                f"missing schemas: {', '.join(missing)}"
            ) from exc
        print(f"[WARN] Preserving complete schemas already present in {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tools = []
    for name in AIRLINE_TOOLS:
        item = {
            "class_name": f"delta_critic_ledger.verl_integration.tools.TauBench_{name}_Tool",
            "config": {"type": "native"},
        }
        if name in schemas:
            item["tool_schema"] = schemas[name]
        tools.append(item)
    out.write_text(yaml.safe_dump({"tools": tools}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
