#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


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
    parser.add_argument("--tau-bench-path", default=str(ROOT.parent / "agentic-grpo-longhorizon-main" / "tau-bench"))
    args = parser.parse_args()
    sys.path.insert(0, args.tau_bench_path)

    ref_tool_config = ROOT.parent / "agentic-grpo-longhorizon-main" / "agentic-grpo-longhorizon" / "configs" / "tool_config" / "tau_bench_airline_tools.yaml"
    try:
        from tau_bench.envs import get_env
        import yaml

        env = get_env(
            env_name="airline",
            user_strategy="llm",
            user_model="dummy",
            user_provider="openai",
            task_split="test",
            task_index=0,
        )
        schemas = {tool["function"]["name"]: tool for tool in env.tools_info}
    except Exception as exc:
        print(f"[WARN] Could not import tau-bench tools: {exc}")
        try:
            import yaml

            with open(ref_tool_config, "r", encoding="utf-8") as f:
                ref = yaml.safe_load(f)
            for item in ref["tools"]:
                name = item["tool_schema"]["function"]["name"]
                item["class_name"] = f"delta_critic_ledger.verl_integration.tools.TauBench_{name}_Tool"
                item["config"] = {"type": "native"}
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump(ref, sort_keys=False, allow_unicode=True), encoding="utf-8")
            print(f"Wrote {out} from reference tool schema fallback")
            return
        except Exception as fallback_exc:
            print(f"[WARN] Could not read reference tool schema, writing class-only fallback: {fallback_exc}")
        schemas = {}
        yaml = None

    out = Path(args.output)
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
    if yaml is not None:
        out.write_text(yaml.safe_dump({"tools": tools}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        lines = ["tools:"]
        for item in tools:
            lines.extend([f"  - class_name: {item['class_name']}", "    config:", "      type: native"])
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
