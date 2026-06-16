#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required: {exc}")

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from delta_critic_ledger.adaptive_control import (  # noqa: E402
    AdaptiveKlEntropyController,
    decision_as_dict,
    load_recent_trace_payloads,
    summarize_trace_payloads,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a YAML mapping.")
    return data


def hydra_overrides(decision) -> list[str]:
    return [
        f"actor_rollout_ref.actor.kl_loss_coef={decision.actor_kl_loss_coef:.6g}",
        f"algorithm.kl_ctrl.kl_coef={decision.algorithm_kl_coef:.6g}",
        f"actor_rollout_ref.rollout.temperature={decision.temperature:.6g}",
        f"actor_rollout_ref.rollout.top_p={decision.top_p:.6g}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend adaptive KL/sampling overrides from Ledger traces.")
    parser.add_argument("--trace-dir", default="outputs/grpo_delta_traces")
    parser.add_argument("--config", default="configs/train/grpo/adaptive_kl_entropy.yaml")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--format", choices=["json", "summary", "hydra-lines"], default="summary")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    trace_dir = Path(args.trace_dir)
    if not trace_dir.is_absolute():
        trace_dir = ROOT / trace_dir

    controller = AdaptiveKlEntropyController.from_config(load_yaml(config_path))
    payloads = load_recent_trace_payloads(trace_dir, window_size=args.window_size)
    progress = summarize_trace_payloads(payloads)
    decision = controller.recommend(progress)
    result = {
        "trace_dir": str(trace_dir),
        "trace_count": len(payloads),
        "progress": progress.__dict__,
        "decision": decision_as_dict(decision),
        "hydra_overrides": hydra_overrides(decision),
    }

    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.format == "hydra-lines":
        for item in result["hydra_overrides"]:
            print(item)
    else:
        print(
            "adaptive_kl_entropy: "
            f"mode={decision.mode} "
            f"traces={len(payloads)} "
            f"progress={progress.progress_ratio:.3f} "
            f"bad_write={progress.bad_write_ratio:.3f} "
            f"outcome={progress.outcome_reward:.3f} "
            f"actor_kl={decision.actor_kl_loss_coef:.6g} "
            f"kl_coef={decision.algorithm_kl_coef:.6g} "
            f"temperature={decision.temperature:.3f} "
            f"top_p={decision.top_p:.3f}"
        )


if __name__ == "__main__":
    main()
