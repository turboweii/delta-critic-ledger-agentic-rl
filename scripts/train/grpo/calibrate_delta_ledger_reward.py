#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BAD_WRITE_STATUSES = {"ungrounded_write", "conflicting_write"}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def load_traces(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not trace_dir.is_dir():
        return rows
    for path in sorted(trace_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            data["_path"] = str(path)
            rows.append(data)
    return rows


def point_biserial(feature: list[int], outcome: list[float]) -> float:
    n = len(feature)
    if n == 0 or n != len(outcome):
        return 0.0
    positives = [y for x, y in zip(feature, outcome) if x]
    negatives = [y for x, y in zip(feature, outcome) if not x]
    if not positives or not negatives:
        return 0.0
    mean_pos = sum(positives) / len(positives)
    mean_neg = sum(negatives) / len(negatives)
    mean_y = sum(outcome) / n
    var_y = sum((y - mean_y) ** 2 for y in outcome) / n
    if var_y <= 0.0:
        return 0.0
    p = len(positives) / n
    q = 1.0 - p
    return (mean_pos - mean_neg) * math.sqrt(p * q) / math.sqrt(var_y)


def trace_features(trace: dict[str, Any]) -> dict[str, int]:
    delta_steps = trace.get("delta_steps") or []
    ledger_steps = trace.get("ledger_steps") or []
    action_history = trace.get("action_history") or []
    statuses = [str(step.get("write_grounding_status", "")) for step in ledger_steps if isinstance(step, dict)]
    deltas = [as_float(step.get("delta_reward")) for step in delta_steps if isinstance(step, dict)]
    return {
        "has_positive_delta": int(any(value > 0 for value in deltas)),
        "has_negative_delta": int(any(value < 0 for value in deltas)),
        "has_grounded_write": int("evidence_grounded_write" in statuses),
        "has_bad_write": int(any(status in BAD_WRITE_STATUSES for status in statuses)),
        "has_ungrounded_write": int("ungrounded_write" in statuses),
        "has_conflicting_write": int("conflicting_write" in statuses),
        "has_tool_error": int(any(bool(row.get("is_error")) for row in action_history if isinstance(row, dict))),
        "has_tool_loop": int(_has_tool_loop(action_history)),
    }


def _has_tool_loop(action_history: list[Any]) -> bool:
    counts: Counter[str] = Counter()
    for row in action_history:
        if not isinstance(row, dict):
            continue
        key = json.dumps({"tool": row.get("tool"), "parameters": row.get("parameters")}, sort_keys=True)
        counts[key] += 1
    return any(count >= 3 for count in counts.values())


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [as_float(row.get("outcome_reward")) for row in traces]
    combined = [as_float(row.get("combined_reward")) for row in traces]
    dense = [as_float(row.get("dense_reward")) for row in traces]
    raw_delta = [as_float(row.get("raw_delta_reward_sum", row.get("delta_reward_sum"))) for row in traces]
    raw_evidence = [as_float(row.get("raw_evidence_bonus_sum", row.get("evidence_bonus_sum"))) for row in traces]
    features_by_name: dict[str, list[int]] = defaultdict(list)
    for row in traces:
        for name, value in trace_features(row).items():
            features_by_name[name].append(value)

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}

    feature_rows = []
    for name, values in sorted(features_by_name.items()):
        count = sum(values)
        feature_rows.append({
            "feature": name,
            "count": count,
            "rate": count / len(values) if values else 0.0,
            "point_biserial_with_success": point_biserial(values, outcomes),
        })

    warnings = []
    by_feature = {row["feature"]: row for row in feature_rows}
    if by_feature.get("has_positive_delta", {}).get("point_biserial_with_success", 0.0) < 0.05 and traces:
        warnings.append("positive_delta is weakly correlated with success; reduce beta_delta or inspect goal fields.")
    if by_feature.get("has_grounded_write", {}).get("point_biserial_with_success", 0.0) < 0.0 and traces:
        warnings.append("grounded_write is negatively correlated with success; evidence reward may be miscalibrated.")
    if by_feature.get("has_bad_write", {}).get("point_biserial_with_success", 0.0) > 0.0 and traces:
        warnings.append("bad_write appears in successful traces; verify ledger false positives before increasing penalties.")
    if max(dense or [0.0]) > 0.5:
        warnings.append("dense reward exceeds 0.5; it may overpower terminal reward under GRPO.")

    return {
        "num_traces": len(traces),
        "success_rate": sum(outcomes) / len(outcomes) if outcomes else 0.0,
        "combined_reward": stats(combined),
        "dense_reward": stats(dense),
        "raw_delta_reward_sum": stats(raw_delta),
        "raw_evidence_bonus_sum": stats(raw_evidence),
        "feature_correlations": feature_rows,
        "warnings": warnings,
        "recommendation": {
            "keep_delta_ledger": not warnings or len(warnings) <= 1,
            "default_beta_delta": 0.3,
            "default_beta_evidence": 0.1,
            "note": "Use this after each real rollout batch; dense reward tiers should be empirically discriminative, not only intuitive.",
        },
    }


def log_to_wandb(summary: dict[str, Any], project: str, run_name: str) -> None:
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("wandb is required for --wandb") from exc
    run = wandb.init(project=project, name=run_name, group="reward-calibration", job_type="analysis")
    metrics = {
        "calibration/num_traces": summary["num_traces"],
        "calibration/success_rate": summary["success_rate"],
        "calibration/dense_reward_mean": summary["dense_reward"]["mean"],
        "calibration/raw_delta_mean": summary["raw_delta_reward_sum"]["mean"],
        "calibration/raw_evidence_mean": summary["raw_evidence_bonus_sum"]["mean"],
        "calibration/warning_count": len(summary["warnings"]),
    }
    for row in summary["feature_correlations"]:
        metrics[f"calibration/corr/{row['feature']}"] = row["point_biserial_with_success"]
    wandb.log(metrics)
    table = wandb.Table(columns=["feature", "count", "rate", "point_biserial_with_success"])
    for row in summary["feature_correlations"]:
        table.add_data(row["feature"], row["count"], row["rate"], row["point_biserial_with_success"])
    wandb.log({"calibration/feature_correlations": table})
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether Delta/Ledger dense rewards are empirically aligned with success.")
    parser.add_argument("--trace-dir", default="outputs/grpo_delta_traces_2xa800")
    parser.add_argument("--output", default="outputs/reward_calibration/delta_ledger_summary.json")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="delta-critic-ledger-agentic-rl")
    parser.add_argument("--wandb-run-name", default="delta-ledger-reward-calibration")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    if not trace_dir.is_absolute():
        trace_dir = ROOT / trace_dir
    traces = load_traces(trace_dir)
    summary = summarize(traces)
    summary["trace_dir"] = str(trace_dir)
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.wandb:
        log_to_wandb(summary, args.wandb_project, args.wandb_run_name)


if __name__ == "__main__":
    main()
