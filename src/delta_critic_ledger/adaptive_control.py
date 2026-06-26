from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class RolloutProgress:
    progress_ratio: float
    positive_goal_fields: int
    goal_fields: int
    delta_reward_sum: float
    evidence_bonus_sum: float
    outcome_reward: float
    num_tool_calls: int
    bad_write_count: int
    ledger_write_count: int
    tool_error_rate: float = 0.0
    loop_detected: float = 0.0
    repeated_tool_args_rate: float = 0.0

    @property
    def bad_write_ratio(self) -> float:
        return max(self.tool_error_rate, self.repeated_tool_args_rate, self.loop_detected)


@dataclass(frozen=True)
class AdaptiveDecision:
    mode: str
    temperature: float
    top_p: float
    progress_ratio: float
    bad_write_ratio: float
    actor_kl_loss_coef: float | None = None
    algorithm_kl_coef: float | None = None

    def to_metrics(self) -> dict[str, float]:
        mode_codes = {"disabled": 0.0, "base": 1.0, "explore": 2.0, "repair": 3.0, "consolidate": 4.0}
        metrics = {
            "adaptive_control/mode_code": mode_codes.get(self.mode, -1.0),
            "adaptive_control/temperature": self.temperature,
            "adaptive_control/top_p": self.top_p,
            "adaptive_control/progress_ratio": self.progress_ratio,
            "adaptive_control/bad_write_ratio": self.bad_write_ratio,
        }
        if self.actor_kl_loss_coef is not None:
            metrics["adaptive_control/actor_kl_loss_coef"] = self.actor_kl_loss_coef
        if self.algorithm_kl_coef is not None:
            metrics["adaptive_control/algorithm_kl_coef"] = self.algorithm_kl_coef
        return metrics


@dataclass(frozen=True)
class AdaptiveEntropyConfig:
    enabled: bool = True
    base_temperature: float = 0.7
    min_temperature: float = 0.45
    max_temperature: float = 0.95
    base_top_p: float = 0.9
    min_top_p: float = 0.75
    max_top_p: float = 0.95
    explore_temperature_delta: float = 0.15
    consolidate_temperature_delta: float = -0.12
    repair_temperature_delta: float = -0.10
    explore_top_p_delta: float = 0.03
    consolidate_top_p_delta: float = -0.05
    repair_top_p_delta: float = -0.08
    min_tool_calls: int = 2
    progress_low: float = 0.15
    progress_high: float = 0.65
    bad_write_ratio_high: float = 0.35


@dataclass(frozen=True)
class AdaptiveKlConfig:
    enabled: bool = True
    actor_kl_loss_base: float = 0.01
    actor_kl_loss_min: float = 0.003
    actor_kl_loss_max: float = 0.03
    algorithm_kl_base: float = 0.01
    algorithm_kl_min: float = 0.003
    algorithm_kl_max: float = 0.03
    temperature_base: float = 0.7
    temperature_min: float = 0.45
    temperature_max: float = 0.95
    top_p_base: float = 0.9
    top_p_min: float = 0.75
    top_p_max: float = 0.95
    progress_low: float = 0.15
    progress_high: float = 0.65
    outcome_high: float = 0.6
    bad_write_ratio_high: float = 0.30


def _process_features(state: dict[str, Any]) -> dict[str, Any]:
    reward_components = state.get("reward_components") or {}
    if isinstance(reward_components, dict) and isinstance(reward_components.get("process_features"), dict):
        return reward_components["process_features"]
    lh_state = state.get("long_horizon_state") or {}
    tracker = lh_state.get("feature_tracker") if isinstance(lh_state, dict) else None
    if tracker is not None and hasattr(tracker, "to_dict"):
        return tracker.to_dict()
    features = state.get("process_features")
    return features if isinstance(features, dict) else {}


def summarize_rollout_state(state: dict[str, Any] | None) -> RolloutProgress:
    state = state or {}
    features = _process_features(state)
    outcome_reward = 1.0 if float(state.get("total_reward", state.get("outcome_reward", 0.0)) or 0.0) >= 1.0 else 0.0
    tool_calls = int(features.get("tool_calls", state.get("num_tool_calls", 0)) or 0)
    tool_error_rate = float(features.get("tool_error_rate", 0.0) or 0.0)
    repeat_rate = float(features.get("repeat_tool_args_rate", 0.0) or 0.0)
    loop_detected = float(features.get("loop_detected", 0.0) or 0.0)
    progress_ratio = 1.0 if outcome_reward >= 1.0 else 0.0
    if tool_calls > 0 and not loop_detected:
        progress_ratio = max(progress_ratio, min(0.5, float(features.get("read_tool_diversity", 0.0) or 0.0) * 0.5))
    return RolloutProgress(
        progress_ratio=_clamp(progress_ratio, 0.0, 1.0),
        positive_goal_fields=0,
        goal_fields=0,
        delta_reward_sum=0.0,
        evidence_bonus_sum=0.0,
        outcome_reward=outcome_reward,
        num_tool_calls=tool_calls,
        bad_write_count=int(features.get("tool_errors", 0) or 0),
        ledger_write_count=tool_calls,
        tool_error_rate=tool_error_rate,
        loop_detected=loop_detected,
        repeated_tool_args_rate=repeat_rate,
    )


class AdaptiveEntropyController:
    def __init__(self, config: AdaptiveEntropyConfig | None = None):
        self.config = config or AdaptiveEntropyConfig()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "AdaptiveEntropyController":
        env_enabled = os.environ.get("DCL_ADAPTIVE_ENTROPY")
        values = {key: value for key, value in dict(config or {}).items() if key in AdaptiveEntropyConfig.__dataclass_fields__}
        if env_enabled is not None:
            values["enabled"] = env_enabled not in {"0", "false", "False"}
        return cls(AdaptiveEntropyConfig(**values))

    def adjust_sampling_params(self, sampling_params: dict[str, Any], state: dict[str, Any] | None) -> tuple[dict[str, Any], AdaptiveDecision]:
        cfg = self.config
        params = dict(sampling_params)
        base_temperature = float(params.get("temperature", cfg.base_temperature))
        base_top_p = float(params.get("top_p", cfg.base_top_p))
        progress = summarize_rollout_state(state)
        mode = "base"
        temperature = base_temperature
        top_p = base_top_p
        if not cfg.enabled:
            mode = "disabled"
        elif progress.num_tool_calls >= cfg.min_tool_calls and progress.bad_write_ratio >= cfg.bad_write_ratio_high:
            mode = "repair"
            temperature = base_temperature + cfg.repair_temperature_delta
            top_p = base_top_p + cfg.repair_top_p_delta
        elif progress.outcome_reward >= 1.0 or progress.progress_ratio >= cfg.progress_high:
            mode = "consolidate"
            temperature = base_temperature + cfg.consolidate_temperature_delta
            top_p = base_top_p + cfg.consolidate_top_p_delta
        elif progress.num_tool_calls >= cfg.min_tool_calls and progress.progress_ratio <= cfg.progress_low:
            mode = "explore"
            temperature = base_temperature + cfg.explore_temperature_delta
            top_p = base_top_p + cfg.explore_top_p_delta
        temperature = _clamp(temperature, cfg.min_temperature, cfg.max_temperature)
        top_p = _clamp(top_p, cfg.min_top_p, cfg.max_top_p)
        params["temperature"] = temperature
        params["top_p"] = top_p
        return params, AdaptiveDecision(mode, temperature, top_p, progress.progress_ratio, progress.bad_write_ratio)


def summarize_trace_payloads(payloads: list[dict[str, Any]]) -> RolloutProgress:
    if not payloads:
        return RolloutProgress(0.0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0)
    items = [summarize_rollout_state(payload) for payload in payloads]
    count = len(items)
    return RolloutProgress(
        progress_ratio=sum(item.progress_ratio for item in items) / count,
        positive_goal_fields=0,
        goal_fields=0,
        delta_reward_sum=0.0,
        evidence_bonus_sum=0.0,
        outcome_reward=sum(item.outcome_reward for item in items) / count,
        num_tool_calls=round(sum(item.num_tool_calls for item in items) / count),
        bad_write_count=sum(item.bad_write_count for item in items),
        ledger_write_count=sum(item.ledger_write_count for item in items),
        tool_error_rate=sum(item.tool_error_rate for item in items) / count,
        loop_detected=sum(item.loop_detected for item in items) / count,
        repeated_tool_args_rate=sum(item.repeated_tool_args_rate for item in items) / count,
    )


def load_recent_trace_payloads(trace_dir: str | Path, window_size: int = 64) -> list[dict[str, Any]]:
    root = Path(trace_dir)
    if not root.is_dir():
        return []
    files = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    payloads = []
    for path in files[:window_size]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            payloads.append(data)
    return payloads


class AdaptiveKlEntropyController:
    def __init__(self, config: AdaptiveKlConfig | None = None):
        self.config = config or AdaptiveKlConfig()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "AdaptiveKlEntropyController":
        values = {key: value for key, value in dict(config or {}).items() if key in AdaptiveKlConfig.__dataclass_fields__}
        return cls(AdaptiveKlConfig(**values))

    def recommend(self, progress: RolloutProgress) -> AdaptiveDecision:
        cfg = self.config
        mode = "base"
        kl = cfg.actor_kl_loss_base
        algorithm_kl = cfg.algorithm_kl_base
        temperature = cfg.temperature_base
        top_p = cfg.top_p_base
        if not cfg.enabled:
            mode = "disabled"
        elif progress.bad_write_ratio >= cfg.bad_write_ratio_high:
            mode = "repair"
            kl = min(cfg.actor_kl_loss_max, cfg.actor_kl_loss_base * 1.8)
            algorithm_kl = min(cfg.algorithm_kl_max, cfg.algorithm_kl_base * 1.8)
            temperature = max(cfg.temperature_min, cfg.temperature_base - 0.12)
            top_p = max(cfg.top_p_min, cfg.top_p_base - 0.08)
        elif progress.outcome_reward >= cfg.outcome_high or progress.progress_ratio >= cfg.progress_high:
            mode = "consolidate"
            kl = min(cfg.actor_kl_loss_max, cfg.actor_kl_loss_base * 1.4)
            algorithm_kl = min(cfg.algorithm_kl_max, cfg.algorithm_kl_base * 1.4)
            temperature = max(cfg.temperature_min, cfg.temperature_base - 0.10)
            top_p = max(cfg.top_p_min, cfg.top_p_base - 0.04)
        elif progress.num_tool_calls >= 2 and progress.progress_ratio <= cfg.progress_low:
            mode = "explore"
            kl = max(cfg.actor_kl_loss_min, cfg.actor_kl_loss_base * 0.5)
            algorithm_kl = max(cfg.algorithm_kl_min, cfg.algorithm_kl_base * 0.5)
            temperature = min(cfg.temperature_max, cfg.temperature_base + 0.15)
            top_p = min(cfg.top_p_max, cfg.top_p_base + 0.03)
        return AdaptiveDecision(mode, temperature, top_p, progress.progress_ratio, progress.bad_write_ratio, kl, algorithm_kl)


def decision_as_dict(decision: AdaptiveDecision) -> dict[str, Any]:
    return asdict(decision)
