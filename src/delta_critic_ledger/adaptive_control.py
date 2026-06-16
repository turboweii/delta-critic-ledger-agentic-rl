from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BAD_WRITE_STATUSES = {"ungrounded_write", "conflicting_write"}


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


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

    @property
    def bad_write_ratio(self) -> float:
        if self.ledger_write_count <= 0:
            return 0.0
        return self.bad_write_count / self.ledger_write_count


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


def summarize_rollout_state(state: dict[str, Any] | None) -> RolloutProgress:
    state = state or {}
    reward_state = _as_mapping(state.get("delta_reward_state"))
    goal_fields = list(reward_state.get("goal_fields") or [])
    delta_steps = list(state.get("delta_steps") or [])
    ledger_steps = list(state.get("ledger_steps") or [])

    positive_fields: set[str] = set()
    for raw_step in delta_steps:
        step = _as_mapping(raw_step)
        if float(step.get("delta_reward", 0.0) or 0.0) <= 0.0:
            continue
        for field in step.get("changed_goal_fields") or []:
            positive_fields.add(str(field))

    ledger_write_count = 0
    bad_write_count = 0
    for raw_step in ledger_steps:
        step = _as_mapping(raw_step)
        status = str(step.get("write_grounding_status", ""))
        if status:
            ledger_write_count += 1
        if status in BAD_WRITE_STATUSES:
            bad_write_count += 1

    goal_count = len(goal_fields)
    if goal_count > 0:
        progress_ratio = len(positive_fields) / goal_count
    else:
        progress_ratio = max(0.0, float(state.get("delta_reward_sum", 0.0) or 0.0))

    outcome_reward = 1.0 if float(state.get("total_reward", 0.0) or 0.0) >= 1.0 else 0.0
    return RolloutProgress(
        progress_ratio=_clamp(progress_ratio, 0.0, 1.0),
        positive_goal_fields=len(positive_fields),
        goal_fields=goal_count,
        delta_reward_sum=float(state.get("delta_reward_sum", 0.0) or 0.0),
        evidence_bonus_sum=float(state.get("evidence_bonus_sum", 0.0) or 0.0),
        outcome_reward=outcome_reward,
        num_tool_calls=int(state.get("num_tool_calls", 0) or 0),
        bad_write_count=bad_write_count,
        ledger_write_count=ledger_write_count,
    )


class AdaptiveEntropyController:
    def __init__(self, config: AdaptiveEntropyConfig | None = None):
        self.config = config or AdaptiveEntropyConfig()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "AdaptiveEntropyController":
        if not config:
            return cls()
        known = {field.name for field in AdaptiveEntropyConfig.__dataclass_fields__.values()}
        values = {key: value for key, value in dict(config).items() if key in known}
        return cls(AdaptiveEntropyConfig(**values))

    def adjust_sampling_params(
        self,
        sampling_params: dict[str, Any],
        state: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], AdaptiveDecision]:
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
        return params, AdaptiveDecision(
            mode=mode,
            temperature=temperature,
            top_p=top_p,
            progress_ratio=progress.progress_ratio,
            bad_write_ratio=progress.bad_write_ratio,
        )


def summarize_trace_payloads(payloads: list[dict[str, Any]]) -> RolloutProgress:
    if not payloads:
        return RolloutProgress(0.0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0)

    progress_items = [summarize_rollout_state(payload) for payload in payloads]
    count = len(progress_items)
    return RolloutProgress(
        progress_ratio=sum(item.progress_ratio for item in progress_items) / count,
        positive_goal_fields=sum(item.positive_goal_fields for item in progress_items),
        goal_fields=sum(item.goal_fields for item in progress_items),
        delta_reward_sum=sum(item.delta_reward_sum for item in progress_items) / count,
        evidence_bonus_sum=sum(item.evidence_bonus_sum for item in progress_items) / count,
        outcome_reward=sum(item.outcome_reward for item in progress_items) / count,
        num_tool_calls=round(sum(item.num_tool_calls for item in progress_items) / count),
        bad_write_count=sum(item.bad_write_count for item in progress_items),
        ledger_write_count=sum(item.ledger_write_count for item in progress_items),
    )


def load_recent_trace_payloads(trace_dir: str | Path, window_size: int = 64) -> list[dict[str, Any]]:
    root = Path(trace_dir)
    if not root.is_dir():
        return []
    files = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    payloads: list[dict[str, Any]] = []
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
        if not config:
            return cls()
        known = {field.name for field in AdaptiveKlConfig.__dataclass_fields__.values()}
        values = {key: value for key, value in dict(config).items() if key in known}
        return cls(AdaptiveKlConfig(**values))

    def recommend(self, progress: RolloutProgress) -> AdaptiveDecision:
        cfg = self.config
        actor_kl = cfg.actor_kl_loss_base
        algorithm_kl = cfg.algorithm_kl_base
        temperature = cfg.temperature_base
        top_p = cfg.top_p_base
        mode = "base"

        if not cfg.enabled:
            mode = "disabled"
        elif progress.ledger_write_count > 0 and progress.bad_write_ratio >= cfg.bad_write_ratio_high:
            mode = "repair"
            actor_kl *= 1.5
            algorithm_kl *= 1.5
            temperature -= 0.10
            top_p -= 0.08
        elif progress.outcome_reward >= cfg.outcome_high or progress.progress_ratio >= cfg.progress_high:
            mode = "consolidate"
            actor_kl *= 1.25
            algorithm_kl *= 1.25
            temperature -= 0.08
            top_p -= 0.04
        elif (
            progress.num_tool_calls > 0
            and progress.progress_ratio <= cfg.progress_low
            and progress.outcome_reward <= 0.05
        ):
            mode = "explore"
            actor_kl *= 0.65
            algorithm_kl *= 0.65
            temperature += 0.12
            top_p += 0.03

        return AdaptiveDecision(
            mode=mode,
            temperature=_clamp(temperature, cfg.temperature_min, cfg.temperature_max),
            top_p=_clamp(top_p, cfg.top_p_min, cfg.top_p_max),
            progress_ratio=progress.progress_ratio,
            bad_write_ratio=progress.bad_write_ratio,
            actor_kl_loss_coef=_clamp(actor_kl, cfg.actor_kl_loss_min, cfg.actor_kl_loss_max),
            algorithm_kl_coef=_clamp(algorithm_kl, cfg.algorithm_kl_min, cfg.algorithm_kl_max),
        )


def decision_as_dict(decision: AdaptiveDecision) -> dict[str, Any]:
    return asdict(decision)
