from __future__ import annotations

import copy
import fnmatch
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .schemas import Action, DeltaStep

MISSING = "<MISSING>"


@dataclass(frozen=True)
class GoalField:
    path: tuple[Any, ...]
    initial_value: Any
    target_value: Any


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def values_equal(a: Any, b: Any) -> bool:
    return stable_json(a) == stable_json(b)


def flatten_leaves(value: Any, prefix: tuple[Any, ...] = ()) -> dict[tuple[Any, ...], Any]:
    leaves: dict[tuple[Any, ...], Any] = {}
    if isinstance(value, dict):
        if not value:
            leaves[prefix] = {}
        for key in sorted(value):
            leaves.update(flatten_leaves(value[key], prefix + (key,)))
    elif isinstance(value, list):
        if not value:
            leaves[prefix] = []
        for idx, item in enumerate(value):
            leaves.update(flatten_leaves(item, prefix + (idx,)))
    else:
        leaves[prefix] = value
    return leaves


def value_at(data: Any, path: tuple[Any, ...]) -> Any:
    cur = data
    for key in path:
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            return MISSING
    return cur


def path_to_str(path: tuple[Any, ...]) -> str:
    parts = []
    for item in path:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            if parts:
                parts.append(".")
            parts.append(str(item))
    return "".join(parts)


def path_matches(path: tuple[Any, ...], patterns: Iterable[str]) -> bool:
    text = path_to_str(path)
    normalized = ".".join("*" if isinstance(item, int) else str(item) for item in path)
    return any(fnmatch.fnmatch(text, pat) or fnmatch.fnmatch(normalized, pat) for pat in patterns)


def build_goal_fields(
    initial_data: dict[str, Any],
    target_data: dict[str, Any],
    include_paths: Iterable[str] | None = None,
    exclude_paths: Iterable[str] | None = None,
) -> list[GoalField]:
    before = flatten_leaves(initial_data)
    after = flatten_leaves(target_data)
    paths = sorted(set(before) | set(after), key=path_to_str)
    include = list(include_paths or [])
    exclude = list(exclude_paths or [])
    fields = []
    for path in paths:
        if include and not path_matches(path, include):
            continue
        if exclude and path_matches(path, exclude):
            continue
        initial_value = before.get(path, MISSING)
        target_value = after.get(path, MISSING)
        if not values_equal(initial_value, target_value):
            fields.append(GoalField(path, initial_value, target_value))
    return fields


class DeltaCritic:
    """State-delta credit assignment over oracle goal fields."""

    def __init__(
        self,
        initial_data: dict[str, Any],
        target_data: dict[str, Any],
        include_paths: Iterable[str] | None = None,
        exclude_paths: Iterable[str] | None = None,
    ):
        self.initial_data = copy.deepcopy(initial_data)
        self.target_data = copy.deepcopy(target_data)
        self.include_paths = list(include_paths or [])
        self.exclude_paths = list(exclude_paths or [])
        self.goal_fields = build_goal_fields(
            self.initial_data,
            self.target_data,
            include_paths=self.include_paths,
            exclude_paths=self.exclude_paths,
        )

    @classmethod
    def from_target_actions(
        cls,
        initial_data: dict[str, Any],
        target_actions: Iterable[Action],
        execute_tool: Callable[[dict[str, Any], Action], str],
        include_paths: Iterable[str] | None = None,
        exclude_paths: Iterable[str] | None = None,
    ) -> "DeltaCritic":
        target_data = copy.deepcopy(initial_data)
        for action in target_actions:
            execute_tool(target_data, action)
        return cls(initial_data, target_data, include_paths=include_paths, exclude_paths=exclude_paths)

    def phi(self, data: dict[str, Any]) -> float:
        if not self.goal_fields:
            return 1.0
        matched = 0
        for field in self.goal_fields:
            if values_equal(value_at(data, field.path), field.target_value):
                matched += 1
        return matched / len(self.goal_fields)

    def score_step(self, step_idx: int, tool: str, before: dict[str, Any], after: dict[str, Any]) -> DeltaStep:
        phi_before = self.phi(before)
        phi_after = self.phi(after)
        changed, regressed = [], []
        for field in self.goal_fields:
            was_good = values_equal(value_at(before, field.path), field.target_value)
            now_good = values_equal(value_at(after, field.path), field.target_value)
            if not was_good and now_good:
                changed.append(path_to_str(field.path))
            elif was_good and not now_good:
                regressed.append(path_to_str(field.path))
        return DeltaStep(
            step_idx=step_idx,
            tool=tool,
            phi_before=round(phi_before, 6),
            phi_after=round(phi_after, 6),
            delta_reward=round(phi_after - phi_before, 6),
            changed_goal_fields=changed,
            regressed_goal_fields=regressed,
        )

    def goal_field_names(self) -> list[str]:
        return [path_to_str(field.path) for field in self.goal_fields]
