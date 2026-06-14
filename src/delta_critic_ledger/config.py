from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_defaults(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")

    merged: dict[str, Any] = {}
    for item in data.get("defaults", []):
        if not isinstance(item, str) or item == "_self_" or item.startswith("ppo_trainer"):
            continue
        default_path = path.parent / f"{item}.yaml"
        if default_path.exists():
            merged = _deep_merge(merged, _load_yaml_with_defaults(default_path))
    return _deep_merge(merged, data)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        return _load_yaml_with_defaults(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
