from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def validate_config(cfg: dict[str, Any]) -> None:
    required = [
        "ground_filter",
        "clustering",
        "pairing",
        "histogram_init",
        "icp",
        "association",
        "output",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    if cfg["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if cfg["clustering"]["max_clusters"] <= 0:
        raise ValueError("clustering.max_clusters must be positive")
