"""Configuration loading and path resolution for the UHI baseline.

The configuration is a plain YAML file (see ``configs/baseline.yaml``). Paths
may be absolute or relative to ``project_root``. Nothing here reads or writes
the frozen preprocessing artifacts; it only resolves where things live.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "resolve_path", "get_paths", "deep_update"]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg)}")
    return cfg


def resolve_path(value: str | Path, project_root: str | Path) -> Path:
    """Resolve ``value`` against ``project_root`` unless it is already absolute."""
    p = Path(value)
    if p.is_absolute():
        return p
    return (Path(project_root) / p).resolve()


def get_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Return the ``paths`` section with every entry resolved to a ``Path``.

    Unlike the output directories, absolute data paths are left as written so
    that a missing frozen dataset fails loudly rather than silently resolving
    somewhere unexpected.
    """
    project_root = cfg["project_root"]
    raw = cfg.get("paths", {})
    resolved: dict[str, Path] = {}
    for key, value in raw.items():
        resolved[key] = resolve_path(value, project_root)
    return resolved


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns ``base``)."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base
