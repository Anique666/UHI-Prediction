"""Run metadata capture for reproducibility.

Records the seed, resolved configuration, data paths, package versions and GPU
information alongside every experiment so a run can be reproduced and audited.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

__all__ = ["collect_environment", "make_run_id", "write_json", "read_json"]


def collect_environment() -> dict:
    """Collect interpreter, package and GPU metadata."""
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
    }
    for name in ("numpy", "rasterio", "pandas", "yaml"):
        try:
            mod = __import__(name)
            env[f"{name}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            env[f"{name}_version"] = None
    if torch.cuda.is_available():
        env["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        props = torch.cuda.get_device_properties(0)
        env["gpu_total_memory_mib"] = round(props.total_memory / 1024**2, 1)
    return env


def make_run_id(prefix: str = "run") -> str:
    """A filesystem-safe, sortable run identifier."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_json(path: str | Path, obj) -> None:
    """Write ``obj`` as pretty JSON (creates parent directories)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str | Path):
    """Read a JSON file."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)