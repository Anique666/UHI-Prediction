"""Normalization statistics handling.

The statistics live in the frozen ``normalization_stats.json``. This module is
the single place that reads them and converts between normalized space and
physical units, so no literals are duplicated across the codebase.

Convention (matches the dataset contract):

    x_normalized = (x_raw - train_mean) / train_std
    x_raw        = x_normalized * train_std + train_mean
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "NormalizationStats",
    "load_normalization_stats",
    "normalized_to_celsius",
    "celsius_to_normalized",
]


@dataclass(frozen=True)
class NormalizationStats:
    """Per-channel TRAIN normalization statistics."""

    channel_names: list[str]
    fill_values: list[float]
    means: list[float]
    stds: list[float]

    def index(self, channel: str) -> int:
        try:
            return self.channel_names.index(channel)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown channel {channel!r}") from exc


def load_normalization_stats(dataset_root: str | Path) -> NormalizationStats:
    """Read ``normalization_stats.json`` from the frozen dataset root."""
    path = Path(dataset_root) / "normalization_stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"normalization_stats.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    required = ["channel_names", "fill_values", "train_means", "train_stds"]
    missing = [k for k in required if k not in doc]
    if missing:
        raise KeyError(f"normalization_stats.json missing keys: {missing}")

    stats = NormalizationStats(
        channel_names=list(doc["channel_names"]),
        fill_values=list(doc["fill_values"]),
        means=list(doc["train_means"]),
        stds=list(doc["train_stds"]),
    )
    lens = {
        len(stats.channel_names),
        len(stats.fill_values),
        len(stats.means),
        len(stats.stds),
    }
    if len(lens) != 1:
        raise ValueError("normalization_stats arrays have mismatched lengths")
    if any(s == 0 for s in stats.stds):
        raise ValueError("normalization_stats contains a zero std")
    return stats


def normalized_to_celsius(
    normalized,
    stats: NormalizationStats,
    lst_channel: str = "LST_Celsius",
):
    """Convert normalized LST to degrees Celsius using the LST channel stats.

    Works with numpy arrays or torch tensors.
    """
    i = stats.index(lst_channel)
    return normalized * stats.stds[i] + stats.means[i]


def celsius_to_normalized(
    celsius,
    stats: NormalizationStats,
    lst_channel: str = "LST_Celsius",
):
    """Convert degrees Celsius back to normalized LST."""
    i = stats.index(lst_channel)
    return (celsius - stats.means[i]) / stats.stds[i]