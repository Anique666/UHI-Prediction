"""Persistence baseline for next-month LST forecasting.

    LST_hat(t+1, x, y) = LST_observed(t, x, y)

i.e. the most recent observed LST is used unchanged as the prediction of the
next month. The most recent input frame is index ``-1`` and the LST channel is
channel ``0`` in the frozen tensor contract.

Important: the frozen ``UHIDataset`` fills missing input LST with the TRAIN
mean before normalization, so a filled latest-input pixel would masquerade as a
genuine persistence prediction. This module reads the *raw* latest input month
window to recover a proper input-validity mask. Those invalid input pixels are
excluded from the persistence evaluation and the exclusion is reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from utils.normalization import NormalizationStats

from .metrics import MetricAccumulator

__all__ = ["last_frame_lst_normalized", "evaluate_persistence", "PersistenceResult"]


def last_frame_lst_normalized(X: torch.Tensor) -> torch.Tensor:
    """Return the latest input LST frame in normalized space.

    ``X`` is ``[B, T, C, H, W]``; returns ``[B, 1, H, W]``.
    """
    if X.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got {tuple(X.shape)}")
    return X[:, -1, 0:1]


@dataclass
class PersistenceResult:
    """Aggregate persistence metrics plus exclusion bookkeeping."""

    metrics: dict
    n_samples: int
    target_valid_pixels: int
    combined_valid_pixels: int
    input_invalid_excluded_pixels: int
    samples_with_input_invalid_pixels: int


def evaluate_persistence(
    dataset,
    stats: NormalizationStats,
    reader,
    lst_band_index: int = 1,
    max_samples: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> PersistenceResult:
    """Evaluate the persistence baseline over ``dataset``.

    Parameters
    ----------
    dataset : UHIDataset
        A split-specific frozen dataset (must expose ``records`` and valid
        ``__getitem__``).
    stats : NormalizationStats
        LST statistics for converting normalized values to Celsius.
    reader : MonthRasterReader
        Raw windowed reader used to recover latest-input LST validity.
    lst_band_index : int
        1-based band index of LST in the monthly rasters.
    """
    lst_i = stats.index("LST_Celsius")
    mean = stats.means[lst_i]
    std = stats.stds[lst_i]

    acc = MetricAccumulator()
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    target_valid = 0
    combined_valid = 0
    excluded = 0
    samples_with_exclusion = 0

    for idx in range(n):
        rec = dataset.records[idx]
        X, y, mask = dataset[idx]
        pred_norm = last_frame_lst_normalized(X.unsqueeze(0))[0]  # [1,H,W]
        pred = pred_norm * std + mean
        target = y * std + mean
        tmask = mask.to(dtype=torch.bool)

        r0, c0 = int(rec["row_off"]), int(rec["col_off"])
        s = int(rec["height"])
        window = ((r0, r0 + s), (c0, c0 + s))
        latest_label = rec["input_months"][-1]
        input_valid = torch.from_numpy(
            reader.read_validity(latest_label, lst_band_index, window)
        ).unsqueeze(0)

        combined = tmask & input_valid
        target_valid += int(tmask.sum())
        combined_valid += int(combined.sum())
        n_excl = int((tmask & ~input_valid).sum())
        excluded += n_excl
        if n_excl > 0:
            samples_with_exclusion += 1

        acc.update(pred, target, combined)
        if progress is not None:
            progress(idx)

    return PersistenceResult(
        metrics=acc.as_dict(),
        n_samples=n,
        target_valid_pixels=target_valid,
        combined_valid_pixels=combined_valid,
        input_invalid_excluded_pixels=excluded,
        samples_with_input_invalid_pixels=samples_with_exclusion,
    )