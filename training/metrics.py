"""Aggregated error metrics for masked pixel-wise regression.

Metrics must be aggregated over **all** valid pixels across the whole
evaluation population, never as an unweighted average of per-batch metrics.
This module accumulates error sums and valid-pixel counts so batches with
different valid-pixel counts are weighted correctly.

The accumulator is unit-agnostic: feed it tensors already expressed in the
desired units (normalized space or degrees Celsius).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["MetricAccumulator"]


@dataclass
class MetricAccumulator:
    """Accumulates masked error statistics across batches."""

    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_err: float = 0.0
    sum_target: float = 0.0
    sum_target_sq: float = 0.0
    count: int = 0
    _batches: int = field(default=0, repr=False)

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Accumulate error statistics for one batch over valid pixels."""
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must match")
        if mask.shape != prediction.shape:
            raise ValueError("mask must match prediction")
        pred = prediction.detach().double()
        tgt = target.detach().double()
        m = mask.to(dtype=torch.bool).detach()
        if not bool(m.any()):
            self._batches += 1
            return
        p = pred[m]
        t = tgt[m]
        diff = p - t
        self.sum_abs += float(diff.abs().sum())
        self.sum_sq += float((diff ** 2).sum())
        self.sum_err += float(diff.sum())
        self.sum_target += float(t.sum())
        self.sum_target_sq += float((t ** 2).sum())
        self.count += int(m.sum())
        self._batches += 1

    @property
    def mae(self) -> float:
        return self.sum_abs / self.count if self.count else float("nan")

    @property
    def rmse(self) -> float:
        return (self.sum_sq / self.count) ** 0.5 if self.count else float("nan")

    @property
    def bias(self) -> float:
        return self.sum_err / self.count if self.count else float("nan")

    @property
    def r2(self) -> float:
        """Coefficient of determination against the global valid-target mean."""
        if self.count == 0:
            return float("nan")
        mean_t = self.sum_target / self.count
        ss_tot = self.sum_target_sq - self.count * mean_t ** 2
        if ss_tot <= 0:
            return float("nan")
        ss_res = self.sum_sq
        return 1.0 - ss_res / ss_tot

    def as_dict(self) -> dict[str, float]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "bias": self.bias,
            "r2": self.r2,
            "valid_pixels": self.count,
            "batches": self._batches,
        }