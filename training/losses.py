"""Masked regression losses for next-month LST prediction.

The target validity mask ``M`` marks genuine observations (1) versus
missing/pasted pixels (0). Only genuine pixels contribute to the loss.
"""
from __future__ import annotations

import torch

__all__ = ["masked_mae", "masked_mse", "masked_huber", "MaskedMAELoss"]


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean absolute error over valid pixels.

    ``L = sum(M * |pred - target|) / sum(M)``

    The mask is cast to the prediction dtype and gradients flow only through
    valid pixels. If a batch contains no valid pixels the loss is the mean of
    ``prediction * 0`` so the graph stays connected and finite (0.0) while
    contributing no gradient signal.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} and target {tuple(target.shape)} must match"
        )
    if mask.shape != prediction.shape:
        raise ValueError(
            f"mask {tuple(mask.shape)} must broadcast-match prediction {tuple(prediction.shape)}"
        )

    m = mask.to(dtype=prediction.dtype)
    abs_err = (prediction - target).abs()
    denom = m.sum()
    if float(denom.detach()) == 0.0:
        return (prediction * 0.0).mean()
    return (abs_err * m).sum() / denom


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean squared error over valid pixels (same safety rules)."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must match")
    if mask.shape != prediction.shape:
        raise ValueError("mask must match prediction")
    m = mask.to(dtype=prediction.dtype)
    err = (prediction - target) ** 2
    denom = m.sum()
    if float(denom.detach()) == 0.0:
        return (prediction * 0.0).mean()
    return (err * m).sum() / denom


class MaskedMAELoss(torch.nn.Module):
    """``nn.Module`` wrapper around :func:`masked_mae`."""

    def forward(self, prediction, target, mask):
        return masked_mae(prediction, target, mask)


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Masked Huber (smooth-L1) loss over valid pixels.

    ``L = sum(M * h(pred - y)) / sum(M)`` with, per pixel,

        h(e) = 0.5 * e^2                 if |e| <= delta
        h(e) = delta * (|e| - 0.5*delta) otherwise

    ``delta`` is expressed in the units of the target. Because the model is
    trained on **normalized** LST, ``delta`` is in normalized units (1 delta =
    1 training standard deviation of LST). Invalid pixels contribute nothing;
    an all-invalid batch yields a finite connected 0.0.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must match")
    if mask.shape != prediction.shape:
        raise ValueError("mask must match prediction")
    m = mask.to(dtype=prediction.dtype)
    diff = prediction - target
    abs_diff = diff.abs()
    quadratic = 0.5 * diff * diff
    linear = delta * (abs_diff - 0.5 * delta)
    per_pixel = torch.where(abs_diff <= delta, quadratic, linear)
    denom = m.sum()
    if float(denom.detach()) == 0.0:
        return (prediction * 0.0).mean()
    return (per_pixel * m).sum() / denom