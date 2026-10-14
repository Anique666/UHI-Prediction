"""Hotspot-weighted regression loss for the UHI ConvLSTM (opt-in experiment).

This module adds a **weighted equivalent of the existing masked-MAE control**
without touching the frozen dataset, model, or the original loss functions in
``training/losses.py``. It gives a modest, capped extra weight to
high-temperature **urban** target pixels, keeps ordinary valid pixels at weight
1.0, and normalizes by the sum of valid weights.

Design
------
The control objective (``masked_mae``) is::

    L = sum(M * |pred - y|) / sum(M)

The hotspot variant is the same L1 objective with a per-pixel weight ``W``::

    W = 1                       for ordinary valid pixels
    W = w_hot     (>1)          for valid URBAN pixels with target z >= hot_z
    W = w_extreme (>w_hot)      for valid URBAN pixels with target z >= extreme_z
    W = 0                       where the target is invalid (M == 0)

    L = sum(W * |pred - y|) / sum(W)

``z`` is the *normalized* target using the frozen **train-only** LST statistics,
so the thresholds are training-data-only. With the frozen LST stats
(mean=34.3276, std=4.2946) the default thresholds correspond to
``hot_z = 1.0 -> ~38.6 C`` and ``extreme_z = 1.5 -> ~40.8 C``.

The urban reference mask is the frozen ``uhi_masks/urban_mask.tif``. Because the
loss cannot know a sample's grid offset on its own, ``UrbanPatchDataset`` wraps
the frozen ``UHIDataset`` and attaches the correct urban sub-window to each
training sample as a fourth return value. Validation is left untouched and keeps
using the control ``masked_mae`` so results stay comparable to the baseline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch

__all__ = [
    "HotspotWeightedMAELoss",
    "UrbanPatchDataset",
    "load_urban_mask",
    "DEFAULT_WEIGHTING",
]

#: Modest, capped defaults. Ordinary valid pixels keep weight 1.0.
DEFAULT_WEIGHTING = {
    "hot_z": 1.0,
    "extreme_z": 1.5,
    "w_hot": 2.0,
    "w_extreme": 5.0,
}


def load_urban_mask(mask_path: str | Path) -> np.ndarray:
    """Read the frozen binary urban mask (uint8, >0 = urban) into memory."""
    with rasterio.open(mask_path) as ds:
        return ds.read(1)


class UrbanPatchDataset(torch.utils.data.Dataset):
    """Attach the frozen urban mask sub-window to each frozen dataset sample.

    ``__getitem__`` returns ``(X, y, mask, urban)`` where ``urban`` is a
    ``[1, H, W]`` uint8 tensor (1 = urban). The wrapped dataset must expose
    ``records`` (patch ``row_off``/``col_off``/``height``/``width``) and a
    3-tuple ``__getitem__``, i.e. the frozen ``UHIDataset``.
    """

    def __init__(self, base, urban_mask_path: str | Path):
        self.base = base
        self.urban = load_urban_mask(urban_mask_path)
        self.mask_height, self.mask_width = self.urban.shape

    def __len__(self) -> int:
        return len(self.base)

    @property
    def records(self):
        return self.base.records

    def __getitem__(self, idx: int):
        X, y, mask = self.base[idx]
        rec = self.base.records[idx]
        r0, c0 = int(rec["row_off"]), int(rec["col_off"])
        s = int(rec["height"])
        if s != int(rec["width"]):
            raise ValueError(f"non-square patch in sample {rec.get('sample_id')}")
        if r0 < 0 or c0 < 0 or r0 + s > self.mask_height or c0 + s > self.mask_width:
            raise ValueError(
                f"patch window ({r0}:{r0 + s}, {c0}:{c0 + s}) outside urban mask "
                f"({self.mask_height}, {self.mask_width})"
            )
        ua = np.ascontiguousarray(self.urban[r0 : r0 + s, c0 : c0 + s] > 0)
        urban = torch.from_numpy(ua.astype(np.uint8)[None])
        return X, y, mask, urban

    def close(self) -> None:
        closer = getattr(self.base, "close", None)
        if closer is not None:
            closer()


class HotspotWeightedMAELoss(torch.nn.Module):
    """Masked L1 loss with capped extra weight on hot urban target pixels.

    Signature: ``forward(prediction, target, mask, urban) -> scalar``.
    ``target`` is in normalized LST space; ``mask`` and ``urban`` are boolean
    ``[B, 1, H, W]``. Invalid targets contribute zero loss and zero weight.
    """

    requires_urban = True

    def __init__(
        self,
        hot_z: float = 1.0,
        extreme_z: float = 1.5,
        w_hot: float = 2.0,
        w_extreme: float = 5.0,
    ):
        super().__init__()
        if not (w_hot >= 1.0):
            raise ValueError("w_hot must be >= 1.0")
        if not (w_extreme >= w_hot):
            raise ValueError("w_extreme must be >= w_hot")
        if not (extreme_z >= hot_z):
            raise ValueError("extreme_z must be >= hot_z")
        self.hot_z = float(hot_z)
        self.extreme_z = float(extreme_z)
        self.w_hot = float(w_hot)
        self.w_extreme = float(w_extreme)

    def weight_map(self, target: torch.Tensor, mask: torch.Tensor, urban: torch.Tensor) -> torch.Tensor:
        """Per-pixel weight ``W`` (float, same dtype/device as ``target``)."""
        dtype = target.dtype
        m = mask.to(dtype=dtype)
        u = urban.to(dtype=dtype)
        hot = (target >= self.hot_z).to(dtype=dtype)
        extreme = (target >= self.extreme_z).to(dtype=dtype)
        w = torch.ones_like(target, dtype=dtype)
        w = w + u * hot * (self.w_hot - 1.0) + u * extreme * (self.w_extreme - self.w_hot)
        return w * m

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        urban: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction {tuple(prediction.shape)} and target {tuple(target.shape)} must match"
            )
        if mask.shape != prediction.shape:
            raise ValueError("mask must match prediction")
        if urban.shape != prediction.shape:
            raise ValueError("urban must match prediction")
        w = self.weight_map(target, mask, urban)
        abs_err = (prediction - target).abs()
        denom = w.sum()
        if float(denom.detach()) == 0.0:
            return (prediction * 0.0).mean()
        return (abs_err * w).sum() / denom