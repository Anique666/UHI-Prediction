"""Opt-in urban-rural contrast auxiliary loss for the UHI ConvLSTM.

Motivation
----------
The baseline objective is a per-pixel masked L1 on normalized next-month LST.
UHI intensity is a *contrast* between an LST field and the spatial mean LST over
a fixed rural reference mask, so a per-pixel L1 objective has no term that
couples the rural reference to the urban pixels. This module adds an auxiliary
term that penalises a mismatch in the per-sample urban-minus-rural mean LST
contrast::

    C_pred = mean(pred  over urban ∩ valid) - mean(pred  over rural ∩ valid)
    C_obs  = mean(y     over urban ∩ valid) - mean(y     over rural ∩ valid)
    L_aux  = |C_pred - C_obs|
    L      = masked_MAE(pred, y, mask) + lambda * L_aux

Both contrasts are computed in the same normalized LST space the baseline loss
uses, with the frozen urban/rural reference masks. Samples with no valid urban
or no valid rural pixels are skipped (their auxiliary term contributes no
gradient). The base term is always the original masked MAE, so the baseline
implementation in :mod:`training.losses` is untouched; enabling this loss is
opt-in only.

This differs from the (failed) hotspot-weighted experiment: it does **not**
re-weight individual hot urban pixels. It targets exactly the spatial contrast
that the UHI metric measures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
import torch

from .losses import masked_mae

__all__ = [
    "load_binary_mask",
    "UrbanRuralPatchDataset",
    "UrbanRuralContrastLoss",
    "DEFAULT_CONTRAST_LAMBDA",
]

#: Prescribed auxiliary weight used by the controlled experiment.
DEFAULT_CONTRAST_LAMBDA = 0.1


def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    """Read a frozen binary mask (uint8, ``> 0`` marks the class) into memory."""
    with rasterio.open(mask_path) as ds:
        return ds.read(1)


def _mask_patch(full_mask: np.ndarray, rec: dict, name: str) -> np.ndarray:
    """Extract the patch-window boolean sub-mask for one sample record."""
    r0, c0 = int(rec["row_off"]), int(rec["col_off"])
    s = int(rec["height"])
    if s != int(rec["width"]):
        raise ValueError(f"non-square patch in sample {rec.get('sample_id')}")
    h, w = full_mask.shape
    if r0 < 0 or c0 < 0 or r0 + s > h or c0 + s > w:
        raise ValueError(
            f"patch window ({r0}:{r0 + s}, {c0}:{c0 + s}) outside {name} mask ({h}, {w})"
        )
    return np.ascontiguousarray(full_mask[r0 : r0 + s, c0 : c0 + s] > 0)


class UrbanRuralPatchDataset(torch.utils.data.Dataset):
    """Attach the frozen urban **and** rural mask sub-windows to each sample.

    ``__getitem__`` returns ``(X, y, mask, urban, rural)`` where both mask
    tensors are ``[1, H, W]`` uint8 (1 = member of the reference mask). The
    wrapped dataset must be the frozen ``UHIDataset`` (exposes ``records`` and a
    3-tuple ``__getitem__``). This mirrors :class:`training.hotspot_loss.
    UrbanPatchDataset` but carries both reference masks.
    """

    def __init__(self, base, urban_mask_path: str | Path, rural_mask_path: str | Path):
        self.base = base
        self.urban = load_binary_mask(urban_mask_path)
        self.rural = load_binary_mask(rural_mask_path)
        if self.urban.shape != self.rural.shape:
            raise ValueError("urban and rural masks must share the same grid shape")

    def __len__(self) -> int:
        return len(self.base)

    @property
    def records(self):
        return self.base.records

    def __getitem__(self, idx: int):
        X, y, mask = self.base[idx]
        rec = self.base.records[idx]
        ua = _mask_patch(self.urban, rec, "urban")
        ra = _mask_patch(self.rural, rec, "rural")
        urban = torch.from_numpy(ua.astype(np.uint8)[None])
        rural = torch.from_numpy(ra.astype(np.uint8)[None])
        return X, y, mask, urban, rural

    def close(self) -> None:
        closer = getattr(self.base, "close", None)
        if closer is not None:
            closer()


class UrbanRuralContrastLoss(torch.nn.Module):
    """Masked MAE plus ``lambda *`` urban-rural contrast mismatch.

    Signature: ``forward(prediction, target, mask, urban, rural) -> scalar``.
    All tensors are ``[B, 1, H, W]``; ``target`` is normalized LST. The
    contrast mismatch is the absolute difference of the per-sample urban-minus-
    rural mean predicted and observed LST, in normalized units. Samples whose
    urban or rural valid-pixel count is zero are skipped.
    """

    requires_contrast = True

    def __init__(self, lam: float = 0.1, base_loss: Callable | None = None):
        super().__init__()
        if lam < 0:
            raise ValueError("lambda must be non-negative")
        self.lam = float(lam)
        self._base_loss = base_loss if base_loss is not None else masked_mae
        # Detached scalars from the most recent forward, for separate logging of
        # the base LST loss and the auxiliary contrast term.
        self.last_base = float("nan")
        self.last_contrast = float("nan")

    def base_loss(self, prediction, target, mask):
        return self._base_loss(prediction, target, mask)

    def contrast_mismatch(self, prediction, target, mask, urban, rural) -> tuple[torch.Tensor, int]:
        """Return ``(mean |dC|, n_valid_samples)`` over samples with both masks."""
        b = prediction.shape[0]
        dtype = prediction.dtype

        def _flat(t):
            return t.reshape(b, -1).to(dtype=dtype)

        m = _flat(mask)
        u = _flat(urban) * m
        r = _flat(rural) * m
        p = _flat(prediction)
        y = _flat(target)

        nu = u.sum(dim=1)
        nr = r.sum(dim=1)
        valid = (nu > 0) & (nr > 0)

        du = nu.clamp_min(1e-6)
        dr = nr.clamp_min(1e-6)
        pred_contrast = (u * p).sum(dim=1) / du - (r * p).sum(dim=1) / dr
        obs_contrast = (u * y).sum(dim=1) / du - (r * y).sum(dim=1) / dr
        diff = (pred_contrast - obs_contrast).abs()

        if not bool(valid.any()):
            return (prediction * 0.0).mean(), 0
        return diff[valid].mean(), int(valid.sum())

    def forward(self, prediction, target, mask, urban, rural) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction {tuple(prediction.shape)} and target {tuple(target.shape)} must match"
            )
        if mask.shape != prediction.shape:
            raise ValueError("mask must match prediction")
        if urban.shape != prediction.shape or rural.shape != prediction.shape:
            raise ValueError("urban/rural masks must match prediction")
        base = self.base_loss(prediction, target, mask)
        c, _ = self.contrast_mismatch(prediction, target, mask, urban, rural)
        self.last_base = float(base.detach())
        self.last_contrast = float(c.detach())
        return base + self.lam * c

    def component_losses(self) -> dict:
        """Return the detached component scalars from the most recent forward.

        Used by the engine to log the base LST loss and the auxiliary contrast
        term separately. Keys: ``base`` (masked MAE) and ``contrast`` (the
        unweighted contrast mismatch).
        """
        return {"base": self.last_base, "contrast": self.last_contrast}