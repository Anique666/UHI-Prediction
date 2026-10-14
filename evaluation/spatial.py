"""Spatial reconstruction and UHI derivation.

The model operates on 128x128 patches with stride 64 (overlapping). The urban
and rural reference masks span the full 1608x1493 study grid. To derive UHI we
must place patch predictions back into their true grid windows and reconcile
overlaps deterministically before computing a rural reference mean.

Design
------
* ``OverlapGrid`` accumulates weighted predictions for overlapping patches.
  Each patch contributes through a smooth, deterministic separable window
  (a triangular/Bartlett taper) so that overlap regions are blended and patch
  borders do not dominate the reconstruction. The final map is
  ``sum(pred * w) / sum(w)``; a ``weight`` map records coverage.
* UHI is defined citywide as::

      LST_rural_mean(t) = mean of valid predicted LST over the rural mask
      UHI(t, x, y)      = LST(t, x, y) - LST_rural_mean(t)

* The rural mean uses only valid predicted pixels inside the fixed rural mask.
  Urban and rural reference masks are never used to define the supervised
  target or loss; they are used only for this downstream diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "patch_weight",
    "OverlapGrid",
    "rural_reference_mean",
    "derive_uhi",
    "compute_coverage",
]


def patch_weight(size: int = 128, floor: float = 1e-6) -> np.ndarray:
    """Deterministic separable triangular taper weight for one patch.

    Returns a ``(size, size)`` float64 array with peak 1.0 at the centre and a
    small positive ``floor`` at the edges so edge pixels still contribute.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    one_d = np.minimum(np.arange(size), np.arange(size)[::-1]) + 1
    one_d = one_d.astype(np.float64)
    w = np.outer(one_d, one_d)
    w = w / w.max()
    return np.maximum(w, floor)


@dataclass
class _Grid:
    sum: np.ndarray
    weight: np.ndarray


class OverlapGrid:
    """Weighted accumulator that places patch predictions onto a full grid."""

    def __init__(self, height: int, width: int, patch_size: int = 128):
        self.height = int(height)
        self.width = int(width)
        self.patch_size = int(patch_size)
        self.acc_sum = np.zeros((self.height, self.width), dtype=np.float64)
        self.acc_w = np.zeros((self.height, self.width), dtype=np.float64)
        self._weight = patch_weight(self.patch_size)

    def add(self, prediction: np.ndarray, row_off: int, col_off: int, valid: np.ndarray | None = None) -> None:
        """Add one ``patch_size`` prediction window at ``(row_off, col_off)``.

        ``prediction`` and ``valid`` are 2D arrays of shape
        ``(patch_size, patch_size)``. Pixels with ``valid == False`` are not
        accumulated.
        """
        p = np.asarray(prediction, dtype=np.float64)
        if p.shape != (self.patch_size, self.patch_size):
            raise ValueError(
                f"prediction shape {p.shape} != ({self.patch_size}, {self.patch_size})"
            )
        r0, c0 = int(row_off), int(col_off)
        r1, c1 = r0 + self.patch_size, c0 + self.patch_size
        if r0 < 0 or c0 < 0 or r1 > self.height or c1 > self.width:
            raise ValueError(
                f"patch window ({r0}:{r1}, {c0}:{c1}) exceeds grid "
                f"({self.height}, {self.width})"
            )
        w = self._weight
        if valid is not None:
            v = np.asarray(valid, dtype=bool)
            if v.shape != p.shape:
                raise ValueError("valid mask shape must match prediction")
            w = w * v
        self.acc_sum[r0:r1, c0:c1] += p * w
        self.acc_w[r0:r1, c0:c1] += w

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(blended, weight)`` with NaN where no valid contribution."""
        weight = self.acc_w
        blended = np.full_like(self.acc_sum, np.nan)
        nz = weight > 0
        blended[nz] = self.acc_sum[nz] / weight[nz]
        return blended, weight


def rural_reference_mean(lst_map: np.ndarray, valid: np.ndarray, rural_mask: np.ndarray) -> float:
    """Spatial mean of valid LST over the fixed rural reference mask.

    ``valid`` marks pixels with a genuine value; ``rural_mask`` is the fixed
    binary rural reference. Returns NaN when there are no valid rural pixels.
    """
    m = np.asarray(valid, dtype=bool) & (np.asarray(rural_mask) > 0)
    if not m.any():
        return float("nan")
    return float(np.asarray(lst_map, dtype=np.float64)[m].mean())


def derive_uhi(lst_map: np.ndarray, rural_mean: float, valid: np.ndarray) -> np.ndarray:
    """UHI intensity map ``LST - rural_mean``; NaN where ``valid`` is False."""
    lst = np.asarray(lst_map, dtype=np.float64)
    uhi = lst - float(rural_mean)
    out = np.where(np.asarray(valid, dtype=bool), uhi, np.nan)
    return out


def compute_coverage(
    valid: np.ndarray,
    urban_mask: np.ndarray,
    rural_mask: np.ndarray,
) -> dict:
    """Report how much of the reference masks a reconstruction covers."""
    v = np.asarray(valid, dtype=bool)
    urb = np.asarray(urban_mask) > 0
    rur = np.asarray(rural_mask) > 0
    study = urb | rur
    out = {
        "grid_pixels": int(v.size),
        "valid_grid_pixels": int(v.sum()),
        "grid_fraction": float(v.mean()),
        "study_pixels": int(study.sum()),
    }
    for name, m in (("urban", urb), ("rural", rur), ("study", study)):
        total = int(m.sum())
        covered = int((m & v).sum())
        out[f"{name}_pixels"] = total
        out[f"{name}_covered"] = covered
        out[f"{name}_coverage"] = float(covered / total) if total else 0.0
    return out