"""Thin windowed raster access helpers.

This module provides raw GeoTIFF reads that the frozen ``UHIDataset`` does not
expose (e.g. the *input validity* of the latest month, or the raw target LST
for physical-unit evaluation and UHI derivation). It does **not** reimplement
the dataset's fill/normalization logic and never modifies the frozen artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

__all__ = ["MonthRasterReader", "read_mask_raster", "load_mask_metadata"]


class MonthRasterReader:
    """Windowed reader over the canonical monthly feature GeoTIFFs.

    Month paths are resolved from the frozen ``month_catalog.json`` relative
    paths, overridden by ``input_dir`` exactly like ``UHIDataset`` does.
    Readers are cached and closed explicitly via :meth:`close`.
    """

    def __init__(self, dataset_root: str | Path, input_dir: str | Path | None = None):
        self.root = Path(dataset_root)
        with (self.root / "month_catalog.json").open(encoding="utf-8") as f:
            cat = json.load(f)
        self._saved_input_dir = Path(cat["input_dir"])
        self._input_dir = Path(input_dir) if input_dir else self._saved_input_dir
        self._relpath = {m["label"]: m["relative_path"] for m in cat["months"]}
        self._open: dict[str, rasterio.DatasetReader] = {}

    def month_path(self, label: str) -> str:
        if label not in self._relpath:
            raise KeyError(f"Unknown month label {label!r}")
        rel = Path(self._relpath[label])
        if rel.is_absolute():
            return str(rel)
        return str(self._input_dir / rel)

    def _reader(self, label: str) -> rasterio.DatasetReader:
        ds = self._open.get(label)
        if ds is None:
            ds = rasterio.open(self.month_path(label))
            self._open[label] = ds
        return ds

    def read_band(
        self, label: str, band_index: int, window=None
    ) -> np.ndarray:
        """Read a single 1-based band as float32 for ``window``."""
        ds = self._reader(label)
        return ds.read(band_index, window=window).astype(np.float32, copy=True)

    def read_validity(self, label: str, band_index: int, window=None) -> np.ndarray:
        """Boolean mask: ``True`` where the raw band value is finite."""
        return np.isfinite(self.read_band(label, band_index, window))

    def close(self) -> None:
        for ds in self._open.values():
            try:
                ds.close()
            except Exception:
                pass
        self._open.clear()

    def __enter__(self) -> "MonthRasterReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_mask_raster(path: str | Path):
    """Read a binary mask raster. Returns ``(array[uint8 2D], profile_dict)``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask raster not found: {path}")
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = {
            "crs": ds.crs,
            "transform": ds.transform,
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": ds.dtypes[0],
        }
    return arr, profile


def load_mask_metadata(mask_dir: str | Path) -> dict:
    """Load ``mask_metadata.json`` from the frozen mask directory."""
    path = Path(mask_dir) / "mask_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"mask_metadata.json not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)