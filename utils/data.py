"""Dataset construction and split bookkeeping.

Constructs the frozen ``UHIDataset`` with the Windows temporal-path override,
and exposes split-level metadata (patch ids, target months, window offsets)
used by the spatial reconstruction and leakage audits.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import get_paths

__all__ = ["import_uhidataset", "make_dataset", "make_dataloader", "SplitSummary", "summarize_splits"]


def import_uhidataset(dataset_root: Path):
    """Import the authoritative frozen ``UHIDataset`` from ``dataset_root``."""
    dataset_root = Path(dataset_root)
    dataset_py = dataset_root / "dataset.py"
    if not dataset_py.is_file():
        raise FileNotFoundError(f"Frozen dataset.py not found: {dataset_py}")
    root_str = str(dataset_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from dataset import UHIDataset  # type: ignore

    return UHIDataset


def make_dataset(cfg: dict, split: str):
    """Build a split-specific ``UHIDataset`` using resolved Windows paths."""
    paths = get_paths(cfg)
    UHIDataset = import_uhidataset(paths["dataset_root"])
    return UHIDataset(
        dataset_root=paths["dataset_root"],
        split=split,
        input_dir=paths["temporal_root"],
    )


def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with Windows-safe initial defaults."""
    persistent = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent,
    )


@dataclass
class SplitSummary:
    """Aggregate description of one split's samples."""

    split: str
    n_samples: int
    n_patches: int
    patch_ids: list[int]
    target_months: list[int]
    row_offsets: dict[int, int]
    col_offsets: dict[int, int]


def summarize_splits(dataset_root: str | Path, input_dir: str | Path) -> dict[str, SplitSummary]:
    """Summarize every split from the frozen sample index (no tensor reads)."""
    UHIDataset = import_uhidataset(Path(dataset_root))
    out: dict[str, SplitSummary] = {}
    for split in ("train", "val", "test"):
        ds = UHIDataset(
            dataset_root=dataset_root, split=split, input_dir=input_dir
        )
        recs = ds.records
        patches = sorted({r["patch_id"] for r in recs})
        rows = {r["patch_id"]: int(r["row_off"]) for r in recs}
        cols = {r["patch_id"]: int(r["col_off"]) for r in recs}
        out[split] = SplitSummary(
            split=split,
            n_samples=len(recs),
            n_patches=len(patches),
            patch_ids=patches,
            target_months=sorted({int(r["target_month_index"]) for r in recs}),
            row_offsets=rows,
            col_offsets=cols,
        )
        ds.close()
    return out