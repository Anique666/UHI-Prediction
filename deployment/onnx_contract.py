"""Frozen runtime contract for the UHI ConvLSTM deployment.

Every value is derived from existing source files and frozen artifacts (never
duplicated as a literal). Missing or inconsistent artifacts raise ``ContractError``.

Authoritative sources
---------------------
* ``configs/baseline.yaml``                     -> paths, seq len, patch, channels
* ``<dataset_root>/normalization_stats.json``   -> channel order, fill/mean/std
* ``<dataset_root>/month_catalog.json``         -> 72 chronological months
* ``<dataset_root>/dataset.py``                 -> fill/normalize/terrain rules
* ``<mask_dir>/mask_metadata.json``             -> grid, CRS, transform, masks
* ``models/convlstm.py``                         -> architecture (NOT modified)
* ``checkpoints/baseline_full/best.pt``          -> retained weights (read-only)
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = [
    "ContractError",
    "DeploymentContract",
    "load_contract",
    "sha256_file",
    "dependency_versions",
]

ONNX_OPSET = 17
INPUT_NAME = "input"
OUTPUT_NAME = "lst_normalized"
INPUT_SHAPE_TAIL = (12, 18, 128, 128)   # fixed dims after the dynamic batch axis
OUTPUT_SHAPE_TAIL = (1, 128, 128)


class ContractError(RuntimeError):
    """Raised when a required artifact is missing or inconsistent."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dependency_versions() -> dict:
    """Best-effort versions of runtime dependencies (recorded, not enforced)."""
    import importlib

    out = {}
    for mod in ("torch", "onnx", "onnxruntime", "numpy", "rasterio", "pyproj", "yaml"):
        try:
            out[mod] = getattr(importlib.import_module(mod), "__version__", "unknown")
        except Exception:
            out[mod] = None
    return out


@dataclass(frozen=True)
class DeploymentContract:
    """Resolved, verified runtime contract."""

    paths: dict
    channel_names: list
    fill_values: list
    means: list
    stds: list
    lst_index: int
    sequence_length: int
    patch_size: int
    stride: int
    grid_height: int
    grid_width: int
    crs: str
    transform: tuple
    month_labels: list
    checkpoint_path: str
    checkpoint_sha256: str
    model: dict
    onnx_opset: int
    input_name: str
    output_name: str
    input_shape_tail: tuple
    output_shape_tail: tuple
    dependency_versions: dict

    # -- convenience accessors --------------------------------------------
    def index(self, channel: str) -> int:
        return self.channel_names.index(channel)

    @property
    def lst_mean(self) -> float:
        return float(self.means[self.lst_index])

    @property
    def lst_std(self) -> float:
        return float(self.stds[self.lst_index])

    def normalized_to_celsius(self, normalized):
        return normalized * self.lst_std + self.lst_mean

    def latest_target_month(self) -> str:
        return self.month_labels[-1]

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["paths"] = {k: str(v) for k, v in self.paths.items()}
        d["transform"] = [float(v) for v in self.transform]
        return d


def load_contract(
    *,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    verify_raster_bands: bool = True,
) -> DeploymentContract:
    """Load and validate the frozen deployment contract."""
    from utils.config import get_paths, load_config

    cfg = load_config(config_path or (PROJECT_ROOT / "configs" / "baseline.yaml"))
    paths = get_paths(cfg)

    dataset_root = Path(paths["dataset_root"])
    mask_dir = Path(paths["mask_dir"])
    temporal_root = Path(paths["temporal_root"])

    stats_path = dataset_root / "normalization_stats.json"
    catalog_path = dataset_root / "month_catalog.json"
    mask_meta_path = mask_dir / "mask_metadata.json"
    for p in (stats_path, catalog_path, mask_meta_path):
        if not p.is_file():
            raise ContractError(f"Required frozen artifact missing: {p}")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    channels = list(stats["channel_names"])
    lengths = {len(channels), len(stats["fill_values"]), len(stats["train_means"]),
               len(stats["train_stds"])}
    if len(lengths) != 1:
        raise ContractError("normalization_stats.json arrays have mismatched lengths")
    if any(s == 0 for s in stats["train_stds"]):
        raise ContractError("normalization_stats.json contains a zero std")

    cat = json.loads(catalog_path.read_text(encoding="utf-8"))
    month_labels = [m["label"] for m in cat["months"]]
    if month_labels != sorted(month_labels):
        raise ContractError("month_catalog.json is not chronologically ordered")

    mask_meta = json.loads(mask_meta_path.read_text(encoding="utf-8"))

    model_cfg = dict(cfg["model"])
    seq_len = int(cfg["data"]["sequence_length"])
    patch = int(cfg["data"]["patch_size"])
    n_ch = int(cfg["data"]["input_channels"])

    patch_index_path = dataset_root / "patch_index.json"
    if not patch_index_path.is_file():
        raise ContractError(f"Required frozen artifact missing: {patch_index_path}")
    patch_index = json.loads(patch_index_path.read_text(encoding="utf-8"))
    stride = int(patch_index["stride"])
    if int(patch_index["patch_size"]) != patch:
        raise ContractError("patch_index.json patch_size is inconsistent with config")

    ckpt = Path(checkpoint_path) if checkpoint_path else (
        PROJECT_ROOT / "checkpoints" / "baseline_full" / "best.pt"
    )
    if not ckpt.is_file():
        raise ContractError(f"Production checkpoint not found: {ckpt}")
    ckpt_sha = sha256_file(ckpt)

    if n_ch != len(channels) or model_cfg.get("output_channels") != 1:
        raise ContractError("config data/model channels inconsistent with frozen stats")
    lst_channel = str(cfg["data"].get("lst_channel", ""))
    lst_index = int(cfg["data"].get("lst_channel_index", -1))
    if not (0 <= lst_index < len(channels)) or channels[lst_index] != lst_channel:
        raise ContractError("config LST channel/index inconsistent with frozen stats")
    if seq_len != INPUT_SHAPE_TAIL[0] or patch != INPUT_SHAPE_TAIL[2]:
        raise ContractError("config sequence/patch inconsistent with the ONNX contract")

    transform = tuple(float(v) for v in mask_meta["transform"])
    grid_h, grid_w = int(mask_meta["height"]), int(mask_meta["width"])

    if verify_raster_bands:
        import rasterio

        with rasterio.open(temporal_root / cat["months"][-1]["relative_path"]) as ds:
            desc = list(ds.descriptions)
            if desc and desc != channels:
                raise ContractError("Raster band descriptions do not match channel order")
            if (ds.height, ds.width) != (grid_h, grid_w):
                raise ContractError("Raster grid does not match mask grid")
            if str(ds.crs) != str(mask_meta["crs"]):
                raise ContractError("Raster CRS does not match mask CRS")

    return DeploymentContract(
        paths=paths,
        channel_names=channels,
        fill_values=list(stats["fill_values"]),
        means=list(stats["train_means"]),
        stds=list(stats["train_stds"]),
        lst_index=lst_index,
        sequence_length=seq_len,
        patch_size=patch,
        stride=stride,
        grid_height=grid_h,
        grid_width=grid_w,
        crs=str(mask_meta["crs"]),
        transform=transform,
        month_labels=month_labels,
        checkpoint_path=str(ckpt),
        checkpoint_sha256=ckpt_sha,
        model=model_cfg,
        onnx_opset=ONNX_OPSET,
        input_name=INPUT_NAME,
        output_name=OUTPUT_NAME,
        input_shape_tail=INPUT_SHAPE_TAIL,
        output_shape_tail=OUTPUT_SHAPE_TAIL,
        dependency_versions=dependency_versions(),
    )
