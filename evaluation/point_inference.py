"""Standalone point-based inference for the frozen UHI ConvLSTM.

Given a geographic point (latitude, longitude), this module locates the aligned
128x128 window on the frozen study grid, builds the correct 12-month / 18-channel
input tensor by reusing the frozen ``UHIDataset`` internals (identical windowed
reads, terrain-mean replacement, train fill and train normalization), runs the
frozen checkpoint, converts the raw prediction to degrees Celsius with the frozen
target statistics, and renders / saves the resulting LST map.

Design constraints (Milestone 1)
--------------------------------
* Inference only. No training, fine-tuning or weight updates.
* The checkpoint, frozen dataset, normalization statistics, split definitions and
  existing evaluation artifacts are read-only and are never modified.
* The feature order, terrain treatment, normalization and inverse-normalization
  are reused exactly; nothing is reimplemented or guessed.
* The grid / CRS are taken from the frozen rasters themselves (verified), not
  assumed. The point is projected to the raster CRS with pyproj.
* A predicted pixel is valid only where the 12-month input window contains at
  least one genuine (finite) LST observation; invalid pixels are left as NaN.
  This mirrors the existing full-grid inference validity rule.

Reusable API
------------
``PointInference`` is modular so a future mitigation layer can pass an
alternative 12-month input tensor through the same ``run`` method without
changing anything else.

CLI
---
    python -m evaluation.point_inference --lat 13.11323 --lon 77.59155 \
        --checkpoint checkpoints/baseline_full/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.spatial import patch_weight  # noqa: E402
from models.convlstm import ConvLSTM  # noqa: E402
from training.engine import load_checkpoint  # noqa: E402
from utils.config import get_paths, load_config  # noqa: E402
from utils.data import import_uhidataset  # noqa: E402
from utils.normalization import load_normalization_stats  # noqa: E402
from utils.runmeta import collect_environment, write_json  # noqa: E402

PATCH_SIZE = 128
STRIDE = 64
LST_CHANNEL = "LST_Celsius"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed (used to pin the frozen checkpoint)."""
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Window / grid helpers (pure, unit-testable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowInfo:
    """A resolved 128x128 inference window and the point's position inside it."""

    row_off: int
    col_off: int
    raw_row_off: int
    raw_col_off: int
    in_window_row: int
    in_window_col: int
    row_inside: bool
    col_inside: bool
    edge_snapped: bool

    @property
    def contains_point(self) -> bool:
        return self.row_inside and self.col_inside


def snap_to_grid(rc: int, patch: int = PATCH_SIZE, stride: int = STRIDE) -> int:
    """Snap a pixel index to the underlying stride grid (floor to multiple)."""
    return (int(rc) // int(stride)) * int(stride)


def resolve_window(
    row: int,
    col: int,
    height: int,
    width: int,
    patch: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> WindowInfo:
    """Resolve the aligned inference window for a pixel ``(row, col)``.

    Top-left offsets are snapped to the frozen stride grid (all frozen patches
    are multiples of the stride), then clamped so the whole ``patch`` window
    stays inside the raster. ``raw_*`` record the pre-clamp alignment so an
    edge-of-raster adjustment can be reported explicitly rather than hidden.
    """
    raw_r = snap_to_grid(row, patch, stride)
    raw_c = snap_to_grid(col, patch, stride)
    r0 = min(max(raw_r, 0), height - patch)
    c0 = min(max(raw_c, 0), width - patch)
    return WindowInfo(
        row_off=int(r0),
        col_off=int(c0),
        raw_row_off=int(raw_r),
        raw_col_off=int(raw_c),
        in_window_row=int(row - r0),
        in_window_col=int(col - c0),
        row_inside=bool(r0 <= row < r0 + patch),
        col_inside=bool(c0 <= col < c0 + patch),
        edge_snapped=bool(r0 != raw_r or c0 != raw_c),
    )


def latlon_to_pixel(lat: float, lon: float, transform, crs, shape) -> tuple[int, int, float, float]:
    """Project ``(lat, lon)`` (EPSG:4326) to the raster CRS and return
    ``(row, col, x, y)`` for the raster ``transform``/``shape``."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = rasterio.transform.rowcol(transform, x, y)
    return int(row), int(col), float(x), float(y)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
class PointInference:
    """Reusable point-inference engine over the frozen dataset + checkpoint.

    ``build_input(record)`` reproduces the frozen ``UHIDataset`` input path
    exactly (no target read), so a future mitigation layer can override the
    returned tensor before calling :meth:`run`.
    """

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        input_dir: str | Path,
        stats,
        model,
        device: torch.device,
        month_labels: list[str],
        sequence_length: int = 12,
        grid_path: str | Path | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.input_dir = Path(input_dir)
        self.stats = stats
        self.device = device
        self.model = model
        self.month_labels = list(month_labels)
        self.sequence_length = int(sequence_length)

        UHIDataset = import_uhidataset(self.dataset_root)
        # split=None loads the frozen metadata/stats/readers without split samples.
        self.base = UHIDataset(
            dataset_root=str(self.dataset_root), split=None, input_dir=str(self.input_dir)
        )
        if "LST_Celsius" not in self.base.channel_names:
            raise KeyError("Frozen dataset is missing the LST_Celsius channel")
        if int(self.base._n_ch) != 18:
            raise ValueError(f"Expected 18 channels, frozen dataset exposes {self.base._n_ch}")

        # Grid provenance comes from the rasters, never assumed.
        grid_ref = Path(grid_path) if grid_path else self._first_month_path()
        with rasterio.open(grid_ref) as ds:
            self.grid_crs = ds.crs
            self.grid_transform = ds.transform
            self.grid_width = ds.width
            self.grid_height = ds.height
            self.grid_band_names = list(ds.descriptions)
        if self.grid_band_names and self.grid_band_names != list(self.base.channel_names):
            raise ValueError("Frozen raster band descriptions do not match channel order")

    # -- path helpers ------------------------------------------------------
    def _first_month_path(self) -> str:
        return self.base._month_path(self.month_labels[0])

    def close(self) -> None:
        self.base.close()

    def __enter__(self) -> "PointInference":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- month window ------------------------------------------------------
    def resolve_target_month(self, target_month: str | None = None):
        """Resolve the input/target month window.

        Default: the latest valid 12 consecutive months available in the frozen
        dataset, predicting the following month. Returns
        ``(input_labels, target_label, target_index)``.
        """
        labels = self.month_labels
        seq = self.sequence_length
        if target_month is None:
            if len(labels) < seq + 1:
                raise ValueError(
                    f"Frozen dataset has {len(labels)} months; need at least {seq + 1} "
                    "to form a 12-month window and a target month."
                )
            target_index = len(labels) - 1
            target_label = labels[target_index]
        else:
            if target_month not in labels:
                raise ValueError(f"Unknown target month {target_month!r}")
            target_index = labels.index(target_month)
            target_label = target_month
        if target_index < seq:
            raise ValueError(
                f"Target month {target_label!r} needs {seq} preceding months; "
                f"only {target_index} available."
            )
        input_labels = labels[target_index - seq: target_index]
        # Guard against silent gaps: the frozen catalog must be contiguous.
        input_indices = list(range(target_index - seq, target_index))
        expected = [labels[i] for i in input_indices]
        if input_labels != expected:
            raise ValueError("Non-contiguous month window in frozen catalog")
        return input_labels, target_label, target_index

    def build_record(self, lat: float, lon: float, target_month: str | None = None) -> dict:
        """Resolve the aligned window + month window for a point into a record.

        Raises ``ValueError`` if the point falls outside the study raster.
        """
        input_labels, target_label, target_index = self.resolve_target_month(target_month)
        row, col, x, y = latlon_to_pixel(
            lat, lon, self.grid_transform, self.grid_crs,
            (self.grid_height, self.grid_width),
        )
        if not (0 <= row < self.grid_height and 0 <= col < self.grid_width):
            raise ValueError(
                f"Point ({lat}, {lon}) falls outside the frozen raster "
                f"(row={row}, col={col}, grid={self.grid_height}x{self.grid_width})."
            )
        win = resolve_window(row, col, self.grid_height, self.grid_width)
        if not win.contains_point:
            raise ValueError(
                "Resolved window does not contain the requested point "
                f"(edge adjustment: raw=({win.raw_row_off},{win.raw_col_off}) "
                f"-> ({win.row_off},{win.col_off}))."
            )
        return {
            "sample_id": -1,
            "split": "inference",
            "patch_id": -1,
            "row_off": win.row_off,
            "col_off": win.col_off,
            "height": PATCH_SIZE,
            "width": PATCH_SIZE,
            "input_months": list(input_labels),
            "input_month_indices": list(range(target_index - self.sequence_length, target_index)),
            "target_month": target_label,
            "target_month_index": int(target_index),
            "requested_lat": float(lat),
            "requested_lon": float(lon),
            "projected_x": float(x),
            "projected_y": float(y),
            "point_pixel_row": int(row),
            "point_pixel_col": int(col),
            "window": win,
        }

    # -- input construction (reuses the frozen dataset logic) --------------
    def build_input(self, record: dict) -> np.ndarray:
        """``[12, 18, 128, 128]`` normalized float32 (frozen contract)."""
        base = self.base
        r0, c0 = int(record["row_off"]), int(record["col_off"])
        s = int(record["height"])
        window = ((r0, r0 + s), (c0, c0 + s))
        input_labels = record["input_months"]
        C = base._n_ch
        X = np.empty((len(input_labels), C, s, s), dtype=np.float32)
        dynamic = [c for c in range(C) if c not in base._terrain_idx]
        for t, label in enumerate(input_labels):
            ds = base._readers.reader(base._month_path(label))
            win = ds.read(indexes=[c + 1 for c in dynamic], window=window).astype(
                np.float32, copy=True
            )
            for j, c in enumerate(dynamic):
                X[t, c] = win[j]
        if base._terrain_idx:
            tds = base._readers.reader(str(base.terrain_mean_path))
            tw = tds.read(window=window).astype(np.float32, copy=True)
            for j, c in enumerate(base._terrain_idx):
                X[:, c] = tw[j][None, :, :]
        fills = base.fill_values[None, :, None, None]
        means = base.means[None, :, None, None]
        stds = base.stds[None, :, None, None]
        Xf = np.where(np.isfinite(X), X, fills)
        Xn = (Xf - means) / stds
        return np.ascontiguousarray(Xn, dtype=np.float32)

    def input_lst_validity(self, record: dict) -> np.ndarray:
        """Boolean ``[128,128]``: >=1 finite LST observation in the window."""
        base = self.base
        r0, c0 = int(record["row_off"]), int(record["col_off"])
        s = int(record["height"])
        window = ((r0, r0 + s), (c0, c0 + s))
        count = np.zeros((s, s), dtype=np.int32)
        for label in record["input_months"]:
            ds = base._readers.reader(base._month_path(label))
            band = ds.read(base._lst_idx + 1, window=window).astype(np.float32)
            count += np.isfinite(band).astype(np.int32)
        return count > 0

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def run(self, record: dict, input_tensor: np.ndarray | None = None) -> dict:
        """Run the frozen model on one window and return a result dict.

        ``input_tensor`` lets a future mitigation layer substitute an alternative
        ``[12,18,128,128]`` normalized input; by default it is built by
        :meth:`build_input` (frozen contract).
        """
        X = input_tensor if input_tensor is not None else self.build_input(record)
        X = np.ascontiguousarray(X, dtype=np.float32)
        if tuple(X.shape) != (self.sequence_length, 18, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Unexpected input shape {tuple(X.shape)}")

        valid = self.input_lst_validity(record)
        lst_i = self.stats.index(LST_CHANNEL)
        mean, std = self.stats.means[lst_i], self.stats.stds[lst_i]

        self.model.eval()
        Xt = torch.from_numpy(X[None]).to(self.device)
        raw = self.model(Xt)  # [1,1,128,128] normalized
        if tuple(raw.shape) != (1, 1, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Unexpected model output shape {tuple(raw.shape)}")
        raw_np = raw[0, 0].float().cpu().numpy()
        pred_c = raw_np * float(std) + float(mean)  # Celsius (frozen inverse transform)

        # Weighted-overlap taper (matches the frozen full-grid blender so a single
        # window is directly comparable); applied only where the input is valid.
        w = patch_weight(PATCH_SIZE) * valid
        valid_c = np.where(valid, pred_c, np.nan)

        return {
            "record": record,
            "input": X,
            "raw_normalized": raw_np,
            "pred_celsius": pred_c,
            "valid_mask": valid,
            "weight": w.astype(np.float32),
            "pred_celsius_masked": valid_c,
            "lst_mean": float(mean),
            "lst_std": float(std),
        }


def prediction_stats(pred_masked: np.ndarray, valid: np.ndarray) -> dict:
    """Compact numeric summary over valid (finite) pixels."""
    vals = pred_masked[np.isfinite(pred_masked)]
    n_valid = int(vals.size)
    return {
        "valid_pixel_count": n_valid,
        "valid_pixel_fraction": float(n_valid / pred_masked.size),
        "mean_c": float(vals.mean()) if n_valid else float("nan"),
        "median_c": float(np.median(vals)) if n_valid else float("nan"),
        "min_c": float(vals.min()) if n_valid else float("nan"),
        "max_c": float(vals.max()) if n_valid else float("nan"),
        "std_c": float(vals.std()) if n_valid else float("nan"),
    }


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------
def window_transform(transform, row_off: int, col_off: int):
    """Geotransform for a window at ``(row_off, col_off)`` of ``transform``."""
    from rasterio.windows import Window, transform as rio_window_transform

    return rio_window_transform(Window(col_off, row_off, PATCH_SIZE, PATCH_SIZE), transform)


def write_tif(path: str | Path, arr: np.ndarray, crs, transform) -> None:
    a = np.asarray(arr, dtype=np.float32)
    profile = {
        "driver": "GTiff", "height": a.shape[0], "width": a.shape[1],
        "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
        "nodata": float("nan"), "compress": "lzw",
    }
    path = Path(path)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(a, 1)


def plot_lst_map(
    *,
    pred_masked: np.ndarray,
    record: dict,
    stats: dict,
    out_png: str | Path,
    crs,
    transform,
    point_xy: tuple[float, float] | None = None,
    cmap: str = "turbo",
) -> None:
    """Render a labelled 128x128 predicted-LST map with colour bar and stats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    font_manager._load_fontmanager(try_read_cache=False)

    finite = pred_masked[np.isfinite(pred_masked)]
    if finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    masked = np.ma.masked_invalid(pred_masked)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#4d4d4d")

    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    extent = (
        rasterio.transform.xy(transform, PATCH_SIZE, 0)[0],
        rasterio.transform.xy(transform, PATCH_SIZE, PATCH_SIZE)[0],
        rasterio.transform.xy(transform, PATCH_SIZE, 0)[1],
        rasterio.transform.xy(transform, 0, 0)[1],
    )
    im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, extent=extent,
                   interpolation="nearest", origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted LST (\u00b0C)")

    title = (
        f"ConvLSTM predicted LST \u2014 target {record['target_month']}\n"
        f"input {record['input_months'][0]} \u2026 {record['input_months'][-1]}  "
        f"window (row {record['row_off']}, col {record['col_off']})"
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"easting (m, {crs})")
    ax.set_ylabel("northing (m)")

    if point_xy is not None:
        ax.plot(point_xy[0], point_xy[1], marker="*", markersize=15,
                markeredgecolor="white", markerfacecolor="red", zorder=5)
        ax.plot([], [], marker="*", markersize=10, markeredgecolor="white",
                markerfacecolor="red", linestyle="none",
                label=f"selected point ({record['requested_lat']:.5f}, "
                      f"{record['requested_lon']:.5f})")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    invalid_handle = ax.imshow(
        np.ma.masked_where(np.isfinite(pred_masked), np.zeros_like(pred_masked)),
        cmap=plt.get_cmap("gray"), vmin=0, vmax=1, extent=extent,
        interpolation="nearest", origin="upper",
    )
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#4d4d4d", lw=8, label="invalid (no LST input)"),
        ] + ([plt.Line2D([0], [0], marker="*", color="red", lw=0,
                         markeredgecolor="white",
                         label="selected point")] if point_xy is not None else []),
        loc="lower right", fontsize=8, framealpha=0.85,
    )

    summary = (
        f"valid {stats['valid_pixel_count']}/{PATCH_SIZE * PATCH_SIZE} "
        f"({100 * stats['valid_pixel_fraction']:.1f}%)   "
        f"mean {stats['mean_c']:.2f}  median {stats['median_c']:.2f}  "
        f"min {stats['min_c']:.2f}  max {stats['max_c']:.2f} \u00b0C"
    )
    fig.text(0.5, 0.005, summary, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Baseline reproducibility
# ---------------------------------------------------------------------------
def reference_model_prediction(model, dataset, idx: int, stats, device):
    """Raw/inverse-normalized prediction for a frozen ``UHIDataset`` sample.

    This is the exact frozen evaluation path: it reads the sample through
    ``dataset[idx]`` (identical windowed reads / terrain / fill / normalization)
    and applies ``pred_norm * lst_std + lst_mean``.
    """
    X, y, m = dataset[idx]
    lst_i = stats.index(LST_CHANNEL)
    mean, std = stats.means[lst_i], stats.stds[lst_i]
    with torch.no_grad():
        pred_n = model(X.unsqueeze(0).to(device))[0, 0].float().cpu().numpy()
    return pred_n, pred_n * float(std) + float(mean), X, m[0].numpy().astype(bool)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Point-based frozen ConvLSTM LST inference")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    ap.add_argument("--lat", type=float, required=True, help="latitude (EPSG:4326)")
    ap.add_argument("--lon", type=float, required=True, help="longitude (EPSG:4326)")
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "baseline_full" / "best.pt"))
    ap.add_argument("--target-month", default=None,
                    help="Target calendar month (default: latest available, predicts following month)")
    ap.add_argument("--out-dir", default=None,
                    help="Default: outputs/point_inference/<tag>")
    ap.add_argument("--tag", default=None, help="Output subdirectory name")
    ap.add_argument("--skip-repro", action="store_true", help="Skip baseline reproducibility check")
    return ap.parse_args(argv)


def build_model(cfg, checkpoint, device):
    model = ConvLSTM(
        input_channels=cfg["data"]["input_channels"],
        hidden_channels=cfg["model"]["hidden_channels"],
        kernel_size=cfg["model"]["kernel_size"],
        output_channels=cfg["model"]["output_channels"],
    ).to(device)
    ckpt = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    return model, ckpt


def main(argv=None) -> int:
    t0 = time.perf_counter()
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    stats = load_normalization_stats(paths["dataset_root"])
    model, ckpt = build_model(cfg, checkpoint, device)

    with (paths["dataset_root"] / "month_catalog.json").open(encoding="utf-8") as f:
        month_labels = [m["label"] for m in json.load(f)["months"]]

    engine = PointInference(
        dataset_root=paths["dataset_root"],
        input_dir=paths["temporal_root"],
        stats=stats,
        model=model,
        device=device,
        month_labels=month_labels,
        sequence_length=int(cfg["data"]["sequence_length"]),
    )

    tag = args.tag or f"pt_{args.lat:.5f}_{args.lon:.5f}"
    out_dir = Path(args.out_dir) if args.out_dir else paths["outputs_dir"] / "point_inference" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_sha_before = sha256_file(checkpoint)

    try:
        record = engine.build_record(args.lat, args.lon, args.target_month)
        print(f"Point (lat, lon)    : ({args.lat}, {args.lon})  [EPSG:4326]")
        print(f"Projected to CRS    : {engine.grid_crs}  ->  x={record['projected_x']:.1f}, "
              f"y={record['projected_y']:.1f}")
        print(f"Aligned window      : row {record['row_off']}:{record['row_off'] + PATCH_SIZE}, "
              f"col {record['col_off']}:{record['col_off'] + PATCH_SIZE}  "
              f"(point at window [{record['point_pixel_row'] - record['row_off']}, "
              f"{record['point_pixel_col'] - record['col_off']}])")
        win = record["window"]
        if win.edge_snapped:
            print(f"NOTE: window clamped to raster edge "
                  f"(raw ({win.raw_row_off},{win.raw_col_off}) -> ({win.row_off},{win.col_off}))")
        print(f"Input months        : {record['input_months'][0]} .. {record['input_months'][-1]} "
              f"({len(record['input_months'])})")
        print(f"Target month        : {record['target_month']}")

        result = engine.run(record)
        print(f"Input tensor shape  : {tuple(result['input'].shape)}")
        print(f"Raw output shape    : {tuple(result['raw_normalized'].shape)}")
        print(f"LST inverse transform: Celsius = norm * {result['lst_std']:.10g} + {result['lst_mean']:.10g}")

        st = prediction_stats(result["pred_celsius_masked"], result["valid_mask"])
        print(f"Valid pixels        : {st['valid_pixel_count']}/{PATCH_SIZE * PATCH_SIZE} "
              f"({100 * st['valid_pixel_fraction']:.1f}%)")
        print(f"Predicted LST (C)   : mean {st['mean_c']:.3f}  median {st['median_c']:.3f}  "
              f"min {st['min_c']:.3f}  max {st['max_c']:.3f}")

        # --- artifacts
        wtransform = window_transform(engine.grid_transform, record["row_off"], record["col_off"])
        lst_tif = out_dir / "lst_pred_celsius.tif"
        valid_tif = out_dir / "lst_valid_uint8.tif"
        npy_path = out_dir / "lst_pred_celsius.npy"
        png_path = out_dir / "lst_pred_map.png"

        write_tif(lst_tif, result["pred_celsius_masked"], engine.grid_crs, wtransform)
        write_tif(valid_tif, result["valid_mask"].astype(np.float32), engine.grid_crs, wtransform)
        np.save(npy_path, result["pred_celsius_masked"].astype(np.float32))

        selected_xy = rasterio.transform.xy(
            engine.grid_transform, record["point_pixel_row"], record["point_pixel_col"]
        )
        plot_lst_map(
            pred_masked=result["pred_celsius_masked"],
            record=record,
            stats=st,
            out_png=png_path,
            crs=engine.grid_crs,
            transform=wtransform,
            point_xy=selected_xy,
        )
        print(f"PNG map             : {png_path}")

        # --- baseline reproducibility vs the frozen UHIDataset reference path
        repro = None
        if not args.skip_repro:
            print("\n[reproducibility] comparing against frozen UHIDataset reference ...")
            repro = _run_reproducibility(
                engine=engine, cfg=cfg, stats=stats, model=model, device=device,
                record=record, point_result=result,
            )
            if repro is not None:
                print(f"  reference sample  : split={repro['split']} sample_id={repro['sample_id']} "
                      f"patch_id={repro['patch_id']} at ({repro['row_off']},{repro['col_off']})")
                print(f"  input max|diff|   : {repro['input_max_abs_diff']:.3e}")
                print(f"  raw max|diff|     : {repro['raw_max_abs_diff']:.3e}  "
                      f"mean|diff| {repro['raw_mean_abs_diff']:.3e}")
                print(f"  celsius max|diff| : {repro['celsius_max_abs_diff']:.3e}  "
                      f"mean|diff| {repro['celsius_mean_abs_diff']:.3e}")
                print(f"  valid agreement   : {repro['valid_agreement']:.6f} "
                      f"(input-window validity, frozen full-grid reference)")

        ckpt_sha_after = sha256_file(checkpoint)

        report = {
            "milestone": "M1_standalone_point_inference",
            "generated_utc": collect_environment()["timestamp_utc"],
            "request": {
                "latitude": float(args.lat),
                "longitude": float(args.lon),
                "ephemeral_point": True,
                "target_month": record["target_month"],
                "target_month_selection": (
                    "default latest available" if args.target_month is None
                    else "explicit --target-month"
                ),
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256_before": ckpt_sha_before,
                "sha256_after": ckpt_sha_after,
                "unchanged": ckpt_sha_before == ckpt_sha_after,
                "epoch": ckpt.get("epoch"),
                "best_val_normalized": ckpt.get("best_val"),
            },
            "input_months": record["input_months"],
            "target_month": record["target_month"],
            "target_month_index": record["target_month_index"],
            "input_shape": list(result["input"].shape),
            "output_shape": [1, 1, PATCH_SIZE, PATCH_SIZE],
            "crs": str(engine.grid_crs),
            "grid": {"width": engine.grid_width, "height": engine.grid_height},
            "window": {
                "row_off": record["row_off"],
                "col_off": record["col_off"],
                "raw_row_off": win.raw_row_off,
                "raw_col_off": win.raw_col_off,
                "edge_snapped": win.edge_snapped,
                "transform": [float(v) for v in wtransform[:6]],
            },
            "point_pixel": {
                "row": record["point_pixel_row"],
                "col": record["point_pixel_col"],
                "projected_x": record["projected_x"],
                "projected_y": record["projected_y"],
            },
            "normalization": {
                "source": str(paths["dataset_root"] / "normalization_stats.json"),
                "lst_channel": LST_CHANNEL,
                "lst_mean": result["lst_mean"],
                "lst_std": result["lst_std"],
                "inverse_transform": "celsius = normalized * lst_std + lst_mean",
            },
            "validity": {
                "rule": "valid where >=1 finite LST observation across the 12 input months",
                "valid_pixel_count": st["valid_pixel_count"],
                "valid_pixel_fraction": st["valid_pixel_fraction"],
            },
            "prediction_stats": st,
            "outputs": {
                "png": str(png_path),
                "lst_geotiff": str(lst_tif),
                "valid_geotiff": str(valid_tif),
                "npy": str(npy_path),
                "json": str(out_dir / "point_inference_report.json"),
            },
            "baseline_reproducibility": repro,
            "environment": collect_environment(),
            "elapsed_seconds": time.perf_counter() - t0,
        }
        write_json(out_dir / "point_inference_report.json", report)
        print(f"JSON report         : {out_dir / 'point_inference_report.json'}")
        print(f"Checkpoint unchanged: {report['checkpoint']['unchanged']} "
              f"(sha256 {ckpt_sha_before[:12]}...)")
    finally:
        engine.close()

    return 0


def _run_reproducibility(*, engine, cfg, stats, model, device, record, point_result):
    """Compare the point engine input/prediction to the frozen dataset path.

    Uses the frozen ``UHIDataset`` (all splits) and selects the sample whose
    patch window and input months exactly match the point's resolved window, so
    the comparison is on the identical input window and checkpoint. Returns
    ``None`` (with a printed note) if no exact match exists.
    """
    from utils.data import make_dataset

    ds = make_dataset(cfg, None)
    fg = None
    try:
        idx = None
        for i, rec in enumerate(ds.records):
            if (
                int(rec["row_off"]) == record["row_off"]
                and int(rec["col_off"]) == record["col_off"]
                and list(rec["input_months"]) == list(record["input_months"])
            ):
                idx = i
                break
        if idx is None:
            print("  no frozen sample matches this window/month; "
                  "reproducibility against stored artifacts is unavailable.")
            return None
        ref_n, ref_c, ref_X, target_mask = reference_model_prediction(model, ds, idx, stats, device)
        rec = ds.records[idx]

        # Frozen input-window validity, from the existing full-grid implementation
        # (same input-validity semantics as the point engine's valid_mask).
        from evaluation.full_grid import FullGridInferenceDataset

        fg = FullGridInferenceDataset(cfg["paths"]["dataset_root"], cfg["paths"]["temporal_root"])
        fg_record = {
            "row_off": record["row_off"], "col_off": record["col_off"],
            "height": PATCH_SIZE, "width": PATCH_SIZE,
            "input_months": list(record["input_months"]),
        }
        ref_input_valid = fg.input_lst_validity(fg_record)

        inp = point_result["input"]
        refX = ref_X.numpy()
        input_diff = np.abs(inp - refX)
        raw_diff = np.abs(point_result["raw_normalized"] - ref_n)
        cel_diff = np.abs(point_result["pred_celsius"] - ref_c)
        valid_agree = float((point_result["valid_mask"] == ref_input_valid).mean())
        return {
            "reference_path": "frozen UHIDataset.__getitem__ + model + frozen inverse transform",
            "input_validity_reference": "evaluation.full_grid.FullGridInferenceDataset.input_lst_validity",
            "split": rec["split"],
            "sample_id": int(rec["sample_id"]),
            "patch_id": int(rec["patch_id"]),
            "row_off": int(rec["row_off"]),
            "col_off": int(rec["col_off"]),
            "input_max_abs_diff": float(input_diff.max()),
            "input_mean_abs_diff": float(input_diff.mean()),
            "raw_max_abs_diff": float(raw_diff.max()),
            "raw_mean_abs_diff": float(raw_diff.mean()),
            "celsius_max_abs_diff": float(cel_diff.max()),
            "celsius_mean_abs_diff": float(cel_diff.mean()),
            "valid_agreement": valid_agree,
            "valid_mask_identical": bool(valid_agree == 1.0),
            "target_month_validity_agreement": float(
                (point_result["valid_mask"] == target_mask).mean()
            ),
            "note": (
                "valid_agreement compares input-window validity to the frozen "
                "full-grid implementation; target_month_validity_agreement is a "
                "different quantity (target-month LST validity) and is informational only."
            ),
        }
    finally:
        ds.close()
        if fg is not None:
            fg.close()


if __name__ == "__main__":
    raise SystemExit(main())