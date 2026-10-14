"""Full-grid inference for the UHI ConvLSTM baseline.

This module tiles the *complete study grid* (independent of the frozen
train/validation/test split), builds the 12-month / 18-channel input tensor for
each patch by reusing the frozen ``UHIDataset`` internals, runs batched CUDA
inference, and reconstructs the full 1608x1493 grid with deterministic
weighted-overlap blending. It writes a georeferenced forecast LST raster in
Celsius, a coverage raster, and a forecast-derived UHI raster built from the
fixed urban/rural reference masks.

Design notes
------------
* ``FullGridInferenceDataset`` subclasses the frozen ``UHIDataset`` so the
  windowed reads, terrain-mean replacement, train fill and train normalization
  are *exactly* the frozen contract. It only adds an inference patch index and
  an input-only builder (the target raster is never read for prediction).
* Patch tiling uses stride 64 with an extra edge patch per axis so the whole
  grid is covered; only patches overlapping the study region (urban|rural) are
  kept, so areas genuinely outside the study region remain nodata.
* A predicted pixel is valid only where the 12-month input window contains at
  least one genuine (finite) LST observation. Missing areas are left as nodata;
  no values are invented.
* ``--smoke`` runs the same pipeline with a deterministic persistence
  predictor instead of a trained checkpoint. Smoke output is labelled and must
  not be presented as a scientific prediction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.spatial import (  # noqa: E402
    OverlapGrid,
    compute_coverage,
    derive_uhi,
    rural_reference_mean,
)
from models.convlstm import ConvLSTM  # noqa: E402
from training.engine import load_checkpoint  # noqa: E402
from utils.config import get_paths, load_config  # noqa: E402
from utils.data import import_uhidataset  # noqa: E402
from utils.normalization import load_normalization_stats  # noqa: E402
from utils.rasters import load_mask_metadata  # noqa: E402
from utils.runmeta import collect_environment, write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402

PATCH_SIZE = 128
STRIDE = 64


# ---------------------------------------------------------------------------
# Month / patch-index helpers (pure, unit-testable)
# ---------------------------------------------------------------------------
def load_month_labels(dataset_root: str | Path) -> list[str]:
    """Return ordered month labels from the frozen ``month_catalog.json``."""
    with (Path(dataset_root) / "month_catalog.json").open(encoding="utf-8") as f:
        cat = json.load(f)
    return [m["label"] for m in cat["months"]]


def resolve_month_window(
    month_labels: list[str], target_label: str, seq_len: int = 12
) -> tuple[list[int], list[str], int]:
    """Resolve the input month window for a target month.

    Returns ``(input_indices, input_labels, target_index)``. Raises if the
    target month is unknown or its preceding ``seq_len`` months are not
    available, so a forecast is only ever made when a full history exists.
    """
    if target_label not in month_labels:
        raise ValueError(f"Unknown target month {target_label!r}")
    target_index = month_labels.index(target_label)
    start = target_index - seq_len
    if start < 0:
        raise ValueError(
            f"Target month {target_label!r} needs {seq_len} preceding months; "
            f"only {target_index} available."
        )
    input_indices = list(range(start, target_index))
    input_labels = [month_labels[i] for i in input_indices]
    return input_indices, input_labels, target_index


def axis_offsets(total: int, patch: int = PATCH_SIZE, stride: int = STRIDE) -> list[int]:
    """Offsets along one axis that tile ``total`` including the far edge."""
    if total < patch:
        raise ValueError(f"total {total} smaller than patch {patch}")
    offs = list(range(0, total - patch + 1, stride))
    if offs[-1] != total - patch:
        offs.append(total - patch)
    return offs


def build_patch_index(
    height: int,
    width: int,
    study_mask: np.ndarray,
    patch: int = PATCH_SIZE,
    stride: int = STRIDE,
    min_study_overlap: int = 1,
) -> list[dict]:
    """Tiling patch index over the study grid, independent of split membership.

    Every patch overlaps at least ``min_study_overlap`` study pixels
    (``study_mask > 0``). Returns records with ``row_off``/``col_off`` and the
    number of overlapping study pixels.
    """
    study = np.asarray(study_mask) > 0
    if study.shape != (height, width):
        raise ValueError("study_mask shape does not match grid")
    rows = axis_offsets(height, patch, stride)
    cols = axis_offsets(width, patch, stride)
    index = []
    for r in rows:
        for c in cols:
            overlap = int(study[r : r + patch, c : c + patch].sum())
            if overlap >= min_study_overlap:
                index.append(
                    {"row_off": int(r), "col_off": int(c),
                     "height": patch, "width": patch, "study_overlap": overlap}
                )
    return index


def coverage_of_patches(
    height: int, width: int, patch_index: list[dict], patch: int = PATCH_SIZE
) -> np.ndarray:
    """Boolean map marking every grid pixel covered by at least one patch."""
    cover = np.zeros((height, width), dtype=bool)
    for rec in patch_index:
        r0, c0 = rec["row_off"], rec["col_off"]
        cover[r0 : r0 + patch, c0 : c0 + patch] = True
    return cover


# ---------------------------------------------------------------------------
# Inference dataset (reuses frozen UHIDataset internals; input-only)
# ---------------------------------------------------------------------------
class FullGridInferenceDataset:
    """Builds inference input tensors using the frozen ``UHIDataset`` state.

    The frozen dataset object supplies readers, month paths, terrain mean and
    train fill/normalization statistics. Only the input builder is provided
    here; the future target raster is never read during prediction.
    """

    def __init__(self, dataset_root: str | Path, input_dir: str | Path):
        UHIDataset = import_uhidataset(Path(dataset_root))
        # split=None loads the frozen metadata/stats/readers; records are not
        # used for inference and are replaced by explicit inference records.
        self.base = UHIDataset(dataset_root=dataset_root, split=None, input_dir=input_dir)

    @property
    def channel_names(self) -> list[str]:
        return self.base.channel_names

    def close(self) -> None:
        self.base.close()

    def build_input(self, record: dict) -> np.ndarray:
        """Return ``X`` ``[12, 18, 128, 128]`` normalized, mirroring the frozen
        ``UHIDataset.__getitem__`` input path exactly (no target read)."""
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
            win = ds.read(
                indexes=[c + 1 for c in dynamic], window=window
            ).astype(np.float32, copy=True)
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
        """Boolean map: at least one genuine LST observation in the window.

        A patch prediction is only considered valid where the model has seen at
        least one real LST value for that pixel (channel 0, the LST channel).
        """
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


class PersistencePredictor(torch.nn.Module):
    """Deterministic smoke predictor: the latest input LST frame.

    Used only when no trained checkpoint is supplied (``--smoke``). It is a
    real deterministic baseline, not random weights, but smoke output is still
    labelled and must not be presented as a trained-model prediction.
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X[:, -1, 0:1]


# ---------------------------------------------------------------------------
# Raster writing
# ---------------------------------------------------------------------------
def write_float32_tif(path: str | Path, arr: np.ndarray, crs, transform) -> None:
    a = np.asarray(arr, dtype=np.float32)
    profile = {
        "driver": "GTiff", "height": a.shape[0], "width": a.shape[1],
        "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
        "nodata": np.nan, "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(a, 1)


def write_uint8_tif(path: str | Path, arr: np.ndarray, crs, transform, nodata=0) -> None:
    a = np.asarray(arr, dtype=np.uint8)
    profile = {
        "driver": "GTiff", "height": a.shape[0], "width": a.shape[1],
        "count": 1, "dtype": "uint8", "crs": crs, "transform": transform,
        "nodata": nodata, "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(a, 1)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_full_grid_inference(
    *,
    model,
    inference_dataset: FullGridInferenceDataset,
    record_template: dict,
    patch_index: list[dict],
    stats,
    urban: np.ndarray,
    rural: np.ndarray,
    crs,
    transform,
    device,
    batch_size: int,
    height: int,
    width: int,
    save_dir: Path,
    checkpoint_label: str,
    predictor_label: str,
    is_smoke: bool,
    limit_patches: int | None = None,
    log=print,
) -> dict:
    """Run the full pipeline and write rasters + summary. Returns the summary."""
    lst_i = stats.index("LST_Celsius")
    mean, std = stats.means[lst_i], stats.stds[lst_i]

    index = patch_index if limit_patches is None else patch_index[: int(limit_patches)]
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    pred_grid = OverlapGrid(height, width, PATCH_SIZE)

    buf_X: list[np.ndarray] = []
    buf_valid: list[np.ndarray] = []
    buf_rc: list[tuple[int, int]] = []

    def flush():
        if not buf_X:
            return
        Xt = torch.from_numpy(np.stack(buf_X)).to(device)
        out = model(Xt)[:, 0].float().cpu().numpy()  # [B,H,W] normalized
        for k in range(out.shape[0]):
            pred_c = out[k] * std + mean
            r0, c0 = buf_rc[k]
            pred_grid.add(pred_c, r0, c0, valid=buf_valid[k])
        buf_X.clear()
        buf_valid.clear()
        buf_rc.clear()
        del Xt, out

    t0 = time.perf_counter()
    for n, rec in enumerate(index):
        full_rec = dict(record_template)
        full_rec.update(rec)
        X = inference_dataset.build_input(full_rec)
        valid = inference_dataset.input_lst_validity(full_rec)
        buf_X.append(X)
        buf_valid.append(valid)
        buf_rc.append((rec["row_off"], rec["col_off"]))
        if len(buf_X) >= batch_size:
            flush()
        if log and (n + 1) % max(batch_size * 10, 1) == 0:
            log(f"  patch {n + 1}/{len(index)}")
    flush()
    elapsed = time.perf_counter() - t0

    lst_map, weight = pred_grid.finalize()
    valid_map = weight > 0

    coverage = compute_coverage(valid_map, urban, rural)
    rural_mean = rural_reference_mean(lst_map, valid_map, rural)
    uhi_map = derive_uhi(lst_map, rural_mean, valid_map)

    write_float32_tif(save_dir / "lst_forecast_celsius.tif", lst_map, crs, transform)
    write_float32_tif(save_dir / "uhi_forecast_celsius.tif", uhi_map, crs, transform)
    write_float32_tif(save_dir / "coverage_weight.tif", weight, crs, transform)
    write_uint8_tif(save_dir / "lst_valid_uint8.tif", valid_map.astype(np.uint8), crs, transform)

    study = (np.asarray(urban) > 0) | (np.asarray(rural) > 0)
    study_uncovered = int((study & ~valid_map).sum())
    outside_study = int((~study).sum())

    summary = {
        "run_id": save_dir.name,
        "predictor": predictor_label,
        "checkpoint": checkpoint_label,
        "is_smoke": bool(is_smoke),
        "science_warning": (
            "SMOKE run: uses a deterministic smoke predictor, not a trained "
            "model. Outputs are pipeline checks only and are NOT scientific "
            "predictions." if is_smoke else
            "Historical hindcast from a spatially trained model. Full-grid "
            "inference is not proof of citywide predictive accuracy."
        ),
        "target_month": record_template["target_month"],
        "target_month_index": record_template["target_month_index"],
        "input_months": record_template["input_months"],
        "methodology": (
            "ConvLSTM forecasts next-month LST on 128x128 patches tiled across "
            "the study grid with stride 64; overlapping predictions are blended "
            "with a deterministic triangular weighted average. UHI is derived "
            "downstream as LST - mean(valid predicted LST over the fixed rural "
            "mask). UHI is not a supervised target."
        ),
        "overlap_method": "triangular separable weighted average (deterministic)",
        "validity_rule": (
            "A predicted pixel is valid only where the 12-month input window "
            "contains at least one finite LST observation; otherwise nodata."
        ),
        "grid": {"width": int(width), "height": int(height), "nodata": "NaN (float rasters)"},
        "crs": str(crs),
        "transform": [float(v) for v in transform[:6]],
        "n_patches": len(index),
        "patches_available": len(patch_index),
        "batch_size": int(batch_size),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "rural_reference_mean_celsius": rural_mean,
        "coverage": coverage,
        "unsupported_areas": {
            "outside_study_pixels": outside_study,
            "study_pixels_without_valid_prediction": study_uncovered,
            "study_coverage_fraction": float(1.0 - study_uncovered / max(int(study.sum()), 1)),
        },
        "artifacts": {
            "lst_forecast_celsius": "lst_forecast_celsius.tif",
            "uhi_forecast_celsius": "uhi_forecast_celsius.tif",
            "coverage_weight": "coverage_weight.tif",
            "lst_valid_uint8": "lst_valid_uint8.tif",
        },
    }
    write_json(save_dir / "inference_summary.json", summary)
    write_json(save_dir / "environment.json", collect_environment())
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Full-grid inference for UHI")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    ap.add_argument("--target-month", default="2024-12")
    ap.add_argument("--checkpoint", default=None, help="Trained best.pt (required unless --smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="Run without a checkpoint using a deterministic persistence predictor")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-patches", type=int, default=None)
    ap.add_argument("--min-study-overlap", type=int, default=1)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.smoke and not args.checkpoint:
        raise SystemExit("Provide --checkpoint PATH or use --smoke for a pipeline check.")

    month_labels = load_month_labels(paths["dataset_root"])
    input_indices, input_labels, target_index = resolve_month_window(
        month_labels, args.target_month, int(cfg["data"]["sequence_length"])
    )

    stats = load_normalization_stats(paths["dataset_root"])
    meta = load_mask_metadata(paths["mask_dir"])
    H, W = int(meta["height"]), int(meta["width"])
    with rasterio.open(paths["mask_dir"] / "urban_mask.tif") as ds:
        urban, transform, crs = ds.read(1), ds.transform, ds.crs
    with rasterio.open(paths["mask_dir"] / "rural_mask.tif") as ds:
        rural = ds.read(1)
    study = (urban > 0) | (rural > 0)

    patch_index = build_patch_index(
        H, W, study, PATCH_SIZE, STRIDE, int(args.min_study_overlap)
    )
    cover = coverage_of_patches(H, W, patch_index, PATCH_SIZE)
    study_pixels = int(study.sum())
    study_covered_pixels = int((study & cover).sum())
    print(f"Target month        : {args.target_month} (idx {target_index})")
    print(f"Input months        : {input_labels[0]} .. {input_labels[-1]} ({len(input_labels)})")
    print(f"Study pixels        : {study_pixels}  covered by tiling: {study_covered_pixels}")
    print(f"Inference patches   : {len(patch_index)}")

    predictor_label = "persistence-last-input"
    checkpoint_label = "none"
    if args.checkpoint:
        model = ConvLSTM(
            input_channels=cfg["data"]["input_channels"],
            hidden_channels=cfg["model"]["hidden_channels"],
            kernel_size=cfg["model"]["kernel_size"],
            output_channels=cfg["model"]["output_channels"],
        ).to(device)
        ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
        checkpoint_label = str(args.checkpoint)
        predictor_label = "convlstm"
        print(f"Checkpoint          : {args.checkpoint} (epoch {ckpt.get('epoch')}, "
              f"best_val={ckpt.get('best_val')})")
    else:
        model = PersistencePredictor().to(device)
        print("Predictor           : SMOKE persistence-last-input (no checkpoint)")

    record_template = {
        "sample_id": -1, "split": "inference", "patch_id": -1,
        "input_month_indices": input_indices,
        "input_months": input_labels,
        "target_month_index": target_index,
        "target_month": args.target_month,
    }

    run_id = args.run_id or (
        f"fullgrid_smoke_{args.target_month}" if args.smoke
        else f"fullgrid_{args.target_month}_{Path(args.checkpoint).parent.name}"
    )
    save_dir = paths["evaluation_dir"] / run_id

    inference_dataset = FullGridInferenceDataset(paths["dataset_root"], paths["temporal_root"])
    try:
        summary = run_full_grid_inference(
            model=model,
            inference_dataset=inference_dataset,
            record_template=record_template,
            patch_index=patch_index,
            stats=stats,
            urban=urban,
            rural=rural,
            crs=crs,
            transform=transform,
            device=device,
            batch_size=int(args.batch_size or cfg["evaluation"]["batch_size"]),
            height=H,
            width=W,
            save_dir=save_dir,
            checkpoint_label=checkpoint_label,
            predictor_label=predictor_label,
            is_smoke=bool(args.smoke),
            limit_patches=args.limit_patches,
        )
    finally:
        inference_dataset.close()

    print(f"\nRural reference mean: {summary['rural_reference_mean_celsius']:.3f} C")
    cov = summary["coverage"]
    print(f"Urban coverage      : {cov['urban_coverage']*100:.1f}% "
          f"({cov['urban_covered']}/{cov['urban_pixels']})")
    print(f"Rural coverage      : {cov['rural_coverage']*100:.1f}% "
          f"({cov['rural_covered']}/{cov['rural_pixels']})")
    print(f"Elapsed             : {summary['elapsed_seconds']:.1f}s")
    print(f"Artifacts           : {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())