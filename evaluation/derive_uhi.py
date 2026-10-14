"""Spatial reconstruction and citywide UHI derivation from forecast LST.

For one selected target month this script:

1. runs the trained ConvLSTM on the **test-split** samples whose target month
   matches the selected month (chronological 12-month windows);
2. converts predicted normalized LST to degrees Celsius;
3. places predictions into the full 1608x1493 grid at their recorded patch
   offsets using a deterministic triangular weighted-overlap average;
4. masks to genuine predicted pixels (target-valid);
5. computes the rural reference mean over valid predicted pixels in the fixed
   rural mask;
6. derives forecast UHI = predicted LST - rural mean;
7. independently derives **observed** UHI from the raw target LST map using the
   same fixed rural reference definition.

Honesty about coverage
----------------------
The test split is a spatially disjoint region, so the reconstruction covers only
test patches. Coverage of the urban/rural masks is measured and written to the
output metadata; areas not covered by test patches are left as nodata. Training
or validation patches are never used to fill test areas.
"""
from __future__ import annotations

import argparse
import sys
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
from utils.data import make_dataset  # noqa: E402
from utils.normalization import load_normalization_stats  # noqa: E402
from utils.rasters import MonthRasterReader, load_mask_metadata  # noqa: E402
from utils.runmeta import collect_environment, write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Derive UHI from forecast LST")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-month", default="2024-12",
                    help="Calendar month label to reconstruct (must be a test target)")
    ap.add_argument("--run-id", default=None)
    return ap.parse_args(argv)


@torch.no_grad()
def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import json

    with (paths["dataset_root"] / "month_catalog.json").open(encoding="utf-8") as f:
        catalog = json.load(f)
    month_to_index = {m["label"]: int(m["index"]) for m in catalog["months"]}
    if args.target_month not in month_to_index:
        raise ValueError(f"Unknown target month {args.target_month!r}")
    target_index = month_to_index[args.target_month]

    stats = load_normalization_stats(paths["dataset_root"])
    lst_i = stats.index("LST_Celsius")
    mean, std = stats.means[lst_i], stats.stds[lst_i]

    meta = load_mask_metadata(paths["mask_dir"])
    H, W = int(meta["height"]), int(meta["width"])
    with rasterio.open(paths["mask_dir"] / "urban_mask.tif") as ds:
        urban = ds.read(1)
        transform = ds.transform
        crs = ds.crs
    with rasterio.open(paths["mask_dir"] / "rural_mask.tif") as ds:
        rural = ds.read(1)

    run_id = args.run_id or f"uhi_{args.target_month}_{Path(args.checkpoint).parent.name}"
    out_dir = paths["evaluation_dir"] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ConvLSTM(
        input_channels=cfg["data"]["input_channels"],
        hidden_channels=cfg["model"]["hidden_channels"],
        kernel_size=cfg["model"]["kernel_size"],
        output_channels=cfg["model"]["output_channels"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    test_ds = make_dataset(cfg, "test")
    reader = MonthRasterReader(paths["dataset_root"], paths["temporal_root"])
    lst_band = cfg["data"]["lst_channel_index"] + 1

    pred_grid = OverlapGrid(H, W, 128)
    obs_grid = OverlapGrid(H, W, 128)
    n_matched = 0
    try:
        for idx in range(len(test_ds)):
            rec = test_ds.records[idx]
            if int(rec["target_month_index"]) != target_index:
                continue
            X, y, m = test_ds[idx]
            Xb = X.unsqueeze(0).to(device)
            pred = model(Xb)[0, 0].cpu().numpy() * std + mean  # Celsius
            valid = m[0].numpy().astype(bool)
            # observed target LST (Celsius) from the raw raster
            obs = y[0].numpy() * std + mean
            r0, c0 = int(rec["row_off"]), int(rec["col_off"])
            pred_grid.add(pred, r0, c0, valid=valid)
            obs_grid.add(obs, r0, c0, valid=valid)
            n_matched += 1
            del X, y, m, Xb, pred, obs, valid
    finally:
        test_ds.close()
        reader.close()

    pred_map, weight = pred_grid.finalize()
    obs_map, obs_weight = obs_grid.finalize()
    pred_valid = weight > 0
    obs_valid = obs_weight > 0

    coverage = compute_coverage(pred_valid, urban, rural)
    rural_mean_forecast = rural_reference_mean(pred_map, pred_valid, rural)
    rural_mean_observed = rural_reference_mean(obs_map, obs_valid, rural)
    uhi_forecast = derive_uhi(pred_map, rural_mean_forecast, pred_valid)
    uhi_observed = derive_uhi(obs_map, rural_mean_observed, obs_valid)

    _write_raster(out_dir / "lst_forecast_celsius.tif", pred_map, crs, transform)
    _write_raster(out_dir / "uhi_forecast_celsius.tif", uhi_forecast, crs, transform)
    _write_raster(out_dir / "lst_observed_celsius.tif", obs_map, crs, transform)
    _write_raster(out_dir / "uhi_observed_celsius.tif", uhi_observed, crs, transform)
    _write_raster(out_dir / "coverage_weight.tif", weight, crs, transform)

    summary = {
        "run_id": run_id,
        "target_month": args.target_month,
        "target_month_index": target_index,
        "methodology": (
            "Forecast LST is the ConvLSTM prediction; UHI is derived downstream as "
            "LST - mean(valid predicted LST over the fixed rural mask). Observed UHI "
            "uses the raw target LST map with the same rural reference definition."
        ),
        "distinction": (
            "The network forecasts next-month LST. UHI is NOT a supervised target; "
            "it is derived from the forecast field using urban/rural reference masks."
        ),
        "n_test_samples_matched": n_matched,
        "crs": str(crs),
        "transform": list(transform)[:6],
        "grid": {"width": W, "height": H, "nodata": "NaN"},
        "overlap_method": "triangular separable weighted average (deterministic)",
        "rural_reference_mean_forecast_celsius": rural_mean_forecast,
        "rural_reference_mean_observed_celsius": rural_mean_observed,
        "coverage": coverage,
        "coverage_limitation": (
            "Reconstruction uses test-split patches only. The test split is a "
            "spatially disjoint region, so only test patches contribute; uncovered "
            "pixels are nodata. This is not full-city coverage."
        ),
    }
    write_json(out_dir / "uhi_summary.json", summary)
    write_json(out_dir / "environment.json", collect_environment())

    print(f"Target month            : {args.target_month}")
    print(f"Matched test samples    : {n_matched}")
    print(f"Rural mean (forecast)   : {rural_mean_forecast:.3f} C")
    print(f"Rural mean (observed)   : {rural_mean_observed:.3f} C")
    print(f"Urban mask coverage     : {coverage['urban_coverage']*100:.1f}%")
    print(f"Rural mask coverage     : {coverage['rural_coverage']*100:.1f}%")
    print(f"Artifacts               : {out_dir}")
    return 0


def _write_raster(path, arr, crs, transform):
    a = np.asarray(arr, dtype=np.float32)
    profile = {
        "driver": "GTiff",
        "height": a.shape[0],
        "width": a.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(a, 1)


if __name__ == "__main__":
    raise SystemExit(main())