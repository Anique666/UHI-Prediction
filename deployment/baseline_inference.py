"""ONNX Runtime CPU baseline inference using the frozen UHI preprocessing path."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.onnx_contract import ContractError, load_contract, sha256_file
from deployment.export_onnx import DEFAULT_ONNX


def _next_month(label: str) -> str:
    d = datetime.strptime(label, "%Y-%m")
    return f"{d.year + (d.month == 12):04d}-{1 if d.month == 12 else d.month + 1:02d}"


def resolve_forecast_month(contract, target_month: str) -> dict:
    """Resolve a stored hindcast or the single supported next-month forecast."""
    try:
        datetime.strptime(target_month, "%Y-%m")
    except ValueError as exc:
        raise ValueError("target_month must use YYYY-MM") from exc
    labels = contract.month_labels
    if target_month in labels:
        target_index = labels.index(target_month)
        if target_index < contract.sequence_length:
            raise ValueError(f"{target_month} has fewer than {contract.sequence_length} historical months")
        inputs = labels[target_index - contract.sequence_length : target_index]
        is_forward = False
    elif target_month == _next_month(labels[-1]):
        target_index = len(labels)
        inputs = labels[-contract.sequence_length :]
        is_forward = True
    else:
        raise ValueError(
            f"Unsupported target {target_month}; choose a stored month from {labels[contract.sequence_length]} "
            f"through {labels[-1]}, or the next forecast month {_next_month(labels[-1])}."
        )
    return {"target_month": target_month, "target_month_index": target_index,
            "input_months": inputs, "is_forward_forecast": is_forward}


def validate_runtime_artifacts(onnx_path: str | Path = DEFAULT_ONNX) -> dict:
    """Validate frozen checkpoint identity, graph validity, and CPU Runtime I/O."""
    import onnx
    import onnxruntime as ort

    contract = load_contract()
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise ContractError(f"ONNX artifact missing: {onnx_path}; run deployment.export_onnx")
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if "CPUExecutionProvider" not in session.get_providers():
        raise ContractError("ONNX Runtime CPUExecutionProvider is unavailable")
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    if (inp.name, out.name) != (contract.input_name, contract.output_name):
        raise ContractError("ONNX I/O names do not match the deployment contract")
    if tuple(inp.shape[1:]) != contract.input_shape_tail or tuple(out.shape[1:]) != contract.output_shape_tail:
        raise ContractError("ONNX I/O shapes do not match the deployment contract")
    if inp.type != "tensor(float)" or out.type != "tensor(float)":
        raise ContractError("ONNX I/O must be float32")
    return {"contract": contract, "session": session, "onnx_path": str(onnx_path),
            "onnx_sha256": sha256_file(onnx_path), "checkpoint_sha256": sha256_file(contract.checkpoint_path)}


def _load_masks(contract):
    from utils.rasters import read_mask_raster

    urban, profile = read_mask_raster(contract.paths["mask_dir"] / "urban_mask.tif")
    rural, rural_profile = read_mask_raster(contract.paths["mask_dir"] / "rural_mask.tif")
    if urban.shape != rural.shape or profile["transform"] != rural_profile["transform"]:
        raise ContractError("Frozen urban/rural masks are not spatially aligned")
    if urban.shape != (contract.grid_height, contract.grid_width):
        raise ContractError("Frozen masks do not match the deployment grid")
    return urban, rural, profile


def run_baseline(
    target_month: str = "2025-01", *, onnx_path: str | Path = DEFAULT_ONNX,
    batch_size: int = 4, input_transform=None,
) -> dict:
    """Forecast/reconstruct one target month and return aligned arrays plus metrics."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    from evaluation.full_grid import FullGridInferenceDataset, build_patch_index
    from evaluation.spatial import OverlapGrid, compute_coverage, derive_uhi, rural_reference_mean

    runtime = validate_runtime_artifacts(onnx_path)
    contract, session = runtime["contract"], runtime["session"]
    window = resolve_forecast_month(contract, target_month)
    urban, rural, profile = _load_masks(contract)
    patches = build_patch_index(contract.grid_height, contract.grid_width, (urban > 0) | (rural > 0),
                                contract.patch_size, contract.stride)
    dataset = FullGridInferenceDataset(contract.paths["dataset_root"], contract.paths["temporal_root"])
    grid = OverlapGrid(contract.grid_height, contract.grid_width, contract.patch_size)
    xs, valid_masks, offsets = [], [], []

    def flush():
        if not xs:
            return
        x = np.ascontiguousarray(np.stack(xs), dtype=np.float32)
        pred = session.run([contract.output_name], {contract.input_name: x})[0][:, 0]
        for i, values in enumerate(pred):
            celsius = values.astype(np.float64) * contract.lst_std + contract.lst_mean
            grid.add(celsius, *offsets[i], valid=valid_masks[i])
        xs.clear(); valid_masks.clear(); offsets.clear()

    try:
        for patch in patches:
            rec = {**patch, **window}
            x = dataset.build_input(rec)
            valid_patch = dataset.input_lst_validity(rec)
            if input_transform is not None:
                x = input_transform(x, rec, valid_patch, dataset)
                if x.shape != (contract.sequence_length, len(contract.channel_names), contract.patch_size, contract.patch_size):
                    raise ContractError("Input transform returned an invalid inference tensor shape")
                if x.dtype != np.float32:
                    raise ContractError("Input transform must return float32")
            xs.append(x)
            valid_masks.append(valid_patch)
            offsets.append((patch["row_off"], patch["col_off"]))
            if len(xs) == batch_size:
                flush()
        flush()
    finally:
        dataset.close()
    lst_celsius, overlap_weight = grid.finalize()
    valid = overlap_weight > 0
    rural_mean = rural_reference_mean(lst_celsius, valid, rural)
    uhi_celsius = derive_uhi(lst_celsius, rural_mean, valid)
    coverage = compute_coverage(valid, urban, rural)
    urban_valid = valid & (urban > 0)
    metrics = {
        "rural_reference_mean_celsius": rural_mean,
        "urban_mean_uhi_celsius": float(np.nanmean(uhi_celsius[urban_valid])) if urban_valid.any() else float("nan"),
        "predicted_lst_mean_celsius": float(np.nanmean(lst_celsius)),
        "valid_prediction_pixels": int(valid.sum()),
        "coverage": coverage,
    }
    return {**window, "contract": contract, "onnx": {k: runtime[k] for k in ("onnx_path", "onnx_sha256", "checkpoint_sha256")},
            "lst_celsius": lst_celsius.astype(np.float32), "uhi_celsius": uhi_celsius.astype(np.float32),
            "valid_mask": valid, "overlap_weight": overlap_weight.astype(np.float32), "urban_mask": urban,
            "rural_mask": rural, "crs": profile["crs"], "transform": profile["transform"], "metrics": metrics,
            "patch_count": len(patches), "batch_size": batch_size}


def write_baseline_result(result: dict, output_dir: str | Path) -> dict:
    """Write GeoTIFF result files and a JSON summary under ``deployment/``."""
    from evaluation.full_grid import write_float32_tif, write_uint8_tif

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    write_float32_tif(out / "lst_forecast_celsius.tif", result["lst_celsius"], result["crs"], result["transform"])
    write_float32_tif(out / "uhi_forecast_celsius.tif", result["uhi_celsius"], result["crs"], result["transform"])
    write_float32_tif(out / "coverage_weight.tif", result["overlap_weight"], result["crs"], result["transform"])
    write_uint8_tif(out / "lst_valid_uint8.tif", result["valid_mask"], result["crs"], result["transform"])
    summary = {k: result[k] for k in ("target_month", "target_month_index", "input_months", "is_forward_forecast", "onnx", "metrics", "patch_count", "batch_size")}
    summary.update({"crs": str(result["crs"]), "transform": [float(v) for v in result["transform"][:6]],
                    "artifacts": {"lst_celsius": "lst_forecast_celsius.tif", "uhi_celsius": "uhi_forecast_celsius.tif",
                                  "overlap_weight": "coverage_weight.tif", "valid_mask": "lst_valid_uint8.tif"}})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def compare_fixture(result: dict, fixture_lst_path: str | Path) -> dict:
    """Compare an ONNX reconstructed field with a stored geospatial fixture."""
    import rasterio
    with rasterio.open(fixture_lst_path) as ds:
        fixture = ds.read(1).astype(np.float64)
        if fixture.shape != result["lst_celsius"].shape or ds.transform != result["transform"] or ds.crs != result["crs"]:
            raise ContractError("Stored fixture is not geospatially aligned with the result")
    valid = result["valid_mask"] & np.isfinite(fixture)
    err = np.abs(result["lst_celsius"].astype(np.float64)[valid] - fixture[valid])
    return {"fixture": str(fixture_lst_path), "compared_pixels": int(valid.sum()),
            "max_abs_celsius": float(err.max()), "mean_abs_celsius": float(err.mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run CPU ONNX UHI baseline inference")
    ap.add_argument("--target-month", default="2025-01")
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--fixture-lst", default=None, help="stored aligned LST fixture for validation")
    args = ap.parse_args(argv)
    result = run_baseline(args.target_month, onnx_path=args.onnx, batch_size=args.batch_size)
    output = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "artifacts" / f"baseline_{args.target_month}"
    summary = write_baseline_result(result, output)
    if args.fixture_lst:
        summary["fixture_comparison"] = compare_fixture(result, args.fixture_lst)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
