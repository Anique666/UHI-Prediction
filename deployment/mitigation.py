"""Thin ONNX inference adapter over the registered UHI mitigation strategies."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from deployment.baseline_inference import run_baseline
from deployment.onnx_contract import ContractError
from scenarios.strategies import STRATEGIES, get_strategy
from scenarios.vegetation_restoration import polygon_to_mask, select_by_priority


def _global_treatment(strategy_name, raw, polygon_mask, valid_mask, params, pixel_area_m2):
    """Use each existing strategy's own eligibility/priority implementation once."""
    if strategy_name == "vegetation_restoration":
        from scenarios.vegetation_restoration import select_treatment_globally
        return select_treatment_globally(raw, polygon_mask=polygon_mask, valid_mask=valid_mask,
                                         params=params, pixel_area_m2=pixel_area_m2)
    if strategy_name == "reflective_surface":
        from scenarios.reflective_surface import _eligibility
        eligible, priority, diagnostics = _eligibility(
            raw, polygon_mask=polygon_mask, valid_mask=valid_mask,
            target_albedo=params.target_albedo, min_impervious_fraction=params.min_impervious_fraction)
    elif strategy_name == "bare_soil_restoration":
        from scenarios.bare_soil_restoration import _eligibility
        eligible, priority, diagnostics = _eligibility(
            raw, polygon_mask=polygon_mask, valid_mask=valid_mask, params=params)
    else:  # guarded by registry lookup; defensive if new strategies are added
        raise KeyError(strategy_name)
    max_pixels = None if params.max_treatment_area_m2 is None else int(np.floor(params.max_treatment_area_m2 / pixel_area_m2))
    treated = select_by_priority(eligible, priority, coverage_pct=params.treatment_coverage_pct, max_pixels=max_pixels)
    diagnostics = dict(diagnostics)
    diagnostics.update({"requested_max_area_pixels": max_pixels, "selected_treated_pixels": int(treated.sum()),
                        "selected_treated_area_m2": float(treated.sum() * pixel_area_m2)})
    return treated, eligible, diagnostics


def simulate_mitigation(
    strategy_name: str, polygon_geojson: dict, params, *, target_month: str = "2025-01",
    onnx_path=None, batch_size: int = 4,
) -> dict:
    """Run baseline then a spatially aligned registered mitigation simulation.

    ``polygon_geojson`` is GeoJSON longitude/latitude (EPSG:4326).  Strategy
    parameters must be the registry strategy's existing parameter dataclass.
    """
    strategy = get_strategy(strategy_name)
    if not isinstance(params, strategy.params_cls):
        raise TypeError(f"{strategy_name} requires {strategy.params_cls.__name__}")
    params.validate()
    kwargs = {"batch_size": batch_size}
    if onnx_path is not None:
        kwargs["onnx_path"] = onnx_path
    baseline = run_baseline(target_month, **kwargs)
    contract = baseline["contract"]
    h, w = baseline["lst_celsius"].shape
    polygon_mask = polygon_to_mask(polygon_geojson, h, w, baseline["transform"], baseline["crs"])
    pixel_area_m2 = abs(float(baseline["transform"].a * baseline["transform"].e))

    # Read the latest frozen physical frame only for existing strategy eligibility.
    # It is discarded before model inference; frozen preprocessing still builds X.
    import rasterio
    latest_label = baseline["input_months"][-1]
    from evaluation.full_grid import FullGridInferenceDataset
    reader_ds = FullGridInferenceDataset(contract.paths["dataset_root"], contract.paths["temporal_root"])
    try:
        path = reader_ds.base._month_path(latest_label)
        with rasterio.open(path) as src:
            latest_raw = src.read().astype(np.float32, copy=True)
    finally:
        reader_ds.close()
    if latest_raw.shape != (len(contract.channel_names), h, w):
        raise ContractError("Latest frozen raster does not match the deployment grid/channel contract")
    treated, eligible, selection = _global_treatment(strategy_name, latest_raw, polygon_mask,
                                                       baseline["valid_mask"], params, pixel_area_m2)
    del latest_raw

    checks = {"input_isolation": True, "eligible_pixel_restriction": bool(not (treated & ~eligible).any()),
              "polygon_restriction": bool(not (treated & ~polygon_mask).any()), "applications": 0}

    def transform(x, record, valid_patch, dataset):
        r, c, s = record["row_off"], record["col_off"], record["height"]
        raw = dataset.base._readers.reader(dataset.base._month_path(latest_label)).read(
            window=((r, r + s), (c, c + s))).astype(np.float32, copy=True)
        original = x.copy()
        app = strategy.apply_fn(x, raw, polygon_mask=polygon_mask[r:r+s, c:c+s],
                                valid_mask=valid_patch, params=params, pixel_area_m2=pixel_area_m2,
                                means=contract.means, stds=contract.stds,
                                treatment_override=treated[r:r+s, c:c+s])
        checks["applications"] += 1
        checks["input_isolation"] &= bool(np.array_equal(x, original)) and app.input is not x
        checks["eligible_pixel_restriction"] &= bool(not (app.treatment_mask & ~app.eligible_mask).any())
        checks["polygon_restriction"] &= bool(not (app.treatment_mask & ~app.polygon_mask).any())
        return app.input

    scenario = run_baseline(target_month, input_transform=transform, **kwargs)
    common = baseline["valid_mask"] & scenario["valid_mask"]
    if not np.array_equal(baseline["valid_mask"], scenario["valid_mask"]):
        raise ContractError("Scenario changed the frozen input-validity spatial domain")
    delta_lst = np.where(common, scenario["lst_celsius"] - baseline["lst_celsius"], np.nan).astype(np.float32)
    delta_uhi = np.where(common, scenario["uhi_celsius"] - baseline["uhi_celsius"], np.nan).astype(np.float32)
    metadata = {"strategy": strategy_name, "params": asdict(params) if is_dataclass(params) else str(params),
                "polygon_input_crs": "EPSG:4326", "raster_crs": str(baseline["crs"]), "pixel_area_m2": pixel_area_m2,
                "polygon_pixels": int(polygon_mask.sum()), "eligible_pixels": int(eligible.sum()),
                "treated_pixels": int(treated.sum()), "treated_area_m2": float(treated.sum() * pixel_area_m2),
                "selection": selection,
                "known_feature_limitation": "Dependent spectral indices and reflectance features may remain unchanged after intervention; this is a one-step model counterfactual, not a physical or causal prediction."}
    return {"baseline": baseline, "scenario": scenario, "delta_lst_celsius": delta_lst, "delta_uhi_celsius": delta_uhi,
            "treated_mask": treated, "eligible_mask": eligible, "polygon_mask": polygon_mask, "metadata": metadata,
            "validation": {**checks, "same_valid_domain": True,
                           "repeatability": "Deterministic ONNX Runtime and registered strategy selection; exercised by focused tests."}}


def write_simulation_result(result: dict, output_dir: str | Path) -> dict:
    """Write aligned scenario rasters and a compact JSON handoff record."""
    from evaluation.full_grid import write_float32_tif, write_uint8_tif
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    base, scen = result["baseline"], result["scenario"]
    for name, arr in (("baseline_lst_celsius.tif", base["lst_celsius"]), ("scenario_lst_celsius.tif", scen["lst_celsius"]),
                      ("delta_lst_celsius.tif", result["delta_lst_celsius"]), ("delta_uhi_celsius.tif", result["delta_uhi_celsius"])):
        write_float32_tif(out / name, arr, base["crs"], base["transform"])
    write_uint8_tif(out / "treated_mask_uint8.tif", result["treated_mask"], base["crs"], base["transform"])
    summary = {"target_month": base["target_month"], "input_months": base["input_months"], "baseline_metrics": base["metrics"],
               "scenario_metrics": scen["metrics"], "metadata": result["metadata"], "validation": result["validation"],
               "delta_sign": "scenario minus baseline; negative temperature/UHI values indicate cooling"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run a registered UHI mitigation through ONNX Runtime")
    ap.add_argument("strategy", choices=sorted(STRATEGIES))
    ap.add_argument("--polygon", required=True, help="path to an EPSG:4326 GeoJSON geometry")
    ap.add_argument("--target-month", default="2025-01"); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--params-json", default=None, help="JSON object of fields accepted by the selected parameter dataclass")
    # The existing parameter dataclasses retain their defaults; CLI callers can
    # use the Python interface for strategy-specific nondefault parameters.
    args = ap.parse_args(argv)
    geometry = json.loads(Path(args.polygon).read_text(encoding="utf-8"))
    params = get_strategy(args.strategy).params_cls(**(json.loads(args.params_json) if args.params_json else {}))
    result = simulate_mitigation(args.strategy, geometry, params, target_month=args.target_month, batch_size=args.batch_size)
    out = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "artifacts" / f"{args.strategy}_{args.target_month}"
    print(json.dumps(write_simulation_result(result, out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
