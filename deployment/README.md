# UHI ConvLSTM deployment handoff

This directory provides the deployment layer for the retained baseline model. It contains Python and CLI interfaces only; no HTTP server or webpage is implemented.

## Contents and setup

```
deployment/
  artifacts/convlstm_lst.onnx       validated production graph
  onnx_contract.py                  frozen paths, shapes, statistics, hashes
  export_onnx.py / parity_test.py   export and CPU validation gate
  baseline_inference.py             full-grid ONNX Runtime CPU forecast
  mitigation.py                     registered-strategy simulation adapter
  tests/test_deployment.py          focused checks
```

The model is the original two-layer ConvLSTM from `models/convlstm.py`; weights are loaded strictly from `checkpoints/baseline_full/best.pt`. Do not use `hotspot_w5` or contrast-loss checkpoints.

```powershell
cd C:\uhi-model
.\.venv\Scripts\python.exe -m pip install -r deployment\requirements.txt
.\.venv\Scripts\python.exe -m deployment.export_onnx
.\.venv\Scripts\python.exe -m deployment.parity_test
.\.venv\Scripts\python.exe -m deployment.baseline_inference --target-month 2025-01
```

The frozen source root is resolved through `configs/baseline.yaml`: `C:\Users\PC-012\Desktop\uhi`, which must contain `processed_dataset`, `processed_temporal`, and `uhi_masks`. Missing frozen inputs are hard errors. `requirements.txt` pins the validated ONNX/runtime additions and deliberately does not replace PyTorch.

## Model and data contract

| Item | Frozen value |
| --- | --- |
| ONNX input | `input`, float32 `[B,12,18,128,128]` |
| ONNX output | `lst_normalized`, float32 `[B,1,128,128]` |
| Dynamic dimension | batch `B` only |
| Opset | 17 |
| Horizon | twelve chronological input months → the following month |
| Grid | EPSG:32643; 30 m; 1608 columns × 1493 rows |
| Reconstruction | stride 64 with positive separable triangular overlap blending |
| LST conversion | `normalized * 4.2945619079543595 + 34.327584344740316` °C |

The channel order is read from frozen `normalization_stats.json` at runtime:

1. `LST_Celsius`, 2. `NDVI`, 3. `NDBI`, 4. `MNDWI`, 5. `EVI`, 6. `Albedo`, 7. `Vegetation_Frac`, 8. `Impervious_Frac`, 9. `Soil_Frac`, 10. `Elevation`, 11. `Slope`, 12. `S2_B2`, 13. `S2_B3`, 14. `S2_B4`, 15. `S2_B8`, 16. `S2_B11`, 17. `AirTemp_2m_C`, 18. `Dewpoint_2m_C`.

All per-channel fill values, means, and standard deviations are read from that same frozen file. The preprocessing is the authoritative `FullGridInferenceDataset` path: nonfinite features are replaced with their train fill values and normalized with train statistics; `Elevation` and `Slope` come from frozen `terrain_mean.tif`. Do not reconstruct this feature stack in a frontend.

The catalogue runs from 2019-01 through 2024-12. Stored hindcasts are available from 2020-01 through 2024-12. The supported forward target is 2025-01, supplied by January–December 2024. It is a forecast, not an observed January 2025 temperature or accuracy result.

## Baseline inference

```python
from deployment.baseline_inference import run_baseline, write_baseline_result

result = run_baseline("2025-01", batch_size=4)
write_baseline_result(result, "deployment/artifacts/baseline_2025-01")
```

`result` contains aligned `lst_celsius`, `uhi_celsius`, `valid_mask`, `overlap_weight`, urban/rural masks, CRS, affine transform, input month list, and metrics. The writer produces `lst_forecast_celsius.tif`, `uhi_forecast_celsius.tif`, `coverage_weight.tif`, `lst_valid_uint8.tif`, and `summary.json`.

A predicted pixel is valid when at least one raw LST value is finite in its twelve input months. UHI is preserved exactly from the project: `UHI(x) = LST(x) - mean(valid predicted LST inside the fixed rural mask)`. The rural reference is recomputed for every baseline and scenario. Urban/rural masks are diagnostics rather than model targets.

The validated 2024-12 hindcast uses 2023-12…2024-11 and produced this summary shape:

```json
{"target_month":"2024-12","metrics":{"rural_reference_mean_celsius":28.99440302757333,"urban_mean_uhi_celsius":1.1869011178898057,"valid_prediction_pixels":1290240}}
```

## Mitigation simulation

`deployment.mitigation` is a thin adapter over `scenarios.strategies.STRATEGIES`. It obtains a baseline, applies the existing strategy to a copied latest input frame, runs the same ONNX model, and returns scenario outputs on the same valid domain.

```python
from deployment.mitigation import simulate_mitigation, write_simulation_result
from scenarios.vegetation_restoration import VegetationRestorationParams

result = simulate_mitigation(
    "vegetation_restoration", polygon_geojson,
    VegetationRestorationParams(target_vegetation_fraction=0.6, treatment_coverage_pct=40),
    target_month="2025-01")
write_simulation_result(result, "deployment/artifacts/vegetation_2025-01")
```

Input geometry is a GeoJSON `Polygon` or `MultiPolygon` in EPSG:4326 (longitude, latitude). It is projected to EPSG:32643 then rasterized with `all_touched=False` on the frozen grid. Output includes baseline/scenario metrics, `delta_lst_celsius`, `delta_uhi_celsius`, and polygon/eligible/treated masks. Deltas are `scenario - baseline`; negative temperature or UHI indicates cooling.

| Strategy | Existing parameters |
| --- | --- |
| `vegetation_restoration` | `target_vegetation_fraction`, `treatment_coverage_pct`, `max_treatment_area_m2`, `soil_weight` |
| `reflective_surface` | `target_albedo`, `treatment_coverage_pct`, `max_treatment_area_m2`, `s_down_w_m2`, `min_impervious_fraction` |
| `bare_soil_restoration` | `target_vegetation_fraction`, `treatment_coverage_pct`, `max_treatment_area_m2`, `min_soil_fraction`, `max_vegetation_fraction`, `exclude_water`, `mndwi_water_threshold` |

CLI example; `--params-json` maps directly to the selected existing parameter dataclass:

```powershell
.\.venv\Scripts\python.exe -m deployment.mitigation vegetation_restoration `
  --polygon request_polygon.geojson `
  --params-json '{"treatment_coverage_pct":25,"target_vegetation_fraction":0.6}'
```

Simulation writes baseline/scenario LST, delta LST, delta UHI, treated-mask GeoTIFFs, and `summary.json`. Global selection applies each strategy's existing polygon, eligibility, coverage and area constraints before patch inference. Only its registered channels in the newest frame can change.

Dependent spectral indices and reflectance features may remain unchanged after intervention. These outputs are one-step model counterfactuals, not physical energy-balance simulations, causal estimates, or measured cooling.

## Proposed API contracts

These are proposals for a web service; no endpoints exist here.

`POST /forecast` request:

```json
{"target_month":"2025-01"}
```

Return target/input months, metrics, CRS/transform, and service-managed raster asset URLs or IDs. Keep the 2.4-million-pixel arrays out of JSON.

`POST /simulate` request:

```json
{"target_month":"2025-01","strategy":"reflective_surface","polygon":{"type":"Polygon","coordinates":[...]},"parameters":{"target_albedo":0.30,"treatment_coverage_pct":40}}
```

Return `baseline_metrics`, `scenario_metrics`, `delta_sign`, intervention metadata, and asset IDs/URLs for baseline, scenario, deltas, validity, and treatment masks. Validate date, geometry type/CRS, numeric parameters, and strategy name; propagate an inference error instead of fabricating data.

## Validation and updates

| Artifact | SHA-256 |
| --- | --- |
| checkpoint `best.pt` | `72cb9a6422f073aa5419a12f806c8fbb2a38fc46332c4a0dad82249c423fa91b` |
| `artifacts/convlstm_lst.onnx` | `5ba7395c217eb7d97534d710c8de8c4bf349eadad59bc4989bf6cbf88b06bd20` |

CPU PyTorch versus CPU ONNX Runtime on three real frozen patches and dynamic batches 1/2/3: maximum absolute error `5.1195167397908684e-06 °C`, mean `4.28068996386593e-07 °C`, repeatability passed. The checkpoint was unchanged before and after export/parity. A 2024-12 full-grid ONNX comparison against the stored CUDA fixture over 1,290,240 pixels measured maximum `0.008586883544921875 °C`, mean `0.00046936794671983945 °C`.

Run relevant checks:

```powershell
.\.venv\Scripts\python.exe -m deployment.parity_test
.\.venv\Scripts\python.exe -m tests.test_full_grid
.\.venv\Scripts\python.exe -m tests.test_reconstruction_masks
.\.venv\Scripts\python.exe -m tests.test_spatial_uhi
```

The validated environment does not include `pytest`; repository tests run as modules. For an intentional checkpoint, frozen-data, grid, channel, or normalization update: export again, pass parity, run a stored-month fixture comparison, record new hashes/results, and update this README. Never overwrite the retained checkpoint or frozen inputs.
