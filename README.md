# UHI ConvLSTM — Model Source of Truth

This repository is the **source of truth for the trained UHI/ConvLSTM model and
its model-side implementation**: the architecture, the retained trained
artifact, the ONNX export, model-side inference/deployment logic, the input
contract, and the building-footprint/land-cover **mitigation counterfactual**
capability.

It is intentionally scoped. It does **not** contain the preprocessing pipeline
or the teammate-built website/frontend (separate repositories), nor the
datasets, tests, evaluation workspaces, experiments, or training logs/checkpoints
that live around the model. If you are looking for data acquisition,
preprocessing, or the web UI, use those separate repositories.

---

## 1. What the model does

The model forecasts **next-month land-surface temperature (LST)** over the
Bengaluru study area. Given **12 consecutive monthly 18-channel spatial rasters**
it predicts the **13th** month's LST field.

```
18-channel input  →  12-month temporal sequence  →  ConvLSTM  →  predicted LST (normalized)
                                                              →  physical temperature (°C)
```

Urban Heat Island (UHI) intensity is **not** a supervised target. It is derived
downstream from a predicted LST field as the contrast against a fixed rural
reference:

```
UHI(t, x, y) = LST(t, x, y) − mean(LST(t) over the fixed rural mask)
```

The mitigation capability edits the model's **input features** inside a user
polygon (e.g. raise vegetation fraction, raise albedo) on the latest frame,
runs baseline and scenario inference through the **same frozen model**, and
reports the difference (`ΔLST`, `ΔUHI`). It is a one-step model-estimated
counterfactual, **not** a measured or causally validated physical effect.

---

## 2. Model architecture

Source of truth: `models/convlstm.py` (frozen, reused unchanged).

| Property | Value |
|---|---|
| Type | Stacked ConvLSTM |
| Layers | 2 `ConvLSTMCell` |
| Hidden channels | 64 (each layer) |
| Kernel size | 3×3 (padding 1, H×W preserved) |
| Gates | one `Conv2d(18+64 → 4·64)` (layer 1); `Conv2d(64+64 → 256)` (layer 2) |
| Activations | `sigmoid` on i/f/o gates; `tanh` on candidate `g` and on `c_next` |
| Head | `Conv2d(64→32, 3×3) → ReLU → Conv2d(32→1, 1×1)` |
| Parameters | **502,849** |
| State init | zero hidden/cell state; only the final hidden state feeds the head |
| Input check | raises unless exactly 18 channels and a 5-D input |

The 12 timesteps are consumed sequentially, carrying the spatial `h`/`c` state.
The model uses only standard ops (`Conv2d`, `sigmoid`, `tanh`, elementwise
mul/add, `cat`, `ReLU`, `torch.chunk`), so it is ONNX-exportable.

An unused residual-head variant lives in `models/residual.py` (same backbone,
predicts a residual over the last LST frame; same parameter count and I/O
contract). It is **not** the retained model.

---

## 3. Input contract

| Item | Shape / value |
|---|---|
| Input `X` | `[B, 12, 18, 128, 128]` float32 — 12 months × 18 channels × 128×128 patch |
| Target `y` | `[B, 1, 128, 128]` float32 — normalized next-month LST |
| Target-validity mask | `[B, 1, 128, 128]` bool |
| Sequence length | 12 (preceding months) |
| Patch size | 128 |

### Input-channel order

The channel order is read at runtime from the frozen
`normalization_stats.json` (`channel_names`). It is fixed as:

```
 0 LST_Celsius        (target channel / latest-frame persistence source)
 1 NDVI
 2 NDBI
 3 MNDWI
 4 EVI
 5 Albedo
 6 Vegetation_Frac
 7 Impervious_Frac
 8 Soil_Frac
 9 Elevation          (terrain; replaced by a per-pixel temporal mean)
10 Slope              (terrain; replaced by a per-pixel temporal mean)
11 S2_B2
12 S2_B3
13 S2_B4
14 S2_B8
15 S2_B11
16 AirTemp_2m_C
17 Dewpoint_2m_C
```

Terrain channels (9, 10) are **not** time-varying: they are replaced by a
precomputed per-pixel temporal mean repeated across all 12 steps.

### Temporal convention

The model uses the **preceding 12 months** to predict the **next** month.
Target month `2024-12` uses inputs `2023-12 … 2024-11`. A forecast is only made
when a full 12-month history exists. The frozen data spans 72 months
(2019-01 … 2024-12); the latest supported target is **2024-12**, and the single
supported forward month is **2025-01** (inputs January–December 2024).

### Normalization

```
x_norm = (x_raw − train_mean) / train_std     (train-split-only per-channel stats)
LST_C  = LST_norm × 4.2945619079543595 + 34.327584344740316
```

Missing/invalid inputs are first filled with the frozen per-channel fill value,
then normalized. Inverse normalization recovers °C.

---

## 4. Retained model artifact

| | |
|---|---|
| Run id | `baseline_full` |
| Checkpoint | `checkpoints/baseline_full/best.pt` |
| Checkpoint SHA-256 | `72cb9a6422f073aa5419a12f806c8fbb2a38fc46332c4a0dad82249c423fa91b` |
| Epoch | 48 (0-indexed) |
| Best val masked MAE (normalized) | 0.3101475038587043 |
| Config | `configs/baseline.yaml` |

This is the **only retained scientific model** — the baseline/full model. Two
training experiments (hotspot-weighted loss; urban–rural contrast loss) were
evaluated and **rejected**; neither replaced the baseline. Their checkpoints are
not part of this repository.

The deployment ONNX artifact is the export of this checkpoint:

| | |
|---|---|
| ONNX | `deployment/artifacts/convlstm_lst.onnx` |
| ONNX SHA-256 | `5ba7395c217eb7d97534d710c8de8c4bf349eadad59bc4989bf6cbf88b06bd20` |
| ONNX input | `input` float32 `[B, 12, 18, 128, 128]` (dynamic batch) |
| ONNX output | `lst_normalized` float32 `[B, 1, 128, 128]` |
| Opset | 17 |

The checkpoint and ONNX hashes are mutually consistent: the ONNX was exported
from this exact `best.pt`, and `deployment/artifacts/export_report.json` /
`parity_report.json` record the read-only export and CPU parity result
(max abs error ≈ `5.1e-06 °C`).

### Training setup (high level)

Masked MAE in normalized LST, Adam (lr 1e-3, weight decay 1e-4), batch size 4,
gradient clipping max-norm 1.0, early stopping (patience 10), 60-epoch budget,
seed 42, spatial (directional-quantile) train/val/test partition
(5160 / 780 / 1140 samples). The test split is never used for selection.

### Final results (reference)

- **Validation** (hotspot protocol): LST MAE 1.3318 °C, R² 0.798; UHI MAE
  1.1577 °C; persistence LST MAE 2.8782 °C.
- **Test** (rigorous protocol, unused in selection): LST MAE 1.1065 °C, R²
  0.8337; UHI MAE 1.0109 °C; persistence LST MAE 2.3216 °C.

Do not mix validation and test numbers — they are different populations. A
known observed-LST anomaly in July 2024 dominates validation error and is
retained honestly in the all-month score.

---

## 5. Repository layout

```text
models/
  convlstm.py              # retained 2-layer ConvLSTM (architecture source of truth)
  residual.py              # unused residual-head variant
configs/
  baseline.yaml            # single configuration: paths, model, data contract
utils/
  config.py                # YAML config + path resolution
  normalization.py         # normalization stats, normalized<->Celsius
  data.py                  # dataset construction, split summaries
  rasters.py               # windowed raw raster reads
  runmeta.py               # environment capture
  seed.py                  # deterministic seeding
training/
  train.py                 # training CLI / orchestration
  engine.py                # loops, checkpoint save/load, selection
  losses.py                # masked MAE / MSE / Huber
  metrics.py               # weighted metric aggregation
  baselines.py             # persistence + input-validity baseline
  hotspot_loss.py          # rejected experiment 1 (documented, default OFF)
  contrast_loss.py         # rejected experiment 2 (documented, default OFF)
evaluation/
  spatial.py               # OverlapGrid blending, UHI derivation, coverage
  rigorous.py              # rigorous evaluation routines
  full_grid.py             # split-independent full-grid inference
  derive_uhi.py            # citywide UHI rasters
  hotspots.py              # hotspot metrics
  point_inference.py       # single-point (128x128 patch) inference
scenarios/
  strategies.py            # mitigation strategy registry + shared contract
  vegetation_restoration.py
  reflective_surface.py
  bare_soil_restoration.py
  run_scenario.py          # mitigation counterfactual runner
deployment/
  artifacts/convlstm_lst.onnx   # retained ONNX artifact
  artifacts/export_report.json  # export provenance + hashes
  artifacts/parity_report.json  # CPU PyTorch vs ONNX Runtime parity
  onnx_contract.py              # frozen paths, shapes, stats, hashes
  export_onnx.py                # export the checkpoint to ONNX
  parity_test.py                # CPU parity validation gate
  baseline_inference.py         # full-grid ONNX Runtime CPU inference
  mitigation.py                 # registered-strategy simulation adapter
  requirements.txt              # ONNX/runtime additions
  README.md                     # deployment contract and interface detail
checkpoints/baseline_full/best.pt  # retained trained model (6.0 MB, git-lfs)
evaluate.py / evaluate_hotspots.py # root evaluation entry points
README.md
```

Not included by design: `tests/`, the `evaluation/` and `outputs/` artifact
trees, `experiments/`, `audit_reports/`, `scripts/`, `notebooks/`, all
`checkpoints/` other than `baseline_full/best.pt`, datasets, and logs.

---

## 6. Inference and ONNX deployment

The deployment layer is described in full in `deployment/README.md`. Summary:

```python
from deployment.baseline_inference import run_baseline, write_baseline_result

result = run_baseline("2025-01", batch_size=4)          # ONNX Runtime CPU
write_baseline_result(result, "deployment/artifacts/baseline_2025-01")
```

`result` contains aligned `lst_celsius`, `uhi_celsius`, `valid_mask`,
`overlap_weight`, urban/rural masks, CRS, affine transform, input months and
metrics. Full-grid inference tiles the study grid with stride 64 and blends
overlapping predictions with a deterministic triangular weight. A predicted
pixel is valid only where the 12-month input window contains at least one
genuine LST observation; everything else stays nodata (never extrapolated).

Export and parity (requires CPU PyTorch as the reference):

```powershell
pip install -r deployment/requirements.txt
python -m deployment.export_onnx     # export + graph validation
python -m deployment.parity_test     # CPU PyTorch vs ONNX Runtime gate
```

Contract, shapes, hashes and the validated parity numbers live in
`deployment/onnx_contract.py` and `deployment/artifacts/*.json`.

---

## 7. Building-footprint / land-cover mitigation workflow

`deployment.mitigation` is a thin adapter over the registered mitigation
strategies. It obtains a baseline, applies an existing strategy to a **copied**
latest input frame, runs the **same ONNX model**, and returns scenario outputs on
the same valid domain:

```python
from deployment.mitigation import simulate_mitigation, write_simulation_result
from scenarios.vegetation_restoration import VegetationRestorationParams

result = simulate_mitigation(
    "vegetation_restoration", polygon_geojson,
    VegetationRestorationParams(target_vegetation_fraction=0.6, treatment_coverage_pct=40),
    target_month="2025-01")
write_simulation_result(result, "deployment/artifacts/vegetation_2025-01")
```

Input geometry is a GeoJSON `Polygon`/`MultiPolygon` in EPSG:4326; it is
projected to EPSG:32643 and rasterized on the frozen grid. Registered strategies
are `vegetation_restoration`, `reflective_surface`, and
`bare_soil_restoration`. Only each strategy's registered channels can change, and
only in the newest frame; historical LST, meteorology, terrain and all other
channels are preserved. Deltas are `scenario − baseline`; negative values
indicate modelled cooling.

> The strategy transformation modules themselves (`scenarios/`) are preserved in
> the separate model/scenario source used to produce the deployment outputs.
> This repository ships the deployment adapter, its ONNX contract and the
> retained model; the preprocessing/dataset that supplies the input stack is a
> separate repository.

Notes and honest limitations: the intervention modifies **input features**, never
temperatures, and no fixed cooling coefficient is subtracted. Dependent spectral
features (NDVI/EVI/Albedo/NDBI/S2) may remain unchanged, so the transforms are
not fully physically consistent. These outputs are one-step model
counterfactuals — not physical energy-balance simulations, causal estimates, or
measured cooling.

---

## 8. Using / inspecting the model

- **Architecture**: read `models/convlstm.py`.
- **Input/output contract**: `deployment/onnx_contract.py` and
  `configs/baseline.yaml`.
- **Retained weights**: `checkpoints/baseline_full/best.pt` (git-lfs).
- **Run it**: `deployment/baseline_inference.py` (ONNX Runtime CPU) or
  `evaluation/full_grid.py` (PyTorch).
- **Mitigation**: `deployment/mitigation.py`.
- **Provenance**: hashes in `deployment/artifacts/export_report.json` and
  `deployment/artifacts/parity_report.json`.

### Environment

Windows 11, Python 3.12, PyTorch 2.13.0+cu132 (CUDA 13.2), NumPy 2.5.3,
rasterio 1.5.2, pyproj 3.8.0, PyYAML 6.0.3. Deployment additionally uses
`onnx==1.23.2` and `onnxruntime==1.31.0`. See `deployment/requirements.txt`.

The frozen preprocessing/dataset that supplies the 18-band monthly rasters is
resolved through `configs/baseline.yaml` paths and lives **outside** this
repository.

---

## 9. Important limitations

- The model predicts LST; UHI is a **downstream** diagnostic, not a model
  target.
- All splits are **spatial**; temporal/future generalization is not
  demonstrated.
- Hot urban pixels are systematically underpredicted (range compression).
- July 2024 is a confirmed observed-LST anomaly dominating validation error; its
  cause (genuine cool event vs interpolation artifact) is undetermined.
- Mitigation outputs are model counterfactuals, not causal or measured effects.

---

## 10. Relationship to sibling repositories

| Repository | Responsibility |
|---|---|
| **this repo** | Retained model, architecture, ONNX, inference/deployment, mitigation adapter, model contracts |
| preprocessing (separate) | Raw acquisition, preprocessing, frozen dataset, masks, normalization stats |
| website/frontend (separate) | User interface; calls the model backend, never recomputes ΔLST/UHI |

This repository is the authoritative description of the model itself.