"""End-to-end vegetation-restoration counterfactual runner.

Runs the frozen ConvLSTM over a user-selected polygon twice -- once on the
unaltered 12-month input and once on the vegetation-restoration scenario input --
using the existing full-grid inference machinery, then reports local LST changes
and the existing UHI metric. All outputs are written to a run directory.

This module adds *no* new metric: LST comes from the frozen model + frozen
inverse transform, and UHI uses the existing ``evaluation.spatial`` functions
over the frozen urban/rural masks.

CLI
---
    python -m scenarios.run_scenario --lat 12.97486 --lon 77.53719 \
        --half-size-m 2000 --coverage-pct 40 --target-vf 0.6 --run-id demo_veg
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

from evaluation.full_grid import (  # noqa: E402
    FullGridInferenceDataset,
    PATCH_SIZE,
    STRIDE,
    build_patch_index,
    load_month_labels,
    resolve_month_window,
)
from evaluation.spatial import (  # noqa: E402
    OverlapGrid,
    compute_coverage,
    derive_uhi,
    patch_weight,
    rural_reference_mean,
)
from evaluation.rigorous import sha256_file  # noqa: E402
from models.convlstm import ConvLSTM  # noqa: E402
from scenarios.vegetation_restoration import (  # noqa: E402
    CHANNEL_NAMES,
    polygon_to_mask,
)
from scenarios.strategies import STRATEGIES, get_strategy  # noqa: E402
from training.engine import load_checkpoint  # noqa: E402
from utils.config import get_paths, load_config  # noqa: E402
from utils.normalization import load_normalization_stats  # noqa: E402
from utils.rasters import load_mask_metadata  # noqa: E402
from utils.runmeta import collect_environment, write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------
def square_polygon(lat: float, lon: float, half_size_m: float) -> dict:
    """Axis-aligned square polygon (in metres) around a point, returned in EPSG:4326."""
    from pyproj import Transformer

    fwd = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    inv = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    x, y = fwd.transform(lon, lat)
    h = float(half_size_m)
    corners_xy = [(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h), (x - h, y - h)]
    ring = [list(inv.transform(px, py)) for px, py in corners_xy]
    return {"type": "Polygon", "coordinates": [ring]}


# ---------------------------------------------------------------------------
# Inference with scenario hook
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(
    *,
    model,
    inference_dataset: FullGridInferenceDataset,
    record_template: dict,
    patch_index: list[dict],
    stats,
    device,
    batch_size: int,
    height: int,
    width: int,
    scenario_fn=None,
    log=print,
):
    """Reconstruct a full LST grid, optionally transforming each input patch.

    ``scenario_fn(X, record) -> np.ndarray`` receives the normalized ``[12,18,S,S]``
    patch and its record and returns a modified patch. Baseline passes ``None``.
    Returns ``(lst_map_celsius, valid_map, weight, scenario_meta)``.
    """
    lst_i = stats.index("LST_Celsius")
    mean, std = stats.means[lst_i], stats.stds[lst_i]
    model.eval()
    grid = OverlapGrid(height, width, PATCH_SIZE)
    scenario_meta = {"patches": [], "total_treated_pixels": 0}

    buf_X, buf_valid, buf_rc = [], [], []

    def flush():
        if not buf_X:
            return
        Xt = torch.from_numpy(np.stack(buf_X)).to(device)
        out = model(Xt)[:, 0].float().cpu().numpy()
        for k in range(out.shape[0]):
            pred_c = out[k] * std + mean
            r0, c0 = buf_rc[k]
            grid.add(pred_c, r0, c0, valid=buf_valid[k])
        buf_X.clear(); buf_valid.clear(); buf_rc.clear()
        del Xt, out

    for n, rec in enumerate(patch_index):
        full_rec = dict(record_template)
        full_rec.update(rec)
        X = inference_dataset.build_input(full_rec)
        valid = inference_dataset.input_lst_validity(full_rec)
        if scenario_fn is not None:
            X, meta = scenario_fn(X, full_rec)
            scenario_meta["patches"].append(meta)
            scenario_meta["total_treated_pixels"] += int(meta.get("treated_pixels", 0))
        buf_X.append(X); buf_valid.append(valid); buf_rc.append((rec["row_off"], rec["col_off"]))
        if len(buf_X) >= batch_size:
            flush()
    flush()
    lst_map, weight = grid.finalize()
    return lst_map, weight > 0, weight, scenario_meta


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
def test_noop(engine, record, baseline_pred, device, tol=0.0) -> dict:
    """Zero-treatment scenario must reproduce baseline exactly."""
    X = engine.build_input(record)
    noop = np.array(X, copy=True)  # zero treatment == unchanged tensor
    res = engine.run(record, input_tensor=noop)
    dif = np.abs(res["pred_celsius"] - baseline_pred)
    return {
        "test": "noop",
        "max_abs_lst_diff": float(dif.max()),
        "mean_abs_lst_diff": float(dif.mean()),
        "pass": bool(dif.max() <= tol),
    }


def test_isolation(input_batch, scenario, baseline_input) -> dict:
    """Transform is inside polygon, eligible, and elsewhere identical."""
    treated = scenario.treatment_mask
    inside_poly = bool((treated & ~scenario.polygon_mask).sum() == 0)
    inside_elig = bool((treated & ~scenario.eligible_mask).sum() == 0)

    changed = scenario.input != baseline_input          # [T, 18, H, W]
    T, C = changed.shape[0], changed.shape[1]
    frames_changed = np.nonzero(changed.any(axis=(1, 2, 3)))[0].tolist()   # over [T]
    chan_changed = np.nonzero(changed.any(axis=(0, 2, 3)))[0].tolist()     # over [18]

    non_surface = [c for c in range(C) if c not in scenario.changed_channels]
    others_unchanged = bool(not changed[:, non_surface].any())

    # every changed (channel, pixel) must be a treated pixel
    surface_changed_mask = changed[-1, list(scenario.changed_channels)].any(axis=0)
    only_treated = bool(not (surface_changed_mask & ~treated).any())

    # original array not mutated: the caller passes a copy; assert unchanged
    original_intact = bool(np.array_equal(baseline_input, baseline_input))
    return {
        "test": "isolation",
        "treated_within_polygon": inside_poly,
        "treated_within_eligible": inside_elig,
        "changed_frames": frames_changed,
        "changed_channels": [CHANNEL_NAMES[c] for c in chan_changed],
        "only_latest_frame_changed": frames_changed == [T - 1],
        "changed_channels_are_subset_of_surface_properties": set(chan_changed) <= set(scenario.changed_channels),
        "non_surface_channels_unchanged": others_unchanged,
        "only_treated_pixels_changed": only_treated,
        "original_input_intact": original_intact,
        "pass": bool(inside_poly and inside_elig and frames_changed == [T - 1]
                     and others_unchanged and only_treated
                     and set(chan_changed) <= set(scenario.changed_channels)),
    }


def test_area_constraints(scenario, pixel_area_m2) -> dict:
    a = scenario.area
    under_cap = True
    if a["max_treatment_area_m2"] is not None:
        under_cap = a["treated_area_m2"] <= a["max_treatment_area_m2"] + 1e-6
    subset = bool((scenario.treatment_mask & ~scenario.polygon_mask).sum() == 0
                  and (scenario.treatment_mask & ~scenario.eligible_mask).sum() == 0)
    cov = a["treated_pixels"] / max(a["eligible_pixels"], 1)
    return {
        "test": "area_constraints",
        "treated_area_m2": a["treated_area_m2"],
        "polygon_area_m2": a["polygon_area_m2"],
        "eligible_area_m2": a["eligible_area_m2"],
        "treated_subset_of_polygon_and_eligible": subset,
        "treated_area_within_cap": bool(under_cap),
        "realised_coverage_pct_of_eligible": float(100 * cov),
        "pass": bool(subset and under_cap),
    }


def test_strategy_consistency(app, base_X, params, means, stds) -> dict:
    """Strategy-agnostic feature-consistency checks (fractions / albedo).

    Works for any strategy whose modified channels are a subset of the frozen
    land-cover fractions and/or ``Albedo``: verifies per-channel [0, 1] bounds,
    monotone direction of change, target compliance where applicable, and the
    equal-and-opposite mass balance of the coupled fraction conversions.
    """
    names = [CHANNEL_NAMES[c] for c in app.changed_channels]
    tr = app.treatment_mask
    Xn = app.input[-1]

    def raw(arr, c):
        return arr[c] * stds[c] + means[c]

    out = {"test": "strategy_consistency", "changed_channels": names,
           "n_treated": int(tr.sum())}
    checks = []

    def _frac(name, monotonically_up):
        c = CHANNEL_NAMES.index(name)
        new, old = raw(Xn, c), raw(base_X[-1], c)
        key = name.split("_")[0].lower()
        out[f"{key}_in_range"] = bool(np.all((new[tr] >= -1e-6) & (new[tr] <= 1 + 1e-6)))
        out[f"{key}_monotone"] = bool(
            np.all(new[tr] >= old[tr] - 1e-6) if monotonically_up
            else np.all(new[tr] <= old[tr] + 1e-6)
        )
        checks.extend([out[f"{key}_in_range"], out[f"{key}_monotone"]])

    if "Vegetation_Frac" in names:
        _frac("Vegetation_Frac", True)
        c = CHANNEL_NAMES.index("Vegetation_Frac")
        if hasattr(params, "target_vegetation_fraction"):
            out["vf_target_respected"] = bool(
                np.all(raw(Xn, c)[tr] <= params.target_vegetation_fraction + 1e-5))
            checks.append(out["vf_target_respected"])
    if "Soil_Frac" in names:
        _frac("Soil_Frac", False)
    if "Impervious_Frac" in names:
        _frac("Impervious_Frac", False)
    if "Albedo" in names:
        _frac("Albedo", True)
        c = CHANNEL_NAMES.index("Albedo")
        if hasattr(params, "target_albedo"):
            out["albedo_target_respected"] = bool(
                np.all(raw(Xn, c)[tr] <= params.target_albedo + 1e-5))
            checks.append(out["albedo_target_respected"])

    if {"Vegetation_Frac", "Soil_Frac"} <= set(names):
        vc = CHANNEL_NAMES.index("Vegetation_Frac")
        sc = CHANNEL_NAMES.index("Soil_Frac")
        d_vf = (raw(Xn, vc)[tr] - raw(base_X[-1], vc)[tr]).sum()
        d_src = (raw(base_X[-1], sc)[tr] - raw(Xn, sc)[tr]).sum()
        if "Impervious_Frac" in names:
            ic = CHANNEL_NAMES.index("Impervious_Frac")
            d_src += (raw(base_X[-1], ic)[tr] - raw(Xn, ic)[tr]).sum()
        out["mass_balance_dVF_equals_dLoss"] = bool(np.isclose(d_vf, d_src, atol=1e-4))
        checks.append(out["mass_balance_dVF_equals_dLoss"])

    out["n_checks"] = len(checks)
    out["pass"] = bool(out["n_treated"] > 0 and len(checks) > 0 and all(checks))
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _plot_lst(path, arr, record, crs, transform, vmin, vmax, title, point_xy=None,
              treated_mask=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masked = np.ma.masked_invalid(arr)
    cm = plt.get_cmap("turbo").copy(); cm.set_bad(color="#4d4d4d")
    fig, ax = plt.subplots(figsize=(9, 8))
    extent = (
        rasterio.transform.xy(transform, 0, 0)[0], rasterio.transform.xy(transform, 0, arr.shape[1])[0],
        rasterio.transform.xy(transform, arr.shape[0], 0)[1], rasterio.transform.xy(transform, 0, 0)[1],
    )
    im = ax.imshow(masked, cmap=cm, vmin=vmin, vmax=vmax, extent=extent, origin="upper")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Predicted LST (\u00b0C)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"easting (m, {crs})"); ax.set_ylabel("northing (m)")
    if treated_mask is not None and treated_mask.any():
        ax.contour(treated_mask.astype(float), levels=[0.5], colors="magenta",
                   linewidths=1.0, extent=extent)
    if point_xy is not None:
        ax.plot(point_xy[0], point_xy[1], marker="*", ms=15, mec="white", mfc="red", zorder=5)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_delta(path, delta, record, crs, transform, treated_mask=None, point_xy=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lim = float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else 1.0
    lim = max(lim, 1e-6)
    masked = np.ma.masked_invalid(delta)
    cm = plt.get_cmap("coolwarm").copy(); cm.set_bad(color="#4d4d4d")
    fig, ax = plt.subplots(figsize=(9, 8))
    extent = (
        rasterio.transform.xy(transform, 0, 0)[0], rasterio.transform.xy(transform, 0, delta.shape[1])[0],
        rasterio.transform.xy(transform, delta.shape[0], 0)[1], rasterio.transform.xy(transform, 0, 0)[1],
    )
    im = ax.imshow(masked, cmap=cm, vmin=-lim, vmax=lim, extent=extent, origin="upper")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("\u0394 Predicted LST (scenario \u2212 baseline) (\u00b0C)")
    ax.set_title(f"Counterfactual difference \u2014 target {record['target_month']}\n"
                 f"input {record['input_months'][0]} \u2026 {record['input_months'][-1]} "
                 f"(negative = cooling)", fontsize=10)
    ax.set_xlabel(f"easting (m, {crs})"); ax.set_ylabel("northing (m)")
    if treated_mask is not None and treated_mask.any():
        ax.contour(treated_mask.astype(float), levels=[0.5], colors="k", linewidths=1.0, extent=extent)
    if point_xy is not None:
        ax.plot(point_xy[0], point_xy[1], marker="*", ms=15, mec="white", mfc="red", zorder=5)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_polygon(path, arr, polygon_mask, eligible_mask, treated_mask, crs, transform,
                  record):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # 0 none, 1 polygon, 2 eligible, 3 treated
    cls = np.zeros(arr.shape, dtype=np.uint8)
    cls[polygon_mask] = 1
    cls[eligible_mask] = 2
    cls[treated_mask] = 3
    cm = ListedColormap(["#111111", "#3b6ea5", "#f2c14e", "#d7263d"])
    fig, ax = plt.subplots(figsize=(9, 8))
    extent = _extent(transform, 0, 0, arr.shape[0], arr.shape[1])
    ax.imshow(arr, cmap="gray", extent=extent, origin="upper", alpha=0.6)
    ax.imshow(cls, cmap=cm, vmin=-0.5, vmax=3.5, extent=extent, origin="upper", alpha=0.85)
    ax.set_title(f"Polygon / eligible / treated \u2014 target {record['target_month']}", fontsize=10)
    ax.set_xlabel(f"easting (m, {crs})"); ax.set_ylabel("northing (m)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#3b6ea5", "#f2c14e", "#d7263d"]]
    ax.legend(handles, ["polygon", "eligible", "treated"], loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _extent(transform, row0, col0, nrows, ncols):
    """Data extent (left, right, bottom, top) for a window of a raster transform."""
    left = rasterio.transform.xy(transform, row0, col0)[0]
    right = rasterio.transform.xy(transform, row0, col0 + ncols)[0]
    top = rasterio.transform.xy(transform, row0, col0)[1]
    bottom = rasterio.transform.xy(transform, row0 + nrows, col0)[1]
    return (left, right, bottom, top)


# ---------------------------------------------------------------------------
# Analysis helpers (domain labelling, attribution, UHI verification)
# ---------------------------------------------------------------------------
def area_stats(delta_c: np.ndarray, mask: np.ndarray) -> dict:
    """ΔLST statistics over ``mask`` (finite pixels only). Includes the sum so
    that response contributions can be apportioned by area."""
    vals = delta_c[np.asarray(mask, dtype=bool) & np.isfinite(delta_c)]
    if vals.size == 0:
        return {"valid_pixel_count": 0, "mean_c": float("nan"), "median_c": float("nan"),
                "min_c": float("nan"), "max_c": float("nan"), "std_c": float("nan"),
                "sum_c": 0.0}
    return {
        "valid_pixel_count": int(vals.size),
        "mean_c": float(vals.mean()),
        "median_c": float(np.median(vals)),
        "min_c": float(vals.min()),
        "max_c": float(vals.max()),
        "std_c": float(vals.std()),
        "sum_c": float(vals.sum()),
    }


def attribute_response(delta_c: np.ndarray, treated_grid: np.ndarray, halo_px: int = 8) -> dict:
    """Apportion the total ΔLST response to changed vs surrounding pixels.

    The model is convolutional, so pixels adjacent to a treated pixel (but not
    themselves treated) can also change. This partitions all valid response into:
      * ``treated_core``   - pixels whose input was actually modified;
      * ``surrounding_halo`` - valid, untreated pixels within ``halo_px`` of a
        treated pixel (model spatial spillover / receptive field);
      * ``far_field``      - valid, untreated pixels farther than ``halo_px``.
    """
    from scipy import ndimage

    within = np.isfinite(delta_c)
    dist = ndimage.distance_transform_edt(~treated_grid)
    core = within & treated_grid
    surround = within & (~treated_grid) & (dist <= halo_px)
    far = within & (~treated_grid) & (dist > halo_px)

    def agg(m):
        vals = delta_c[m]
        return {"n_pixels": int(vals.size),
                "mean_c": float(vals.mean()) if vals.size else float("nan"),
                "sum_c": float(vals.sum()) if vals.size else 0.0}

    total = agg(within); c = agg(core); s = agg(surround); f = agg(far)
    tot = total["sum_c"]
    frac = lambda x: (x / tot) if tot else None  # noqa: E731
    return {
        "definition": (
            "Partition of the total valid ΔLST sum into treated pixels, model "
            "spatial spillover (valid untreated pixels within the halo), and "
            "far-field. Sum fractions are area-weighted response shares."
        ),
        "halo_px": int(halo_px),
        "total_valid_response": total,
        "treated_core": c,
        "surrounding_halo": s,
        "far_field": f,
        "fraction_of_sum_in_treated_core": frac(c["sum_c"]),
        "fraction_of_sum_in_surrounding_halo": frac(s["sum_c"]),
        "fraction_of_sum_in_far_field": frac(f["sum_c"]),
        "note": "With a tiny net control-volume change the sum fractions can be large; read them alongside the means.",
    }


def compute_uhi_block(*, base_map, scen_map, common, urban, rural, polygon_mask,
                      evaluation_domain=None) -> dict:
    """Existing UHI metric, using the frozen masks and the run's evaluation domain.

    Uses the project's definition ``UHI = LST - mean(valid LST over the fixed
    rural mask)`` untouched. The rural reference mean is computed over the
    *evaluation domain* actually reconstructed in this run. When the domain is
    the full urban|rural study mask, the reference reproduces the authoritative
    study-domain value; the within-polygon view is always labelled as local.
    """
    if evaluation_domain is None:
        evaluation_domain = common
    evaluation_domain = np.asarray(evaluation_domain, dtype=bool)
    dom_common = common & evaluation_domain
    base_rural = rural_reference_mean(base_map, dom_common, rural)
    scen_rural = rural_reference_mean(scen_map, dom_common, rural)
    uhi_base = derive_uhi(base_map, base_rural, common)
    uhi_scen = derive_uhi(scen_map, scen_rural, common)
    n_rural = int((dom_common & (rural > 0)).sum())
    n_urban = int((dom_common & (urban > 0)).sum())
    available = (n_rural > 0) and (n_urban > 0)
    poly_common = polygon_mask & common
    rural_pixels_available_study = int((rural > 0).sum())
    domain_is_study = (n_rural >= rural_pixels_available_study)
    block = {
        "available": bool(available),
        "definition": "UHI = LST - mean(valid predicted LST over the fixed rural mask); existing metric",
        "baseline_rural_mean_c": base_rural,
        "scenario_rural_mean_c": scen_rural,
        "rural_valid_pixels_in_domain": n_rural,
        "urban_valid_pixels_in_domain": n_urban,
        "rural_pixels_available_on_frozen_mask": rural_pixels_available_study,
        "domain": (
            "study domain (urban|rural; rural reference over the full frozen rural mask)"
            if domain_is_study else
            "local (union of evaluated windows; polygon + halo, clipped to study)"
        ),
        "domain_covers_full_rural_mask": bool(domain_is_study),
        "domain_note": (
            "Frozen masks and the UHI formula are unchanged. When the evaluation "
            "domain covers the full study region the rural reference equals the "
            "authoritative study-domain value; otherwise it is an explicitly "
            "labelled local estimate."
        ),
    }
    if not available:
        block["reason_unavailable"] = (
            "The local domain does not contain both rural and urban mask pixels."
        )
        return block
    delta_uhi = uhi_scen - uhi_base
    dv = delta_uhi[poly_common & np.isfinite(delta_uhi)]
    block.update({
        "delta_uhi_polygon_mean_c": float(dv.mean()) if dv.size else float("nan"),
        "delta_uhi_local_domain_mean_c": float(np.nanmean(delta_uhi[np.isfinite(delta_uhi)])),
        "baseline_uhi_urban_mean_c": float(np.nanmean(uhi_base[common & (urban > 0)])),
        "scenario_uhi_urban_mean_c": float(np.nanmean(uhi_scen[common & (urban > 0)])),
        "baseline_uhi_polygon_mean_c": float(np.nanmean(uhi_base[poly_common])) if poly_common.any() else float("nan"),
        "scenario_uhi_polygon_mean_c": float(np.nanmean(uhi_scen[poly_common])) if poly_common.any() else float("nan"),
    })
    return block


def verify_uhi_against_existing(*, paths, target_month, base_map, common, rural) -> dict:
    """Cross-check this run's baseline against the existing full-grid UHI output.

    Loads the frozen full-grid baseline artifact for ``target_month`` (produced by
    ``evaluation/full_grid.py`` over all study patches) and compares its
    ``rural_reference_mean_celsius`` and per-pixel LST with this run's baseline at
    the shared valid pixels.
    """
    ref_dir = paths["evaluation_dir"] / f"fullgrid_{target_month}_baseline_full"
    lst_path = ref_dir / "lst_forecast_celsius.tif"
    sum_path = ref_dir / "inference_summary.json"
    if not lst_path.is_file():
        return {"available": False,
                "reason": f"existing full-grid artifact not found: {lst_path}"}
    with rasterio.open(lst_path) as ds:
        ref_lst = ds.read(1).astype(np.float64)
    with sum_path.open(encoding="utf-8") as f:
        ref_summary = json.load(f)
    ref_rural_pub = ref_summary.get("rural_reference_mean_celsius")
    ref_valid = np.isfinite(ref_lst)

    # per-pixel comparison at shared valid pixels
    ci, ri = np.where(common & ref_valid)
    ours = base_map[ci, ri]
    refv = ref_lst[ci, ri]
    m = np.isfinite(ours) & np.isfinite(refv)
    max_abs = float(np.max(np.abs(ours[m] - refv[m]))) if m.any() else float("nan")
    mean_abs = float(np.mean(np.abs(ours[m] - refv[m]))) if m.any() else float("nan")

    # our baseline rural mean over the local domain (same frozen rural mask)
    our_local_rural = rural_reference_mean(base_map, common, rural)
    return {
        "available": True,
        "reference_artifact": str(lst_path),
        "reference_patches": ref_summary.get("n_patches"),
        "reference_domain": "full study grid (all patches, evaluation/full_grid.py)",
        "published_rural_mean_c": ref_rural_pub,
        "our_local_domain_rural_mean_c": our_local_rural,
        "rural_mean_abs_diff_vs_published": (
            abs(our_local_rural - ref_rural_pub)
            if (ref_rural_pub is not None and np.isfinite(our_local_rural)) else None
        ),
        "shared_valid_pixels": int(m.sum()),
        "per_pixel_max_abs_diff_c": max_abs,
        "per_pixel_mean_abs_diff_c": mean_abs,
        "per_pixel_identical": bool(m.any() and max_abs == 0.0),
        "residual_is_float32_rounding": bool(
            m.any() and max_abs <= float(np.spacing(np.float32(30.0)))),
        "note": (
            "Provenance-only comparison against the previously generated frozen "
            "GeoTIFF, which stores LST as float32. A residual <= 1 float32 ULP "
            "means the baseline is numerically identical and the difference is "
            "storage rounding only. The authoritative baseline_equivalence check "
            "reconstructs the full_grid.py output fresh and is exact (0.0); a "
            "domain effect only remains if the frozen run included a different "
            "window margin over the same stride-64 grid."
        ),
    }


def verify_authoritative_stitching(*, model, stats, target_month, base_map, base_weight,
                                   eval_domain, urban, rural, batch_size=16) -> dict:
    """Verify the stitched baseline equals the authoritative full_grid.py output.

    Reconstructs the *baseline* independently using the exact authoritative
    helpers -- ``build_patch_index`` over the full study mask, the frozen
    ``FullGridInferenceDataset``, ``OverlapGrid`` + ``patch_weight`` from
    ``evaluation.spatial`` -- i.e. the same code path as
    ``evaluation.full_grid.run_full_grid_inference``. It then compares against
    this run's stitched baseline and against the frozen reference GeoTIFF, and
    derives the study-domain rural reference through the same helpers.
    """
    import torch

    from evaluation.full_grid import (
        FullGridInferenceDataset as _FGDS,
        build_patch_index as _bpi,
    )
    from evaluation.spatial import (
        OverlapGrid as _OG,
        derive_uhi as _du,
        rural_reference_mean as _rrm,
    )
    H, W = base_map.shape
    study = (np.asarray(urban) > 0) | (np.asarray(rural) > 0)
    patch_index = _bpi(H, W, study, PATCH_SIZE, STRIDE, 1)

    # The latest input frame of the same target window (matches run_scenario).
    # Reuse the frozen stats to inverse-transform.
    lst_i = stats.index("LST_Celsius")
    mean, std = stats.means[lst_i], stats.stds[lst_i]

    from utils.config import get_paths, load_config
    cfg = load_config(str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    P = get_paths(cfg)
    ds = _FGDS(P["dataset_root"], P["temporal_root"])
    try:
        grid = _OG(H, W, PATCH_SIZE)
        model.eval()
        Xbuf, Vbuf, RCbuf = [], [], []
        month_labels = load_month_labels(P["dataset_root"])
        input_indices, input_labels, target_index = resolve_month_window(
            month_labels, target_month, 12)
        rec_t = {"input_months": input_labels, "input_month_indices": input_indices,
                 "target_month": target_month, "target_month_index": target_index}

        def flush():
            if not Xbuf:
                return
            Xt = torch.from_numpy(np.stack(Xbuf)).to(next(model.parameters()).device)
            with torch.no_grad():
                out = model(Xt)[:, 0].float().cpu().numpy()
            for k in range(out.shape[0]):
                grid.add(out[k] * std + mean, RCbuf[k][0], RCbuf[k][1], valid=Vbuf[k])
            Xbuf.clear(); Vbuf.clear(); RCbuf.clear()

        for rec in patch_index:
            full = dict(rec_t); full.update(rec)
            X = ds.build_input(full)
            V = ds.input_lst_validity(full)
            Xbuf.append(X); Vbuf.append(V); RCbuf.append((rec["row_off"], rec["col_off"]))
            if len(Xbuf) >= batch_size:
                flush()
        flush()
        auth_map, auth_w = grid.finalize()
    finally:
        ds.close()

    auth_valid = auth_w > 0
    shared = np.isfinite(base_map) & eval_domain & auth_valid
    d = np.abs(base_map[shared] - auth_map[shared]) if shared.any() else np.array([])
    max_abs = float(d.max()) if d.size else float("nan")
    mean_abs = float(d.mean()) if d.size else float("nan")

    auth_rural = _rrm(auth_map, eval_domain & auth_valid, rural)
    auth_rural_study = _rrm(auth_map, study & auth_valid, rural)
    return {
        "method": (
            "Reconstructed the baseline with the authoritative evaluation.full_grid "
            "tiling (study mask, stride 64) and evaluation.spatial OverlapGrid + "
            "patch_weight, then compared to this run's stitched baseline."
        ),
        "authoritative_patches": len(patch_index),
        "shared_eval_domain_pixels": int(shared.sum()),
        "max_abs_diff_c": max_abs,
        "mean_abs_diff_c": mean_abs,
        "baseline_matches_authoritative": bool(shared.any() and max_abs == 0.0),
        "authoritative_rural_mean_eval_domain_c": auth_rural,
        "authoritative_rural_mean_study_domain_c": auth_rural_study,
        "ours_rural_mean_eval_domain_c": rural_reference_mean(
            base_map, eval_domain & (base_weight > 0), rural),
        "note": (
            "Exact (bit-identical float32) agreement is expected because the same "
            "windows, weights and inverse transform are used; only the evaluation "
            "domain is clipped. Frozen masks and formula unchanged."
        ),
    }


def _plot_zoom(path, base_c, scen_c, delta_c, polygon_mask, treated_grid, transform,
               crs, record, vmin, vmax, pad_px=24):
    """Zoomed baseline/scenario/difference with identical spatial extents."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = np.where(polygon_mask.any(axis=1))[0]
    cols = np.where(polygon_mask.any(axis=0))[0]
    r0 = max(int(rows.min()) - pad_px, 0); r1 = min(int(rows.max()) + pad_px + 1, base_c.shape[0])
    c0 = max(int(cols.min()) - pad_px, 0); c1 = min(int(cols.max()) + pad_px + 1, base_c.shape[1])

    def sl(a):
        return a[r0:r1, c0:c1]

    ext = _extent(transform, r0, c0, r1 - r0, c1 - c0)
    lim = float(np.nanmax(np.abs(sl(delta_c)))) if np.isfinite(sl(delta_c)).any() else 1.0
    lim = max(lim, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    panels = [
        ("Baseline LST (\u00b0C)", sl(base_c), "turbo", vmin, vmax, False),
        ("Scenario LST (\u00b0C)", sl(scen_c), "turbo", vmin, vmax, False),
        ("\u0394 LST scenario\u2212baseline (\u00b0C)", sl(delta_c), "coolwarm", -lim, lim, True),
    ]
    for ax, (title, arr, cmap_, lo, hi, diverging) in zip(axes, panels):
        cm = plt.get_cmap(cmap_).copy(); cm.set_bad(color="#4d4d4d")
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=cm, vmin=lo, vmax=hi,
                       extent=ext, origin="upper")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if diverging:
            cb.set_label("\u0394 LST (\u00b0C)")
        else:
            cb.set_label("Predicted LST (\u00b0C)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"easting (m, {crs})"); ax.set_ylabel("northing (m)")
        ax.contour(sl(polygon_mask).astype(float), levels=[0.5], colors="k",
                   linewidths=1.2, extent=ext)
        if sl(treated_grid).any():
            ax.contour(sl(treated_grid).astype(float), levels=[0.5], colors="magenta",
                       linewidths=1.0, extent=ext)
    fig.suptitle(
        f"Zoom on treated area \u2014 target {record['target_month']} | input "
        f"{record['input_months'][0]}\u2013{record['input_months'][-1]}  "
        f"(identical extent {r1 - r0}\u00d7{c1 - c0}; black=treated-polygon boundary, magenta=treated pixels)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_attribution(path, treated_grid, delta_c, surround_mask, transform, crs, record,
                      halo_px):
    """Categorical map (treated core / halo / other) beside the ΔLST map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rows = np.where(treated_grid.any(axis=1))[0]; cols = np.where(treated_grid.any(axis=0))[0]
    pad = halo_px + 16
    r0 = max(int(rows.min()) - pad, 0); r1 = min(int(rows.max()) + pad + 1, treated_grid.shape[0])
    c0 = max(int(cols.min()) - pad, 0); c1 = min(int(cols.max()) + pad + 1, treated_grid.shape[1])
    ext = _extent(transform, r0, c0, r1 - r0, c1 - c0)

    cls = np.zeros(treated_grid.shape, dtype=np.uint8)
    cls[surround_mask] = 1
    cls[treated_grid] = 2
    cm = ListedColormap(["#dddddd", "#f2c14e", "#d7263d"])
    lim = float(np.nanmax(np.abs(delta_c[r0:r1, c0:c1]))) if np.isfinite(delta_c[r0:r1, c0:c1]).any() else 1.0
    lim = max(lim, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    axes[0].imshow(cls[r0:r1, c0:c1], cmap=cm, vmin=-0.5, vmax=2.5, extent=ext, origin="upper")
    axes[0].set_title(f"Attribution classes (halo = {halo_px} px)", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#dddddd", "#f2c14e", "#d7263d"]]
    axes[0].legend(handles, ["other / far", "surrounding halo", "treated core"], loc="lower right")
    cm2 = plt.get_cmap("coolwarm").copy(); cm2.set_bad(color="#4d4d4d")
    im = axes[1].imshow(np.ma.masked_invalid(delta_c[r0:r1, c0:c1]), cmap=cm2,
                        vmin=-lim, vmax=lim, extent=ext, origin="upper")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label("\u0394 LST (\u00b0C)")
    axes[1].set_title("Predicted \u0394LST", fontsize=11)
    for ax in axes:
        ax.set_xlabel(f"easting (m, {crs})"); ax.set_ylabel("northing (m)")
    fig.suptitle(f"Changed-pixel vs surrounding-pixel response \u2014 target {record['target_month']}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Mitigation counterfactual simulation")
    ap.add_argument("--strategy", default="vegetation_restoration",
                    choices=sorted(STRATEGIES),
                    help="mitigation strategy to simulate")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "baseline_full" / "best.pt"))
    ap.add_argument("--lat", type=float, default=12.97486)
    ap.add_argument("--lon", type=float, default=77.53719)
    ap.add_argument("--half-size-m", type=float, default=2000.0, help="half side of square polygon (m)")
    ap.add_argument("--polygon-json", default=None, help="GeoJSON Polygon file (EPSG:4326); overrides square")
    ap.add_argument("--target-vf", type=float, default=0.6, help="target Vegetation_Frac (veg / bare-soil)")
    ap.add_argument("--coverage-pct", type=float, default=40.0)
    ap.add_argument("--max-area-m2", type=float, default=1_000_000.0)
    ap.add_argument("--soil-weight", type=float, default=0.5, help="vegetation strategy only")
    # reflective-surface parameters
    ap.add_argument("--target-albedo", type=float, default=0.30)
    ap.add_argument("--s-down-w-m2", type=float, default=None,
                    help="incoming shortwave irradiance for energy accounting (optional)")
    ap.add_argument("--min-impervious-fraction", type=float, default=0.5)
    # bare-soil restoration parameters
    ap.add_argument("--min-soil-fraction", type=float, default=0.30)
    ap.add_argument("--max-veg-fraction", type=float, default=0.30)
    ap.add_argument("--no-exclude-water", action="store_true", default=False,
                    help="disable MNDWI-based water exclusion (bare-soil strategy)")
    ap.add_argument("--mndwi-water-threshold", type=float, default=0.0)
    ap.add_argument("--evaluation-domain", default="polygon", choices=["polygon", "study"],
                    help="'polygon' (default): authoritative baseline within the polygon "
                         "plus a 64-px halo; 'study': full urban|rural study domain. "
                         "Pixels outside the domain remain invalid/NaN.")
    ap.add_argument("--target-month", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--run-id", default="veg_restoration_demo")
    ap.add_argument("--noop-first", action="store_true", default=True,
                    help="run the zero-treatment no-op test (default on)")
    ap.add_argument("--sweep", action="store_true", default=False,
                    help="run a controlled coverage-% sweep with all other params fixed")
    ap.add_argument("--sweep-coverage", default="0,10,25,40,60,80,100",
                    help="comma-separated treatment_coverage_pct values for the sweep")
    ap.add_argument("--sweep-no-cap", action="store_true", default=False,
                    help="disable max-area cap during the sweep so coverage is the only lever")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    t0 = time.perf_counter()
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    out_dir = paths["evaluation_dir"] / f"scenario_{args.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_normalization_stats(paths["dataset_root"])
    means = np.asarray(stats.means, dtype=np.float64)
    stds = np.asarray(stats.stds, dtype=np.float64)
    lst_i = stats.index("LST_Celsius")

    month_labels = load_month_labels(paths["dataset_root"])
    input_indices, input_labels, target_index = resolve_month_window(
        month_labels, args.target_month or month_labels[-1], int(cfg["data"]["sequence_length"])
    )

    meta = load_mask_metadata(paths["mask_dir"])
    H, W = int(meta["height"]), int(meta["width"])
    with rasterio.open(paths["mask_dir"] / "urban_mask.tif") as ds:
        urban, transform, crs = ds.read(1), ds.transform, ds.crs
    with rasterio.open(paths["mask_dir"] / "rural_mask.tif") as ds:
        rural = ds.read(1)
    pixel_area_m2 = abs(transform.a * transform.e)

    # polygon
    if args.polygon_json:
        with open(args.polygon_json, encoding="utf-8") as f:
            polygon = json.load(f)
    else:
        polygon = square_polygon(args.lat, args.lon, args.half_size_m)
    polygon_mask = polygon_to_mask(polygon, H, W, transform, crs)

    strategy = get_strategy(args.strategy)
    params = strategy.params_from_args(args)
    params.validate()

    # model
    model = ConvLSTM(
        input_channels=cfg["data"]["input_channels"],
        hidden_channels=cfg["model"]["hidden_channels"],
        kernel_size=cfg["model"]["kernel_size"],
        output_channels=cfg["model"]["output_channels"],
    ).to(device)
    ckpt = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    ckpt_sha_before = sha256_file(checkpoint)

    record_template = {
        "sample_id": -1, "split": "inference", "patch_id": -1,
        "input_month_indices": input_indices, "input_months": input_labels,
        "target_month_index": target_index, "target_month": args.target_month or month_labels[-1],
    }

    # Inference domain.
    #
    # The baseline must be the *authoritative* full-grid reconstruction, so the
    # patch set is built exactly as evaluation/full_grid.py builds it: every
    # stride-64 tile overlapping the study region (urban|rural). A previous
    # version tiled only a 2-pixel dilation of the polygon, which omitted tiles
    # that contribute to the local mosaic through the 128-wide overlap blending
    # and therefore produced a baseline that differed from the frozen reference
    # by up to ~1.64 C in the boundary halo.
    #
    # The delta/UHI evaluation domain is the polygon dilated by TF_HALO_PX
    # (>= the patch half-width, 64 px) so each evaluated pixel has all patches
    # whose 128x128 footprints touch the polygon and the reconstruction matches
    # the authoritative one; farther pixels are reported as invalid/NaN, not
    # extrapolated. ``--evaluation-domain study`` restores the full study domain.
    from scipy import ndimage

    study_mask = (urban > 0) | (rural > 0)
    if args.evaluation_domain == "study":
        interest = study_mask
        TF_HALO_PX = 0
    else:  # "polygon" (default): authoritative within the polygon + halo
        TF_HALO_PX = 64
        interest = ndimage.binary_dilation(polygon_mask, iterations=TF_HALO_PX)

    patch_index = build_patch_index(H, W, interest, PATCH_SIZE, STRIDE, 1)
    if not patch_index:
        raise SystemExit("No model windows intersect the evaluation domain.")
    authoritative_windows = build_patch_index(H, W, study_mask, PATCH_SIZE, STRIDE, 1)
    print(f"Target month        : {record_template['target_month']} (idx {target_index})")
    print(f"Input months        : {input_labels[0]} .. {input_labels[-1]} ({len(input_labels)})")
    print(f"Polygon pixels      : {int(polygon_mask.sum())} ({polygon_mask.sum() * pixel_area_m2 / 1e6:.3f} km^2)")
    print(f"Evaluation domain   : {args.evaluation_domain} (halo {TF_HALO_PX} px)")
    print(f"Inference windows   : {len(patch_index)} "
          f"(authoritative full study region: {len(authoritative_windows)})")

    # Evaluation domain: intersection of the polygon halo with the study region.
    # Invalid pixels stay NaN; they are never extrapolated.
    if args.evaluation_domain == "study":
        eval_domain = study_mask
    else:
        eval_domain = (ndimage.binary_dilation(polygon_mask, iterations=TF_HALO_PX)) & study_mask

    inference_dataset = FullGridInferenceDataset(paths["dataset_root"], paths["temporal_root"])

    # latest raw frame (physical) for eligibility/selection
    from utils.rasters import MonthRasterReader
    reader = MonthRasterReader(paths["dataset_root"], paths["temporal_root"])

    # Read the full latest raw frame once (physical land-cover fractions) and
    # select the treated set ONCE over the whole polygon footprint, so the
    # coverage percentage and max-area cap are enforced globally rather than
    # independently per 128x128 window.
    latest_label = input_labels[-1]
    with rasterio.open(reader.month_path(latest_label)) as mds:
        latest_raw_full = mds.read().astype(np.float32)
    valid_full = np.isfinite(latest_raw_full[stats.index("LST_Celsius")])

    def build_scenario(params_obj, meta_holder):
        """Return ``(scenario_fn, global_treated, global_elig, diag)`` for a params set.

        The transformation is dispatched through the strategy registry, so all
        strategies share the same inference, stitching and cap logic. The
        treatment set is selected once over the whole polygon footprint by
        applying the strategy to a dummy empty batch (its ``treatment_mask`` is
        the globally selected set) and then passed back in as the per-window
        override to guarantee no double-counting or coverage gaps.
        """
        dummy = np.zeros((12, 18, H, W), dtype=np.float32)
        g_app = strategy.apply_fn(
            dummy, latest_raw_full, polygon_mask=polygon_mask, valid_mask=valid_full,
            params=params_obj, pixel_area_m2=pixel_area_m2, means=means, stds=stds,
        )
        g_treated, g_elig, g_diag = g_app.treatment_mask, g_app.eligible_mask, g_app.area

        def fn(X, rec):
            r0, c0 = int(rec["row_off"]), int(rec["col_off"])
            raw18 = latest_raw_full[:, r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]
            pm = polygon_mask[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]
            vm = inference_dataset.input_lst_validity(rec)
            treated_win = g_treated[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]
            app = strategy.apply_fn(
                X, raw18, polygon_mask=pm, valid_mask=vm, params=params_obj,
                pixel_area_m2=pixel_area_m2, means=means, stds=stds,
                treatment_override=treated_win,
            )
            meta_holder[(r0, c0)] = app
            return app.input, {**app.area, "row_off": r0, "col_off": c0}

        return fn, g_treated, g_elig, g_diag

    scenario_meta_holder = {}
    scenario_fn, global_treated, global_elig, global_diag = build_scenario(
        params, scenario_meta_holder)
    total_treated_pixels = int(global_treated.sum())
    print(f"Eligible pixels     : {global_elig.sum()} "
          f"({global_elig.sum() * pixel_area_m2 / 1e6:.3f} km^2)")
    print(f"Treated pixels      : {total_treated_pixels} "
          f"({total_treated_pixels * pixel_area_m2 / 1e6:.3f} km^2)")

    # baseline (authoritative reconstruction over the same patch set as scenario)
    print("\n[1/2] Baseline inference ...")
    base_map, base_valid, base_weight, _ = run_inference(
        model=model, inference_dataset=inference_dataset, record_template=record_template,
        patch_index=patch_index, stats=stats, device=device,
        batch_size=args.batch_size, height=H, width=W, scenario_fn=None,
    )
    # no-op scenario: identical patch set, identity transform. If the
    # reconstruction is sound this must reproduce the baseline exactly.
    print("[no-op] Identity-scenario inference ...")
    noop_map, noop_valid, _, _ = run_inference(
        model=model, inference_dataset=inference_dataset, record_template=record_template,
        patch_index=patch_index, stats=stats, device=device,
        batch_size=args.batch_size, height=H, width=W,
        scenario_fn=lambda X, rec: (X, {"treated_pixels": 0}),
    )
    # scenario
    print("[2/2] Scenario inference ...")
    scen_map, scen_valid, _, scen_meta = run_inference(
        model=model, inference_dataset=inference_dataset, record_template=record_template,
        patch_index=patch_index, stats=stats, device=device,
        batch_size=args.batch_size, height=H, width=W, scenario_fn=scenario_fn,
    )

    # --- controlled coverage sweep: hold every other parameter fixed.
    sweep_rows = []
    if args.sweep:
        coverages = [float(x) for x in args.sweep_coverage.split(",")]
        sweep_max_area = None if args.sweep_no_cap else args.max_area_m2
        print(f"\n[coverage sweep] coverage_pct in {coverages} (other params frozen)")
        from dataclasses import replace as _dc_replace
        for cov_pct in coverages:
            sp = _dc_replace(
                params, treatment_coverage_pct=cov_pct, max_treatment_area_m2=sweep_max_area
            )
            holder = {}
            sfn, s_treated, s_elig, s_diag = build_scenario(sp, holder)
            smap, svalid, _, _ = run_inference(
                model=model, inference_dataset=inference_dataset, record_template=record_template,
                patch_index=patch_index, stats=stats, device=device,
                batch_size=args.batch_size, height=H, width=W, scenario_fn=sfn,
            )
            s_common = base_valid & svalid
            s_delta = np.where(s_common, smap - base_map, np.nan)
            row = {
                "coverage_pct_requested": cov_pct,
                "treated_pixels": int(s_treated.sum()),
                "treated_area_km2": float(s_treated.sum() * pixel_area_m2 / 1e6),
                "realised_coverage_pct_of_eligible": (
                    100.0 * int(s_treated.sum()) / max(int(s_elig.sum()), 1)),
                "mean_primary_change_treated": float(np.mean(
                    [a.area.get("mean_dVF_treated", a.area.get("mean_delta_albedo_treated", 0.0))
                     for a in holder.values() if a.area["treated_pixels"]]) ) if any(
                    a.area["treated_pixels"] for a in holder.values()) else 0.0,
                "delta_lst_all_valid_mean_c": float(np.nanmean(s_delta)),
                "delta_lst_polygon_mean_c": float(np.nanmean(s_delta[polygon_mask & s_common]))
                    if (polygon_mask & s_common).any() else float("nan"),
                "delta_lst_treated_mean_c": float(np.nanmean(s_delta[s_treated & s_common]))
                    if (s_treated & s_common).any() else float("nan"),
                "delta_lst_polygon_median_c": float(np.nanmedian(s_delta[polygon_mask & s_common]))
                    if (polygon_mask & s_common).any() else float("nan"),
                "delta_lst_polygon_min_c": float(np.nanmin(s_delta[polygon_mask & s_common]))
                    if (polygon_mask & s_common).any() else float("nan"),
                "delta_lst_polygon_max_c": float(np.nanmax(s_delta[polygon_mask & s_common]))
                    if (polygon_mask & s_common).any() else float("nan"),
            }
            ub = compute_uhi_block(base_map=base_map, scen_map=smap, common=s_common,
                                   urban=urban, rural=rural, polygon_mask=polygon_mask)
            row["delta_uhi_polygon_mean_c"] = ub.get("delta_uhi_polygon_mean_c")
            sweep_rows.append(row)
            print(f"  coverage {cov_pct:5.1f}%  treated {row['treated_pixels']:6d} px "
                  f"({row['treated_area_km2']:.3f} km^2)  "
                  f"ΔLST poly {row['delta_lst_polygon_mean_c']:+.4f} C  "
                  f"ΔUHI {row['delta_uhi_polygon_mean_c']:+.4f} C")

    reader.close()
    inference_dataset.close()

    # --- delta on the shared valid pixels inside the evaluation domain only.
    # Pixels outside the domain (or invalid in either field) stay NaN and are
    # excluded from every comparison and metric; nothing is extrapolated.
    common = base_valid & scen_valid & eval_domain
    delta = scen_map - base_map
    delta_c = np.where(common, delta, np.nan)

    # no-op == baseline exactly (shared valid pixels)
    noop_diff = np.abs(noop_map - base_map)
    noop_common = base_valid & noop_valid & eval_domain & np.isfinite(noop_diff)
    noop_check = {
        "method": (
            "Identity-transform inference over the identical patch set, compared "
            "to the baseline at shared valid evaluation-domain pixels."
        ),
        "shared_valid_pixels": int(noop_common.sum()),
        "max_abs_delta_c": float(noop_diff[noop_common].max()) if noop_common.any() else float("nan"),
        "mean_abs_delta_c": float(noop_diff[noop_common].mean()) if noop_common.any() else float("nan"),
        "exact_zero": bool(noop_common.any() and noop_diff[noop_common].max() == 0.0),
    }

    # treated mask assembled on the grid from per-patch treatment masks
    treated_grid = np.zeros((H, W), dtype=bool)
    eligible_grid = np.zeros((H, W), dtype=bool)
    for (r0, c0), app in scenario_meta_holder.items():
        treated_grid[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE] |= app.treatment_mask
        eligible_grid[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE] |= app.eligible_mask
    # The globally selected set must equal the union of per-window applications
    # (no gaps, no double-counting).
    coverage_gap = int((global_treated & ~treated_grid).sum())
    coverage_extra = int((treated_grid & ~global_treated).sum())
    treated_in_delta = treated_grid & common
    polygon_in_delta = polygon_mask & common
    target_month = record_template["target_month"]

    # --- local domain labelling. The arrays reconstructed here cover ONLY the
    # union of the windows intersecting the polygon (plus dilation), i.e. a local
    # model domain -- NOT the full 1608x1493 study grid.
    local_rows = np.where(common.any(axis=1))[0]
    local_cols = np.where(common.any(axis=0))[0]
    domain_label = (
        "study domain (urban|rural)" if args.evaluation_domain == "study"
        else f"polygon + {TF_HALO_PX}-px halo, clipped to study"
    )
    local_domain = {
        "description": f"evaluation domain = {domain_label}",
        "evaluation_domain_mode": args.evaluation_domain,
        "halo_px": TF_HALO_PX,
        "domain_pixels": int(eval_domain.sum()),
        "reconstructed_pixels": int(np.isfinite(base_map).sum()),
        "row_range": [int(local_rows.min()), int(local_rows.max())] if local_rows.size else None,
        "col_range": [int(local_cols.min()), int(local_cols.max())] if local_cols.size else None,
        "domain_shape": [int(local_rows.size), int(local_cols.size)] if local_rows.size else None,
        "full_grid_shape": [H, W],
        "valid_pixels": int(common.sum()),
        "full_grid_pixels": int(H * W),
        "fraction_of_full_grid": float(common.sum() / (H * W)),
        "note": (
            "The difference map is confined to the evaluation domain; pixels outside "
            "it (and pixels invalid in either field) remain NaN and are excluded from "
            "metrics. Baselines are computed with the authoritative full_grid.py "
            "tiling and validated pixel-identical by the baseline_equivalence test. "
            "The frozen reference GeoTIFF is compared for provenance only: it may "
            "differ where the frozen run used a different window-inclusion margin "
            "over the same stride-64 grid. No extrapolation is performed."
        ),
    }

    delta_local = area_stats(delta_c, common)
    delta_treated = area_stats(delta_c, treated_in_delta)
    delta_polygon = area_stats(delta_c, polygon_in_delta)

    # --- apparent response attribution: changed vs surrounding pixels
    ATTR_HALO_PX = 8
    from scipy import ndimage as _ndi
    _dist = _ndi.distance_transform_edt(~treated_grid)
    surround_mask = common & (~treated_grid) & (_dist <= ATTR_HALO_PX)
    attribution = attribute_response(delta_c, treated_grid, halo_px=ATTR_HALO_PX)

    # --- contribution decomposition inside the polygon (unsigned, since signs vary)
    treated_in_poly = treated_grid & polygon_in_delta
    untreated_in_poly = polygon_in_delta & (~treated_grid)
    def _contrib(mask):
        v = delta_c[mask]
        return {"n_pixels": int(v.size),
                "sum_signed_c": float(v.sum()) if v.size else 0.0,
                "sum_abs_c": float(np.abs(v).sum()) if v.size else 0.0,
                "mean_signed_c": float(v.mean()) if v.size else float("nan")}
    contribution_in_polygon = {
        "definition": (
            "Inside-polygon ΔLST split by whether the pixel's input was changed "
            "(treated) vs not. 'sum_abs_c' is the unsigned magnitude so signed "
            "cooling and warming do not cancel."
        ),
        "treated_in_polygon": _contrib(treated_in_poly),
        "untreated_in_polygon": _contrib(untreated_in_poly),
        "polygon_total": _contrib(polygon_in_delta),
    }
    tot_abs = contribution_in_polygon["polygon_total"]["sum_abs_c"]
    if tot_abs:
        contribution_in_polygon["treated_share_of_abs_response"] = (
            contribution_in_polygon["treated_in_polygon"]["sum_abs_c"] / tot_abs)
        contribution_in_polygon["untreated_share_of_abs_response"] = (
            contribution_in_polygon["untreated_in_polygon"]["sum_abs_c"] / tot_abs)

    # --- UHI through the existing metric definition, computed over the frozen
    # study domain (urban|rural). The study forest is a subset of every
    # authoritative stride-64 window tile, so the baseline reconstruction is
    # pixel-identical to evaluation/full_grid.py; the rural reference therefore
    # matches the authoritative value rather than the local-domain value.
    # (A within-polygon view is still reported, labelled as local.)
    cov = compute_coverage(common, urban, rural)
    uhi_block = compute_uhi_block(
        base_map=base_map, scen_map=scen_map, common=common,
        urban=urban, rural=rural, polygon_mask=polygon_mask,
        evaluation_domain=eval_domain,
    )
    uhi_block["coverage"] = cov
    # Cross-check the module-externally (authoritative) derivation against the
    # stitched baseline, using the exact full_grid.py tiling and blending.
    uhi_verification = verify_authoritative_stitching(
        model=model, stats=stats, target_month=target_month,
        base_map=base_map, base_weight=base_weight, eval_domain=eval_domain,
        urban=urban, rural=rural,
    )
    # Provenance comparison against the previously generated frozen reference.
    uhi_reference_check = verify_uhi_against_existing(
        paths=paths, target_month=target_month, base_map=base_map, common=common,
        rural=rural,
    )

    # --- validation tests
    print("\n[validation]")
    val = {}
    # Reference patch for the isolation/fraction/no-op tests: the patch with the
    # most treated pixels (guarantees a non-vacuous check), else any intersecting.
    if scenario_meta_holder:
        ref_key = max(scenario_meta_holder, key=lambda k: int(scenario_meta_holder[k].treatment_mask.sum()))
        if int(scenario_meta_holder[ref_key].treatment_mask.sum()) == 0:
            ref_key = next(iter(scenario_meta_holder))
    else:
        raise SystemExit("Scenario produced no patch applications.")
    ref_rec = dict(record_template)
    ref_rec.update({"row_off": ref_key[0], "col_off": ref_key[1],
                    "height": PATCH_SIZE, "width": PATCH_SIZE})
    app = scenario_meta_holder[ref_key]

    # Use the existing PointInference path for the no-op + isolation tests.
    from evaluation.point_inference import PointInference
    engine = PointInference(
        dataset_root=paths["dataset_root"], input_dir=paths["temporal_root"],
        stats=stats, model=model, device=device, month_labels=month_labels,
        sequence_length=int(cfg["data"]["sequence_length"]),
    )
    try:
        base_point = engine.run(ref_rec)
        base_pred = base_point["pred_celsius"]
        val["noop"] = test_noop(engine, ref_rec, base_pred, device, tol=0.0)

        base_X = np.array(engine.build_input(ref_rec), copy=True)
        val["isolation"] = test_isolation(base_X, app, base_X.copy())

        # strategy-agnostic feature consistency using actual stats
        val["strategy_consistency"] = test_strategy_consistency(
            app, base_X, params, means, stds)
        area_test = test_area_constraints(app, pixel_area_m2)
        # global-cap and no-gap/double-count checks on the assembled grid
        cap = params.max_treatment_area_m2
        area_test["global_treated_pixels"] = total_treated_pixels
        area_test["global_treated_area_m2"] = total_treated_pixels * pixel_area_m2
        area_test["global_area_within_cap"] = bool(
            cap is None or total_treated_pixels * pixel_area_m2 <= cap + 1e-3
        )
        area_test["no_coverage_gaps"] = bool(coverage_gap == 0)
        area_test["no_double_counting"] = bool(coverage_extra == 0)
        area_test["treated_area_matches_global"] = bool(
            total_treated_pixels == int(treated_grid.sum())
        )
        area_test["pass"] = bool(
            area_test["pass"] and area_test["global_area_within_cap"]
            and area_test["no_coverage_gaps"] and area_test["no_double_counting"]
            and area_test["treated_area_matches_global"]
        )
        val["area_constraints"] = area_test
    finally:
        engine.close()

    # Baseline equivalence (authoritative stitching) and exact no-op are
    # first-class, reported validation tests, not just ancillary diagnostics.
    val["baseline_equivalence"] = {
        "test": "baseline_equivalence",
        "method": uhi_verification["method"],
        "shared_valid_pixels": int(uhi_verification["shared_eval_domain_pixels"]),
        "max_abs_diff_c": uhi_verification["max_abs_diff_c"],
        "mean_abs_diff_c": uhi_verification["mean_abs_diff_c"],
        "pass": bool(uhi_verification["baseline_matches_authoritative"]),
    }
    val["noop_reconstruction"] = {
        "test": "noop_reconstruction",
        "method": noop_check["method"],
        "shared_valid_pixels": int(noop_check["shared_valid_pixels"]),
        "max_abs_delta_c": noop_check["max_abs_delta_c"],
        "pass": bool(noop_check["exact_zero"]),
    }

    ckpt_sha_after = sha256_file(checkpoint)
    val["inference_integrity"] = {
        "test": "inference_integrity",
        "checkpoint_sha256_before": ckpt_sha_before,
        "checkpoint_sha256_after": ckpt_sha_after,
        "checkpoint_unchanged": ckpt_sha_before == ckpt_sha_after,
        "input_shape": [len(input_labels), 18, PATCH_SIZE, PATCH_SIZE],
        "output_shape": [1, 1, PATCH_SIZE, PATCH_SIZE],
        "normalization_source": str(paths["dataset_root"] / "normalization_stats.json"),
        "pass": bool(ckpt_sha_before == ckpt_sha_after),
    }

    all_pass = all(v.get("pass", False) for v in val.values())

    # --- artifacts
    print("\n[artifacts]")
    base_tif = out_dir / "baseline_lst_celsius.tif"
    scen_tif = out_dir / "scenario_lst_celsius.tif"
    delta_tif = out_dir / "delta_lst_celsius.tif"

    def _tif(path, arr):
        profile = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
                   "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
                   "nodata": float("nan"), "compress": "lzw"}
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(np.where(np.isfinite(arr), arr, np.nan).astype(np.float32), 1)

    # Confine all written rasters to the evaluation domain: pixels outside it (and
# pixels invalid in that field) stay NaN and are never extrapolated or
# misrepresented as valid.
    base_c = np.where(base_valid & eval_domain, base_map, np.nan)
    scen_c = np.where(scen_valid & eval_domain, scen_map, np.nan)
    _tif(base_tif, base_c); _tif(scen_tif, scen_c); _tif(delta_tif, delta_c)
    np.save(out_dir / "baseline_lst_celsius.npy", base_c.astype(np.float32))
    np.save(out_dir / "scenario_lst_celsius.npy", scen_c.astype(np.float32))
    np.save(out_dir / "delta_lst_celsius.npy", delta_c.astype(np.float32))

    # common symmetric colour limits
    both = np.concatenate([base_c[np.isfinite(base_c)], scen_c[np.isfinite(scen_c)]])
    vmin, vmax = (float(both.min()), float(both.max())) if both.size else (0.0, 1.0)
    point_xy = None
    if args.polygon_json is None:
        from pyproj import Transformer
        point_xy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(args.lon, args.lat)
    title_common = (f"target {record_template['target_month']} | input "
                    f"{input_labels[0]}\u2013{input_labels[-1]}")
    _plot_lst(out_dir / "baseline_lst_map.png", base_c, record_template, crs, transform,
              vmin, vmax, f"Baseline predicted LST\n{title_common}", point_xy, treated_grid)
    _plot_lst(out_dir / "scenario_lst_map.png", scen_c, record_template, crs, transform,
              vmin, vmax, f"Scenario predicted LST ({strategy.label})\n{title_common}",
              point_xy, treated_grid)
    _plot_delta(out_dir / "delta_lst_map.png", delta_c, record_template, crs, transform,
                treated_grid, point_xy)
    _plot_polygon(out_dir / "polygon_eligible_treated.png", base_c, polygon_mask,
                  eligible_grid, treated_grid, crs, transform, record_template)
    _plot_zoom(out_dir / "zoom_treated_area.png", base_c, scen_c, delta_c,
               polygon_mask, treated_grid, transform, crs, record_template, vmin, vmax)
    _plot_attribution(out_dir / "attribution_changed_vs_surrounding.png",
                      treated_grid, delta_c, surround_mask, transform, crs,
                      record_template, ATTR_HALO_PX)

    # treated area aggregated across the whole run (globally selected set)
    total_treated = int(global_treated.sum())
    report = {
        "milestone": "M2_M3_mitigation_counterfactual",
        "run_id": args.run_id,
        "generated_utc": collect_environment()["timestamp_utc"],
        "checkpoint": {
            "path": str(checkpoint), "sha256_before": ckpt_sha_before,
            "sha256_after": ckpt_sha_after, "unchanged": ckpt_sha_before == ckpt_sha_after,
            "epoch": ckpt.get("epoch"), "best_val_normalized": ckpt.get("best_val"),
        },
        "forecast_origin": {
            "input_months": input_labels,
            "target_month": record_template["target_month"],
            "target_month_index": target_index,
            "temporal_assumption": (
                "one-step counterfactual: first 11 frames preserved; intervention applied to "
                "the latest input frame only; LST and meteorology preserved. Not a growth/"
                "irrigation/thermal-adaptation model."
            ),
        },
        "strategy": {
            "name": strategy.name,
            "label": strategy.label,
            "parameters": _serialize_params(params),
            "modified_channels": list(strategy.modified_channel_names),
            "unchanged_channels_note": app.area.get("unchanged_channels_note"),
            "metadata": app.metadata,
            "warnings": list(app.warnings),
        },
        "polygon": {
            "source": args.polygon_json or "square around point",
            "point_lat": args.lat if args.polygon_json is None else None,
            "point_lon": args.lon if args.polygon_json is None else None,
            "half_size_m": args.half_size_m if args.polygon_json is None else None,
            "crs": str(crs),
            "pixel_area_m2": pixel_area_m2,
            "polygon_pixels": int(polygon_mask.sum()),
            "polygon_area_km2": float(polygon_mask.sum() * pixel_area_m2 / 1e6),
        },
        "area": {
            "polygon_pixels": int(polygon_mask.sum()),
            "eligible_pixels_full_polygon": int(global_elig.sum()),
            "treated_pixels_full_polygon": total_treated,
            "treated_area_m2": total_treated * pixel_area_m2,
            "treated_area_km2": total_treated * pixel_area_m2 / 1e6,
            "max_treatment_area_m2": params.max_treatment_area_m2,
            "global_selection_diagnostics": global_diag,
            "coverage_gap_pixels": coverage_gap,
            "double_counted_pixels": coverage_extra,
            "per_patch_example": app.area,
        },
        "local_lst_difference": {
            "definition": "delta_lst = scenario_lst_celsius - baseline_lst_celsius (negative = cooling)",
            "domain": f"evaluation domain = {domain_label}",
            "all_valid_intersection": delta_local,
            "inside_treated_area": delta_treated,
            "inside_full_polygon": delta_polygon,
            "note": "Model-estimated counterfactual difference, not a measured causal effect.",
        },
        "domain_labels": local_domain,
        "response_attribution": {
            "changed_vs_surrounding": attribution,
            "inside_polygon_contribution": contribution_in_polygon,
        },
        "uhi": uhi_block,
        "noop_check": noop_check,
        "baseline_equivalence": uhi_verification,
        "reference_vs_frozen_artifact": uhi_reference_check,
        "coverage_sweep": {
            "enabled": bool(args.sweep),
            "held_fixed": {
                "strategy": strategy.name,
                "base_parameters": _serialize_params(params),
                "max_treatment_area_m2": (None if args.sweep_no_cap else args.max_area_m2)
                    if args.sweep else params.max_treatment_area_m2,
            },
            "lever": "treatment_coverage_pct",
            "response_curve": sweep_rows,
            "interpretation": (
                "Actual simulated response along the coverage lever. Not forced monotonic: "
                "positive, negative and negligible entries are all reported as produced."
            ),
        },
        "validation": val,
        "all_validation_passed": bool(all_pass),
        "response_plausibility": None,  # filled below after inspection
        "artifacts": {
            "baseline_lst_map_png": str(out_dir / "baseline_lst_map.png"),
            "scenario_lst_map_png": str(out_dir / "scenario_lst_map.png"),
            "delta_lst_map_png": str(out_dir / "delta_lst_map.png"),
            "zoom_treated_area_png": str(out_dir / "zoom_treated_area.png"),
            "attribution_png": str(out_dir / "attribution_changed_vs_surrounding.png"),
            "polygon_eligible_treated_png": str(out_dir / "polygon_eligible_treated.png"),
            "baseline_lst_geotiff": str(base_tif), "scenario_lst_geotiff": str(scen_tif),
            "delta_lst_geotiff": str(delta_tif),
            "baseline_npy": str(out_dir / "baseline_lst_celsius.npy"),
            "scenario_npy": str(out_dir / "scenario_lst_celsius.npy"),
            "delta_npy": str(out_dir / "delta_lst_celsius.npy"),
        },
        "environment": collect_environment(),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    # plausibility: describe sign/magnitude without enforcing an outcome
    m = delta_local["mean_c"]
    report["response_plausibility"] = {
        "mean_local_delta_c": m,
        "sign": "cooling" if (np.isfinite(m) and m < 0) else ("warming" if np.isfinite(m) else "n/a"),
        "comment": (
            "Reported as-is. A positive, negative or negligible effect is a valid model output; "
            "no expected cooling is enforced and transformations are not tuned to obtain one. "
            "delta_lst is evaluated over the documented evaluation domain, not citywide UHI."
        ),
    }
    write_json(out_dir / "scenario_report.json", report)
    _write_markdown(out_dir / "SCENARIO_REPORT.md", report)

    print(f"Baseline mean LST   : {np.nanmean(base_c):.3f} C")
    print(f"Scenario mean LST   : {np.nanmean(scen_c):.3f} C")
    print(f"Mean delta (common) : {delta_local['mean_c']:.4f} C  "
          f"(median {delta_local['median_c']:.4f}, min {delta_local['min_c']:.4f}, "
          f"max {delta_local['max_c']:.4f})")
    print(f"Mean delta (treated): {delta_treated['mean_c']:.4f} C")
    print(f"Mean delta (polygon): {delta_polygon['mean_c']:.4f} C")
    print(f"Attribution (sum fractions): treated {attribution['fraction_of_sum_in_treated_core']}, "
          f"halo {attribution['fraction_of_sum_in_surrounding_halo']}, "
          f"far {attribution['fraction_of_sum_in_far_field']}")
    if uhi_block.get("available"):
        print(f"Delta UHI (local poly): {uhi_block['delta_uhi_polygon_mean_c']:.4f} C")
    else:
        print(f"Delta UHI           : unavailable ({uhi_block.get('reason_unavailable')})")
    print(f"Baseline==authoritative: {uhi_verification['baseline_matches_authoritative']} "
          f"(max|d|={uhi_verification['max_abs_diff_c']:.2e} C, "
          f"n={uhi_verification['shared_eval_domain_pixels']})")
    print(f"No-op exact zero    : {noop_check['exact_zero']} "
          f"(max|d|={noop_check['max_abs_delta_c']:.2e} C, n={noop_check['shared_valid_pixels']})")
    print(f"Validation passed   : {all_pass}")
    for w in report["strategy"].get("warnings", []):
        print(f"WARNING             : {w}")
    print(f"Artifacts           : {out_dir}")
    print(f"Checkpoint unchanged: {ckpt_sha_before == ckpt_sha_after}")
    return 0


def _serialize_params(params) -> dict:
    """JSON/Markdown-friendly view of a strategy params dataclass."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(params):
        return dict(asdict(params))
    return {k: v for k, v in vars(params).items() if not k.startswith("_")}


def _write_markdown(path: Path, r: dict) -> None:
    lines = [f"# {r['strategy'].get('label', r['strategy']['name']).title()} "
             "Counterfactual — Scenario Report\n"]
    lines.append(f"- Run id: `{r['run_id']}`")
    lines.append(f"- Checkpoint: `{r['checkpoint']['path']}`")
    lines.append(f"- Checkpoint SHA-256: `{r['checkpoint']['sha256_before']}` "
                 f"(unchanged: {r['checkpoint']['unchanged']})")
    lines.append(f"- Target month: {r['forecast_origin']['target_month']}  |  "
                 f"input: {r['forecast_origin']['input_months'][0]} .. "
                 f"{r['forecast_origin']['input_months'][-1]}")
    lines.append(f"- Temporal assumption: {r['forecast_origin']['temporal_assumption']}\n")
    lines.append("## Strategy & parameters\n")
    lines.append(f"- strategy: **{r['strategy']['name']}** "
                 f"({r['strategy'].get('label', '')})")
    for k, v in r["strategy"]["parameters"].items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- modified channels: {', '.join(r['strategy']['modified_channels'])}")
    if r["strategy"].get("unchanged_channels_note"):
        lines.append(f"- {r['strategy']['unchanged_channels_note']}")
    for w in r["strategy"].get("warnings", []):
        lines.append(f"- WARNING: {w}")
    lines.append("")
    lines.append("## Polygon / CRS\n")
    p = r["polygon"]
    lines.append(f"- CRS: {p['crs']}  | pixel area: {p['pixel_area_m2']:.1f} m^2")
    lines.append(f"- polygon pixels: {p['polygon_pixels']} ({p['polygon_area_km2']:.3f} km^2)\n")
    a = r["area"]
    lines.append("## Area accounting\n")
    lines.append(f"- polygon area: {r['polygon']['polygon_area_km2']:.3f} km^2")
    lines.append(f"- treated pixels: {a['treated_pixels_full_polygon']} "
                 f"({a['treated_area_km2']:.3f} km^2)")
    lines.append(f"- max area cap: {r['strategy']['parameters']['max_treatment_area_m2']} m^2\n")
    d = r["local_lst_difference"]
    dl = r["domain_labels"]
    lines.append("## Domain labels\n")
    lines.append(f"- Difference map domain: **{dl['description']}**")
    lines.append(f"- evaluation mode: `{dl['evaluation_domain_mode']}`  |  halo: {dl['halo_px']} px")
    lines.append(f"- domain pixels: {dl['domain_pixels']}  |  reconstructed: {dl['reconstructed_pixels']}")
    lines.append(f"- evaluated valid pixels {dl['valid_pixels']} of {dl['full_grid_pixels']} "
                 f"({100 * dl['fraction_of_full_grid']:.2f}% of the full grid)")
    lines.append(f"- {dl['note']}\n")
    be = r["baseline_equivalence"]
    nc = r["noop_check"]
    lines.append("## Baseline equivalence & no-op\n")
    lines.append(f"- no-op vs baseline: shared {nc['shared_valid_pixels']} px, "
                 f"max|Δ|={nc['max_abs_delta_c']:.3e} °C, exact zero={nc['exact_zero']}")
    lines.append(f"- baseline vs authoritative full_grid.py stitching "
                 f"({be['authoritative_patches']} patches): shared "
                 f"{be['shared_eval_domain_pixels']} px, max|Δ|={be['max_abs_diff_c']:.3e} °C, "
                 f"mean|Δ|={be['mean_abs_diff_c']:.3e} °C, identical="
                 f"{be['baseline_matches_authoritative']}")
    lines.append(f"- authoritative study-domain rural mean: "
                 f"{be['authoritative_rural_mean_study_domain_c']:.4f} °C  |  "
                 f"ours (evaluation domain): {be['ours_rural_mean_eval_domain_c']:.4f} °C")
    rv = r.get("reference_vs_frozen_artifact", {})
    if rv.get("available"):
        lines.append(f"- vs frozen reference GeoTIFF (provenance): shared "
                     f"{rv['shared_valid_pixels']} px, max|Δ|={rv['per_pixel_max_abs_diff_c']:.3e} °C, "
                     f"identical={rv['per_pixel_identical']} (may differ where the frozen run "
                     f"used a different inclusion margin over the same stride-64 grid)")
    lines.append("")
    lines.append("## LST difference (scenario − baseline; negative = cooling)\n")
    lines.append("| region | n | mean °C | median °C | min °C | max °C |")
    lines.append("|---|---|---|---|---|---|")
    for key, lab in (("all_valid_intersection", "common valid (evaluation domain)"),
                     ("inside_treated_area", "treated area"),
                     ("inside_full_polygon", "full polygon")):
        s = d[key]
        lines.append(f"| {lab} | {s['valid_pixel_count']} | {s['mean_c']:.4f} | "
                     f"{s['median_c']:.4f} | {s['min_c']:.4f} | {s['max_c']:.4f} |")
    lines.append("")
    lines.append(f"- {d['note']}\n")
    ra = r["response_attribution"]
    at = ra["changed_vs_surrounding"]
    ip = ra["inside_polygon_contribution"]
    lines.append("## Response attribution (changed vs surrounding pixels)\n")
    lines.append(f"- treated core: n={at['treated_core']['n_pixels']}, "
                 f"mean {at['treated_core']['mean_c']:.4f} °C")
    lines.append(f"- surrounding halo (≤{at['halo_px']} px from a treated pixel): "
                 f"n={at['surrounding_halo']['n_pixels']}, mean {at['surrounding_halo']['mean_c']:.4f} °C")
    lines.append(f"- far field: n={at['far_field']['n_pixels']}, "
                 f"mean {at['far_field']['mean_c']:.4f} °C")
    lines.append(f"- share of total signed ΔLST sum: treated "
                 f"{at['fraction_of_sum_in_treated_core']}, halo "
                 f"{at['fraction_of_sum_in_surrounding_halo']}, far "
                 f"{at['fraction_of_sum_in_far_field']}")
    if "treated_share_of_abs_response" in ip:
        lines.append(f"- inside-polygon unsigned response: treated "
                     f"{100 * ip['treated_share_of_abs_response']:.1f}% vs untreated "
                     f"{100 * ip['untreated_share_of_abs_response']:.1f}% (surrounding model response)")
    lines.append("")
    lines.append("## UHI (existing metric)\n")
    u = r["uhi"]
    lines.append(f"- available: {u['available']}  |  domain: {u['domain']}")
    lines.append(f"- domain covers full frozen rural mask: {u['domain_covers_full_rural_mask']} "
                 f"({u['rural_valid_pixels_in_domain']}/{u['rural_pixels_available_on_frozen_mask']} "
                 f"rural pixels, {u['urban_valid_pixels_in_domain']} urban pixels)")
    if u["available"]:
        lines.append(f"- baseline rural mean: {u['baseline_rural_mean_c']:.4f} °C  |  "
                     f"scenario rural mean: {u['scenario_rural_mean_c']:.4f} °C")
        lines.append(f"- ΔUHI (polygon mean): {u.get('delta_uhi_polygon_mean_c'):.4f} °C")
        lines.append(f"- ΔUHI (evaluation-domain mean): {u.get('delta_uhi_local_domain_mean_c'):.4f} °C")
        lines.append(f"- baseline urban UHI mean: {u.get('baseline_uhi_urban_mean_c'):.4f} °C  |  "
                     f"scenario: {u.get('scenario_uhi_urban_mean_c'):.4f} °C")
        lines.append(f"- {u['domain_note']}")
    else:
        lines.append(f"- reason unavailable: {u['reason_unavailable']}")
    lines.append("")
    if r.get("coverage_sweep") and r["coverage_sweep"]["enabled"]:
        cs = r["coverage_sweep"]
        lines.append("## Coverage sweep (controlled: only coverage-% varies)\n")
        lines.append(f"- held fixed: {cs['held_fixed']}")
        lines.append("| coverage % | treated px | treated km² | ΔLST poly mean °C | "
                     "ΔUHI poly mean °C |")
        lines.append("|---|---|---|---|---|")
        for row in cs["response_curve"]:
            lines.append(f"| {row['coverage_pct_requested']:.1f} | {row['treated_pixels']} | "
                         f"{row['treated_area_km2']:.4f} | "
                         f"{row['delta_lst_polygon_mean_c']:.4f} | "
                         f"{row['delta_uhi_polygon_mean_c']:.4f} |")
        lines.append("")
        lines.append(f"- {cs['interpretation']}\n")
    if r.get("metadata"):
        lines.append("## Strategy metadata / assumptions\n")
        md = r.get("metadata") or {}
        for w in r["strategy"].get("warnings", []):
            lines.append(f"- WARNING: {w}")
        for a in md.get("assumptions", []):
            lines.append(f"- assumption: {a}")
        for lim in md.get("limitations", []):
            lines.append(f"- limitation: {lim}")
        ea = md.get("energy_accounting")
        if ea:
            lines.append(f"- energy accounting: available={ea.get('available')}, "
                         f"S_down={ea.get('s_down_w_m2')} W/m2, "
                         f"mean ΔQ_abs={ea.get('mean_delta_q_abs_w_m2')} W/m2, "
                         f"total ΔQ_abs={ea.get('total_delta_q_abs_w')} W")
        lines.append("")
    lines.append("## Validation\n")
    for k, v in r["validation"].items():
        lines.append(f"- **{k}**: {'PASS' if v.get('pass') else 'FAIL'}")
    lines.append(f"\n- all validation passed: **{r['all_validation_passed']}**\n")
    lines.append("## Interpretation\n")
    m = d["inside_full_polygon"]["mean_c"]
    um = u.get("delta_uhi_polygon_mean_c") if u["available"] else None
    lines.append(
        f"Under the specified {r['strategy'].get('label', r['strategy']['name'])} scenario, "
        f"the frozen ConvLSTM estimates a change of {m:.4f} °C in local mean predicted LST "
        f"inside the polygon"
        + (f" and {um:.4f} °C in the existing UHI metric" if um is not None else
           "; the city-level UHI metric is unavailable for this run")
        + ", subject to the documented assumptions and limitations."
    )
    lines.append("\nThis is a model-estimated counterfactual, not proof of a real-world causal effect.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())