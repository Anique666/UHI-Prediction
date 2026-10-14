"""Vegetation-restoration scenario transformer (model-independent).

Given a 12-month / 18-channel normalized input batch and a geographic polygon,
this module selects a deterministic, eligible subset of land-cover pixels inside
the polygon and converts impervious/soil surface fraction into vegetation
fraction on the **latest** input frame only, leaving historical LST and
meteorological observations untouched. It edits only the surface-property
channels and returns a *new* tensor (the original is never mutated).

Physical scope (deliberately narrow, see the scenario report):
  * Only ``Vegetation_Frac`` (6), ``Impervious_Frac`` (7) and ``Soil_Frac`` (8)
    are modified. These are independent sub-pixel fractions in ``[0, 1]`` in the
    frozen dataset (the sum is *not* a strict simplex), so the composition rule
    enforced here is per-channel ``[0, 1]``.
  * ``NDVI`` / ``EVI`` / ``Albedo`` are deliberately **not** modified: the frozen
    dataset does not expose a defensible reflectance -> index transformation
    (the stored indices are inconsistent with the stored S2 bands), so retaining
    them unchanged is the scientifically honest option and is reported.
  * Elevation, slope, LST and the two meteorological channels are never touched.

This is a one-step counterfactual approximation, not a vegetation-growth or
irrigation model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Authoritative channel order (frozen dataset README / normalization_stats.json).
CHANNEL_NAMES = [
    "LST_Celsius", "NDVI", "NDBI", "MNDWI", "EVI", "Albedo",
    "Vegetation_Frac", "Impervious_Frac", "Soil_Frac", "Elevation", "Slope",
    "S2_B2", "S2_B3", "S2_B4", "S2_B8", "S2_B11", "AirTemp_2m_C", "Dewpoint_2m_C",
]
VF_IDX = CHANNEL_NAMES.index("Vegetation_Frac")
IF_IDX = CHANNEL_NAMES.index("Impervious_Frac")
SF_IDX = CHANNEL_NAMES.index("Soil_Frac")
MODIFIED_CHANNELS = (VF_IDX, IF_IDX, SF_IDX)


class ParameterError(ValueError):
    """Raised when a scenario parameter is outside its valid domain/units."""


@dataclass
class VegetationRestorationParams:
    """User-configurable, unit-validated scenario parameters.

    target_vegetation_fraction : fraction in (0, 1]. The maximum vegetation
        fraction a treated pixel is raised toward. A treated pixel receives
        ``min(target, existing + treatable)`` where ``treatable`` is bounded by
        the convertible impervious/soil fraction.
    treatment_coverage_pct : percent in (0, 100] of *eligible* pixels to treat.
    max_treatment_area_m2 : hard cap on treated area in square metres (> 0), or
        ``None`` for no cap.
    soil_weight : share of conversion drawn from ``Soil_Frac`` vs
        ``Impervious_Frac`` (in [0, 1]); conversion is drawn pro-rata when both
        are available, using these weights.
    """

    target_vegetation_fraction: float = 0.6
    treatment_coverage_pct: float = 40.0
    max_treatment_area_m2: float | None = None
    soil_weight: float = 0.5

    def validate(self) -> None:
        if not (0.0 < self.target_vegetation_fraction <= 1.0):
            raise ParameterError(
                "target_vegetation_fraction must be in (0, 1] "
                f"(got {self.target_vegetation_fraction})"
            )
        if not (0.0 <= self.treatment_coverage_pct <= 100.0):
            raise ParameterError(
                "treatment_coverage_pct must be in [0, 100] "
                f"(got {self.treatment_coverage_pct})"
            )
        if self.max_treatment_area_m2 is not None and self.max_treatment_area_m2 <= 0:
            raise ParameterError(
                f"max_treatment_area_m2 must be > 0 (got {self.max_treatment_area_m2})"
            )
        if not (0.0 <= self.soil_weight <= 1.0):
            raise ParameterError(
                f"soil_weight must be in [0, 1] (got {self.soil_weight})"
            )


@dataclass
class ScenarioApplication:
    """Result of applying the scenario to one input batch.

    A common contract shared by every mitigation strategy: the modified input,
    the polygon/eligible/treated masks, the modified-channel list, area
    accounting, and (added here for strategies that need them) free-form
    ``metadata`` and validation ``warnings``. The vegetation strategy leaves
    ``metadata``/``warnings`` empty so its behaviour and outputs are unchanged.
    """

    input: np.ndarray                       # modified copy, same shape
    treatment_mask: np.ndarray              # [H, W] bool, treated pixels
    eligible_mask: np.ndarray               # [H, W] bool, eligible-and-selected
    polygon_mask: np.ndarray                # [H, W] bool
    changed_channels: tuple[int, ...]
    area: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def polygon_to_mask(polygon_geojson: dict, height: int, width: int, transform, crs) -> np.ndarray:
    """Rasterize a GeoJSON Polygon in geographic coords to the study grid.

    The polygon coordinates are projected from EPSG:4326 to ``crs`` before
    rasterization. Raises if the geometry is not a (Multi)Polygon.
    """
    from pyproj import Transformer
    from rasterio.features import geometry_mask

    geom_type = polygon_geojson.get("type")
    if geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Expected a (Multi)Polygon geometry, got {geom_type!r}")
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def _ring(ring):
        out = []
        for coord in ring:
            x, y = transformer.transform(coord[0], coord[1])
            out.append([float(x), float(y)])
        return out

    if geom_type == "Polygon":
        projected = {"type": "Polygon", "coordinates": [_ring(r) for r in polygon_geojson["coordinates"]]}
    else:
        projected = {
            "type": "MultiPolygon",
            "coordinates": [[_ring(r) for r in poly] for poly in polygon_geojson["coordinates"]],
        }
    mask = geometry_mask([projected], (height, width), transform, invert=True, all_touched=False)
    return np.asarray(mask, dtype=bool)


def _eligibility(
    latest_raw: np.ndarray,
    *,
    polygon_mask: np.ndarray,
    valid_mask: np.ndarray,
    target_vegetation_fraction: float,
    suitability_mask: np.ndarray | None = None,
    water_available_mask: np.ndarray | None = None,
    min_convertible: float = 0.02,
    vf_room_eps: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """Eligible pixels: inside polygon, valid, suitable, with room to increase.

    A pixel is eligible only if (a) it has a convertible impervious/soil share at
    least ``min_convertible`` and (b) its current vegetation fraction is strictly
    below the target (so the intervention can actually raise it). Pixels already
    at/above the target are not "eligible" because no vegetation restoration can
    be applied to them.
    """
    vf = latest_raw[VF_IDX]
    imp = latest_raw[IF_IDX]
    soil = latest_raw[SF_IDX]
    finite = np.isfinite(vf) & np.isfinite(imp) & np.isfinite(soil)
    convertible = np.clip(imp, 0, None) + np.clip(soil, 0, None)
    has_source = convertible >= min_convertible
    has_room = vf < (target_vegetation_fraction - vf_room_eps)

    elig = polygon_mask & valid_mask & finite & has_source & has_room
    if suitability_mask is not None:
        elig &= suitability_mask
    if water_available_mask is not None:
        elig &= water_available_mask

    diag = {
        "polygon_pixels": int(polygon_mask.sum()),
        "valid_in_polygon_pixels": int((polygon_mask & valid_mask).sum()),
        "finite_landcover_pixels": int((polygon_mask & valid_mask & finite).sum()),
        "pixels_with_convertible_source": int((polygon_mask & valid_mask & finite & has_source).sum()),
        "pixels_below_target_vf": int((polygon_mask & valid_mask & finite & has_room).sum()),
        "eligible_pixels": int(elig.sum()),
        "suitability_mask_applied": suitability_mask is not None,
        "water_constraint_applied": water_available_mask is not None,
        "min_convertible": float(min_convertible),
        "target_vegetation_fraction": float(target_vegetation_fraction),
    }
    return elig, diag


def _select_deterministic(
    elig: np.ndarray,
    latest_raw: np.ndarray,
    *,
    coverage_pct: float,
    max_pixels: int | None,
) -> np.ndarray:
    """Deterministically choose treated pixels from the eligible set.

    Prioritise pixels with lower existing vegetation fraction (more potential),
    tie-broken by a fixed coordinate order. Selection is fully deterministic.
    """
    idxs = np.argwhere(elig)  # row-major (deterministic)
    if idxs.size == 0:
        return np.zeros_like(elig)
    vf = latest_raw[VF_IDX][idxs[:, 0], idxs[:, 1]]
    order = np.lexsort((idxs[:, 1], idxs[:, 0], vf))  # vf asc, then row, then col
    n_avail = len(order)
    n_target = int(round(coverage_pct / 100.0 * n_avail))
    if max_pixels is not None:
        n_target = min(n_target, int(max_pixels))
    n_target = max(0, min(n_target, n_avail))
    chosen = idxs[order[:n_target]]
    out = np.zeros_like(elig)
    if chosen.size:
        out[chosen[:, 0], chosen[:, 1]] = True
    return out


def select_by_priority(
    elig: np.ndarray,
    priority: np.ndarray,
    *,
    coverage_pct: float,
    max_pixels: int | None,
) -> np.ndarray:
    """Deterministically choose treated pixels, preferring lower ``priority``.

    Shared selection primitive: eligibility is decided by the strategy, and this
    helper applies the same deterministic ranking rule (priority ascending, then
    row, then column) and the same global coverage/area-cap logic used by the
    vegetation module. ``priority`` may contain NaN for ineligible pixels; those
    are never selected.
    """
    idxs = np.argwhere(elig)
    if idxs.size == 0:
        return np.zeros_like(elig)
    p = np.asarray(priority, dtype=np.float64)[idxs[:, 0], idxs[:, 1]]
    order = np.lexsort((idxs[:, 1], idxs[:, 0], p))  # priority asc, then row, then col
    n_avail = len(order)
    n_target = int(round(coverage_pct / 100.0 * n_avail))
    if max_pixels is not None:
        n_target = min(n_target, int(max_pixels))
    n_target = max(0, min(n_target, n_avail))
    out = np.zeros_like(elig)
    chosen = idxs[order[:n_target]]
    if chosen.size:
        out[chosen[:, 0], chosen[:, 1]] = True
    return out


def select_treatment_globally(
    latest_raw_frame: np.ndarray,
    *,
    polygon_mask: np.ndarray,
    valid_mask: np.ndarray,
    params: VegetationRestorationParams,
    pixel_area_m2: float,
    suitability_mask: np.ndarray | None = None,
    water_available_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select the treated set **once over the whole polygon footprint**.

    This is the correct place to enforce ``treatment_coverage_pct`` and
    ``max_treatment_area_m2``: applying them per model window (128x128) and then
    aggregating would exceed the requested cap. Returns
    ``(treated_mask, eligible_mask, diagnostics)`` on the input grid.
    """
    params.validate()
    elig, diag = _eligibility(
        latest_raw_frame,
        polygon_mask=polygon_mask,
        valid_mask=valid_mask,
        target_vegetation_fraction=params.target_vegetation_fraction,
        suitability_mask=suitability_mask,
        water_available_mask=water_available_mask,
    )
    max_pixels = None
    if params.max_treatment_area_m2 is not None:
        max_pixels = int(np.floor(params.max_treatment_area_m2 / float(pixel_area_m2)))
    treated = _select_deterministic(
        elig, latest_raw_frame,
        coverage_pct=params.treatment_coverage_pct, max_pixels=max_pixels,
    )
    diag = dict(diag)
    diag["requested_max_area_pixels"] = max_pixels
    diag["selected_treated_pixels"] = int(treated.sum())
    diag["selected_treated_area_m2"] = float(treated.sum() * pixel_area_m2)
    return treated, elig, diag


def apply_vegetation_restoration(
    input_batch: np.ndarray,
    latest_raw_frame: np.ndarray,
    *,
    polygon_mask: np.ndarray,
    valid_mask: np.ndarray,
    params: VegetationRestorationParams,
    pixel_area_m2: float,
    means,
    stds,
    suitability_mask: np.ndarray | None = None,
    water_available_mask: np.ndarray | None = None,
    treatment_override: np.ndarray | None = None,
) -> ScenarioApplication:
    """Apply the scenario to the latest input frame.

    Parameters
    ----------
    input_batch : ``[T, 18, H, W]`` normalized input (not mutated).
    latest_raw_frame : ``[18, H, W]`` *raw* (physical) latest frame, used for
        eligibility/selection. The model input is normalized, so raw land-cover
        fractions are needed to reason about physical composition.
    pixel_area_m2 : area of one pixel (from raster metadata).
    means, stds : frozen per-channel normalization statistics (len 18).
    """
    params.validate()
    if input_batch.ndim != 4 or input_batch.shape[1] != 18:
        raise ValueError(f"Expected [T,18,H,W] input, got {input_batch.shape}")
    T = input_batch.shape[0]
    if latest_raw_frame.shape != input_batch.shape[1:]:
        raise ValueError("latest_raw_frame spatial shape does not match input_batch")

    if treatment_override is not None:
        # Global treatment mask supplied (e.g. selected once over the full
        # polygon); clip it to this window's eligibility so no invalid pixel is
        # ever treated even if the caller passes a wider mask.
        elig, diag = _eligibility(
            latest_raw_frame,
            polygon_mask=polygon_mask,
            valid_mask=valid_mask,
            target_vegetation_fraction=params.target_vegetation_fraction,
            suitability_mask=suitability_mask,
            water_available_mask=water_available_mask,
        )
        treated = np.asarray(treatment_override, dtype=bool) & elig
    else:
        treated, elig, diag = select_treatment_globally(
            latest_raw_frame, polygon_mask=polygon_mask, valid_mask=valid_mask,
            params=params, pixel_area_m2=pixel_area_m2,
            suitability_mask=suitability_mask, water_available_mask=water_available_mask,
        )

    transformed = np.array(input_batch, dtype=np.float32, copy=True)  # never mutate input
    vf = latest_raw_frame[VF_IDX].astype(np.float64)
    imp = latest_raw_frame[IF_IDX].astype(np.float64)
    soil = latest_raw_frame[SF_IDX].astype(np.float64)

    applied = {
        "n_treated": 0, "sum_dVF": 0.0, "sum_dIF": 0.0, "sum_dSF": 0.0,
        "n_from_impervious": 0, "n_from_soil": 0, "n_both": 0,
        "vf_clipped": 0,
    }
    rows, cols = np.nonzero(treated)
    w_soil = float(params.soil_weight)
    for r, c in zip(rows, cols):
        cur = vf[r, c]
        target = float(params.target_vegetation_fraction)
        need = target - cur
        if need <= 0:
            continue
        i_av = max(imp[r, c], 0.0)
        s_av = max(soil[r, c], 0.0)
        if i_av + s_av <= 0:
            continue
        # Declared source split: take soil_weight of the shortfall from soil, the
        # rest from impervious; each draw bounded by availability, and the total
        # shortfall may be back-filled from whichever source has spare capacity.
        take_soil = min(need * w_soil, s_av)
        take_imp = min(need - take_soil, i_av)
        shortfall = need - (take_soil + take_imp)
        if shortfall > 0:
            extra_soil = min(shortfall, s_av - take_soil)
            take_soil += extra_soil
            shortfall -= extra_soil
        if shortfall > 0:
            take_imp += min(shortfall, i_av - take_imp)
        total = take_imp + take_soil
        if total <= 0:
            continue
        new_vf = cur + total
        new_imp = i_av - take_imp
        new_soil = s_av - take_soil
        # per-channel [0,1] composition guard (frozen dataset rule)
        new_vf = float(np.clip(new_vf, 0.0, 1.0))
        new_imp = float(np.clip(new_imp, 0.0, 1.0))
        new_soil = float(np.clip(new_soil, 0.0, 1.0))

        transformed[-1, VF_IDX, r, c] = np.float32((new_vf - means[VF_IDX]) / stds[VF_IDX])
        transformed[-1, IF_IDX, r, c] = np.float32((new_imp - means[IF_IDX]) / stds[IF_IDX])
        transformed[-1, SF_IDX, r, c] = np.float32((new_soil - means[SF_IDX]) / stds[SF_IDX])

        applied["n_treated"] += 1
        applied["sum_dVF"] += new_vf - cur
        applied["sum_dIF"] += new_imp - i_av
        applied["sum_dSF"] += new_soil - s_av
        if take_imp > 0 and take_soil > 0:
            applied["n_both"] += 1
        elif take_imp > 0:
            applied["n_from_impervious"] += 1
        elif take_soil > 0:
            applied["n_from_soil"] += 1

    treated_area_m2 = applied["n_treated"] * float(pixel_area_m2)
    area = {
        **diag,
        "treated_pixels": int(applied["n_treated"]),
        "treated_area_m2": treated_area_m2,
        "treated_area_km2": treated_area_m2 / 1e6,
        "polygon_area_m2": float(diag["polygon_pixels"] * pixel_area_m2),
        "eligible_area_m2": float(diag["eligible_pixels"] * pixel_area_m2),
        "eligible_area_km2": float(diag["eligible_pixels"] * pixel_area_m2) / 1e6,
        "pixel_area_m2": float(pixel_area_m2),
        "max_treatment_area_m2": params.max_treatment_area_m2,
        "mean_dVF_treated": (applied["sum_dVF"] / applied["n_treated"]) if applied["n_treated"] else 0.0,
        "mean_dIF_treated": (applied["sum_dIF"] / applied["n_treated"]) if applied["n_treated"] else 0.0,
        "mean_dSF_treated": (applied["sum_dSF"] / applied["n_treated"]) if applied["n_treated"] else 0.0,
        "n_from_impervious": applied["n_from_impervious"],
        "n_from_soil": applied["n_from_soil"],
        "n_both": applied["n_both"],
        "temporal_assumption": "latest input frame only; first T-1 frames preserved",
        "modified_channels": [CHANNEL_NAMES[i] for i in MODIFIED_CHANNELS],
        "unchanged_channels_note": (
            "NDVI/EVI/Albedo deliberately unchanged: no defensible reflectance->index "
            "transform is available in the frozen dataset."
        ),
    }
    return ScenarioApplication(
        input=transformed,
        treatment_mask=treated,
        eligible_mask=elig,
        polygon_mask=np.asarray(polygon_mask, dtype=bool),
        changed_channels=MODIFIED_CHANNELS,
        area=area,
    )