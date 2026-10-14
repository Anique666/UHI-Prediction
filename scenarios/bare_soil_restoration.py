"""Bare-soil / degraded-land restoration scenario transformer.

Given a 12-month / 18-channel normalized input batch and a geographic polygon,
this module selects a deterministic, eligible subset of *bare-soil* pixels inside
the polygon and converts a portion of their ``Soil_Frac`` into
``Vegetation_Frac`` on the latest input frame only, leaving historical LST and
meteorological observations untouched. It mirrors the vegetation module's
contract and reuses its shared eligibility/selection primitives, but its
eligibility prioritises soil-dominated, vegetation-poor pixels rather than all
convertible land.

What is (and is not) modified
-----------------------------
* ``Soil_Frac`` (8) is decreased and ``Vegetation_Frac`` (6) increased by the
  same amount on treated pixels, preserving the frozen dataset's per-channel
  ``[0, 1]`` fraction semantics. ``Impervious_Frac`` is never touched.
* ``NDVI`` / ``EVI`` / ``Albedo`` / raw ``S2_*`` bands are **deliberately left
  unchanged**: the frozen dataset exposes no defensible reflectance -> index
  transformation (the stored indices do not exactly match the stored S2 bands),
  so fabricating new values would be unscientific. This is reported explicitly.

Physical honesty (see the scenario report)
------------------------------------------
* The frozen dataset exposes **no soil-moisture, water-availability, irrigation
  or land-use suitability** channel. Bare soil is therefore assumed to be
  restorable *by default*, which is explicitly flagged as an unvalidated
  assumption, not a physical claim.
* Water bodies are excluded where the data supports it, using the stored
  ``MNDWI`` (water index) as a transparent, documented exclusion (the same
  channel the vegetation module does not use).
* No direct temperature reduction is applied; LST and air temperature are
  untouched. A negligible or warming modeled response is a valid result.
* One-step counterfactual approximation on the latest frame only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenarios.vegetation_restoration import (
    CHANNEL_NAMES,
    ParameterError,
    ScenarioApplication,
    select_by_priority,
)

VF_IDX = CHANNEL_NAMES.index("Vegetation_Frac")
SF_IDX = CHANNEL_NAMES.index("Soil_Frac")
IF_IDX = CHANNEL_NAMES.index("Impervious_Frac")
MNDWI_IDX = CHANNEL_NAMES.index("MNDWI")
MODIFIED_CHANNELS = (VF_IDX, SF_IDX)

# Default water exclusion: the frozen MNDWI is a normalized difference index
# for which the conventional open-water threshold is 0. The dataset contains
# rare outliers, so pixels above the threshold are treated as water and excluded.
MNDWI_WATER_THRESHOLD = 0.0


@dataclass
class BareSoilRestorationParams:
    """User-configurable, unit-validated bare-soil restoration parameters.

    target_vegetation_fraction : fraction in (0, 1]. The maximum vegetation
        fraction a treated pixel is raised toward.
    treatment_coverage_pct : percent in [0, 100] of *eligible* pixels to treat.
    max_treatment_area_m2 : hard cap on treated area (m^2, > 0) or ``None``.
    min_soil_fraction : eligibility floor on ``Soil_Frac`` in [0, 1]; only
        soil-dominated pixels are eligible.
    max_vegetation_fraction : eligibility ceiling on the *current*
        ``Vegetation_Frac`` in [0, 1]; pixels already at/above this are treated
        as existing healthy vegetation and excluded.
    exclude_water : whether to exclude probable water pixels via ``MNDWI``.
    mndwi_water_threshold : ``MNDWI`` above this is water (only used when
        ``exclude_water`` is true).
    """

    target_vegetation_fraction: float = 0.6
    treatment_coverage_pct: float = 40.0
    max_treatment_area_m2: float | None = None
    min_soil_fraction: float = 0.30
    max_vegetation_fraction: float = 0.30
    exclude_water: bool = True
    mndwi_water_threshold: float = MNDWI_WATER_THRESHOLD

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
        if not (0.0 <= self.min_soil_fraction <= 1.0):
            raise ParameterError(
                f"min_soil_fraction must be in [0, 1] (got {self.min_soil_fraction})"
            )
        if not (0.0 <= self.max_vegetation_fraction <= 1.0):
            raise ParameterError(
                "max_vegetation_fraction must be in [0, 1] "
                f"(got {self.max_vegetation_fraction})"
            )
        if not (0.0 <= self.mndwi_water_threshold <= 1.0):
            raise ParameterError(
                "mndwi_water_threshold must be in [0, 1] "
                f"(got {self.mndwi_water_threshold})"
            )


def _eligibility(
    latest_raw: np.ndarray,
    *,
    polygon_mask: np.ndarray,
    valid_mask: np.ndarray,
    params: BareSoilRestorationParams,
    soil_room_eps: float = 1e-6,
    vf_room_eps: float = 1e-6,
    suitability_mask: np.ndarray | None = None,
    water_available_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Eligible bare-soil pixels and their deterministic priority.

    Eligible = inside polygon, valid, finite land-cover, soil >= floor,
    vegetation <= ceiling, below the conversion target, soil has room to give,
    and (optionally) not water. Priority is **decreasing soil fraction** then
    **increasing vegetation** (most bare first); the caller converts it to the
    shared ascending-priority selection.
    """
    vf = latest_raw[VF_IDX]
    soil = latest_raw[SF_IDX]
    imp = latest_raw[IF_IDX]
    mndwi = latest_raw[MNDWI_IDX]
    finite = np.isfinite(vf) & np.isfinite(soil) & np.isfinite(imp) & np.isfinite(mndwi)

    soil_ok = np.clip(soil, 0, None) >= params.min_soil_fraction
    low_veg = vf <= params.max_vegetation_fraction
    has_room = soil > soil_room_eps
    below_target = vf < (params.target_vegetation_fraction - vf_room_eps)

    elig = polygon_mask & valid_mask & finite & soil_ok & low_veg & has_room & below_target
    is_water = mndwi > params.mndwi_water_threshold
    n_water_excluded = int((elig & is_water).sum())
    if params.exclude_water:
        elig &= ~is_water
    if suitability_mask is not None:
        elig &= np.asarray(suitability_mask, dtype=bool)
    if water_available_mask is not None:
        elig &= np.asarray(water_available_mask, dtype=bool)

    # priority: ascending = more soil (needs negating) then less vegetation
    priority = np.full(vf.shape, np.inf, dtype=np.float64)
    if elig.any():
        priority[elig] = np.clip(-soil[elig], -1.0, 0.0) + np.clip(vf[elig], 0.0, 1.0)

    diag = {
        "polygon_pixels": int(polygon_mask.sum()),
        "valid_in_polygon_pixels": int((polygon_mask & valid_mask).sum()),
        "finite_landcover_pixels": int((polygon_mask & valid_mask & finite).sum()),
        "soil_dominant_pixels": int((polygon_mask & valid_mask & finite & soil_ok).sum()),
        "low_vegetation_pixels": int((polygon_mask & valid_mask & finite & low_veg).sum()),
        "pixels_below_target_vf": int((polygon_mask & valid_mask & finite & below_target).sum()),
        "water_pixels_excluded": n_water_excluded,
        "eligible_pixels": int(elig.sum()),
        "suitability_mask_applied": suitability_mask is not None,
        "water_constraint_applied": water_available_mask is not None,
        "min_soil_fraction": float(params.min_soil_fraction),
        "max_vegetation_fraction": float(params.max_vegetation_fraction),
        "target_vegetation_fraction": float(params.target_vegetation_fraction),
    }
    return elig, priority, diag


def apply_bare_soil_restoration(
    input_batch: np.ndarray,
    latest_raw_frame: np.ndarray,
    *,
    polygon_mask: np.ndarray,
    valid_mask: np.ndarray,
    params: BareSoilRestorationParams,
    pixel_area_m2: float,
    means,
    stds,
    suitability_mask: np.ndarray | None = None,
    water_available_mask: np.ndarray | None = None,
    treatment_override: np.ndarray | None = None,
) -> ScenarioApplication:
    """Apply the bare-soil restoration scenario to the latest input frame.

    Parameters
    ----------
    input_batch : ``[T, 18, H, W]`` normalized input (not mutated).
    latest_raw_frame : ``[18, H, W]`` *raw* (physical) latest frame, used for
        eligibility/selection and the fraction conversion.
    pixel_area_m2 : area of one pixel (from raster metadata).
    means, stds : frozen per-channel normalization statistics (len 18).
    """
    params.validate()
    if input_batch.ndim != 4 or input_batch.shape[1] != 18:
        raise ValueError(f"Expected [T,18,H,W] input, got {input_batch.shape}")
    T = input_batch.shape[0]
    if latest_raw_frame.shape != input_batch.shape[1:]:
        raise ValueError("latest_raw_frame spatial shape does not match input_batch")

    elig, priority, diag = _eligibility(
        latest_raw_frame,
        polygon_mask=polygon_mask,
        valid_mask=valid_mask,
        params=params,
        suitability_mask=suitability_mask,
        water_available_mask=water_available_mask,
    )

    if treatment_override is not None:
        treated = np.asarray(treatment_override, dtype=bool) & elig
    else:
        max_pixels = None
        if params.max_treatment_area_m2 is not None:
            max_pixels = int(np.floor(params.max_treatment_area_m2 / float(pixel_area_m2)))
        treated = select_by_priority(
            elig, priority,
            coverage_pct=params.treatment_coverage_pct, max_pixels=max_pixels,
        )

    transformed = np.array(input_batch, dtype=np.float32, copy=True)  # never mutate input
    vf = latest_raw_frame[VF_IDX].astype(np.float64)
    soil = latest_raw_frame[SF_IDX].astype(np.float64)

    rows, cols = np.nonzero(treated)
    n_treated = 0
    sum_d_vf = 0.0
    sum_d_sf = 0.0
    n_clipped = 0

    for r, c in zip(rows, cols):
        cur = float(vf[r, c])
        s_av = max(float(soil[r, c]), 0.0)
        target = float(params.target_vegetation_fraction)
        need = target - cur
        if need <= 0 or s_av <= 0:
            continue
        take = min(need, s_av)
        if take <= 0:
            continue
        new_vf = float(np.clip(cur + take, 0.0, 1.0))
        new_soil = float(np.clip(s_av - take, 0.0, 1.0))
        if abs((new_vf - cur) - (s_av - new_soil)) > 1e-9:
            n_clipped += 1

        transformed[-1, VF_IDX, r, c] = np.float32((new_vf - means[VF_IDX]) / stds[VF_IDX])
        transformed[-1, SF_IDX, r, c] = np.float32((new_soil - means[SF_IDX]) / stds[SF_IDX])

        n_treated += 1
        sum_d_vf += new_vf - cur
        sum_d_sf += new_soil - s_av

    treated_area_m2 = n_treated * float(pixel_area_m2)
    area = {
        **diag,
        "requested_max_area_pixels": (
            int(np.floor(params.max_treatment_area_m2 / float(pixel_area_m2)))
            if params.max_treatment_area_m2 is not None else None
        ),
        "treated_pixels": n_treated,
        "treated_area_m2": treated_area_m2,
        "treated_area_km2": treated_area_m2 / 1e6,
        "polygon_area_m2": float(diag["polygon_pixels"] * pixel_area_m2),
        "eligible_area_m2": float(diag["eligible_pixels"] * pixel_area_m2),
        "eligible_area_km2": float(diag["eligible_pixels"] * pixel_area_m2) / 1e6,
        "pixel_area_m2": float(pixel_area_m2),
        "max_treatment_area_m2": params.max_treatment_area_m2,
        "mean_dVF_treated": (sum_d_vf / n_treated) if n_treated else 0.0,
        "mean_dSF_treated": (sum_d_sf / n_treated) if n_treated else 0.0,
        "n_edge_clipped": n_clipped,
        "temporal_assumption": "latest input frame only; first T-1 frames preserved",
        "modified_channels": [CHANNEL_NAMES[i] for i in MODIFIED_CHANNELS],
        "unchanged_channels_note": (
            "NDVI/EVI/Albedo deliberately unchanged: no defensible reflectance->index "
            "transform is available in the frozen dataset."
        ),
        "water_exclusion": (
            "MNDWI > {}".format(params.mndwi_water_threshold)
            if params.exclude_water else "disabled"
        ),
    }

    warnings: list = []
    if not params.exclude_water:
        warnings.append(
            "WATER_EXCLUSION_DISABLED: probable water pixels were not removed; "
            "restoration eligibility may include water."
        )
    if suitability_mask is None:
        warnings.append(
            "NO_SUITABILITY_DATA: the frozen dataset provides no soil-moisture, "
            "water-availability, irrigation or land-use suitability channel. All "
            "eligible bare-soil pixels are assumed restorable by default; this is "
            "an unvalidated assumption, not a physical claim."
        )
    if water_available_mask is None:
        warnings.append(
            "NO_WATER_AVAILABILITY_DATA: irrigation water availability is not "
            "represented in the frozen dataset."
        )

    metadata = {
        "strategy": "bare_soil_restoration",
        "transformation": (
            "Soil_Frac converted to Vegetation_Frac (equal and opposite) on treated "
            "pixels up to target_vegetation_fraction, respecting per-channel [0,1]."
        ),
        "assumptions": [
            "Eligible bare soil is assumed restorable (no suitability data available).",
            "One-step counterfactual on the latest frame only.",
            "No direct temperature reduction is applied.",
        ],
        "limitations": [
            "No soil-moisture / water-availability / land-use suitability data.",
            "NDVI/EVI/Albedo not updated (no defensible transform in frozen data).",
        ],
    }
    return ScenarioApplication(
        input=transformed,
        treatment_mask=treated,
        eligible_mask=elig,
        polygon_mask=np.asarray(polygon_mask, dtype=bool),
        changed_channels=MODIFIED_CHANNELS,
        area=area,
        metadata=metadata,
        warnings=warnings,
    )