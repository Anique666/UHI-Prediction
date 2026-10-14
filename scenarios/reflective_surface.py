"""Reflective-surface (albedo-enhancement) scenario transformer.

Given a 12-month / 18-channel normalized input batch and a geographic polygon,
this module selects a deterministic, eligible subset of impervious/paved pixels
inside the polygon and raises their **latest-frame surface albedo** to the
requested target, leaving historical LST and meteorological observations
untouched. It edits only the ``Albedo`` channel and returns a *new* tensor (the
original is never mutated).

What is (and is not) modified
-----------------------------
* ``Albedo`` (5) is raised to ``target_albedo`` on eligible, treated pixels of
  the latest frame only. Eligibility requires the pixel to be impervious-dominant
  and currently below the target, so the change is always an increase.
* ``NDBI`` and the raw ``S2_*`` bands are **deliberately not modified**. The
  frozen dataset stores Sentinel-2 surface reflectance from which ``NDBI`` and
  broad-band albedo are built, but there is no defensible way to redistribute a
  user-requested albedo change across the individual S2 bands (a reflective
  coating raises visible reflectance without a determined SWIR response). Rather
  than invent reflectance values to force ``NDBI`` to move, they are retained
  unchanged and this limitation is reported, mirroring the vegetation module's
  treatment of ``NDVI``/``EVI``/``Albedo``.
* Nothing else is touched: LST, meteorology, terrain and land-cover fractions
  are preserved.

Physical honesty (see the scenario report)
------------------------------------------
* The albedo change is translated into a *change in absorbed shortwave energy*
  only when an incoming shortwave irradiance ``S_down`` (W/m^2) is supplied by
  the caller: ``delta_Q_abs = -S_down * delta_albedo`` over the treated surface
  fraction. When ``S_down`` is not supplied the energy accounting is **skipped**
  and a warning is emitted; no energy or temperature value is fabricated.
* The frozen dataset exposes **no building footprints and no roof/road
  separation**. ``Impervious_Frac`` and ``NDBI`` cannot distinguish a roof from
  a road, so this is reported as a *broad reflective-surface intervention*, not
  a building-specific scenario.
* No direct temperature reduction is applied and none is expected. This is a
  one-step counterfactual approximation, not a radiative-transfer or thermal
  energy-balance simulation. The result is a model-estimated counterfactual, not
  a measured causal effect.
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

ALBEDO_IDX = CHANNEL_NAMES.index("Albedo")
IMPERVIOUS_IDX = CHANNEL_NAMES.index("Impervious_Frac")
MODIFIED_CHANNELS = (ALBEDO_IDX,)

# The stored Albedo is a physical reflectance channel, so a user-requested
# target in [0, 1] is directly meaningful; no derived-index constant is needed.


@dataclass
class ReflectiveSurfaceParams:
    """User-configurable, unit-validated reflective-surface parameters.

    target_albedo : target broadband albedo in (0, 1]. A treated pixel's albedo
        is raised to this value (eligibility guarantees it was below the target).
    treatment_coverage_pct : percent in [0, 100] of *eligible* pixels to treat.
    max_treatment_area_m2 : hard cap on treated area (m^2, > 0) or ``None``.
    s_down_w_m2 : incoming shortwave irradiance (W/m^2, > 0) used *only* for the
        absorbed-shortwave energy accounting. ``None`` disables that accounting
        and produces an explicit warning (no fabricated energy values).
    min_impervious_fraction : eligibility floor on ``Impervious_Frac`` in
        [0, 1]; only built/paved-dominant pixels are eligible.
    """

    target_albedo: float = 0.30
    treatment_coverage_pct: float = 40.0
    max_treatment_area_m2: float | None = None
    s_down_w_m2: float | None = None
    min_impervious_fraction: float = 0.5

    def validate(self) -> None:
        if not (0.0 < self.target_albedo <= 1.0):
            raise ParameterError(
                f"target_albedo must be in (0, 1] (got {self.target_albedo})"
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
        if self.s_down_w_m2 is not None and self.s_down_w_m2 <= 0:
            raise ParameterError(
                f"s_down_w_m2 must be > 0 (got {self.s_down_w_m2})"
            )
        if not (0.0 <= self.min_impervious_fraction <= 1.0):
            raise ParameterError(
                "min_impervious_fraction must be in [0, 1] "
                f"(got {self.min_impervious_fraction})"
            )


def _eligibility(
    latest_raw: np.ndarray,
    *,
    polygon_mask: np.ndarray,
    valid_mask: np.ndarray,
    target_albedo: float,
    min_impervious_fraction: float,
    albedo_room_eps: float = 1e-6,
    surface_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Eligible reflective-surface pixels and their deterministic priority.

    Eligible = inside polygon, valid, finite inputs, impervious-dominant, with
    albedo strictly below the target. Priority is **decreasing impervious
    fraction** (most built first), tie-broken by fixed coordinate order; the
    caller converts it to the shared ascending-priority selection.
    """
    alb = latest_raw[ALBEDO_IDX]
    imp = latest_raw[IMPERVIOUS_IDX]
    finite = np.isfinite(alb) & np.isfinite(imp)
    impervious_ok = np.clip(imp, 0, None) >= min_impervious_fraction
    has_room = alb < (target_albedo - albedo_room_eps)

    elig = polygon_mask & valid_mask & finite & impervious_ok & has_room
    if surface_mask is not None:
        elig &= np.asarray(surface_mask, dtype=bool)

    # priority: more impervious first -> ascending priority = -impervious
    priority = np.full(alb.shape, np.inf, dtype=np.float64)
    if elig.any():
        priority[elig] = -np.clip(imp[elig], 0.0, 1.0)

    diag = {
        "polygon_pixels": int(polygon_mask.sum()),
        "valid_in_polygon_pixels": int((polygon_mask & valid_mask).sum()),
        "finite_input_pixels": int((polygon_mask & valid_mask & finite).sum()),
        "impervious_dominant_pixels": int((polygon_mask & valid_mask & finite & impervious_ok).sum()),
        "pixels_below_target_albedo": int((polygon_mask & valid_mask & finite & has_room).sum()),
        "eligible_pixels": int(elig.sum()),
        "surface_mask_applied": surface_mask is not None,
        "min_impervious_fraction": float(min_impervious_fraction),
        "target_albedo": float(target_albedo),
    }
    return elig, priority, diag


def apply_reflective_surface(
    input_batch: np.ndarray,
    latest_raw_frame: np.ndarray,
    *,
    polygon_mask: np.ndarray,
    valid_mask: np.ndarray,
    params: ReflectiveSurfaceParams,
    pixel_area_m2: float,
    means,
    stds,
    surface_mask: np.ndarray | None = None,
    treatment_override: np.ndarray | None = None,
) -> ScenarioApplication:
    """Apply the reflective-surface scenario to the latest input frame.

    Parameters
    ----------
    input_batch : ``[T, 18, H, W]`` normalized input (not mutated).
    latest_raw_frame : ``[18, H, W]`` *raw* (physical) latest frame, used for
        eligibility and the albedo transformation.
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
        target_albedo=params.target_albedo,
        min_impervious_fraction=params.min_impervious_fraction,
        surface_mask=surface_mask,
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
    alb = latest_raw_frame[ALBEDO_IDX].astype(np.float64)

    rows, cols = np.nonzero(treated)
    n_treated = 0
    sum_d_albedo = 0.0
    sum_delta_q_abs_wm2 = 0.0  # per treated pixel: -S_down * d_albedo (W/m^2)

    for r, c in zip(rows, cols):
        cur = float(alb[r, c])
        new_alb = float(params.target_albedo)
        d_alb = new_alb - cur
        if d_alb <= 0:
            continue  # eligibility already guarantees this, but stay safe

        transformed[-1, ALBEDO_IDX, r, c] = np.float32(
            (new_alb - means[ALBEDO_IDX]) / stds[ALBEDO_IDX]
        )
        n_treated += 1
        sum_d_albedo += d_alb
        if params.s_down_w_m2 is not None:
            sum_delta_q_abs_wm2 += (-float(params.s_down_w_m2) * d_alb)

    energy = {
        "available": params.s_down_w_m2 is not None,
        "s_down_w_m2": params.s_down_w_m2,
        "definition": (
            "delta_Q_abs = -S_down * delta_albedo, evaluated per treated pixel over "
            "the treated surface fraction (assumed = full pixel here). Negative values "
            "denote reduced absorbed shortwave energy."
        ),
        "units": "W/m^2 (per-pixel mean) and W (area-integrated)",
    }
    if params.s_down_w_m2 is not None and n_treated:
        energy.update({
            "mean_delta_q_abs_w_m2": sum_delta_q_abs_wm2 / n_treated,
            "total_delta_q_abs_w": sum_delta_q_abs_wm2 * float(pixel_area_m2),
        })
    else:
        energy.update({
            "mean_delta_q_abs_w_m2": None,
            "total_delta_q_abs_w": None,
            "reason_unavailable": (
                "No incoming shortwave irradiance (S_down) supplied; absorbed-shortwave "
                "energy change is not computed. No energy or temperature value is fabricated."
            ),
        })

    warnings: list = []
    if params.s_down_w_m2 is None:
        warnings.append(
            "ENERGY_ACCOUNTING_SKIPPED: no S_down supplied. The absorbed-shortwave "
            "energy change is not computed; the scenario is albedo-only and exploratory."
        )
    warnings.append(
        "DERIVED_INDEX_NOT_UPDATED: NDBI and the raw S2 bands are unchanged because the "
        "frozen data does not support a physically consistent redistribution of the "
        "albedo change across individual Sentinel-2 bands."
    )

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
        "mean_delta_albedo_treated": (sum_d_albedo / n_treated) if n_treated else 0.0,
        "temporal_assumption": "latest input frame only; first T-1 frames preserved",
        "modified_channels": [CHANNEL_NAMES[i] for i in MODIFIED_CHANNELS],
        "unchanged_channels_note": (
            "NDBI and S2_B* deliberately unchanged: the frozen dataset has no "
            "defensible S2-band -> albedo redistribution, so forcing NDBI would "
            "invent reflectance values."
        ),
        "surface_type_note": (
            "No building footprints or roof/road separation exist in the frozen "
            "dataset; Impervious_Frac and NDBI cannot distinguish roofs from roads. "
            "Described as a broad reflective-surface intervention."
        ),
        "energy_accounting": energy,
    }

    metadata = {
        "strategy": "reflective_surface",
        "target_albedo": float(params.target_albedo),
        "transformation": (
            "Albedo raised to target_albedo on treated pixels of the latest frame; "
            "no other channel modified."
        ),
        "energy_accounting": energy,
        "assumptions": [
            "delta_Q_abs = -S_down * delta_albedo (treated surface fraction = full pixel).",
            "No direct temperature reduction is applied; LST and air temperature are untouched.",
            "One-step counterfactual on the latest frame only.",
        ],
        "limitations": [
            "Broad reflective-surface intervention; roof/road cannot be separated.",
            "NDBI/S2 bands not updated (no defensible reflectance redistribution).",
            "Not a radiative-transfer or thermal energy-balance simulation.",
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