"""Registry of mitigation strategies for the scenario runner.

Each strategy exposes the same transformation contract used by the vegetation
module: ``apply_fn(input_batch, latest_raw_frame, *, polygon_mask, valid_mask,
params, pixel_area_m2, means, stds, treatment_override=None, ...)`` returning a
:class:`~scenarios.vegetation_restoration.ScenarioApplication`. The runner uses
this registry to dispatch CLI parameters and the transformation function without
any strategy-specific inference code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from scenarios.bare_soil_restoration import (
    BareSoilRestorationParams,
    apply_bare_soil_restoration,
)
from scenarios.reflective_surface import (
    ReflectiveSurfaceParams,
    apply_reflective_surface,
)
from scenarios.vegetation_restoration import (
    VegetationRestorationParams,
    apply_vegetation_restoration,
)


def _veg_params_from_args(a) -> VegetationRestorationParams:
    return VegetationRestorationParams(
        target_vegetation_fraction=a.target_vf,
        treatment_coverage_pct=a.coverage_pct,
        max_treatment_area_m2=a.max_area_m2,
        soil_weight=a.soil_weight,
    )


def _reflective_params_from_args(a) -> ReflectiveSurfaceParams:
    return ReflectiveSurfaceParams(
        target_albedo=a.target_albedo,
        treatment_coverage_pct=a.coverage_pct,
        max_treatment_area_m2=a.max_area_m2,
        s_down_w_m2=a.s_down_w_m2,
        min_impervious_fraction=a.min_impervious_fraction,
    )


def _bare_soil_params_from_args(a) -> BareSoilRestorationParams:
    return BareSoilRestorationParams(
        target_vegetation_fraction=a.target_vf,
        treatment_coverage_pct=a.coverage_pct,
        max_treatment_area_m2=a.max_area_m2,
        min_soil_fraction=a.min_soil_fraction,
        max_vegetation_fraction=a.max_veg_fraction,
        exclude_water=(not a.no_exclude_water),
        mndwi_water_threshold=a.mndwi_water_threshold,
    )


@dataclass(frozen=True)
class StrategySpec:
    name: str
    label: str
    params_cls: type
    apply_fn: Callable
    params_from_args: Callable
    modified_channel_names: tuple


STRATEGIES: dict[str, StrategySpec] = {
    "vegetation_restoration": StrategySpec(
        name="vegetation_restoration",
        label="vegetation restoration",
        params_cls=VegetationRestorationParams,
        apply_fn=apply_vegetation_restoration,
        params_from_args=_veg_params_from_args,
        modified_channel_names=("Vegetation_Frac", "Impervious_Frac", "Soil_Frac"),
    ),
    "reflective_surface": StrategySpec(
        name="reflective_surface",
        label="reflective-surface treatment",
        params_cls=ReflectiveSurfaceParams,
        apply_fn=apply_reflective_surface,
        params_from_args=_reflective_params_from_args,
        modified_channel_names=("Albedo",),
    ),
    "bare_soil_restoration": StrategySpec(
        name="bare_soil_restoration",
        label="bare-soil and degraded-land restoration",
        params_cls=BareSoilRestorationParams,
        apply_fn=apply_bare_soil_restoration,
        params_from_args=_bare_soil_params_from_args,
        modified_channel_names=("Vegetation_Frac", "Soil_Frac"),
    ),
}


def get_strategy(name: str) -> StrategySpec:
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[name]