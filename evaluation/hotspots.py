"""Hotspot-focused validation metrics for urban LST / UHI.

These are pure, unit-testable functions operating on numpy arrays. They are used
by ``evaluate_hotspots.py`` to answer whether the frozen baseline:

1. localizes the hottest urban pixels,
2. predicts LST accurately at those pixels,
3. reproduces the intensity and spatial pattern of derived UHI.

Conventions
-----------
* "Hotspot" = a pixel at or above a high quantile (default 90th percentile) of
  the valid urban-pixel LST distribution for a given month.
* Observed and predicted hotspots are defined **independently** from their own
  value distributions over the *same* valid comparison population.
* All metrics ignore pixels that are NaN or outside the comparison mask.

Nothing here reads or writes frozen artifacts.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "finite",
    "quantile",
    "hotspot_mask",
    "binary_overlap",
    "masked_error_stats",
    "distribution_stats",
    "urban_hotspot_metrics",
    "spatial_rank_correlation",
]


def finite(values) -> np.ndarray:
    """Boolean finite mask."""
    return np.isfinite(np.asarray(values, dtype=np.float64))


def quantile(values, q: float, mask=None) -> float:
    """``q``-th percentile (0-100) of finite values, optionally under a mask."""
    v = np.asarray(values, dtype=np.float64)
    if mask is not None:
        v = v[np.asarray(mask, dtype=bool)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, q))


def hotspot_mask(values, threshold: float) -> np.ndarray:
    """Boolean mask of finite values ``>= threshold``.

    If ``threshold`` is NaN or not finite an all-False mask is returned, so a
    month with no valid pixels produces empty (never crashing) hotspot sets.
    """
    v = np.asarray(values, dtype=np.float64)
    if not np.isfinite(threshold):
        return np.zeros(v.shape, dtype=bool)
    return np.isfinite(v) & (v >= float(threshold))


def binary_overlap(observed, predicted) -> dict:
    """Precision / recall / IoU / Dice for two boolean hotspot sets.

    Empty-set policy: when both sets are empty IoU=Dice=1.0 (perfect vacuous
    agreement); when only one is empty precision/recall may be NaN where their
    denominator is zero, but IoU/Dice are 0.0.
    """
    o = np.asarray(observed, dtype=bool)
    p = np.asarray(predicted, dtype=bool)
    if o.shape != p.shape:
        raise ValueError("observed and predicted must have the same shape")
    tp = int(np.count_nonzero(o & p))
    fp = int(np.count_nonzero(~o & p))
    fn = int(np.count_nonzero(o & ~p))
    prec_den = tp + fp
    rec_den = tp + fn
    union = tp + fp + fn
    precision = tp / prec_den if prec_den else float("nan")
    recall = tp / rec_den if rec_den else float("nan")
    if union == 0:
        iou = 1.0
        dice = 1.0
    else:
        iou = tp / union
        dice = 2.0 * tp / (2.0 * tp + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "observed_hot_pixels": int(np.count_nonzero(o)),
        "predicted_hot_pixels": int(np.count_nonzero(p)),
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
    }


def masked_error_stats(observed, predicted, mask=None) -> dict:
    """MAE / RMSE / bias of ``predicted`` versus ``observed`` under ``mask``.

    Uses only pixels where both fields are finite (and inside ``mask``).
    """
    o = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if o.shape != p.shape:
        raise ValueError("observed and predicted must have the same shape")
    sel = np.isfinite(o) & np.isfinite(p)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != o.shape:
            raise ValueError("mask must match observed shape")
        sel = sel & m
    if not sel.any():
        return {
            "n": 0, "mae": float("nan"), "rmse": float("nan"),
            "bias": float("nan"), "mean_observed": float("nan"),
            "mean_predicted": float("nan"),
        }
    a = o[sel]
    b = p[sel]
    d = b - a
    return {
        "n": int(sel.sum()),
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "bias": float(d.mean()),
        "mean_observed": float(a.mean()),
        "mean_predicted": float(b.mean()),
    }


def distribution_stats(values, mask=None) -> dict:
    """Summary of a value distribution: mean and upper percentiles."""
    v = np.asarray(values, dtype=np.float64)
    if mask is not None:
        v = v[np.asarray(mask, dtype=bool)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "p90": float("nan"),
                "p95": float("nan"), "p99": float("nan")}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
    }


def urban_hotspot_metrics(observed, predicted, urban_mask, urban_valid, q: float = 90.0) -> dict:
    """Hotspot-localization + intensity metrics for one month.

    Parameters
    ----------
    observed, predicted : 2D arrays
        Observed and predicted fields on the shared grid (LST or UHI).
    urban_mask : 2D bool
        Fixed urban reference mask.
    urban_valid : 2D bool
        Shared valid comparison population (e.g. pred-valid AND obs-valid).
    q : float
        Hotspot quantile (default the top 10%).

    Returns a dict with the hotspot overlap metrics plus intensity errors
    measured **at the observed-hotspot locations** and upper-tail percentiles of
    the full valid urban distributions.
    """
    u = np.asarray(urban_mask, dtype=bool)
    v = np.asarray(urban_valid, dtype=bool)
    common = u & v

    obs_thr = quantile(observed, q, common)
    pred_thr = quantile(predicted, q, common)
    obs_hot = hotspot_mask(observed, obs_thr) & common
    pred_hot = hotspot_mask(predicted, pred_thr) & common

    overlap = binary_overlap(obs_hot, pred_hot)
    err_obs_hot = masked_error_stats(observed, predicted, obs_hot)

    obs_dist = distribution_stats(observed, common)
    pred_dist = distribution_stats(predicted, common)

    out = {
        "hotspot_quantile": q,
        "valid_urban_pixels": int(common.sum()),
        "obs_hotspot_threshold": obs_thr,
        "pred_hotspot_threshold": pred_thr,
        "observed_hotspot_n": int(obs_hot.sum()),
        "predicted_hotspot_n": int(pred_hot.sum()),
        **overlap,
        # internal boolean maps (stripped before serialization; used for figures)
        "_obs_hot_bool": obs_hot,
        "_pred_hot_bool": pred_hot,
    }
    out.update({
        "obs_hot_mae": err_obs_hot["mae"],
        "obs_hot_rmse": err_obs_hot["rmse"],
        "obs_hot_bias": err_obs_hot["bias"],
        "obs_hot_mean_obs": err_obs_hot["mean_observed"],
        "obs_hot_mean_pred": err_obs_hot["mean_predicted"],
        "obs_hot_n": err_obs_hot["n"],
    })
    for name, d in (("obs", obs_dist), ("pred", pred_dist)):
        out[f"{name}_urban_mean"] = d["mean"]
        out[f"{name}_p90"] = d["p90"]
        out[f"{name}_p95"] = d["p95"]
        out[f"{name}_p99"] = d["p99"]
    return out


def spatial_rank_correlation(observed, predicted, mask=None) -> float:
    """Spearman rank correlation between observed and predicted over ``mask``.

    Implemented via average ranks + Pearson so ties are handled correctly and a
    large pixel population stays fast. Returns NaN when fewer than 2 comparable
    pixels exist or either field is constant.
    """
    o = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if o.shape != p.shape:
        raise ValueError("observed and predicted must have the same shape")
    sel = np.isfinite(o) & np.isfinite(p)
    if mask is not None:
        sel = sel & np.asarray(mask, dtype=bool)
    a = o[sel]
    b = p[sel]
    if a.size < 2:
        return float("nan")
    ra = _average_rank(a)
    rb = _average_rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _average_rank(x: np.ndarray) -> np.ndarray:
    """Average ranks of ``x`` (1-based, ties averaged)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    n = x.size
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks