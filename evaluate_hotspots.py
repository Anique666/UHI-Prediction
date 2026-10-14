"""Hotspot-focused validation of the frozen UHI ConvLSTM baseline.

Evaluation only: this script never trains, fine-tunes, or writes frozen
artifacts. It runs the existing checkpoint over the **validation** split only
(never the test split), reconstructs per-month LST grids with the existing
deterministic weighted-overlap blender, derives UHI with the frozen urban/rural
masks, and answers four hotspot questions:

1. Does the model identify the correct hottest urban locations?
2. Does it predict temperature accurately at those locations?
3. Does it reproduce the intensity / spatial pattern of derived UHI?
4. What failure pattern should guide the next optimization experiment?

Outputs
-------
``outputs/baseline_full/validation_hotspots/``

* ``hotspot_metrics.json`` — full machine-readable metrics.
* ``hotspot_per_month.csv`` — per-month localization + intensity + UHI rows.
* ``hotspot_overall.csv`` — one-row overall summary.
* ``report.md`` — concise human report (written by ``write_report``).
* ``audit.json`` — sample/pixel coverage, masks, thresholds, skipped months.
* diagnostic PNGs for representative validation months.

Usage (from ``C:\\uhi-model`` with the .venv active):

    python evaluate_hotspots.py --checkpoint checkpoints/baseline_full/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.hotspots import (  # noqa: E402
    binary_overlap,
    spatial_rank_correlation,
    urban_hotspot_metrics,
)
from evaluation.rigorous import (  # noqa: E402
    build_combined_mask,
    load_model,
    sample_meta,
    sha256_file,
)
from evaluation.spatial import (  # noqa: E402
    OverlapGrid,
    compute_coverage,
    derive_uhi,
    rural_reference_mean,
)
from training.baselines import last_frame_lst_normalized  # noqa: E402
from utils.config import get_paths, load_config  # noqa: E402
from utils.data import make_dataset  # noqa: E402
from utils.normalization import load_normalization_stats, normalized_to_celsius  # noqa: E402
from utils.rasters import MonthRasterReader, load_mask_metadata  # noqa: E402
from utils.runmeta import collect_environment, write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402

PATCH_SIZE = 128
HOTSPOT_Q = 90.0


# ---------------------------------------------------------------------------
# Small aggregation helpers
# ---------------------------------------------------------------------------
def _sums(obs, pred, mask):
    """Return (sum_abs, sum_sq, sum_err, sum_t, sum_t2, count) over ``mask``."""
    sel = np.asarray(mask, dtype=bool) & np.isfinite(obs) & np.isfinite(pred)
    if not sel.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0
    o = obs[sel].astype(np.float64)
    p = pred[sel].astype(np.float64)
    d = p - o
    return (float(np.abs(d).sum()), float((d ** 2).sum()), float(d.sum()),
            float(o.sum()), float((o ** 2).sum()), int(sel.sum()))


def _finish(sums):
    s_abs, s_sq, s_err, s_t, s_t2, n = sums
    if n == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "bias": float("nan"), "r2": float("nan")}
    mean_t = s_t / n
    ss_tot = s_t2 - n * mean_t ** 2
    r2 = 1.0 - (s_sq / ss_tot) if ss_tot > 0 else float("nan")
    return {"n": int(n), "mae": s_abs / n, "rmse": (s_sq / n) ** 0.5,
            "bias": s_err / n, "r2": r2}


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Hotspot-focused validation")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    ap.add_argument("--checkpoint",
                    default=str(PROJECT_ROOT / "checkpoints" / "baseline_full" / "best.pt"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--quantile", type=float, default=HOTSPOT_Q,
                    help="Hotspot percentile (default 90 = top 10%%)")
    ap.add_argument("--n-figure-months", type=int, default=3)
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--max-months", type=int, default=None,
                    help="Limit the number of validation months (smoke testing)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Per-month reconstruction
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _reconstruct_month(model, dataset, idxs, stats, reader, device, lst_band,
                       H, W, batch_size):
    """Return pred/obs/pers Celsius grids + combined validity for one month."""
    pred_grid = OverlapGrid(H, W, PATCH_SIZE)
    obs_grid = OverlapGrid(H, W, PATCH_SIZE)
    pers_grid = OverlapGrid(H, W, PATCH_SIZE)
    for b in range(0, len(idxs), batch_size):
        chunk = idxs[b:b + batch_size]
        Xs, ys, ms, metas = [], [], [], []
        for idx in chunk:
            X, y, m = dataset[idx]
            Xs.append(X)
            ys.append(y)
            ms.append(m)
            metas.append(sample_meta(dataset, idx))
        X = torch.stack(Xs)
        y = torch.stack(ys)
        m = torch.stack(ms).to(torch.bool)
        comb = build_combined_mask(reader, metas, m, lst_band)
        pred_c = normalized_to_celsius(model(X.to(device)), stats)[:, 0].float().cpu().numpy()
        obs_c = normalized_to_celsius(y, stats)[:, 0].float().cpu().numpy()
        pers_c = normalized_to_celsius(
            last_frame_lst_normalized(X), stats)[:, 0].float().cpu().numpy()
        comb_np = comb[:, 0].numpy()
        for k, meta in enumerate(metas):
            pred_grid.add(pred_c[k], meta.row_off, meta.col_off, valid=comb_np[k])
            obs_grid.add(obs_c[k], meta.row_off, meta.col_off, valid=comb_np[k])
            pers_grid.add(pers_c[k], meta.row_off, meta.col_off, valid=comb_np[k])
    pred_map, pred_w = pred_grid.finalize()
    obs_map, obs_w = obs_grid.finalize()
    pers_map, _ = pers_grid.finalize()
    common = (pred_w > 0) & (obs_w > 0)
    return pred_map, obs_map, pers_map, common


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _render_figures(out_dir, month_reconstructions, chosen_months, urban, quantization,
                    transform, crs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    out_dir.mkdir(parents=True, exist_ok=True)
    for tag in chosen_months:
        r = month_reconstructions[tag]
        pred = r["pred_lst"]; obs = r["obs_lst"]; uhi_p = r["uhi_pred"]
        uhi_o = r["uhi_obs"]; common = r["common"]; obs_hot = r["obs_hot"]
        pred_hot = r["pred_hot"]
        u = urban.astype(bool)
        err = np.abs(pred - obs)

        vmin = float(np.nanpercentile(obs, 2))
        vmax = float(np.nanpercentile(obs, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.nanmin(obs)), float(np.nanmax(obs))

        # Panel 1: obs vs pred LST + abs error
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (title, arr, cmap, lim) in zip(axes, [
            (f"Observed LST {tag} (C)", np.where(common, obs, np.nan), "turbo", (vmin, vmax)),
            (f"Predicted LST {tag} (C)", np.where(common, pred, np.nan), "turbo", (vmin, vmax)),
            (f"|LST error| {tag} (C)  MAE={r['metadata']['model_mae_c']:.2f}", err, "magma", None),
        ]):
            im = ax.imshow(arr, cmap=cmap, vmin=(lim[0] if lim else None),
                           vmax=(lim[1] if lim else None))
            ax.set_title(title, fontsize=10); ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"lst_obs_pred_err_{tag}.png", dpi=110)
        plt.close(fig)

        # Panel 2: UHI obs vs pred + error
        ulim = float(np.nanpercentile(np.abs(uhi_o), 98)) if np.isfinite(uhi_o).any() else 1.0
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (title, arr, cmap, lim) in zip(axes, [
            (f"Observed UHI {tag} (C)", uhi_o, "coolwarm", (-ulim, ulim)),
            (f"Predicted UHI {tag} (C)", uhi_p, "coolwarm", (-ulim, ulim)),
            (f"|UHI error| {tag} (C)  MAE={r['metadata']['uhi_mae_c']:.2f}",
             np.abs(uhi_p - uhi_o), "magma", None),
        ]):
            im = ax.imshow(arr, cmap=cmap, vmin=(lim[0] if lim else None),
                           vmax=(lim[1] if lim else None))
            ax.set_title(title, fontsize=10); ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"uhi_obs_pred_err_{tag}.png", dpi=110)
        plt.close(fig)

        # Panel 3: hotspot overlap (urban common population only)
        tp = obs_hot & pred_hot & common & u
        fn = obs_hot & ~pred_hot & common & u
        fp = ~obs_hot & pred_hot & common & u
        cats = np.zeros_like(obs, dtype=np.int16)
        cats[common & u] = 1        # urban, neither hotspot
        cats[tp] = 2
        cats[fn] = 3
        cats[fp] = 4
        cmap = ListedColormap(["#00000000", "#d9d9d9", "#2ca02c", "#d62728", "#1f77b4"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(np.where(cats == 0, np.nan, cats), cmap=cmap, norm=norm)
        ax.set_title(f"Hotspot overlap {tag}: green=TP  red=missed(obs only)  "
                     f"blue=false pred   (top {100-quantization:.0f}% urban)", fontsize=10)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, ticks=[1, 2, 3, 4], fraction=0.046, pad=0.04)
        cb.ax.set_yticklabels(["other urban", "TP", "missed", "false pred"])
        fig.tight_layout()
        fig.savefig(out_dir / f"hotspot_overlap_{tag}.png", dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        paths["outputs_dir"] / checkpoint.parent.name / "validation_hotspots")
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_normalization_stats(paths["dataset_root"])
    lst_band = cfg["data"]["lst_channel_index"] + 1

    meta = load_mask_metadata(paths["mask_dir"])
    H, W = int(meta["height"]), int(meta["width"])
    with rasterio.open(paths["mask_dir"] / "urban_mask.tif") as ds:
        urban, transform, crs = ds.read(1), ds.transform, ds.crs
    with rasterio.open(paths["mask_dir"] / "rural_mask.tif") as ds:
        rural = ds.read(1)
    urban_b = urban > 0
    rural_b = rural > 0

    model, ckpt = load_model(cfg, checkpoint, device)
    print(f"Checkpoint : {checkpoint} (epoch {ckpt.get('epoch')}, best_val={ckpt.get('best_val')})")
    print(f"Device     : {device}")

    val_ds = make_dataset(cfg, "val")
    reader = MonthRasterReader(paths["dataset_root"], paths["temporal_root"])
    print(f"Val samples: {len(val_ds)}")

    by_month = defaultdict(list)
    for idx in range(len(val_ds)):
        by_month[int(val_ds.records[idx]["target_month_index"])].append(idx)

    # global accumulators (combined/valid population)
    g_model = (0.0,) * 6
    g_pers = (0.0,) * 6
    g_obs_hot = (0.0,) * 6
    agg_tp = agg_fp = agg_fn = 0
    uhi_sum_abs = uhi_sum_sq = uhi_sum_err = 0.0
    uhi_count = 0
    urban_pred_vals, urban_obs_vals = [], []
    rural_pred_vals, rural_obs_vals = [], []
    rank_corrs = []

    month_rows = []
    recon = {}
    skipped = []
    coverage_once = None
    urban_valid_pixel_counts = []

    ordered = sorted(by_month.items())
    if args.max_months is not None:
        ordered = ordered[: int(args.max_months)]
    for mi, (month_idx, idxs) in enumerate(ordered):
        tag = val_ds.records[idxs[0]]["target_month"]
        pred_map, obs_map, pers_map, common = _reconstruct_month(
            model, val_ds, idxs, stats, reader, device, lst_band, H, W, args.batch_size)

        n_common = int(common.sum())
        n_urban_common = int((common & urban_b).sum())
        if n_urban_common == 0:
            skipped.append({"target_month": tag, "reason": "no valid urban pixels"})
            continue

        if coverage_once is None:
            coverage_once = compute_coverage(common, urban, rural)

        # --- global LST accuracy (combined population) ---
        g_model = _add(g_model, _sums(obs_map, pred_map, common))
        g_pers = _add(g_pers, _sums(obs_map, pers_map, common))

        # --- derived UHI (frozen definition, per-field rural mean) ---
        rural_pred = rural_reference_mean(pred_map, common, rural)
        rural_obs = rural_reference_mean(obs_map, common, rural)
        uhi_pred = derive_uhi(pred_map, rural_pred, common)
        uhi_obs = derive_uhi(obs_map, rural_obs, common)
        urban_common = urban_b & common
        urban_pred = float(pred_map[urban_common].mean()) if urban_common.any() else float("nan")
        urban_obs = float(obs_map[urban_common].mean()) if urban_common.any() else float("nan")

        um = _sums(uhi_obs, uhi_pred, common)
        uhi_sum_abs += um[0]; uhi_sum_sq += um[1]; uhi_sum_err += um[2]; uhi_count += um[5]
        if n_urban_common:
            urban_pred_vals.append((urban_pred, n_urban_common))
            urban_obs_vals.append((urban_obs, n_urban_common))
        n_rural = int((rural_b & common).sum())
        if n_rural:
            rural_pred_vals.append((rural_pred, n_rural))
            rural_obs_vals.append((rural_obs, n_rural))

        # --- hotspots: LST (urban) ---
        lst_hot = urban_hotspot_metrics(obs_map, pred_map, urban_b, common, args.quantile)
        # --- hotspots: derived UHI (urban) ---
        uhi_hot = urban_hotspot_metrics(uhi_obs, uhi_pred, urban_b, common, args.quantile)

        # --- spatial rank correlation on urban LST ---
        rc = spatial_rank_correlation(obs_map, pred_map, urban_common)
        if np.isfinite(rc):
            rank_corrs.append(rc)

        # aggregate hotspot overlap (pixel-weighted across months)
        agg_tp += lst_hot["tp"]; agg_fp += lst_hot["fp"]; agg_fn += lst_hot["fn"]

        mm = _sums(obs_map, pred_map, urban_common)
        urban_valid_pixel_counts.append(n_urban_common)

        r = {
            "target_month": tag,
            "target_month_index": int(month_idx),
            "n_samples": len(idxs),
            "n_valid_pixels_common": n_common,
            "n_urban_valid": n_urban_common,
            "n_rural_valid": n_rural,
            "lst_model": _finish(_sums(obs_map, pred_map, common)),
            "lst_pers": _finish(_sums(obs_map, pers_map, common)),
            "lst_model_urban": _finish(mm),
            "rank_corr_urban_lst": rc,
            "rural_mean_pred_c": rural_pred,
            "rural_mean_obs_c": rural_obs,
            "urban_mean_pred_c": urban_pred,
            "urban_mean_obs_c": urban_obs,
            "urban_rural_contrast_pred_c": urban_pred - rural_pred,
            "urban_rural_contrast_obs_c": urban_obs - rural_obs,
            "contrast_abs_err_c": abs((urban_pred - rural_pred) - (urban_obs - rural_obs)),
            "uhi_mae_c": _finish(um)["mae"],
            "uhi_rmse_c": _finish(um)["rmse"],
            "uhi_bias_c": _finish(um)["bias"],
            "uhi_valid_pixels": _finish(um)["n"],
        }
        r["metadata"] = {"model_mae_c": r["lst_model"]["mae"], "uhi_mae_c": r["uhi_mae_c"]}
        r["lst_hotspot"] = lst_hot
        r["uhi_hotspot"] = uhi_hot

        recon[tag] = {
            "pred_lst": pred_map, "obs_lst": obs_map, "uhi_pred": uhi_pred,
            "uhi_obs": uhi_obs, "common": common,
            "obs_hot": np.asarray(lst_hot["_obs_hot_bool"], dtype=bool),
            "pred_hot": np.asarray(lst_hot["_pred_hot_bool"], dtype=bool),
            "metadata": r["metadata"],
        }
        month_rows.append(r)

        if (mi + 1) % 10 == 0 or mi == 0:
            print(f"  month {mi+1}/{len(ordered)}  {tag}  "
                  f"LST MAE={r['lst_model']['mae']:.3f}  UHI MAE={r['uhi_mae_c']:.3f}")

    val_ds.close()
    reader.close()

    if not month_rows:
        raise SystemExit("No validation months with valid urban pixels; nothing to report.")

    # ---- overall aggregates ----
    model_overall = _finish(g_model)
    pers_overall = _finish(g_pers)
    uhi_mae = uhi_sum_abs / uhi_count if uhi_count else float("nan")
    uhi_rmse = (uhi_sum_sq / uhi_count) ** 0.5 if uhi_count else float("nan")
    uhi_bias = uhi_sum_err / uhi_count if uhi_count else float("nan")

    def _wmean(pairs):
        tot = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / tot if tot else float("nan")

    urban_pred_o = _wmean(urban_pred_vals)
    urban_obs_o = _wmean(urban_obs_vals)
    rural_pred_o = _wmean(rural_pred_vals)
    rural_obs_o = _wmean(rural_obs_vals)

    # hotspot overlap: pixel-weighted overall + month-averaged
    union = agg_tp + agg_fp + agg_fn
    overall_overlap = {
        "tp": agg_tp, "fp": agg_fp, "fn": agg_fn,
        "precision": agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else float("nan"),
        "recall": agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else float("nan"),
        "iou": (agg_tp / union) if union else 1.0,
        "dice": (2.0 * agg_tp / (2.0 * agg_tp + agg_fp + agg_fn)) if union else 1.0,
    }

    def _mavg(key, sub="lst_hotspot"):
        vals = [r[sub][key] for r in month_rows if np.isfinite(r[sub][key])]
        return float(np.mean(vals)) if vals else float("nan")

    month_avg_overlap = {
        "precision": _mavg("precision"),
        "recall": _mavg("recall"),
        "iou": _mavg("iou"),
        "dice": _mavg("dice"),
        "rank_corr_urban_lst": float(np.mean(rank_corrs)) if rank_corrs else float("nan"),
    }
    # month-averaged intensity at observed hotspots
    month_avg_intensity = {
        "obs_hot_mae": _mavg("obs_hot_mae"),
        "obs_hot_rmse": _mavg("obs_hot_rmse"),
        "obs_hot_bias": _mavg("obs_hot_bias"),
        "obs_urban_p90": _mavg("obs_p90"),
        "pred_urban_p90": _mavg("pred_p90"),
        "obs_urban_p95": _mavg("obs_p95"),
        "pred_urban_p95": _mavg("pred_p95"),
        "p90_gap": _mavg("obs_p90") - _mavg("pred_p90"),
        "p95_gap": _mavg("obs_p95") - _mavg("pred_p95"),
    }

    overall = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": "val",
        "hotspot_quantile": args.quantile,
        "n_months": len(month_rows),
        "n_skipped_months": len(skipped),
        "lst_accuracy_celsius_combined_population": {
            "convlstm": model_overall,
            "persistence": pers_overall,
            "improvement_mae_c": pers_overall["mae"] - model_overall["mae"],
            "improvement_rmse_c": pers_overall["rmse"] - model_overall["rmse"],
        },
        "hotspot_localization_lst_urban": {
            "pixel_weighted_overall": overall_overlap,
            "month_averaged": month_avg_overlap,
        },
        "hotspot_intensity_lst_urban_at_observed_hotspots": month_avg_intensity,
        "uhi_derived": {
            "uhi_mae_c": uhi_mae,
            "uhi_rmse_c": uhi_rmse,
            "uhi_bias_c": uhi_bias,
            "uhi_valid_pixels": uhi_count,
            "rural_mean_pred_c_overall": rural_pred_o,
            "rural_mean_obs_c_overall": rural_obs_o,
            "urban_mean_pred_c_overall": urban_pred_o,
            "urban_mean_obs_c_overall": urban_obs_o,
            "urban_rural_contrast_pred_c_overall": urban_pred_o - rural_pred_o,
            "urban_rural_contrast_obs_c_overall": urban_obs_o - rural_obs_o,
            "urban_rural_contrast_abs_err_c": abs((urban_pred_o - rural_pred_o) - (urban_obs_o - rural_obs_o)),
        },
        "uhi_hotspot_month_averaged": {
            "precision": _mavg("precision", "uhi_hotspot"),
            "recall": _mavg("recall", "uhi_hotspot"),
            "iou": _mavg("iou", "uhi_hotspot"),
            "dice": _mavg("dice", "uhi_hotspot"),
            "obs_hot_mae": _mavg("obs_hot_mae", "uhi_hotspot"),
            "obs_hot_bias": _mavg("obs_hot_bias", "uhi_hotspot"),
        },
        "coverage": coverage_once,
        "skipped_months": skipped,
        "per_month": month_rows,
    }
    # strip internal boolean helpers before JSON write
    for r in overall["per_month"]:
        r["lst_hotspot"].pop("_obs_hot_bool", None)
        r["lst_hotspot"].pop("_pred_hot_bool", None)
        r["uhi_hotspot"].pop("_obs_hot_bool", None)
        r["uhi_hotspot"].pop("_pred_hot_bool", None)

    write_json(out_dir / "hotspot_metrics.json", overall)
    _write_per_month_csv(out_dir / "hotspot_per_month.csv", month_rows)
    _write_overall_csv(out_dir / "hotspot_overall.csv", overall)

    # audit
    audit = {
        "split": "val",
        "n_val_samples": sum(r["n_samples"] for r in month_rows),
        "months_evaluated": [r["target_month"] for r in month_rows],
        "months_skipped": skipped,
        "hotspot_quantile": args.quantile,
        "urban_mask_pixels": int(urban_b.sum()),
        "rural_mask_pixels": int(rural_b.sum()),
        "mask_source": str(paths["mask_dir"]),
        "coverage": coverage_once,
        "urban_valid_pixels_per_month": {
            r["target_month"]: r["n_urban_valid"] for r in month_rows},
        "population_note": (
            "Combined population = target-valid AND latest-input-valid. "
            "Hotspot overlap uses the shared valid urban population; observed "
            "and predicted thresholds are set independently."),
        "uhi_reference_note": (
            "UHI = LST - mean(valid LST over the fixed rural mask), computed "
            "separately for observed and predicted fields."),
        "environment": collect_environment(),
    }
    write_json(out_dir / "audit.json", audit)

    # figures for representative months
    if not args.skip_figures and month_rows:
        by_lst = sorted(month_rows, key=lambda r: r["lst_model"]["mae"])
        by_uhi = sorted(month_rows, key=lambda r: r["uhi_mae_c"])
        n = max(1, args.n_figure_months)
        picks = []
        for coll in (by_lst, by_uhi):
            m = len(coll)
            idxs = {0, (m - 1) // 2, m - 1} if n >= 3 else {m - 1}
            picks.extend(coll[i]["target_month"] for i in sorted(idxs))
        chosen = list(dict.fromkeys(picks))[: max(1, 2 * n)]
        _render_figures(out_dir, recon, chosen, urban, args.quantile, transform, crs)
        print(f"Figures    : {out_dir} ({len(chosen)} months)")

    # console summary
    print("\n=== Hotspot validation (VAL split) ===")
    print(f"LST  ConvLSTM   : MAE={model_overall['mae']:.4f} RMSE={model_overall['rmse']:.4f} "
          f"bias={model_overall['bias']:.4f} R2={model_overall['r2']:.4f} n={model_overall['n']}")
    print(f"LST  Persistence: MAE={pers_overall['mae']:.4f} RMSE={pers_overall['rmse']:.4f}")
    print(f"Hotspot IoU (pixel-weighted)={overall_overlap['iou']:.3f} "
          f"precision={overall_overlap['precision']:.3f} recall={overall_overlap['recall']:.3f}")
    print(f"Hotspot IoU (month-avg)      ={month_avg_overlap['iou']:.3f} "
          f"Dice={month_avg_overlap['dice']:.3f} rank_corr={month_avg_overlap['rank_corr_urban_lst']:.3f}")
    print(f"At observed hotspots: MAE={month_avg_intensity['obs_hot_mae']:.3f} "
          f"bias={month_avg_intensity['obs_hot_bias']:.3f} "
          f"p95 gap={month_avg_intensity['p95_gap']:.3f}")
    print(f"UHI: MAE={uhi_mae:.3f} RMSE={uhi_rmse:.3f} bias={uhi_bias:.3f} "
          f"contrast|err|={overall['uhi_derived']['urban_rural_contrast_abs_err_c']:.3f}")
    print(f"Artifacts: {out_dir}")

    try:
        from evaluation.hotspot_report import write_report
        write_report(out_dir, overall)
        print("Report     : report.md")
    except Exception as exc:  # pragma: no cover
        print(f"(report generator skipped: {exc})")
    return 0


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def _write_per_month_csv(path, rows):
    fields = [
        "target_month", "n_samples", "n_valid_pixels_common", "n_urban_valid",
        "model_mae_c", "model_rmse_c", "model_bias_c",
        "pers_mae_c", "pers_rmse_c",
        "hotspot_precision", "hotspot_recall", "hotspot_iou", "hotspot_dice",
        "rank_corr_urban_lst",
        "obs_hot_mae", "obs_hot_rmse", "obs_hot_bias",
        "obs_urban_p90", "pred_urban_p90", "obs_urban_p95", "pred_urban_p95",
        "rural_mean_pred_c", "rural_mean_obs_c", "urban_mean_pred_c", "urban_mean_obs_c",
        "urban_rural_contrast_pred_c", "urban_rural_contrast_obs_c", "contrast_abs_err_c",
        "uhi_mae_c", "uhi_rmse_c", "uhi_bias_c", "uhi_valid_pixels",
        "uhi_hotspot_iou", "uhi_hotspot_recall",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            lh = r["lst_hotspot"]; uh = r["uhi_hotspot"]; lm = r["lst_model"]; pp = r["lst_pers"]
            w.writerow({
                "target_month": r["target_month"], "n_samples": r["n_samples"],
                "n_valid_pixels_common": r["n_valid_pixels_common"],
                "n_urban_valid": r["n_urban_valid"],
                "model_mae_c": lm["mae"], "model_rmse_c": lm["rmse"], "model_bias_c": lm["bias"],
                "pers_mae_c": pp["mae"], "pers_rmse_c": pp["rmse"],
                "hotspot_precision": lh["precision"], "hotspot_recall": lh["recall"],
                "hotspot_iou": lh["iou"], "hotspot_dice": lh["dice"],
                "rank_corr_urban_lst": r["rank_corr_urban_lst"],
                "obs_hot_mae": lh["obs_hot_mae"], "obs_hot_rmse": lh["obs_hot_rmse"],
                "obs_hot_bias": lh["obs_hot_bias"],
                "obs_urban_p90": lh["obs_p90"], "pred_urban_p90": lh["pred_p90"],
                "obs_urban_p95": lh["obs_p95"], "pred_urban_p95": lh["pred_p95"],
                "rural_mean_pred_c": r["rural_mean_pred_c"], "rural_mean_obs_c": r["rural_mean_obs_c"],
                "urban_mean_pred_c": r["urban_mean_pred_c"], "urban_mean_obs_c": r["urban_mean_obs_c"],
                "urban_rural_contrast_pred_c": r["urban_rural_contrast_pred_c"],
                "urban_rural_contrast_obs_c": r["urban_rural_contrast_obs_c"],
                "contrast_abs_err_c": r["contrast_abs_err_c"],
                "uhi_mae_c": r["uhi_mae_c"], "uhi_rmse_c": r["uhi_rmse_c"],
                "uhi_bias_c": r["uhi_bias_c"], "uhi_valid_pixels": r["uhi_valid_pixels"],
                "uhi_hotspot_iou": uh["iou"], "uhi_hotspot_recall": uh["recall"],
            })


def _write_overall_csv(path, overall):
    m = overall["lst_accuracy_celsius_combined_population"]["convlstm"]
    p = overall["lst_accuracy_celsius_combined_population"]["persistence"]
    pw = overall["hotspot_localization_lst_urban"]["pixel_weighted_overall"]
    ma = overall["hotspot_localization_lst_urban"]["month_averaged"]
    inten = overall["hotspot_intensity_lst_urban_at_observed_hotspots"]
    u = overall["uhi_derived"]
    row = {
        "split": overall["split"], "n_months": overall["n_months"],
        "lst_mae_c": m["mae"], "lst_rmse_c": m["rmse"], "lst_bias_c": m["bias"], "lst_r2": m["r2"],
        "pers_mae_c": p["mae"], "pers_rmse_c": p["rmse"],
        "hotspot_precision_pw": pw["precision"], "hotspot_recall_pw": pw["recall"],
        "hotspot_iou_pw": pw["iou"], "hotspot_dice_pw": pw["dice"],
        "hotspot_iou_monthavg": ma["iou"], "hotspot_dice_monthavg": ma["dice"],
        "rank_corr_urban_lst": ma["rank_corr_urban_lst"],
        "obs_hot_mae_c": inten["obs_hot_mae"], "obs_hot_rmse_c": inten["obs_hot_rmse"],
        "obs_hot_bias_c": inten["obs_hot_bias"],
        "obs_urban_p90_c": inten["obs_urban_p90"], "pred_urban_p90_c": inten["pred_urban_p90"],
        "obs_urban_p95_c": inten["obs_urban_p95"], "pred_urban_p95_c": inten["pred_urban_p95"],
        "p90_gap_c": inten["p90_gap"], "p95_gap_c": inten["p95_gap"],
        "uhi_mae_c": u["uhi_mae_c"], "uhi_rmse_c": u["uhi_rmse_c"], "uhi_bias_c": u["uhi_bias_c"],
        "urban_rural_contrast_pred_c": u["urban_rural_contrast_pred_c_overall"],
        "urban_rural_contrast_obs_c": u["urban_rural_contrast_obs_c_overall"],
        "contrast_abs_err_c": u["urban_rural_contrast_abs_err_c"],
    }
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())