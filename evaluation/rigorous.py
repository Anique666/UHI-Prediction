"""Rigorous held-out evaluation for the UHI ConvLSTM baseline.

Implements the evaluation protocol:

* single deterministic streaming pass over the frozen **test** split (no
  shuffling, record order by ``sample_id``);
* masked metric aggregation over valid pixels only (error sums / valid counts,
  never per-batch averages);
* metric reporting in both normalized and physical (degrees Celsius) units,
  using the frozen LST statistics;
* a persistence baseline (last observed input LST) evaluated on the same
  population;
* per-sample and per-month metrics;
* deterministic spatial diagnostic maps for a small, predefined sample set;
* forecast-derived UHI evaluated per target month against ground-truth UHI
  using the frozen urban/rural masks.

The checkpoint and all frozen preprocessing artifacts are read-only. The test
split is used for evaluation only and never for selection or tuning.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window, transform as window_transform

from evaluation.spatial import (
    OverlapGrid,
    compute_coverage,
    derive_uhi,
    rural_reference_mean,
)
from models.convlstm import ConvLSTM
from training.baselines import last_frame_lst_normalized
from training.engine import load_checkpoint
from training.metrics import MetricAccumulator
from utils.config import get_paths, load_config
from utils.data import make_dataset
from utils.normalization import load_normalization_stats, normalized_to_celsius
from utils.rasters import MonthRasterReader, load_mask_metadata
from utils.runmeta import collect_environment, write_json
from utils.seed import set_seed

PATCH_SIZE = 128
LST_CHANNEL = "LST_Celsius"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hash of a file, streamed (for tying results to a checkpoint)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class SampleMeta:
    """Metadata for one test sample, in deterministic record order."""

    index: int
    sample_id: int
    patch_id: int
    row_off: int
    col_off: int
    height: int
    width: int
    target_month: str
    target_month_index: int
    input_months: list[str]

    @property
    def window(self):
        return ((self.row_off, self.row_off + self.height),
                (self.col_off, self.col_off + self.width))


def sample_meta(dataset, idx: int) -> SampleMeta:
    r = dataset.records[idx]
    return SampleMeta(
        index=idx,
        sample_id=int(r["sample_id"]),
        patch_id=int(r["patch_id"]),
        row_off=int(r["row_off"]),
        col_off=int(r["col_off"]),
        height=int(r["height"]),
        width=int(r["width"]),
        target_month=r["target_month"],
        target_month_index=int(r["target_month_index"]),
        input_months=list(r["input_months"]),
    )


def build_combined_mask(reader, metas, target_mask, lst_band):
    """target-valid AND latest-input-valid, per sample. [B,1,H,W] bool."""
    comb = target_mask.clone()
    if target_mask.dtype != torch.bool:
        comb = comb.to(torch.bool)
    for i, meta in enumerate(metas):
        latest = meta.input_months[-1]
        iv = torch.from_numpy(
            reader.read_validity(latest, lst_band, meta.window)
        ).to(comb.device).unsqueeze(0)
        comb[i] = comb[i] & iv
    return comb


def load_model(cfg, checkpoint, device):
    """Instantiate the frozen model and load the checkpoint state dict."""
    model = ConvLSTM(
        input_channels=cfg["data"]["input_channels"],
        hidden_channels=cfg["model"]["hidden_channels"],
        kernel_size=cfg["model"]["kernel_size"],
        output_channels=cfg["model"]["output_channels"],
    ).to(device)
    ckpt = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Per-month metrics containers
# ---------------------------------------------------------------------------
@dataclass
class MonthAcc:
    month: str
    month_index: int
    model_c: MetricAccumulator = field(default_factory=MetricAccumulator)
    pers_c: MetricAccumulator = field(default_factory=MetricAccumulator)
    model_n: MetricAccumulator = field(default_factory=MetricAccumulator)
    pers_n: MetricAccumulator = field(default_factory=MetricAccumulator)
    n_samples: int = 0
    n_combined_valid: int = 0
    n_target_valid: int = 0


# ---------------------------------------------------------------------------
# Phase B/C/D: streaming test evaluation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_lst(
    model,
    dataset,
    stats,
    reader,
    device,
    lst_band,
    batch_size,
    progress=None,
    max_samples=None,
) -> dict:
    """Single deterministic pass over the test split.

    Returns a dict with global/per-sample/per-month metrics for the ConvLSTM and
    the persistence baseline, in normalized and Celsius units.
    """
    model_c = MetricAccumulator()          # model, combined population, Celsius
    model_c_tv = MetricAccumulator()       # model, target-valid population, Celsius
    pers_c = MetricAccumulator()           # persistence, combined, Celsius
    model_n = MetricAccumulator()          # model, combined, normalized
    pers_n = MetricAccumulator()           # persistence, combined, normalized

    per_sample: list[dict] = []
    months: dict[int, MonthAcc] = {}
    skipped: list[dict] = []
    n = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    buf_X, buf_y, buf_m, buf_meta = [], [], [], []

    def flush():
        nonlocal buf_X, buf_y, buf_m, buf_meta
        if not buf_X:
            return
        X = torch.stack(buf_X).to(device)
        y = torch.stack(buf_y).to(device)
        m = torch.stack(buf_m).to(device).to(torch.bool)
        metas = list(buf_meta)

        pred_n = model(X)                       # [B,1,H,W] normalized
        pers_n_t = last_frame_lst_normalized(X)  # [B,1,H,W] normalized
        comb = build_combined_mask(reader, metas, m, lst_band).to(device)

        pred_c = normalized_to_celsius(pred_n, stats)
        tgt_c = normalized_to_celsius(y, stats)
        pers_c_t = normalized_to_celsius(pers_n_t, stats)

        model_c.update(pred_c, tgt_c, comb)
        model_c_tv.update(pred_c, tgt_c, m)
        pers_c.update(pers_c_t, tgt_c, comb)
        model_n.update(pred_n, y, comb)
        pers_n.update(pers_n_t, y, comb)

        # per-sample and per-month
        predc_np = pred_c[:, 0].float().cpu().numpy()
        tgtc_np = tgt_c[:, 0].float().cpu().numpy()
        persc_np = pers_c_t[:, 0].float().cpu().numpy()
        predn_np = pred_n[:, 0].float().cpu().numpy()
        tgtn_np = y[:, 0].float().cpu().numpy()
        persn_np = pers_n_t[:, 0].float().cpu().numpy()
        comb_np = comb[:, 0].cpu().numpy()
        m_np = m[:, 0].cpu().numpy()

        for k, meta in enumerate(metas):
            cvm = comb_np[k]
            tvm = m_np[k]
            n_comb = int(cvm.sum())
            n_tv = int(tvm.sum())
            if n_comb == 0:
                skipped.append({
                    "sample_id": meta.sample_id, "patch_id": meta.patch_id,
                    "target_month": meta.target_month,
                    "reason": "no pixels valid in combined population",
                })
            mc = predc_np[k] - tgtc_np[k]
            pc = persc_np[k] - tgtc_np[k]
            mn = predn_np[k] - tgtn_np[k]
            pn = persn_np[k] - tgtn_np[k]

            def _rmse(a):
                return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")

            per_sample.append({
                "sample_id": meta.sample_id,
                "patch_id": meta.patch_id,
                "row_off": meta.row_off,
                "col_off": meta.col_off,
                "target_month": meta.target_month,
                "target_month_index": meta.target_month_index,
                "n_combined_valid": n_comb,
                "n_target_valid": n_tv,
                "model_mae_c": float(np.abs(mc[cvm]).mean()) if n_comb else float("nan"),
                "model_rmse_c": _rmse(mc[cvm]),
                "model_bias_c": float(mc[cvm].mean()) if n_comb else float("nan"),
                "model_mae_norm": float(np.abs(mn[cvm]).mean()) if n_comb else float("nan"),
                "pers_mae_c": float(np.abs(pc[cvm]).mean()) if n_comb else float("nan"),
                "pers_rmse_c": _rmse(pc[cvm]),
                "pers_bias_c": float(pc[cvm].mean()) if n_comb else float("nan"),
                "pers_mae_norm": float(np.abs(pn[cvm]).mean()) if n_comb else float("nan"),
            })

            ma = months.get(meta.target_month_index)
            if ma is None:
                ma = MonthAcc(meta.target_month, meta.target_month_index)
                months[meta.target_month_index] = ma
            ma.n_samples += 1
            ma.n_combined_valid += n_comb
            ma.n_target_valid += n_tv
            if n_comb:
                ma.model_c.update(
                    torch.from_numpy(predc_np[k:k + 1])[None].to(device),
                    torch.from_numpy(tgtc_np[k:k + 1])[None].to(device),
                    torch.from_numpy(cvm)[None][None].to(device),
                )
                ma.pers_c.update(
                    torch.from_numpy(persc_np[k:k + 1])[None].to(device),
                    torch.from_numpy(tgtc_np[k:k + 1])[None].to(device),
                    torch.from_numpy(cvm)[None][None].to(device),
                )
                ma.model_n.update(
                    torch.from_numpy(predn_np[k:k + 1])[None].to(device),
                    torch.from_numpy(tgtn_np[k:k + 1])[None].to(device),
                    torch.from_numpy(cvm)[None][None].to(device),
                )
                ma.pers_n.update(
                    torch.from_numpy(persn_np[k:k + 1])[None].to(device),
                    torch.from_numpy(tgtn_np[k:k + 1])[None].to(device),
                    torch.from_numpy(cvm)[None][None].to(device),
                )

        buf_X, buf_y, buf_m, buf_meta = [], [], [], []
        del X, y, m, pred_n, pers_n_t, comb, pred_c, tgt_c, pers_c_t

    for idx in range(n):
        X, y, m = dataset[idx]
        if tuple(X.shape) != (12, 18, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"sample {idx}: unexpected X shape {tuple(X.shape)}")
        if tuple(y.shape) != (1, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"sample {idx}: unexpected y shape {tuple(y.shape)}")
        if tuple(m.shape) != (1, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"sample {idx}: unexpected mask shape {tuple(m.shape)}")
        buf_X.append(X)
        buf_y.append(y)
        buf_m.append(m)
        buf_meta.append(sample_meta(dataset, idx))
        if len(buf_X) >= batch_size:
            flush()
        if progress is not None and (idx + 1) % max(batch_size * 25, 1) == 0:
            progress(idx + 1, n)
    flush()

    def _block(acc, units):
        d = acc.as_dict()
        return {
            "mae": d["mae"], "rmse": d["rmse"], "bias": d["bias"], "r2": d["r2"],
            "valid_pixels": d["valid_pixels"], "units": units,
        }

    return {
        "n_samples": n,
        "n_skipped": len(skipped),
        "skipped": skipped,
        "convlstm": {
            "combined_population": _block(model_c, "celsius"),
            "target_valid_population": _block(model_c_tv, "celsius"),
            "combined_normalized": _block(model_n, "normalized"),
        },
        "persistence": {
            "combined_population": _block(pers_c, "celsius"),
            "combined_normalized": _block(pers_n, "normalized"),
        },
        "per_sample": per_sample,
        "per_month": {str(k): v for k, v in sorted(months.items())},
    }


# ---------------------------------------------------------------------------
# Phase F: per-month UHI reconstruction and evaluation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _infer_patch(model, X, device):
    return model(X.to(device))


def reconstruct_and_evaluate_uhi(
    model,
    dataset,
    stats,
    reader,
    device,
    lst_band,
    urban,
    rural,
    height,
    width,
    batch_size,
    crs,
    transform,
    maps_dir: Path,
    progress=None,
) -> dict:
    """Per-month forecast/observed LST grids and UHI comparison.

    Each target month is reconstructed from its test patches with the
    deterministic weighted-overlap blend. UHI uses the agreed definition with
    each field's own rural reference mean over the frozen rural mask.

    Monthly LST/UHI GeoTIFFs are written immediately (no full-grid arrays are
    retained in memory across months). Only compact metric rows are returned.
    """
    by_month: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        by_month[int(dataset.records[idx]["target_month_index"])].append(idx)

    rural_b = rural > 0
    urban_b = urban > 0
    month_rows = []
    g_sum_abs = 0.0
    g_sum_sq = 0.0
    g_sum_err = 0.0
    g_count = 0
    g_urban_sum_pred = g_urban_sum_obs = 0.0
    g_urban_n = 0

    maps_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(by_month.items())
    for mi, (month_idx, idxs) in enumerate(ordered):
        pred_grid = OverlapGrid(height, width, PATCH_SIZE)
        obs_grid = OverlapGrid(height, width, PATCH_SIZE)
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
            pred_c = normalized_to_celsius(_infer_patch(model, X, device), stats)[:, 0].float().cpu().numpy()
            obs_c = normalized_to_celsius(y, stats)[:, 0].float().cpu().numpy()
            comb_np = comb[:, 0].numpy()
            for k, meta in enumerate(metas):
                pred_grid.add(pred_c[k], meta.row_off, meta.col_off, valid=comb_np[k])
                obs_grid.add(obs_c[k], meta.row_off, meta.col_off, valid=comb_np[k])

        pred_map, pred_w = pred_grid.finalize()
        obs_map, obs_w = obs_grid.finalize()
        common = (pred_w > 0) & (obs_w > 0)

        rural_pred = rural_reference_mean(pred_map, common, rural)
        rural_obs = rural_reference_mean(obs_map, common, rural)
        uhi_pred = derive_uhi(pred_map, rural_pred, common)
        uhi_obs = derive_uhi(obs_map, rural_obs, common)
        err = uhi_pred - uhi_obs
        valid_err = np.isfinite(err)

        urban_common = urban_b & common
        urban_pred = float(pred_map[urban_common].mean()) if urban_common.any() else float("nan")
        urban_obs = float(obs_map[urban_common].mean()) if urban_common.any() else float("nan")

        n_rural_valid = int((rural_b & common).sum())
        n_urban_valid = int(urban_common.sum())
        e = err[valid_err]
        if e.size:
            g_sum_abs += float(np.abs(e).sum())
            g_sum_sq += float((e ** 2).sum())
            g_sum_err += float(e.sum())
            g_count += int(e.size)
        if n_urban_valid:
            g_urban_sum_pred += float(pred_map[urban_common].sum())
            g_urban_sum_obs += float(obs_map[urban_common].sum())
            g_urban_n += n_urban_valid

        tag = dataset.records[idxs[0]]["target_month"]
        _write_tif(maps_dir / f"{tag}_lst_pred_c.tif", pred_map.astype(np.float32), crs, transform)
        _write_tif(maps_dir / f"{tag}_lst_gt_c.tif", obs_map.astype(np.float32), crs, transform)
        _write_tif(maps_dir / f"{tag}_uhi_pred_c.tif", uhi_pred.astype(np.float32), crs, transform)
        _write_tif(maps_dir / f"{tag}_uhi_gt_c.tif", uhi_obs.astype(np.float32), crs, transform)
        _write_tif(maps_dir / f"{tag}_uhi_abs_err_c.tif", np.abs(err).astype(np.float32), crs, transform)

        month_rows.append({
            "target_month": tag,
            "target_month_index": int(month_idx),
            "n_samples": len(idxs),
            "n_valid_pixels_common": int(common.sum()),
            "n_rural_valid": n_rural_valid,
            "n_urban_valid": n_urban_valid,
            "rural_mean_pred_c": rural_pred,
            "rural_mean_obs_c": rural_obs,
            "rural_mean_diff_c": rural_pred - rural_obs,
            "urban_mean_pred_c": urban_pred,
            "urban_mean_obs_c": urban_obs,
            "urban_rural_contrast_pred_c": urban_pred - rural_pred,
            "urban_rural_contrast_obs_c": urban_obs - rural_obs,
            "uhi_mae_c": float(np.abs(e).mean()) if e.size else float("nan"),
            "uhi_rmse_c": float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan"),
            "uhi_bias_c": float(e.mean()) if e.size else float("nan"),
            "uhi_valid_pixels": int(e.size),
        })
        if progress is not None:
            progress(mi + 1, len(ordered))
        del pred_map, obs_map, uhi_pred, uhi_obs, err, common

    # Overall urban/rural means and contrasts, pixel-weighted across months.
    rural_pred_vals, rural_obs_vals, urb_pred_vals, urb_obs_vals = [], [], [], []
    for r in month_rows:
        if r["n_urban_valid"] and np.isfinite(r["urban_mean_pred_c"]):
            urb_pred_vals.append((r["urban_mean_pred_c"], r["n_urban_valid"]))
            urb_obs_vals.append((r["urban_mean_obs_c"], r["n_urban_valid"]))
        if r["n_rural_valid"] and np.isfinite(r["rural_mean_pred_c"]):
            rural_pred_vals.append((r["rural_mean_pred_c"], r["n_rural_valid"]))
            rural_obs_vals.append((r["rural_mean_obs_c"], r["n_rural_valid"]))

    def _wmean(pairs):
        tot = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / tot if tot else float("nan")

    urban_pred_overall = _wmean(urb_pred_vals)
    urban_obs_overall = _wmean(urb_obs_vals)
    rural_pred_overall = _wmean(rural_pred_vals)
    rural_obs_overall = _wmean(rural_obs_vals)

    global_uhi = {
        "uhi_mae_c": g_sum_abs / g_count if g_count else float("nan"),
        "uhi_rmse_c": (g_sum_sq / g_count) ** 0.5 if g_count else float("nan"),
        "uhi_bias_c": g_sum_err / g_count if g_count else float("nan"),
        "uhi_valid_pixels": g_count,
        "urban_mean_pred_c_overall": urban_pred_overall,
        "urban_mean_obs_c_overall": urban_obs_overall,
        "rural_mean_pred_c_overall": rural_pred_overall,
        "rural_mean_obs_c_overall": rural_obs_overall,
        "urban_rural_contrast_pred_c_overall": urban_pred_overall - rural_pred_overall,
        "urban_rural_contrast_obs_c_overall": urban_obs_overall - rural_obs_overall,
        "urban_rural_contrast_bias_c_overall": (urban_pred_overall - rural_pred_overall) - (urban_obs_overall - rural_obs_overall),
    }
    return {"per_month": month_rows, "global": global_uhi, "maps_dir": str(maps_dir)}


# ---------------------------------------------------------------------------
# Phase E: deterministic diagnostic maps
# ---------------------------------------------------------------------------
def select_diagnostic_samples(per_sample: list[dict], n_select: int = 4) -> list[dict]:
    """Deterministic selection by predefined rule: min / p50 / p90 / max MAE."""
    valid = [s for s in per_sample if s["n_combined_valid"] > 0
             and np.isfinite(s["model_mae_c"])]
    if not valid:
        return []
    ordered = sorted(valid, key=lambda s: (s["model_mae_c"], s["sample_id"]))
    m = len(ordered)
    picks = {
        "best": 0,
        "typical_median": (m - 1) // 2,
        "difficult_p90": min(m - 1, int(round(0.9 * (m - 1)))),
        "worst": m - 1,
    }
    out = []
    seen = set()
    for tag, pos in picks.items():
        s = ordered[pos]
        key = s["sample_id"]
        if key in seen:
            continue
        seen.add(key)
        out.append({**s, "selection": tag})
    return out


@torch.inference_mode()
def make_diagnostic_maps(
    model,
    dataset,
    stats,
    reader,
    device,
    lst_band,
    selections: list[dict],
    transform,
    crs,
    out_dir: Path,
) -> list[dict]:
    """Save 6 maps per selected sample (PNG) and LST/error GeoTIFFs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sel in selections:
        idx = None
        for i in range(len(dataset)):
            if int(dataset.records[i]["sample_id"]) == sel["sample_id"]:
                idx = i
                break
        if idx is None:
            continue
        X, y, m = dataset[idx]
        meta = sample_meta(dataset, idx)
        comb = build_combined_mask(reader, [meta], m.unsqueeze(0), lst_band)
        pred_c = normalized_to_celsius(model(X.unsqueeze(0).to(device)), stats)[0, 0].float().cpu().numpy()
        tgt_c = normalized_to_celsius(y, stats)[0, 0].float().cpu().numpy()
        pers_c = normalized_to_celsius(
            last_frame_lst_normalized(X.unsqueeze(0)), stats
        )[0, 0].float().cpu().numpy()
        cvm = comb[0, 0].numpy()

        def msk(a):
            return np.where(cvm, a, np.nan)

        gt = msk(tgt_c)
        pr = msk(pred_c)
        pe = msk(pers_c)
        sign_err = pr - gt
        abs_model = np.abs(sign_err)
        abs_pers = np.abs(pe - gt)

        lst_lim = (float(np.nanmin([gt, pr, pe])), float(np.nanmax([gt, pr, pe])))
        if not np.isfinite(lst_lim).all():
            lst_lim = (float(np.nanmin(gt)), float(np.nanmax(gt)))

        panels = [
            ("Ground-truth LST (C)", gt, "turbo", lst_lim),
            ("ConvLSTM predicted LST (C)", pr, "turbo", lst_lim),
            ("Persistence predicted LST (C)", pe, "turbo", lst_lim),
            ("|ConvLSTM error| (C)", abs_model, "magma", None),
            ("|Persistence error| (C)", abs_pers, "magma", None),
            ("ConvLSTM signed error (C)", sign_err, "coolwarm", None),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, (title, arr, cmap, lim) in zip(axes.ravel(), panels):
            if title.startswith("|"):
                vmin, vmax = 0.0, float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
            elif title.endswith("signed error (C)"):
                a = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 1.0
                vmin, vmax = -a, a
            else:
                vmin, vmax = lim
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}\n{sel['selection']}", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"sample_id={meta.sample_id} patch={meta.patch_id} "
            f"target={meta.target_month}  model_MAE={sel['model_mae_c']:.3f} C  "
            f"pers_MAE={sel['pers_mae_c']:.3f} C", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        png = out_dir / f"sample_{meta.sample_id:05d}_{meta.target_month}.png"
        fig.savefig(png, dpi=110)
        plt.close(fig)

        win = Window(meta.col_off, meta.row_off, meta.width, meta.height)
        wtrans = window_transform(win, transform)
        for name, arr in (("gt", gt), ("model", pr), ("pers", pe),
                          ("sign_err", sign_err), ("abs_model", abs_model),
                          ("abs_pers", abs_pers)):
            _write_tif(out_dir / f"sample_{meta.sample_id:05d}_{meta.target_month}_{name}.tif",
                       arr.astype(np.float32), crs, wtrans)
        manifest.append({
            "selection": sel["selection"],
            "sample_id": meta.sample_id,
            "patch_id": meta.patch_id,
            "target_month": meta.target_month,
            "model_mae_c": sel["model_mae_c"],
            "pers_mae_c": sel["pers_mae_c"],
            "png": png.name,
        })
    write_json(out_dir / "diagnostic_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Raster writer
# ---------------------------------------------------------------------------
def _write_tif(path: Path, arr: np.ndarray, crs, transform):
    a = np.asarray(arr, dtype=np.float32)
    profile = {
        "driver": "GTiff", "height": a.shape[0], "width": a.shape[1],
        "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
        "nodata": np.nan, "compress": "lzw",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(a, 1)


def save_uhi_maps(uhi_rows: list[dict], maps_dir: Path, mode: str = "representative"):
    """Save PNG panels of UHI GT/pred/|err| for a deterministic subset of months.

    Monthly GeoTIFFs are already written by ``reconstruct_and_evaluate_uhi``;
    this only renders a compact set of PNG diagnostics by reading them back.
    """
    if mode == "none":
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps_dir = Path(maps_dir)
    valid_rows = [r for r in uhi_rows if np.isfinite(r["uhi_mae_c"])]
    if not valid_rows:
        return
    ordered = sorted(valid_rows, key=lambda r: r["uhi_mae_c"])
    m = len(ordered)
    picks = {0, (m - 1) // 2, min(m - 1, int(round(0.9 * (m - 1)))), m - 1}
    chosen = [ordered[p] for p in sorted(picks)]

    def _read(tag, kind):
        p = maps_dir / f"{tag}_{kind}.tif"
        with rasterio.open(p) as ds:
            return ds.read(1)

    n = len(chosen)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n), squeeze=False)
    for i, r in enumerate(chosen):
        tag = r["target_month"]
        gt = _read(tag, "uhi_gt_c")
        pr = _read(tag, "uhi_pred_c")
        er = _read(tag, "uhi_abs_err_c")
        lim = float(np.nanmax(np.abs(gt))) if np.isfinite(gt).any() else 1.0
        for j, (title, arr, cmap, vmin, vmax) in enumerate([
            (f"GT UHI {tag}", gt, "coolwarm", -lim, lim),
            (f"Predicted UHI {tag}", pr, "coolwarm", -lim, lim),
            (f"|UHI error| {tag} MAE={r['uhi_mae_c']:.2f}C", er, "magma", 0,
             max(lim, float(np.nanmax(er)) if np.isfinite(er).any() else 1.0)),
        ]):
            im = axes[i][j].imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i][j].set_title(title, fontsize=10)
            axes[i][j].axis("off")
            fig.colorbar(im, ax=axes[i][j], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(maps_dir / "uhi_representative_months.png", dpi=110)
    plt.close(fig)