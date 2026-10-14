"""Held-out test evaluation for the UHI ConvLSTM baseline.

Runs the rigorous evaluation protocol defined in ``evaluation/rigorous.py`` on
the frozen test split exactly once and writes a complete artifact set under
``outputs/baseline_full/evaluation/``:

    evaluation_audit.md, metrics.json, metrics_summary.csv,
    per_sample_metrics.csv, per_month_metrics.csv, uhi_per_month.csv,
    comparison_report.md, diagnostic maps, UHI maps, eval_log.json.

The checkpoint and every frozen preprocessing artifact are read-only. The test
split is used for evaluation only.

Usage (from C:\\uhi-model with the .venv active):

    python evaluate.py --checkpoint checkpoints/baseline_full/best.pt
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.rigorous import (  # noqa: E402
    evaluate_lst,
    load_model,
    make_diagnostic_maps,
    reconstruct_and_evaluate_uhi,
    save_uhi_maps,
    select_diagnostic_samples,
    sha256_file,
)
from utils.config import get_paths, load_config  # noqa: E402
from utils.data import make_dataset  # noqa: E402
from utils.normalization import load_normalization_stats  # noqa: E402
from utils.rasters import MonthRasterReader, load_mask_metadata  # noqa: E402
from utils.runmeta import collect_environment, write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Rigorous test evaluation (UHI ConvLSTM)")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "baseline_full" / "best.pt"))
    ap.add_argument("--out-dir", default=None,
                    help="Default: outputs/<run>/evaluation derived from the checkpoint")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-diagnostic-samples", type=int, default=4)
    ap.add_argument("--skip-uhi", action="store_true", help="Skip UHI reconstruction")
    ap.add_argument("--skip-diagnostics", action="store_true", help="Skip spatial maps")
    return ap.parse_args(argv)


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main(argv=None) -> int:
    t_start = time.perf_counter()
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = get_paths(cfg)

    seed = int(cfg["training"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    # Output directory: outputs/<run_id>/evaluation (never overwrites training outputs).
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        run_id = checkpoint.parent.name
        out_dir = paths["outputs_dir"] / run_id / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_dir / "uhi_maps"
    diag_dir = out_dir / "diagnostics"

    # --- audit copy (authored in Phase A; ensure it is present alongside outputs)
    audit_src = paths["outputs_dir"] / checkpoint.parent.name / "evaluation_audit.md"
    if audit_src.is_file():
        shutil.copy2(audit_src, out_dir / "evaluation_audit.md")

    print(f"Checkpoint    : {checkpoint}")
    print(f"Output dir    : {out_dir}")
    print(f"Device        : {device}")

    # --- frozen inputs (read-only)
    stats = load_normalization_stats(paths["dataset_root"])
    lst_band = cfg["data"]["lst_channel_index"] + 1
    ckpt_sha = sha256_file(checkpoint)

    # --- masks + grid
    meta = load_mask_metadata(paths["mask_dir"])
    H, W = int(meta["height"]), int(meta["width"])
    with rasterio.open(paths["mask_dir"] / "urban_mask.tif") as ds:
        urban, transform, crs = ds.read(1), ds.transform, ds.crs
    with rasterio.open(paths["mask_dir"] / "rural_mask.tif") as ds:
        rural = ds.read(1)

    # --- model + dataset
    model, ckpt = load_model(cfg, checkpoint, device)
    print(f"Loaded ckpt   : epoch {ckpt.get('epoch')} best_val={ckpt.get('best_val')}")

    test_ds = make_dataset(cfg, "test")
    reader = MonthRasterReader(paths["dataset_root"], paths["temporal_root"])
    print(f"Test samples  : {len(test_ds)}")

    result: dict = {}
    try:
        # ---------------- Phase B/C/D: LST metrics ----------------
        def prog(i, n):
            print(f"  eval {i}/{n}", end="\r")

        print("\n[1/3] Streaming LST evaluation ...")
        lst = evaluate_lst(
            model, test_ds, stats, reader, device, lst_band,
            batch_size=args.batch_size, progress=prog,
        )
        print("")
        result["lst"] = lst

        # ---------------- Phase F: UHI ----------------
        uhi = None
        if not args.skip_uhi:
            print("[2/3] Per-month UHI reconstruction ...")
            uhi = reconstruct_and_evaluate_uhi(
                model, test_ds, stats, reader, device, lst_band,
                urban, rural, H, W, args.batch_size, crs, transform, maps_dir,
                progress=lambda i, n: print(f"  uhi {i}/{n}", end="\r"),
            )
            print("")
            result["uhi"] = uhi
            save_uhi_maps(uhi["per_month"], maps_dir, mode="representative")

        # ---------------- Phase E: diagnostics ----------------
        if not args.skip_diagnostics:
            print("[3/3] Diagnostic maps ...")
            selections = select_diagnostic_samples(
                lst["per_sample"], n_select=args.n_diagnostic_samples
            )
            make_diagnostic_maps(
                model, test_ds, stats, reader, device, lst_band,
                selections, transform, crs, diag_dir,
            )
    finally:
        test_ds.close()
        reader.close()

    # ---------------- Phase G: artifacts ----------------
    lst = result["lst"]
    mc = lst["convlstm"]["combined_population"]
    mt = lst["convlstm"]["target_valid_population"]
    mn = lst["convlstm"]["combined_normalized"]
    pc = lst["persistence"]["combined_population"]
    pn = lst["persistence"]["combined_normalized"]

    def _pct(pers, model_v):
        if pers and np.isfinite(pers) and pers != 0:
            return 100.0 * (pers - model_v) / pers
        return float("nan")

    comparison = {
        "population": "combined (target-valid AND latest-input-valid)",
        "units": "degrees Celsius",
        "convlstm": {"mae": mc["mae"], "rmse": mc["rmse"], "bias": mc["bias"], "r2": mc["r2"]},
        "persistence": {"mae": pc["mae"], "rmse": pc["rmse"], "bias": pc["bias"], "r2": pc["r2"]},
        "improvement_mae_abs": pc["mae"] - mc["mae"],
        "improvement_rmse_abs": pc["rmse"] - mc["rmse"],
        "improvement_mae_pct": _pct(pc["mae"], mc["mae"]),
        "improvement_rmse_pct": _pct(pc["rmse"], mc["rmse"]),
    }

    metrics = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_val_normalized": ckpt.get("best_val"),
        "split": "test",
        "split_samples": lst["n_samples"],
        "skipped_samples": lst["n_skipped"],
        "units": {"celsius": "degrees Celsius (raw LST)", "normalized": "z-score (train stats)"},
        "normalization": {
            "lst_mean": stats.means[stats.index("LST_Celsius")],
            "lst_std": stats.stds[stats.index("LST_Celsius")],
            "source": str(paths["dataset_root"] / "normalization_stats.json"),
        },
        "convlstm": {
            "combined_population_celsius": mc,
            "target_valid_population_celsius": mt,
            "combined_population_normalized": mn,
        },
        "persistence": {
            "combined_population_celsius": pc,
            "combined_population_normalized": pn,
        },
        "comparison": comparison,
        "omitted_or_unavailable": {
            "per_pixel_urban_mask_unavailable": False,
            "notes": [
                "Persistence validity (latest input LST) is recovered from raw rasters; "
                "excluded pixels are folded into the combined population count.",
                "Per-month metrics require sample metadata (available).",
                "UHI metrics computed only when not skipped.",
            ],
        },
    }
    if uhi is not None:
        metrics["uhi_global"] = uhi["global"]

    write_json(out_dir / "metrics.json", metrics)

    # summary CSV
    summary_rows = [
        {"metric": "MAE_C", "convlstm": mc["mae"], "persistence": pc["mae"],
         "improvement_abs": comparison["improvement_mae_abs"],
         "improvement_pct": comparison["improvement_mae_pct"]},
        {"metric": "RMSE_C", "convlstm": mc["rmse"], "persistence": pc["rmse"],
         "improvement_abs": comparison["improvement_rmse_abs"],
         "improvement_pct": comparison["improvement_rmse_pct"]},
        {"metric": "Bias_C", "convlstm": mc["bias"], "persistence": pc["bias"],
         "improvement_abs": "", "improvement_pct": ""},
        {"metric": "R2", "convlstm": mc["r2"], "persistence": pc["r2"],
         "improvement_abs": "", "improvement_pct": ""},
        {"metric": "valid_pixels", "convlstm": mc["valid_pixels"],
         "persistence": pc["valid_pixels"], "improvement_abs": "", "improvement_pct": ""},
    ]
    _write_csv(out_dir / "metrics_summary.csv", summary_rows)
    _write_csv(out_dir / "per_sample_metrics.csv", lst["per_sample"])

    per_month_rows = []
    for k, ma in lst["per_month"].items():
        per_month_rows.append({
            "target_month_index": int(k),
            "target_month": ma.month,
            "n_samples": ma.n_samples,
            "n_combined_valid": ma.n_combined_valid,
            "n_target_valid": ma.n_target_valid,
            "model_mae_c": ma.model_c.mae,
            "model_rmse_c": ma.model_c.rmse,
            "model_bias_c": ma.model_c.bias,
            "model_mae_norm": ma.model_n.mae,
            "pers_mae_c": ma.pers_c.mae,
            "pers_rmse_c": ma.pers_c.rmse,
            "pers_bias_c": ma.pers_c.bias,
            "pers_mae_norm": ma.pers_n.mae,
        })
    _write_csv(out_dir / "per_month_metrics.csv", per_month_rows)

    if uhi is not None:
        _write_csv(out_dir / "uhi_per_month.csv", uhi["per_month"])

    # comparison report
    _write_comparison_report(
        out_dir / "comparison_report.md", checkpoint, ckpt_sha, ckpt,
        metrics, uhi, stats, seed,
    )

    elapsed = time.perf_counter() - t_start
    write_json(out_dir / "eval_log.json", {
        "status": "completed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": ckpt.get("epoch"),
        "config_path": str(args.config),
        "config_snapshot": cfg,
        "split": "test",
        "seed": seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "environment": collect_environment(),
        "artifacts": sorted(p.name for p in out_dir.iterdir()),
    })

    print("\n=== Test metrics (degrees Celsius, combined population) ===")
    print(f"ConvLSTM   : MAE={mc['mae']:.4f} RMSE={mc['rmse']:.4f} "
          f"bias={mc['bias']:.4f} R2={mc['r2']:.4f} n={mc['valid_pixels']}")
    print(f"Persistence: MAE={pc['mae']:.4f} RMSE={pc['rmse']:.4f} "
          f"bias={pc['bias']:.4f} R2={pc['r2']:.4f} n={pc['valid_pixels']}")
    print(f"Improvement: MAE {comparison['improvement_mae_pct']:.1f}%  "
          f"RMSE {comparison['improvement_rmse_pct']:.1f}%")
    if uhi is not None:
        g = uhi["global"]
        print(f"UHI (forecast vs obs): MAE={g['uhi_mae_c']:.4f} RMSE={g['uhi_rmse_c']:.4f} "
              f"bias={g['uhi_bias_c']:.4f} n={g['uhi_valid_pixels']}")
    print(f"Elapsed: {elapsed:.1f}s   Artifacts: {out_dir}")
    return 0


def _write_comparison_report(path, checkpoint, ckpt_sha, ckpt, metrics, uhi, stats, seed):
    mc = metrics["convlstm"]["combined_population_celsius"]
    mt = metrics["convlstm"]["target_valid_population_celsius"]
    mn = metrics["convlstm"]["combined_population_normalized"]
    pc = metrics["persistence"]["combined_population_celsius"]
    cmp = metrics["comparison"]
    lst_i = stats.index("LST_Celsius")
    lines = []
    lines.append("# Held-out Test Evaluation — Comparison Report\n")
    lines.append(f"- Checkpoint: `{checkpoint}`")
    lines.append(f"- Checkpoint SHA-256: `{ckpt_sha}`")
    lines.append(f"- Checkpoint epoch (0-indexed): {ckpt.get('epoch')} "
                 f"(history epoch {int(ckpt.get('epoch', -1)) + 1})")
    lines.append(f"- Best validation MAE (normalized): {ckpt.get('best_val')}")
    lines.append(f"- Split: **test** only (never used for selection); seed {seed}")
    lines.append(f"- Target: next-month LST, raw units **degrees Celsius**")
    lines.append(f"- Inverse transform: `LST_C = LST_norm * {stats.stds[lst_i]:.10g} + {stats.means[lst_i]:.10g}`\n")
    lines.append("## Metrics (valid pixels; combined population = target-valid ∧ latest-input-valid)\n")
    lines.append("| Model | MAE (C) | RMSE (C) | Bias (C) | R² | Valid pixels |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| ConvLSTM | {mc['mae']:.4f} | {mc['rmse']:.4f} | {mc['bias']:.4f} | {mc['r2']:.4f} | {mc['valid_pixels']} |")
    lines.append(f"| Persistence | {pc['mae']:.4f} | {pc['rmse']:.4f} | {pc['bias']:.4f} | {pc['r2']:.4f} | {pc['valid_pixels']} |")
    lines.append("")
    lines.append(f"- ConvLSTM on the full target-valid population: MAE={mt['mae']:.4f} C, "
                 f"RMSE={mt['rmse']:.4f} C, n={mt['valid_pixels']}")
    lines.append(f"- ConvLSTM normalized (combined): MAE={mn['mae']:.4f}, RMSE={mn['rmse']:.4f}\n")
    lines.append("## Improvement of ConvLSTM over persistence\n")
    lines.append(f"- MAE: {cmp['improvement_mae_abs']:.4f} C "
                 f"({cmp['improvement_mae_pct']:.2f}%)")
    lines.append(f"- RMSE: {cmp['improvement_rmse_abs']:.4f} C "
                 f"({cmp['improvement_rmse_pct']:.2f}%)")
    beats = cmp["improvement_mae_pct"] > 0 and cmp["improvement_rmse_pct"] > 0
    lines.append(f"- Verdict: ConvLSTM {'beats' if beats else 'does NOT beat'} persistence "
                 f"on both MAE and RMSE (positive % means ConvLSTM better).\n")
    if uhi is not None:
        g = uhi["global"]
        lines.append("## UHI / SUHI (forecast-derived vs ground truth)\n")
        lines.append("- Definition: UHI = LST − mean(LST over the fixed rural mask), computed "
                     "separately for each field with its own valid rural pixels.\n")
        lines.append(f"- UHI MAE: {g['uhi_mae_c']:.4f} C | RMSE: {g['uhi_rmse_c']:.4f} C | "
                     f"bias: {g['uhi_bias_c']:.4f} C | valid pixels: {g['uhi_valid_pixels']}")
        lines.append(f"- Rural mean (pred): {g['rural_mean_pred_c_overall']:.3f} C | "
                     f"Rural mean (obs): {g['rural_mean_obs_c_overall']:.3f} C")
        lines.append(f"- Urban mean (pred): {g['urban_mean_pred_c_overall']:.3f} C | "
                     f"Urban mean (obs): {g['urban_mean_obs_c_overall']:.3f} C")
        lines.append(f"- Urban−rural contrast (pred): {g['urban_rural_contrast_pred_c_overall']:.3f} C | "
                     f"(obs): {g['urban_rural_contrast_obs_c_overall']:.3f} C | "
                     f"bias: {g['urban_rural_contrast_bias_c_overall']:.3f} C")
        lines.append("\nThe model forecasts **LST**; UHI is derived downstream, not predicted directly.\n")
    lines.append("## Notes / limitations\n")
    lines.append("- Spatial split is an artificial directional partition (frozen), not administrative.")
    lines.append("- Persistence is meaningful only where the latest input LST is genuine; the "
                 "combined population enforces this for both models.")
    lines.append("- Test split patch overlap is reconciled deterministically for UHI maps (triangular "
                 "weighted overlap).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())