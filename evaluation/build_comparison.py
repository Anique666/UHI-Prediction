"""Build the baseline-vs-hotspot comparison report from evaluator outputs.

Read-only with respect to checkpoints and frozen data. Consumes the two
``hotspot_metrics.json`` files produced by ``evaluate_hotspots.py`` (validation
split) and writes ``comparison.json`` and ``comparison.csv`` next to them.

Units: all accuracy/UHI/hotspot-intensity numbers are degrees Celsius (the
evaluator inverse-transforms normalized LST with the frozen LST stats). Overlap
metrics (precision/recall/IoU/Dice) and the Spearman rank correlation are
dimensionless.

Run from ``C:\\uhi-model``:
    python evaluation/build_comparison.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "baseline_full" / "hotspot_metrics.json"
HOTSPOT = ROOT / "hotspot_w5" / "hotspot_metrics.json"
OUT_JSON = ROOT / "comparison.json"
OUT_CSV = ROOT / "comparison.csv"
OUT_CSV_MONTH = ROOT / "comparison_per_month.csv"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _delta(a, b):
    if a is None or b is None:
        return None
    try:
        return b - a
    except TypeError:
        return None


def _flat_overall(d: dict) -> dict:
    """Flatten the evaluator's overall block into a flat metric dict (Celsius)."""
    acc = d["lst_accuracy_celsius_combined_population"]
    conv = acc["convlstm"]
    pers = acc["persistence"]
    pw = d["hotspot_localization_lst_urban"]["pixel_weighted_overall"]
    ma = d["hotspot_localization_lst_urban"]["month_averaged"]
    inten = d["hotspot_intensity_lst_urban_at_observed_hotspots"]
    u = d["uhi_derived"]
    uh = d["uhi_hotspot_month_averaged"]
    return {
        "lst_mae_c": conv["mae"],
        "lst_rmse_c": conv["rmse"],
        "lst_bias_c": conv["bias"],
        "lst_r2": conv["r2"],
        "lst_valid_pixels": conv["n"],
        "pers_mae_c": pers["mae"],
        "pers_rmse_c": pers["rmse"],
        "hotspot_precision_pw": pw["precision"],
        "hotspot_recall_pw": pw["recall"],
        "hotspot_iou_pw": pw["iou"],
        "hotspot_dice_pw": pw["dice"],
        "hotspot_precision_monthavg": ma["precision"],
        "hotspot_recall_monthavg": ma["recall"],
        "hotspot_iou_monthavg": ma["iou"],
        "hotspot_dice_monthavg": ma["dice"],
        "rank_corr_urban_lst": ma["rank_corr_urban_lst"],
        "obs_hot_mae_c": inten["obs_hot_mae"],
        "obs_hot_rmse_c": inten["obs_hot_rmse"],
        "obs_hot_bias_c": inten["obs_hot_bias"],
        "obs_urban_p90_c": inten["obs_urban_p90"],
        "pred_urban_p90_c": inten["pred_urban_p90"],
        "obs_urban_p95_c": inten["obs_urban_p95"],
        "pred_urban_p95_c": inten["pred_urban_p95"],
        "p90_gap_c": inten["p90_gap"],
        "p95_gap_c": inten["p95_gap"],
        "uhi_mae_c": u["uhi_mae_c"],
        "uhi_rmse_c": u["uhi_rmse_c"],
        "uhi_bias_c": u["uhi_bias_c"],
        "uhi_valid_pixels": u["uhi_valid_pixels"],
        "uhi_hotspot_iou_monthavg": uh["iou"],
        "uhi_hotspot_recall_monthavg": uh["recall"],
        "urban_rural_contrast_pred_c": u["urban_rural_contrast_pred_c_overall"],
        "urban_rural_contrast_obs_c": u["urban_rural_contrast_obs_c_overall"],
        "contrast_abs_err_c": u["urban_rural_contrast_abs_err_c"],
        "checkpoint_sha256": d["checkpoint_sha256"],
        "checkpoint_epoch": d["checkpoint_epoch"],
        "n_months": d["n_months"],
        "n_skipped_months": d["n_skipped_months"],
    }


def _per_month(d: dict) -> dict:
    out = {}
    for r in d["per_month"]:
        lm = r["lst_model"]
        lh = r["lst_hotspot"]
        uh = r["uhi_hotspot"]
        out[r["target_month"]] = {
            "n_samples": r["n_samples"],
            "n_valid_pixels": r["n_valid_pixels_common"],
            "n_urban_valid": r["n_urban_valid"],
            "lst_mae_c": lm["mae"],
            "lst_rmse_c": lm["rmse"],
            "lst_bias_c": lm["bias"],
            "hotspot_precision": lh["precision"],
            "hotspot_recall": lh["recall"],
            "hotspot_iou": lh["iou"],
            "rank_corr_urban_lst": r["rank_corr_urban_lst"],
            "obs_hot_mae_c": lh["obs_hot_mae"],
            "obs_hot_bias_c": lh["obs_hot_bias"],
            "p95_gap_c": lh["obs_p95"] - lh["pred_p95"],
            "uhi_mae_c": r["uhi_mae_c"],
            "uhi_hotspot_iou": uh["iou"],
            "uhi_hotspot_recall": uh["recall"],
        }
    return out


def _best_val(ckpt_path: Path):
    """Read the checkpoint's saved best validation masked-MAE (normalized)."""
    try:
        import torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ck.get("best_val")
    except Exception:
        return None


def main() -> int:
    b = _load(BASELINE)
    h = _load(HOTSPOT)
    bo = _flat_overall(b)
    ho = _flat_overall(h)
    b_best = _best_val(Path("C:/uhi-model") / b["checkpoint"])
    h_best = _best_val(Path("C:/uhi-model") / h["checkpoint"])

    compare_keys = [k for k in bo if k not in ("checkpoint_sha256", "checkpoint_epoch",
                                                "n_months", "n_skipped_months",
                                                "lst_valid_pixels", "uhi_valid_pixels")]
    deltas = {k: _delta(bo[k], ho[k]) for k in compare_keys}
    pct = {k: (None if bo[k] in (None, 0) or deltas[k] is None
               else 100.0 * deltas[k] / abs(bo[k])) for k in compare_keys}

    bm = _per_month(b)
    hm = _per_month(h)
    months = sorted(set(bm) | set(hm))
    per_month = {}
    for m in months:
        row = {"baseline": bm.get(m), "hotspot_w5": hm.get(m)}
        if bm.get(m) and hm.get(m):
            row["delta"] = {k: _delta(bm[m][k], hm[m][k]) for k in bm[m]
                            if k not in ("n_samples", "n_valid_pixels", "n_urban_valid")}
        per_month[m] = row

    # month-averaged deltas (equal weight per month, matching the evaluator's
    # month-averaged aggregation)
    def _mavg(store, key):
        vals = [store[m][key] for m in months
                if store.get(m) and store[m].get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def _davg(key):
        vals = [per_month[m]["delta"][key] for m in months
                if per_month[m].get("delta") and per_month[m]["delta"].get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    report = {
        "protocol": {
            "evaluator": "evaluate_hotspots.py",
            "config": "configs/baseline.yaml",
            "split": "val",
            "test_split_used": False,
            "hotspot_quantile": b["hotspot_quantile"],
            "hotspot_definition": (
                "Top 10% of valid urban pixels: pixel-level 90th percentile of "
                "observed (resp. predicted) LST over the shared valid urban "
                "population; observed and predicted thresholds set independently."),
            "population": (
                "Combined population = target-valid AND latest-input-valid; "
                "overlap uses shared valid urban pixels."),
            "batch_size": 8,
            "units": {
                "lst_mae_c/lst_rmse_c/lst_bias_c/obs_hot_*_c/uhi_*_c/p*_c": "Celsius",
                "precision/recall/iou/dice/rank_corr_urban_lst": "dimensionless",
            },
            "note": (
                "Evaluator does not record wall-clock runtime or emit warnings; "
                "no runtime/warnings fields are available to report."),
        },
        "models": {
            "baseline_full": {
                "checkpoint": b["checkpoint"],
                "checkpoint_sha256": b["checkpoint_sha256"],
                "checkpoint_epoch_0indexed": b["checkpoint_epoch"],
                "best_val_mae_normalized": b_best,
                "overall": bo,
            },
            "hotspot_w5": {
                "checkpoint": h["checkpoint"],
                "checkpoint_sha256": h["checkpoint_sha256"],
                "checkpoint_epoch_0indexed": h["checkpoint_epoch"],
                "best_val_mae_normalized": h_best,
                "overall": ho,
            },
        },
        "delta_hotspot_minus_baseline": deltas,
        "percent_change_vs_baseline": pct,
        "month_averaged_deltas": {
            "lst_mae_c": _davg("lst_mae_c"),
            "lst_rmse_c": _davg("lst_rmse_c"),
            "hotspot_iou": _davg("hotspot_iou"),
            "hotspot_recall": _davg("hotspot_recall"),
            "hotspot_precision": _davg("hotspot_precision"),
            "obs_hot_mae_c": _davg("obs_hot_mae_c"),
            "obs_hot_bias_c": _davg("obs_hot_bias_c"),
            "p95_gap_c": _davg("p95_gap_c"),
            "uhi_mae_c": _davg("uhi_mae_c"),
        },
        "per_month": per_month,
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # --- flat CSV: one row per metric, both models + deltas ---
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "units", "baseline_full", "hotspot_w5",
                    "delta_hotspot_minus_baseline", "pct_change"])
        import re
        for k in compare_keys:
            unit = "celsius" if re.search(r"(_c$|_r2$)|_c_", k) or k.endswith("_c") else "dimensionless"
            if k in ("lst_r2",):
                unit = "dimensionless"
            if re.search(r"(mae|rmse|bias|_c$|contrast)", k):
                unit = "celsius"
            if re.search(r"(precision|recall|iou|dice|rank_corr)", k):
                unit = "dimensionless"
            w.writerow([k, unit, bo[k], ho[k], deltas[k], pct[k]])
        w.writerow([])
        w.writerow(["month_averaged_delta", "", "", "", "", ""])
        for k, v in report["month_averaged_deltas"].items():
            w.writerow([k, "celsius/dimensionless", "", "", v, ""])

    # --- per-month CSV: baseline vs hotspot by month ---
    mfields = ["month", "lst_mae_c_base", "lst_mae_c_hot", "lst_mae_c_delta",
               "lst_rmse_c_base", "lst_rmse_c_hot",
               "hotspot_iou_base", "hotspot_iou_hot", "hotspot_iou_delta",
               "hotspot_recall_base", "hotspot_recall_hot", "hotspot_recall_delta",
               "obs_hot_mae_c_base", "obs_hot_mae_c_hot", "obs_hot_mae_c_delta",
               "obs_hot_bias_c_base", "obs_hot_bias_c_hot", "obs_hot_bias_c_delta",
               "p95_gap_c_base", "p95_gap_c_hot", "p95_gap_c_delta",
               "uhi_mae_c_base", "uhi_mae_c_hot", "uhi_mae_c_delta",
               "rank_corr_base", "rank_corr_hot",
               "n_samples", "n_urban_valid"]
    with OUT_CSV_MONTH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mfields)
        w.writeheader()
        for m in months:
            bmm = bm.get(m) or {}
            hmm = hm.get(m) or {}
            w.writerow({
                "month": m,
                "lst_mae_c_base": bmm.get("lst_mae_c"),
                "lst_mae_c_hot": hmm.get("lst_mae_c"),
                "lst_mae_c_delta": _delta(bmm.get("lst_mae_c"), hmm.get("lst_mae_c")),
                "lst_rmse_c_base": bmm.get("lst_rmse_c"),
                "lst_rmse_c_hot": hmm.get("lst_rmse_c"),
                "hotspot_iou_base": bmm.get("hotspot_iou"),
                "hotspot_iou_hot": hmm.get("hotspot_iou"),
                "hotspot_iou_delta": _delta(bmm.get("hotspot_iou"), hmm.get("hotspot_iou")),
                "hotspot_recall_base": bmm.get("hotspot_recall"),
                "hotspot_recall_hot": hmm.get("hotspot_recall"),
                "hotspot_recall_delta": _delta(bmm.get("hotspot_recall"), hmm.get("hotspot_recall")),
                "obs_hot_mae_c_base": bmm.get("obs_hot_mae_c"),
                "obs_hot_mae_c_hot": hmm.get("obs_hot_mae_c"),
                "obs_hot_mae_c_delta": _delta(bmm.get("obs_hot_mae_c"), hmm.get("obs_hot_mae_c")),
                "obs_hot_bias_c_base": bmm.get("obs_hot_bias_c"),
                "obs_hot_bias_c_hot": hmm.get("obs_hot_bias_c"),
                "obs_hot_bias_c_delta": _delta(bmm.get("obs_hot_bias_c"), hmm.get("obs_hot_bias_c")),
                "p95_gap_c_base": bmm.get("p95_gap_c"),
                "p95_gap_c_hot": hmm.get("p95_gap_c"),
                "p95_gap_c_delta": _delta(bmm.get("p95_gap_c"), hmm.get("p95_gap_c")),
                "uhi_mae_c_base": bmm.get("uhi_mae_c"),
                "uhi_mae_c_hot": hmm.get("uhi_mae_c"),
                "uhi_mae_c_delta": _delta(bmm.get("uhi_mae_c"), hmm.get("uhi_mae_c")),
                "rank_corr_base": bmm.get("rank_corr_urban_lst"),
                "rank_corr_hot": hmm.get("rank_corr_urban_lst"),
                "n_samples": bmm.get("n_samples") or hmm.get("n_samples"),
                "n_urban_valid": bmm.get("n_urban_valid") or hmm.get("n_urban_valid"),
            })

    print("wrote", OUT_JSON)
    print("wrote", OUT_CSV)
    print("wrote", OUT_CSV_MONTH)
    print("\nKey deltas (hotspot_w5 - baseline_full):")
    for k in ("lst_mae_c", "lst_rmse_c", "hotspot_iou_monthavg", "hotspot_recall_monthavg",
              "obs_hot_mae_c", "obs_hot_bias_c", "p95_gap_c", "uhi_mae_c"):
        print(f"  {k:32s} {deltas[k]:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())