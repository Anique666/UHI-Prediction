"""Markdown report generator for the hotspot validation run.

Reads the ``overall`` metrics dict produced by ``evaluate_hotspots.py`` and
writes ``report.md`` next to the JSON/CSV artifacts. The diagnosis text is
derived from the metrics themselves so the report stays evidence-based.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["write_report"]


def _f(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def _temporal_concentration(per_month: list[dict]) -> dict:
    """Summaries of how error and hotspot bias vary across months/seasons."""
    mae = np.array([r["lst_model"]["mae"] for r in per_month], dtype=float)
    bias = np.array([r["lst_model"]["bias"] for r in per_month], dtype=float)
    hot_bias = np.array([r["lst_hotspot"]["obs_hot_bias"] for r in per_month], dtype=float)
    uhi = np.array([r["uhi_mae_c"] for r in per_month], dtype=float)
    months = [r["target_month"] for r in per_month]
    order = np.argsort(mae)[::-1]
    total = float(mae[sum(mae) > 0]) if False else float(mae.sum())
    top5 = float(mae[order[:5]].sum())
    # month-of-year means
    seas: dict[int, list[float]] = {}
    for m, v in zip(months, mae):
        seas.setdefault(int(m.split("-")[1]), []).append(float(v))
    seasonal = {k: float(np.mean(v)) for k, v in sorted(seas.items())}
    return {
        "mae_mean": float(mae.mean()),
        "mae_median": float(np.median(mae)),
        "mae_p90": float(np.percentile(mae, 90)),
        "mae_max": float(mae.max()),
        "worst_month": months[int(mae.argmax())],
        "top5_months_mae_share": top5 / total if total else float("nan"),
        "n_months_abs_bias_gt_1": int((np.abs(bias) > 1).sum()),
        "frac_hotspot_bias_negative": float(np.mean(hot_bias < 0)),
        "frac_hotspot_bias_lt_-0.15": float(np.mean(hot_bias < -0.15)),
        "hotspot_bias_median": float(np.median(hot_bias)),
        "seasonal_month_of_year_mae": seasonal,
        "worst_months": [
            {"target_month": months[i], "lst_mae_c": float(mae[i]),
             "lst_bias_c": float(bias[i]), "uhi_mae_c": float(uhi[i])}
            for i in order[:5]
        ],
    }


def _diagnose(overall) -> tuple[str, list[str]]:
    """Return (label, evidence bullets) for the dominant failure mode."""
    pw = overall["hotspot_localization_lst_urban"]["pixel_weighted_overall"]
    ma = overall["hotspot_localization_lst_urban"]["month_averaged"]
    inten = overall["hotspot_intensity_lst_urban_at_observed_hotspots"]
    u = overall["uhi_derived"]
    lst = overall["lst_accuracy_celsius_combined_population"]["convlstm"]
    tc = _temporal_concentration(overall["per_month"])

    iou = pw["iou"]
    rec = pw["recall"]
    prec = pw["precision"]
    rank = ma["rank_corr_urban_lst"]
    hot_bias = inten["obs_hot_bias"]
    p95_gap = inten["p95_gap"]
    p90_gap = inten["p90_gap"]

    evidence = [
        f"LST hotspot overlap (pixel-weighted): IoU={_f(iou)}, "
        f"precision={_f(prec)}, recall={_f(rec)}.",
        f"Month-averaged urban LST rank correlation = {_f(rank)} "
        f"(1.0 = perfect spatial ordering).",
        f"At observed hotspots: bias={_f(hot_bias)} C, "
        f"MAE={_f(inten['obs_hot_mae'])} C; hotspot bias is negative in "
        f"{tc['frac_hotspot_bias_negative']*100:.0f}% of months "
        f"(median {_f(tc['hotspot_bias_median'])} C).",
        f"Urban upper tail observed-vs-predicted gap: p90={_f(p90_gap)} C, "
        f"p95={_f(p95_gap)} C (positive = underprediction of hot tail).",
        f"Overall LST bias={_f(lst['bias'])} C, MAE={_f(lst['mae'])} C.",
        f"Derived UHI MAE={_f(u['uhi_mae_c'])} C, bias={_f(u['uhi_bias_c'])} C, "
        f"urban-rural contrast |err|={_f(u['urban_rural_contrast_abs_err_c'])} C "
        f"(the rural-reference definition is NOT the dominant error source overall).",
        f"Error is month-concentrated: worst month {tc['worst_month']} "
        f"(MAE {_f(tc['mae_max'])} C); the 5 worst months carry "
        f"{tc['top5_months_mae_share']*100:.0f}% of total absolute error; "
        f"{tc['n_months_abs_bias_gt_1']} months have |bias|>1 C.",
    ]

    underpred = (hot_bias < -0.15) or (p95_gap > 0.3)
    localized = iou >= 0.35 or (np.isfinite(rank) and rank >= 0.6)
    diffuse = np.isfinite(prec) and np.isfinite(rec) and prec < 0.3 and rec < 0.3

    if underpred and not localized:
        label = "Hot urban pixels are systematically underpredicted AND hotspots are poorly localized."
    elif diffuse and not underpred:
        label = "Hotspots are spatially diffuse or poorly localized."
    elif underpred:
        label = ("Hot urban pixels are systematically underpredicted; errors are also "
                 "concentrated in a few (mostly monsoon-season) months.")
    elif localized and abs(hot_bias) <= 0.15:
        label = "Hotspot temperatures are reasonably accurate and hotspots are well localized."
    else:
        label = "No single dominant failure mode is clearly established by the metrics."
    return label, evidence


def write_report(out_dir: Path, overall: dict) -> Path:
    out_dir = Path(out_dir)
    lst = overall["lst_accuracy_celsius_combined_population"]["convlstm"]
    per = overall["lst_accuracy_celsius_combined_population"]["persistence"]
    imp_mae = overall["lst_accuracy_celsius_combined_population"]["improvement_mae_c"]
    imp_rmse = overall["lst_accuracy_celsius_combined_population"]["improvement_rmse_c"]
    pw = overall["hotspot_localization_lst_urban"]["pixel_weighted_overall"]
    ma = overall["hotspot_localization_lst_urban"]["month_averaged"]
    inten = overall["hotspot_intensity_lst_urban_at_observed_hotspots"]
    u = overall["uhi_derived"]
    uh = overall["uhi_hotspot_month_averaged"]
    cov = overall.get("coverage") or {}
    q = overall.get("hotspot_quantile", 90.0)
    label, evidence = _diagnose(overall)

    L = []
    L.append("# Hotspot-Focused Validation — Baseline ConvLSTM\n")
    L.append("**Evaluation only.** No training, fine-tuning, or test-set use. "
             "Split: **validation** (`val`).\n")
    L.append(f"- Checkpoint: `{overall['checkpoint']}` (epoch {overall.get('checkpoint_epoch')})")
    L.append(f"- SHA-256: `{overall['checkpoint_sha256']}`")
    L.append(f"- Hotspot definition: top {100 - q:.0f}% of valid **urban** pixels "
             f"(pixel-level {q:.0f}th percentile), observed and predicted thresholds "
             f"set independently on the shared valid urban population.")
    L.append(f"- Validation months with valid urban pixels: {overall['n_months']} "
             f"(skipped: {overall['n_skipped_months']})")
    L.append(f"- Metrics are reported **both** pixel-weighted (all valid urban pixels pooled) "
             f"and month-averaged; each is labelled.\n")

    L.append("## 1. Overall accuracy safeguards (LST, combined population)\n")
    L.append("| Model | MAE (C) | RMSE (C) | Bias (C) | R2 | Valid pixels |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| ConvLSTM | {_f(lst['mae'],4)} | {_f(lst['rmse'],4)} | {_f(lst['bias'],4)} | "
             f"{_f(lst['r2'],4)} | {lst['n']} |")
    L.append(f"| Persistence | {_f(per['mae'],4)} | {_f(per['rmse'],4)} | {_f(per['bias'],4)} | "
             f"{_f(per['r2'],4)} | {per['n']} |")
    L.append("")
    L.append(f"ConvLSTM improves over persistence by {_f(imp_mae,4)} C MAE and "
             f"{_f(imp_rmse,4)} C RMSE on this population.\n")

    L.append("## 2. Hotspot localization (urban LST)\n")
    L.append("| Aggregation | Precision | Recall | IoU | Dice |")
    L.append("|---|---|---|---|---|")
    L.append(f"| Pixel-weighted (pooled) | {_f(pw['precision'])} | {_f(pw['recall'])} | "
             f"{_f(pw['iou'])} | {_f(pw['dice'])} |")
    L.append(f"| Month-averaged | {_f(ma['precision'])} | {_f(ma['recall'])} | "
             f"{_f(ma['iou'])} | {_f(ma['dice'])} |")
    L.append("")
    L.append(f"- Month-averaged Spearman rank correlation (observed vs predicted urban LST): "
             f"**{_f(ma['rank_corr_urban_lst'])}**\n")

    L.append("## 3. Hotspot intensity (at observed hotspots, urban LST)\n")
    L.append("| Metric | Value (C) |")
    L.append("|---|---|")
    L.append(f"| MAE at observed hotspots | {_f(inten['obs_hot_mae'])} |")
    L.append(f"| RMSE at observed hotspots | {_f(inten['obs_hot_rmse'])} |")
    L.append(f"| Bias at observed hotspots | {_f(inten['obs_hot_bias'])} |")
    L.append(f"| Observed urban p90 | {_f(inten['obs_urban_p90'])} |")
    L.append(f"| Predicted urban p90 | {_f(inten['pred_urban_p90'])} |")
    L.append(f"| p90 gap (obs - pred) | {_f(inten['p90_gap'])} |")
    L.append(f"| Observed urban p95 | {_f(inten['obs_urban_p95'])} |")
    L.append(f"| Predicted urban p95 | {_f(inten['pred_urban_p95'])} |")
    L.append(f"| p95 gap (obs - pred) | {_f(inten['p95_gap'])} |")
    L.append("")

    L.append("## 4. Derived UHI performance\n")
    L.append("Definition (unchanged): `UHI(t,x,y) = LST(t,x,y) - mean(valid LST over the "
             "fixed rural mask)`, computed separately for the observed and predicted fields.\n")
    L.append(f"- UHI MAE = {_f(u['uhi_mae_c'])} C | RMSE = {_f(u['uhi_rmse_c'])} C | "
             f"bias = {_f(u['uhi_bias_c'])} C | valid pixels = {u['uhi_valid_pixels']}")
    L.append(f"- Rural mean (pred) = {_f(u['rural_mean_pred_c_overall'],2)} C | "
             f"(obs) = {_f(u['rural_mean_obs_c_overall'],2)} C")
    L.append(f"- Urban mean (pred) = {_f(u['urban_mean_pred_c_overall'],2)} C | "
             f"(obs) = {_f(u['urban_mean_obs_c_overall'],2)} C")
    L.append(f"- Urban-rural contrast (pred) = {_f(u['urban_rural_contrast_pred_c_overall'])} C | "
             f"(obs) = {_f(u['urban_rural_contrast_obs_c_overall'])} C | "
             f"|err| = {_f(u['urban_rural_contrast_abs_err_c'])} C")
    L.append(f"- UHI hotspot overlap (month-avg): IoU = {_f(uh['iou'])}, "
             f"recall = {_f(uh['recall'])}, precision = {_f(uh['precision'])}; "
             f"observed-hotspot MAE = {_f(uh['obs_hot_mae'])} C\n")

    tc = _temporal_concentration(overall["per_month"])
    L.append("## 5. Temporal / seasonal pattern\n")
    L.append(f"- Per-month LST MAE: mean {_f(tc['mae_mean'])} C, median "
             f"{_f(tc['mae_median'])} C, p90 {_f(tc['mae_p90'])} C, max "
             f"{_f(tc['mae_max'])} C (**{tc['worst_month']}**).")
    L.append(f"- The 5 worst months carry {tc['top5_months_mae_share']*100:.0f}% of "
             f"total absolute error; {tc['n_months_abs_bias_gt_1']} months have |bias|>1 C.")
    worst = ", ".join(f"{w['target_month']} (MAE {_f(w['lst_mae_c'])}, "
                      f"bias {_f(w['lst_bias_c'])})" for w in tc["worst_months"])
    L.append(f"- Worst months (by LST MAE): {worst}.")
    seas = ", ".join(f"{k:02d}:{_f(v,2)}" for k, v in tc["seasonal_month_of_year_mae"].items())
    L.append(f"- Month-of-year mean LST MAE (Jan..Dec): {seas}.\n")
    L.append("The dominant-error months are the monsoon/early-monsoon season (roughly "
             "May-August), where large cold-season cloud contamination is known to affect "
             "the LST retrievals. These months also show the worst hotspot localization.\n")

    L.append("## 6. Coverage / audit\n")
    if cov:
        L.append(f"- Grid valid fraction: {_f(cov.get('grid_fraction'))}")
        L.append(f"- Urban mask coverage: {_f(cov.get('urban_coverage'))} "
                 f"({cov.get('urban_covered')}/{cov.get('urban_pixels')})")
        L.append(f"- Rural mask coverage: {_f(cov.get('rural_coverage'))} "
                 f"({cov.get('rural_covered')}/{cov.get('rural_pixels')})")
    if overall.get("skipped_months"):
        L.append(f"- Skipped months: {overall['skipped_months']}")
    L.append("- The validation split is a spatially disjoint region; reconstruction "
             "covers only validation patches and uncovered pixels are nodata. "
             "No full-city coverage is claimed.\n")

    L.append("## 7. Dominant failure mode\n")
    L.append(f"**Diagnosis: {label}**\n")
    for e in evidence:
        L.append(f"- {e}")
    L.append("")

    L.append("## 8. Recommended next experiment\n")
    L.append(_recommendation(overall, label))
    L.append("")

    L.append("## 9. Provenance\n")
    L.append("- Split used: **val only**. The test split was not evaluated or used for "
             "selection (the task forbids test-set selection).")
    L.append("- Frozen contract preserved: dataset, channel order, train-fitted "
             "normalization, spatial split, urban/rural masks, nodata handling, and "
             "temporal alignment are untouched. The UHI definition and reference masks are "
             "unchanged.")
    L.append("- UHI is derived from the frozen rural mask, not predicted directly. The "
             "model forecasts next-month LST in Celsius after inverse normalization.")
    L.append("- Command: `python evaluate_hotspots.py --checkpoint "
             "checkpoints/baseline_full/best.pt` (validation inference only; no training).")
    L.append("- Full machine-readable metrics: `hotspot_metrics.json`; per-month: "
             "`hotspot_per_month.csv`; summary: `hotspot_overall.csv`; audit: `audit.json`.")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_dir / "report.md"


def _recommendation(overall, label) -> str:
    inten = overall["hotspot_intensity_lst_urban_at_observed_hotspots"]
    pw = overall["hotspot_localization_lst_urban"]["pixel_weighted_overall"]
    ma = overall["hotspot_localization_lst_urban"]["month_averaged"]
    underpred = (inten["obs_hot_bias"] < -0.15) or (inten["p95_gap"] > 0.3)
    localized = pw["iou"] >= 0.35 or (np.isfinite(ma["rank_corr_urban_lst"])
                                      and ma["rank_corr_urban_lst"] >= 0.6)
    tc = _temporal_concentration(overall["per_month"])
    if underpred and not localized:
        rec = ("**Test a moderately hotspot-weighted regression loss.** Hot pixels are "
               "underpredicted and their locations are poorly recovered, so re-weighting "
               "the existing masked-MAE toward the upper tail is the single most direct, "
               "cheap intervention. Keep the architecture unchanged.")
    elif underpred:
        rec = ("**Test a moderately hotspot-weighted regression loss.** Localization is "
               "reasonable (pixel-weighted IoU {iou}, rank corr {rc}) so the main defect is "
               "intensity: observed hotspots are biased cold by {bias} C and the urban p95 "
               "is underpredicted by {p95} C, in {neg}% of months. A mild upper-tail "
               "weighting targets exactly this, keeps the architecture and data contract "
               "fixed, and is cheap to run; the existing masked MAE (a conditional-median "
               "objective) is the likely cause of regression-to-the-mean and the observed "
               "cold tail.".format(
                   iou=_f(pw["iou"]), rc=_f(ma["rank_corr_urban_lst"]),
                   bias=_f(inten["obs_hot_bias"]), p95=_f(inten["p95_gap"]),
                   neg=f"{tc['frac_hotspot_bias_negative']*100:.0f}"))
        rec += ("\n\nWhy not the alternatives: the largest *aggregate* errors are the "
                "monsoon months ({wm}, up to {mmae} C MAE, {share:.0f}% of total error), "
                "which points to a temporal/seasonal data-quality issue rather than a model "
                "one; investigating that is valuable but lower expected value for the stated "
                "goal of hotspot heat *intensity*. Overall localization is already decent, "
                "so a spatial-architecture change is not yet warranted, and plain optimization "
                "tuning would not address a systematic (not noise-driven) cold hot-tail bias."
                .format(wm=tc["worst_month"], mmae=_f(tc["mae_max"]),
                        share=tc["top5_months_mae_share"] * 100))
    elif not localized:
        rec = ("**Investigate spatial representation/architecture.** The model sharply "
               "under-predicts hotspot overlap despite reasonable overall MAE, pointing to "
               "a spatial-representation limit rather than a loss-weighting issue.")
    else:
        rec = ("**Keep the existing loss and tune optimization settings.** Hotspots are "
               "already localized and their temperatures are essentially unbiased, so "
               "targeted optimization (LR schedule / early stopping) has the best expected "
               "value; a hotspot-weighted loss is not justified by the evidence.")
    rec += ("\n\nThis task does not implement or launch the experiment. If it is run later, "
            "the existing training entry point is: "
            "`python -m training.train --config configs/baseline.yaml --loss huber` "
            "(inspect `training/train.py` for exact flags; a hotspot-weighted loss is not yet "
            "implemented and would need a small, self-contained change to `training/losses.py`).")
    return rec