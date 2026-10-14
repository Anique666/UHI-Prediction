"""Final held-out evaluation for the UHI ConvLSTM baseline.

Reports metrics in degrees Celsius for:

* the trained ConvLSTM, and
* the persistence baseline,

on the **test split only**, using streamed batches so the full dataset is never
materialized. Metrics are aggregated over valid pixels (error sums divided by
valid-pixel counts), never as per-batch averages.

Fair comparison
---------------
The frozen dataset has a target-validity mask but no latest-input validity. A
persistence prediction is only meaningful where the latest input LST was
genuine, so we construct a ``combined`` population = target-valid AND
latest-input-valid, and evaluate **both** models on it. The ConvLSTM is also
reported on the full target-valid population. Exclusions are documented.

The test split is never used for selection; this script is evaluation only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.convlstm import ConvLSTM  # noqa: E402
from training.baselines import last_frame_lst_normalized  # noqa: E402
from training.engine import load_checkpoint  # noqa: E402
from training.metrics import MetricAccumulator  # noqa: E402
from utils.config import get_paths, load_config  # noqa: E402
from utils.data import make_dataset  # noqa: E402
from utils.normalization import load_normalization_stats  # noqa: E402
from utils.rasters import MonthRasterReader  # noqa: E402
from utils.runmeta import collect_environment, write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate ConvLSTM vs persistence")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    ap.add_argument("--checkpoint", required=True, help="Path to best.pt")
    ap.add_argument("--run-id", default=None, help="Evaluation run id")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    return ap.parse_args(argv)


@torch.no_grad()
def _evaluate_main(
    model,
    dataset,
    stats,
    reader,
    lst_band_index,
    device,
    batch_size,
    max_samples,
    out_dir,
    save_predictions_month=None,
):
    """Stream the test set; accumulate Celsius metrics for both models."""
    mean = stats.means[stats.index("LST_Celsius")]
    std = stats.stds[stats.index("LST_Celsius")]

    model_mae = MetricAccumulator()  # model on target-valid population
    model_comb = MetricAccumulator()  # model on combined population
    pers_comb = MetricAccumulator()  # persistence on combined population

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    model.eval()
    buf_X, buf_y, buf_m, buf_recs = [], [], [], []
    excluded = 0
    samples_with_excl = 0
    saved = 0

    def flush():
        nonlocal buf_X, buf_y, buf_m, buf_recs
        if not buf_X:
            return
        X = torch.stack(buf_X).to(device)
        y = torch.stack(buf_y).to(device)
        m = torch.stack(buf_m).to(device)
        pred = model(X)
        # Build combined (target-valid AND latest-input-valid) mask.
        comb = m.clone()
        for i, rec in enumerate(buf_recs):
            r0, c0 = int(rec["row_off"]), int(rec["col_off"])
            s = int(rec["height"])
            window = ((r0, r0 + s), (c0, c0 + s))
            latest = rec["input_months"][-1]
            iv = torch.from_numpy(
                reader.read_validity(latest, lst_band_index, window)
            ).to(device).unsqueeze(0)
            comb[i] = m[i] & iv
        pred_c = pred * std + mean
        tgt_c = y * std + mean
        model_mae.update(pred_c, tgt_c, m)
        model_comb.update(pred_c, tgt_c, comb)
        pers_norm = last_frame_lst_normalized(X)  # [B,1,H,W] normalized
        pers_c = pers_norm * std + mean
        pers_comb.update(pers_c, tgt_c, comb)

        nonlocal excluded, samples_with_excl, saved
        for i, rec in enumerate(buf_recs):
            d = int((m[i] & ~comb[i]).sum())
            excluded += d
            if d > 0:
                samples_with_excl += 1
        del X, y, m, pred, pred_c, tgt_c, pers_norm, pers_c, comb

    for idx in range(n):
        X, y, m = dataset[idx]
        buf_X.append(X)
        buf_y.append(y)
        buf_m.append(m)
        buf_recs.append(dataset.records[idx])
        if len(buf_X) >= batch_size:
            flush()
            buf_X, buf_y, buf_m, buf_recs = [], [], [], []
    flush()

    result = {
        "n_samples": n,
        "populations": {
            "target_valid_pixels": model_mae.count,
            "combined_valid_pixels": model_comb.count,
            "input_invalid_excluded_pixels": excluded,
            "samples_with_input_invalid_pixels": samples_with_excl,
        },
        "convlstm_target_valid_population": model_mae.as_dict(),
        "convlstm_combined_population": model_comb.as_dict(),
        "persistence_combined_population": pers_comb.as_dict(),
    }
    write_json(out_dir / "test_metrics.json", result)
    _write_comparison_csv(out_dir / "comparison.csv", result)
    return result


def _write_comparison_csv(path, result):
    rows = [
        ("model", "convlstm", "target_valid", result["convlstm_target_valid_population"]),
        ("model", "convlstm", "combined", result["convlstm_combined_population"]),
        ("model", "persistence", "combined", result["persistence_combined_population"]),
    ]
    header = "kind,model,population,mae_c,rmse_c,bias_c,r2,valid_pixels"
    lines = [header]
    for kind, name, pop, m in rows:
        lines.append(
            f"{kind},{name},{pop},{m['mae']:.6f},{m['rmse']:.6f},"
            f"{m['bias']:.6f},{m['r2']:.6f},{m['valid_pixels']}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = args.run_id or "eval_" + Path(args.checkpoint).parent.name
    out_dir = paths["evaluation_dir"] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_normalization_stats(paths["dataset_root"])
    model = ConvLSTM(
        input_channels=cfg["data"]["input_channels"],
        hidden_channels=cfg["model"]["hidden_channels"],
        kernel_size=cfg["model"]["kernel_size"],
        output_channels=cfg["model"]["output_channels"],
    ).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch')}, "
          f"best_val={ckpt.get('best_val')})")

    test_ds = make_dataset(cfg, "test")
    reader = MonthRasterReader(paths["dataset_root"], paths["temporal_root"])
    print(f"Evaluating on test split: {len(test_ds)} samples")
    try:
        result = _evaluate_main(
            model,
            test_ds,
            stats,
            reader,
            lst_band_index=cfg["data"]["lst_channel_index"] + 1,
            device=device,
            batch_size=int(args.batch_size or cfg["evaluation"]["batch_size"]),
            max_samples=args.max_samples,
            out_dir=out_dir,
        )
    finally:
        test_ds.close()
        reader.close()

    write_json(out_dir / "environment.json", collect_environment())
    write_json(out_dir / "eval_config.json", {"config": cfg, "checkpoint": args.checkpoint})

    mc = result["convlstm_target_valid_population"]
    cc = result["convlstm_combined_population"]
    pc = result["persistence_combined_population"]
    print("\n=== Test metrics (degrees Celsius) ===")
    print(f"ConvLSTM   [target-valid] : MAE={mc['mae']:.4f} RMSE={mc['rmse']:.4f} "
          f"bias={mc['bias']:.4f} R2={mc['r2']:.4f} n={mc['valid_pixels']}")
    print(f"ConvLSTM   [combined]     : MAE={cc['mae']:.4f} RMSE={cc['rmse']:.4f} "
          f"bias={cc['bias']:.4f} R2={cc['r2']:.4f} n={cc['valid_pixels']}")
    print(f"Persistence[combined]     : MAE={pc['mae']:.4f} RMSE={pc['rmse']:.4f} "
          f"bias={pc['bias']:.4f} R2={pc['r2']:.4f} n={pc['valid_pixels']}")
    print(f"Excluded input-invalid pixels: {result['populations']['input_invalid_excluded_pixels']}")
    print(f"Artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())