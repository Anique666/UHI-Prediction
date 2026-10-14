"""Training CLI for the UHI ConvLSTM baseline.

Usage (from C:\\uhi-model with the .venv active):

    python -m training.train --config configs/baseline.yaml
    python -m training.train --config configs/baseline.yaml --epochs 3 --max-train-samples 40

The test split is never loaded here. Model selection uses the validation split
only. Checkpoints are written atomically so an interrupted run cannot corrupt a
previously saved best model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.convlstm import ConvLSTM  # noqa: E402
from models.residual import ResidualConvLSTM  # noqa: E402
from training.engine import (  # noqa: E402
    get_loss_fn,
    load_checkpoint,
    save_candidate_checkpoint,
    save_checkpoint,
    select_candidate_pool,
    train_one_epoch,
    validate,
)
from training.contrast_loss import (  # noqa: E402
    DEFAULT_CONTRAST_LAMBDA,
    UrbanRuralContrastLoss,
    UrbanRuralPatchDataset,
)
from training.hotspot_loss import (  # noqa: E402
    DEFAULT_WEIGHTING,
    HotspotWeightedMAELoss,
    UrbanPatchDataset,
)
from utils.config import get_paths, load_config  # noqa: E402
from utils.data import make_dataloader, make_dataset  # noqa: E402
from utils.normalization import load_normalization_stats  # noqa: E402
from utils.runmeta import collect_environment, make_run_id, write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def _limited_dataset(dataset, max_samples):
    """Return a Subset capped at ``max_samples`` (for pilot runs)."""
    if max_samples is None:
        return dataset
    from torch.utils.data import Subset

    n = min(int(max_samples), len(dataset))
    return Subset(dataset, list(range(n)))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train the UHI ConvLSTM baseline")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    ap.add_argument("--run-id", default=None, help="Override the generated run id")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--eval-workers", type=int, default=None,
                    help="DataLoader workers for validation (performance only)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-val-samples", type=int, default=None)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision")
    ap.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    ap.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    # --- optimization experiment options (defaults reproduce the baseline) ---
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--scheduler", choices=["none", "plateau"], default="none")
    ap.add_argument("--scheduler-factor", type=float, default=0.5)
    ap.add_argument("--scheduler-patience", type=int, default=3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--loss", choices=["mae", "huber"], default="mae")
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--model", choices=["convlstm", "residual"], default="convlstm")
    ap.add_argument("--hidden-channels", type=int, default=None)
    ap.add_argument("--kernel-size", type=int, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    # --- hotspot-weighted loss experiment (defaults OFF = original behavior) ---
    ap.add_argument("--hotspot-loss", action="store_true",
                    help="Enable the hotspot-weighted MAE over hot urban pixels")
    ap.add_argument("--hotspot-hot-z", type=float, default=None,
                    help="Normalized LST threshold for 'hot' urban pixels (train-only z)")
    ap.add_argument("--hotspot-extreme-z", type=float, default=None,
                    help="Normalized LST threshold for 'extreme' urban pixels")
    ap.add_argument("--hotspot-w-hot", type=float, default=None,
                    help="Weight for hot urban pixels (ordinary = 1.0)")
    ap.add_argument("--hotspot-w-extreme", type=float, default=None,
                    help="Weight for extreme urban pixels (>= w_hot, capped)")
    # --- urban-rural contrast auxiliary loss (defaults OFF = original behavior) ---
    ap.add_argument("--contrast-loss", action="store_true",
                    help="Enable the optional urban-rural contrast auxiliary loss")
    ap.add_argument("--contrast-lambda", type=float, default=None,
                    help="Auxiliary weight lambda (total = masked_mae + lambda*contrast)")
    ap.add_argument("--contrast-rural-mask", default=None,
                    help="Override the rural reference mask path (default: mask_dir/rural_mask.tif)")
    return ap.parse_args(argv)


def _apply_overrides(cfg, args):
    t = cfg.setdefault("training", {})
    if args.epochs is not None:
        t["num_epochs"] = args.epochs
    if args.batch_size is not None:
        t["batch_size"] = args.batch_size
    if args.lr is not None:
        t["learning_rate"] = args.lr
    if args.num_workers is not None:
        t["num_workers"] = args.num_workers
    if args.eval_workers is not None:
        cfg.setdefault("evaluation", {})["num_workers"] = args.eval_workers
    if args.seed is not None:
        t["seed"] = args.seed
    if args.amp:
        t["amp"] = True
    if args.no_amp:
        t["amp"] = False
    if args.optimizer:
        t["optimizer"] = args.optimizer
    if args.weight_decay is not None:
        t["weight_decay"] = args.weight_decay
    if args.scheduler:
        t["scheduler"] = args.scheduler
    if args.scheduler_factor is not None:
        t["scheduler_factor"] = args.scheduler_factor
    if args.scheduler_patience is not None:
        t["scheduler_patience"] = args.scheduler_patience
    if args.min_lr is not None:
        t["min_lr"] = args.min_lr
    if args.loss:
        t["loss"] = args.loss
    if args.huber_delta is not None:
        t["huber_delta"] = args.huber_delta
    if args.grad_clip is not None:
        t["gradient_clip"] = args.grad_clip
    hs = t.setdefault("hotspot_loss", {})
    if args.hotspot_loss:
        hs["enabled"] = True
    for key, val in (
        ("hot_z", args.hotspot_hot_z),
        ("extreme_z", args.hotspot_extreme_z),
        ("w_hot", args.hotspot_w_hot),
        ("w_extreme", args.hotspot_w_extreme),
    ):
        if val is not None:
            hs[key] = val
    cl = t.setdefault("contrast_loss", {})
    if args.contrast_loss:
        cl["enabled"] = True
    if args.contrast_lambda is not None:
        cl["lambda"] = args.contrast_lambda
    if args.contrast_rural_mask is not None:
        cl["rural_mask"] = args.contrast_rural_mask
    m = cfg.setdefault("model", {})
    if args.hidden_channels is not None:
        m["hidden_channels"] = args.hidden_channels
    if args.kernel_size is not None:
        m["kernel_size"] = args.kernel_size
    if args.model:
        m["variant"] = args.model
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = _apply_overrides(load_config(args.config), args)
    paths = get_paths(cfg)

    seed = int(cfg["training"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = args.run_id or make_run_id("baseline")
    ckpt_dir = paths["checkpoints_dir"] / run_id
    out_dir = paths["outputs_dir"] / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist config snapshot + environment metadata before training starts.
    write_json(out_dir / "config_snapshot.json", cfg)
    write_json(out_dir / "environment.json", collect_environment())

    stats = load_normalization_stats(paths["dataset_root"])
    if stats.channel_names[cfg["data"]["lst_channel_index"]] != cfg["data"]["lst_channel"]:
        raise ValueError("Configured LST channel index does not match normalization stats")

    train_ds = make_dataset(cfg, "train")
    val_ds = make_dataset(cfg, "val")

    # Hotspot-weighted experiment: wrap the TRAIN dataset so each sample also
    # carries the frozen urban-mask sub-window. Validation is left unchanged so
    # model selection stays comparable to the baseline (control masked-MAE).
    hotspot_cfg = dict(DEFAULT_WEIGHTING)
    hotspot_cfg.update(cfg["training"].get("hotspot_loss", {}) or {})
    hotspot_enabled = bool(hotspot_cfg.get("enabled", False))

    contrast_cfg = {"enabled": False, "lambda": DEFAULT_CONTRAST_LAMBDA, "rural_mask": None}
    contrast_cfg.update(cfg["training"].get("contrast_loss", {}) or {})
    contrast_enabled = bool(contrast_cfg.get("enabled", False))
    if contrast_enabled and hotspot_enabled:
        raise ValueError("hotspot_loss and contrast_loss are mutually exclusive; enable at most one.")
    if contrast_enabled:
        urban_path = paths["mask_dir"] / "urban_mask.tif"
        rural_path = (Path(contrast_cfg["rural_mask"]) if contrast_cfg.get("rural_mask")
                      else paths["mask_dir"] / "rural_mask.tif")
        train_ds = UrbanRuralPatchDataset(train_ds, urban_path, rural_path)
    elif hotspot_enabled:
        train_ds = UrbanPatchDataset(train_ds, paths["mask_dir"] / "urban_mask.tif")

    train_view = _limited_dataset(train_ds, args.max_train_samples)
    val_view = _limited_dataset(val_ds, args.max_val_samples)

    tcfg = cfg["training"]
    train_loader = make_dataloader(
        train_view,
        batch_size=tcfg["batch_size"],
        shuffle=tcfg["shuffle_train"],
        num_workers=tcfg["num_workers"],
        pin_memory=tcfg["pin_memory"],
    )
    val_loader = make_dataloader(
        val_view,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=cfg["evaluation"]["num_workers"],
        pin_memory=tcfg["pin_memory"],
    )

    print(f"Run id            : {run_id}")
    print(f"Device            : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Train samples     : {len(train_view)} / {len(train_ds)}")
    print(f"Val samples       : {len(val_view)} / {len(val_ds)}")
    print(f"Batch size        : {tcfg['batch_size']}")
    print(f"Epochs            : {tcfg['num_epochs']}")
    print(f"AMP               : {tcfg['amp']}")

    mcfg = cfg["model"]
    variant = mcfg.get("variant", "convlstm")
    model_cls = ResidualConvLSTM if variant == "residual" else ConvLSTM
    model = model_cls(
        input_channels=cfg["data"]["input_channels"],
        hidden_channels=mcfg["hidden_channels"],
        kernel_size=mcfg["kernel_size"],
        output_channels=mcfg["output_channels"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model variant     : {variant}")
    print(f"Parameters        : {n_params:,}")

    wd = float(tcfg["weight_decay"])
    opt_name = tcfg.get("optimizer", "adam")
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(tcfg["learning_rate"]), weight_decay=wd
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(tcfg["learning_rate"]), weight_decay=wd
        )
    print(f"Optimizer         : {opt_name} (lr={tcfg['learning_rate']}, wd={wd})")

    scheduler = None
    if tcfg.get("scheduler", "none") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(tcfg.get("scheduler_factor", 0.5)),
            patience=int(tcfg.get("scheduler_patience", 3)),
            min_lr=float(tcfg.get("min_lr", 1e-6)),
        )
        print(f"Scheduler         : ReduceLROnPlateau factor={tcfg.get('scheduler_factor')} "
              f"patience={tcfg.get('scheduler_patience')} min_lr={tcfg.get('min_lr')}")

    loss_fn = get_loss_fn(tcfg.get("loss", "mae"), float(tcfg.get("huber_delta", 1.0)))
    if hotspot_enabled:
        if tcfg.get("loss", "mae") != "mae":
            raise ValueError(
                "hotspot_loss assumes the MAE control objective; got "
                f"loss={tcfg.get('loss')!r}. Use --loss mae."
            )
        loss_fn = HotspotWeightedMAELoss(
            hot_z=float(hotspot_cfg["hot_z"]),
            extreme_z=float(hotspot_cfg["extreme_z"]),
            w_hot=float(hotspot_cfg["w_hot"]),
            w_extreme=float(hotspot_cfg["w_extreme"]),
        )
        print(f"Loss              : hotspot-weighted MAE (hot_z={loss_fn.hot_z}, "
              f"extreme_z={loss_fn.extreme_z}, w_hot={loss_fn.w_hot}, "
              f"w_extreme={loss_fn.w_extreme})")
    elif contrast_enabled:
        if tcfg.get("loss", "mae") != "mae":
            raise ValueError(
                "contrast_loss assumes the MAE control objective; got "
                f"loss={tcfg.get('loss')!r}. Use --loss mae."
            )
        loss_fn = UrbanRuralContrastLoss(lam=float(contrast_cfg["lambda"]))
        print(f"Loss              : masked MAE + {loss_fn.lam} * urban-rural contrast "
              f"(base=masked_mae; rural mask={contrast_cfg.get('rural_mask') or 'mask_dir/rural_mask.tif'})")
    else:
        print(f"Loss              : {tcfg.get('loss', 'mae')}")

    amp = bool(tcfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if device.type == "cuda" else None

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"Resumed from      : {args.resume} (epoch {start_epoch}, best_val {best_val:.5f})")

    history = []
    best_ckpt_path = ckpt_dir / "best.pt"
    latest_ckpt_path = ckpt_dir / "latest.pt"
    # Compact candidate pool (weights only), ranked by validation masked-MAE.
    # Retains up to ``candidates`` best epochs so hotspot metrics can be compared
    # across nearby checkpoints later, without storing optimizer states.
    n_candidates = int(tcfg.get("keep_candidate_checkpoints", 3))
    candidate_pool: list[dict] = []
    if n_candidates > 0:
        print(f"Candidate ckpts  : keep top {n_candidates} by val masked-MAE "
              f"(weights only, in {ckpt_dir.name})")
    patience = int(tcfg["early_stopping_patience"])
    min_delta = float(tcfg["early_stopping_min_delta"])
    epochs_since_improve = 0

    for epoch in range(start_epoch, int(tcfg["num_epochs"])):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{tcfg['num_epochs']}  lr={lr:.6g}")
        t0 = time.perf_counter()
        train_loss, grad_norm = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=float(tcfg["gradient_clip"]),
            amp=amp,
            scaler=scaler,
            log_every=int(tcfg["log_every_n_batches"]),
            logger=print,
            loss_fn=loss_fn,
        )
        _last_components = getattr(train_one_epoch, "last_components", {}) or {}
        val = validate(model, val_loader, device, stats, in_celsius=False, amp=amp)
        if scheduler is not None:
            scheduler.step(val.mae_normalized)
        epoch_seconds = time.perf_counter() - t0
        peak_gpu = (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else 0.0
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_base_loss": _last_components.get("base"),
            "train_contrast_loss": _last_components.get("contrast"),
            "val_loss": val.loss,
            "val_mae_normalized": val.mae_normalized,
            "val_rmse_normalized": val.rmse_normalized,
            "val_mae_celsius": val.mae_celsius,
            "val_rmse_celsius": val.rmse_celsius,
            "val_valid_pixels": val.valid_pixels,
            "lr": lr,
            "grad_norm": grad_norm,
            "epoch_seconds": epoch_seconds,
            "peak_gpu_mib": peak_gpu,
        }
        history.append(row)
        print(
            f"  val: loss={val.loss:.5f} mae={val.mae_normalized:.5f} "
            f"rmse={val.rmse_normalized:.5f} mae_c={val.mae_celsius:.4f} "
            f"valid_px={val.valid_pixels} time={epoch_seconds:.1f}s peak_gpu={peak_gpu:.0f}MiB"
        )

        improved = val.mae_normalized < (best_val - min_delta)
        if improved:
            best_val = val.mae_normalized
            epochs_since_improve = 0
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                epoch,
                best_val,
                cfg,
                extra={
                    "run_id": run_id,
                    "normalization_source": str(paths["dataset_root"] / "normalization_stats.json"),
                    "val_metrics": val.__dict__,
                },
            )
            print(f"  new best -> saved {best_ckpt_path.name} (val_mae={best_val:.5f})")
        else:
            epochs_since_improve += 1

        if n_candidates > 0:
            candidate_pool, evicted = select_candidate_pool(
                candidate_pool, epoch, val.mae_normalized, n_candidates
            )
            if any(r["epoch"] == epoch for r in candidate_pool):
                save_candidate_checkpoint(
                    ckpt_dir / f"candidate_ep{epoch:03d}.pt",
                    model,
                    epoch,
                    val.mae_normalized,
                    cfg,
                    extra={
                        "run_id": run_id,
                        "normalization_source": str(
                            paths["dataset_root"] / "normalization_stats.json"),
                    },
                )
            for ev in evicted:
                stale = ckpt_dir / f"candidate_ep{ev:03d}.pt"
                if stale.is_file():
                    try:
                        stale.unlink()
                    except OSError:
                        pass
            write_json(out_dir / "candidate_pool.json", candidate_pool)

        save_checkpoint(
            latest_ckpt_path, model, optimizer, epoch, best_val, cfg,
            extra={"run_id": run_id},
        )
        write_json(out_dir / "history.json", history)
        _write_history_csv(out_dir / "history.csv", history)

        if epochs_since_improve >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs.")
            break

    summary = {
        "run_id": run_id,
        "status": "completed",
        "best_val_mae_normalized": best_val,
        "epochs_run": len(history),
        "best_checkpoint": str(best_ckpt_path),
        "seed": seed,
        "device": str(device),
        "model_variant": variant,
        "optimizer": opt_name,
        "loss": ("hotspot_weighted_mae" if hotspot_enabled
                 else ("contrast_mae" if contrast_enabled else tcfg.get("loss", "mae"))),
        "hotspot_loss": (hotspot_cfg if hotspot_enabled else {"enabled": False}),
        "contrast_loss": (contrast_cfg if contrast_enabled else {"enabled": False}),
        "candidate_pool": candidate_pool,
        "scheduler": tcfg.get("scheduler", "none"),
        "weight_decay": wd,
        "learning_rate": float(tcfg["learning_rate"]),
        "gradient_clip": float(tcfg["gradient_clip"]),
        "hidden_channels": int(mcfg["hidden_channels"]),
        "kernel_size": int(mcfg["kernel_size"]),
        "n_parameters": n_params,
        "history": history,
    }
    write_json(out_dir / "run_summary.json", summary)
    print(f"\nTraining complete. Best val MAE (normalized): {best_val:.5f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Outputs        : {out_dir}")

    train_ds.close()
    val_ds.close()
    return 0


def _write_history_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    lines = [",".join(keys)]
    for row in history:
        lines.append(",".join(str(row[k]) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())