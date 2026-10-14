"""Shared training/evaluation engine for the UHI ConvLSTM baseline.

Contains the epoch-level train and validation loops, gradient clipping, mixed
precision handling, masked-MAE optimization, and checkpoint save/load. Kept
separate from the CLI (``train.py``) so the evaluation stage can reuse the
identical validation logic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from utils.normalization import NormalizationStats

from .losses import masked_huber, masked_mae
from .metrics import MetricAccumulator

__all__ = [
    "EpochResult",
    "train_one_epoch",
    "validate",
    "save_checkpoint",
    "save_candidate_checkpoint",
    "select_candidate_pool",
    "load_checkpoint",
    "get_loss_fn",
]


def get_loss_fn(name: str, huber_delta: float = 1.0):
    """Return a masked loss ``fn(pred, target, mask)`` by name."""
    name = (name or "mae").lower()
    if name == "mae":
        return masked_mae
    if name == "huber":
        def _fn(pred, target, mask):
            return masked_huber(pred, target, mask, delta=huber_delta)

        return _fn
    raise ValueError(f"Unknown loss {name!r} (expected 'mae' or 'huber')")


@dataclass
class EpochResult:
    """Loss/metric summary for one epoch."""

    loss: float
    mae_normalized: float
    rmse_normalized: float
    valid_pixels: int
    seconds: float
    peak_gpu_mib: float
    mae_celsius: float = float("nan")
    rmse_celsius: float = float("nan")


def _train_steps(model, loader, optimizer, device, grad_clip, amp, scaler, log_every, logger, loss_fn, grad_log_every):
    model.train()
    total_loss = 0.0
    n_batches = 0
    last_grad_norm = 0.0
    needs_urban = bool(getattr(loss_fn, "requires_urban", False))
    needs_contrast = bool(getattr(loss_fn, "requires_contrast", False))
    # Optional per-component logging (e.g. base LST loss vs auxiliary contrast).
    log_components = callable(getattr(loss_fn, "component_losses", None))
    comp_sums: dict[str, float] = {}
    for step, batch in enumerate(loader):
        if needs_contrast:
            if len(batch) < 5:
                raise ValueError(
                    "loss_fn.requires_contrast is set but the loader did not supply "
                    "urban and rural masks (expected (X, y, mask, urban, rural)); "
                    "wrap the dataset in UrbanRuralPatchDataset."
                )
            X, y, mask, urban, rural = batch
            urban = urban.to(device, non_blocking=True).to(torch.bool)
            rural = rural.to(device, non_blocking=True).to(torch.bool)
        elif needs_urban:
            if len(batch) < 4:
                raise ValueError(
                    "loss_fn.requires_urban is set but the loader did not supply "
                    "an urban mask (expected (X, y, mask, urban)); wrap the dataset "
                    "in UrbanPatchDataset."
                )
            X, y, mask, urban = batch
            urban = urban.to(device, non_blocking=True).to(torch.bool)
        else:
            X, y, mask = batch
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            pred = model(X)
            if needs_contrast:
                loss = loss_fn(pred, y, mask, urban, rural)
            elif needs_urban:
                loss = loss_fn(pred, y, mask, urban)
            else:
                loss = loss_fn(pred, y, mask)
        if log_components:
            for k, v in loss_fn.component_losses().items():
                if v == v:  # skip NaN
                    comp_sums[k] = comp_sums.get(k, 0.0) + float(v)
        if amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                last_grad_norm = float(gn)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                last_grad_norm = float(gn)
            optimizer.step()
        total_loss += float(loss.detach())
        n_batches += 1
        if logger and log_every and (step + 1) % log_every == 0:
            extra = ""
            if log_components:
                extra = " " + " ".join(
                    f"{k}={comp_sums.get(k, 0.0) / (step + 1):.5f}"
                    for k in sorted(comp_sums))
            logger(f"    batch {step + 1}/{len(loader)} loss={float(loss.detach()):.5f} "
                   f"grad_norm={last_grad_norm:.3f}{extra}")
    del X, y, mask, pred, loss
    if needs_urban:
        del urban
    if needs_contrast:
        del rural
    _train_steps.last_components = (
        {k: v / max(n_batches, 1) for k, v in comp_sums.items()} if log_components else {}
    )
    return total_loss / max(n_batches, 1), n_batches, last_grad_norm


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip: float = 1.0,
    amp: bool = False,
    scaler=None,
    log_every: int = 0,
    logger=None,
    loss_fn=None,
    grad_log_every: int = 0,
) -> tuple[float, float]:
    """Run one training epoch; returns ``(mean_batch_loss, last_grad_norm)``."""
    if amp and scaler is None:
        raise ValueError("amp=True requires a GradScaler")
    if loss_fn is None:
        loss_fn = masked_mae
    start = time.perf_counter()
    loss, n_batches, last_grad_norm = _train_steps(
        model, loader, optimizer, device, grad_clip, amp, scaler, log_every, logger,
        loss_fn, grad_log_every,
    )
    logger and logger(
        f"  train: mean_loss={loss:.5f} batches={n_batches} "
        f"grad_norm={last_grad_norm:.3f} time={time.perf_counter() - start:.1f}s"
    )
    # Expose per-component training losses (e.g. base vs auxiliary contrast) for
    # the caller to record. Empty when the loss provides no components.
    train_one_epoch.last_components = getattr(_train_steps, "last_components", {})
    return loss, last_grad_norm


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    stats: NormalizationStats,
    in_celsius: bool = False,
    amp: bool = False,
) -> EpochResult:
    """Evaluate ``model`` over ``loader`` with aggregated masked metrics.

    Both normalized and Celsius metrics are always computed (the latter via the
    frozen LST statistics). ``in_celsius`` is retained for backward
    compatibility but no longer changes behavior.
    """
    model.eval()
    acc = MetricAccumulator()
    acc_c = MetricAccumulator()
    total_loss = 0.0
    n_batches = 0
    lst_i = stats.index("LST_Celsius")
    mean, std = stats.means[lst_i], stats.stds[lst_i]
    start = time.perf_counter()
    for X, y, mask in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            pred = model(X)
        total_loss += float(masked_mae(pred, y, mask).detach())
        n_batches += 1
        acc.update(pred, y, mask)
        # Celsius metrics with the verified inverse LST transform.
        acc_c.update(pred * std + mean, y * std + mean, mask)
        del X, y, mask, pred
    peak = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    m = acc.as_dict()
    mc = acc_c.as_dict()
    return EpochResult(
        loss=total_loss / max(n_batches, 1),
        mae_normalized=m["mae"],
        rmse_normalized=m["rmse"],
        valid_pixels=m["valid_pixels"],
        seconds=time.perf_counter() - start,
        peak_gpu_mib=peak,
        mae_celsius=mc["mae"],
        rmse_celsius=mc["rmse"],
    )


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    best_val: float,
    config: dict,
    extra: dict | None = None,
) -> None:
    """Atomically save a checkpoint (write temp then replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "config": config,
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location=None):
    """Load a checkpoint into ``model`` (and optionally ``optimizer``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def save_candidate_checkpoint(
    path: str | Path,
    model,
    epoch: int,
    val_mae_normalized: float,
    config: dict,
    extra: dict | None = None,
) -> None:
    """Atomically save a compact candidate checkpoint (weights only).

    Candidate checkpoints deliberately omit optimizer state so that retaining a
    small pool of alternatives for hotspot evaluation stays lightweight (roughly
    one third the size of a full best/latest checkpoint). They are still fully
    loadable with :func:`load_checkpoint` (which only requires the model state
    dict, epoch and best_val).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "best_val": float(val_mae_normalized),
        "config": config,
        "candidate": True,
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def select_candidate_pool(
    pool: list[dict],
    epoch: int,
    val_mae_normalized: float,
    keep: int = 3,
) -> tuple[list[dict], list[str]]:
    """Update a ranked candidate pool with one epoch's validation score.

    ``pool`` is a list of ``{"epoch": int, "val_mae": float}`` entries (best
    first). Returns ``(new_pool, evicted_epochs)`` where ``evicted_epochs`` lists
    epoch numbers that fell out of the retained top ``keep``. Ties are broken by
    the lower (earlier) epoch. Pure and side-effect free for unit testing.
    """
    new_pool = list(pool) + [{"epoch": int(epoch), "val_mae": float(val_mae_normalized)}]
    new_pool.sort(key=lambda r: (r["val_mae"], r["epoch"]))
    kept = new_pool[: int(keep)]
    evicted = [r["epoch"] for r in new_pool[int(keep):]]
    return kept, evicted