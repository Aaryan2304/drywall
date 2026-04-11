"""
Training loop for CLIPSeg drywall segmentation.

Handles: AMP, gradient accumulation, cosine LR with warmup,
checkpointing on val mIoU, CSV logging, early stopping.
"""

import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPSegProcessor

from src.data.dataset import build_dataset
from src.models.clipseg import load_model, count_parameters
from src.models.loss import CombinedSegLoss
from src.training.config import TrainConfig
from src.eval.metrics import compute_mIoU


def _warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    return LambdaLR(optimizer, lr_lambda)


def _log_to_csv(log_path: str, row: dict):
    """Append a row to CSV log file."""
    file_exists = Path(log_path).exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def validate(model, val_loader, loss_fn, device, processor):
    """Run validation and compute loss + mIoU.

    Returns:
        avg_loss, avg_miou (per-prompt combined).
    """
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    n_batches = 0

    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masks = batch["mask"].to(device)

        with torch.no_grad():
            logits = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        loss_dict = loss_fn(logits, masks)
        total_loss += loss_dict["loss"].item()

        miou = compute_mIoU(torch.sigmoid(logits), masks)
        total_miou += miou

        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1), total_miou / max(n_batches, 1)


def train(cfg: TrainConfig):
    """Main training entrypoint.

    Args:
        cfg: Training configuration.
    """
    # Reproducibility
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load processor
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Build datasets
    train_ds = build_dataset(
        "cracks", "train", cfg.data_root, cfg.masks_root, processor, augment=True, seed=cfg.seed
    )
    # Merge taping train into cracks train via ConcatDataset
    train_ds_tape = build_dataset(
        "taping", "train", cfg.data_root, cfg.masks_root, processor, augment=True, seed=cfg.seed
    )
    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset([train_ds, train_ds_tape])

    val_ds_cracks = build_dataset(
        "cracks", "valid", cfg.data_root, cfg.masks_root, processor, augment=False, seed=cfg.seed
    )
    val_ds_tape = build_dataset(
        "taping", "valid", cfg.data_root, cfg.masks_root, processor, augment=False, seed=cfg.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader_cracks = DataLoader(
        val_ds_cracks,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader_tape = DataLoader(
        val_ds_tape,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)} samples")
    print(f"Val cracks: {len(val_ds_cracks)}, Val taping: {len(val_ds_tape)}")

    # Load model
    model = load_model(device)
    param_counts = count_parameters(model)
    print(f"Parameters: {param_counts['trainable']:,} trainable / {param_counts['frozen']:,} frozen "
          f"({param_counts['trainable_pct']:.1f}% trainable)")

    # Loss
    loss_fn = CombinedSegLoss(bce_weight=0.5, dice_weight=0.5)
    loss_fn.to(device)

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # LR scheduler
    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    if cfg.max_steps is not None:
        total_steps = cfg.max_steps
    else:
        total_steps = steps_per_epoch * cfg.epochs
    scheduler = _warmup_cosine_schedule(optimizer, cfg.warmup_steps, total_steps)

    # AMP
    scaler = torch.amp.GradScaler("cuda") if cfg.use_amp and device.type == "cuda" else None

    # Training loop
    best_miou = 0.0
    patience_counter = 0
    global_step = 0

    # CSV header
    _log_to_csv(cfg.log_path, {
        "epoch": "epoch", "train_loss": "train_loss", "train_bce": "train_bce",
        "train_dice": "train_dice", "val_loss": "val_loss",
        "val_miou_cracks": "val_miou_cracks", "val_miou_tape": "val_miou_tape",
        "lr": "lr", "peak_vram_gb": "peak_vram_gb", "epoch_time_s": "epoch_time_s",
    })

    if cfg.max_steps is not None:
        est_epochs = cfg.max_steps / max(steps_per_epoch, 1)
        print(f"\nStarting training: {cfg.max_steps} steps (~{est_epochs:.1f} epochs), "
              f"effective batch={cfg.batch_size * cfg.grad_accum_steps}")
    else:
        print(f"\nStarting training: {cfg.epochs} epochs, "
              f"effective batch={cfg.batch_size * cfg.grad_accum_steps}")
    print(f"LR: {cfg.lr}, warmup: {cfg.warmup_steps} steps, cosine decay")
    print()

    for epoch in range(1, cfg.epochs + 1):
        # Check step-based termination before starting new epoch
        if cfg.max_steps is not None and global_step >= cfg.max_steps:
            break
        epoch_start = time.time()
        model.train()

        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_dice = 0.0
        n_batches = 0

        optimizer.zero_grad()

        if cfg.max_steps is not None:
            pbar_desc = f"Step {global_step}/{cfg.max_steps}"
        else:
            pbar_desc = f"Epoch {epoch}/{cfg.epochs}"
        pbar = tqdm(train_loader, desc=pbar_desc, leave=False)
        for step, batch in enumerate(pbar, 1):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            masks = batch["mask"].to(device)

            if cfg.use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits
                    loss_dict = loss_fn(logits, masks)
                    loss = loss_dict["loss"] / cfg.grad_accum_steps

                scaler.scale(loss).backward()
            else:
                logits = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                loss_dict = loss_fn(logits, masks)
                loss = loss_dict["loss"] / cfg.grad_accum_steps
                loss.backward()

            epoch_loss += loss_dict["loss"].item()
            epoch_bce += loss_dict["bce"].item()
            epoch_dice += loss_dict["dice"].item()
            n_batches += 1

            if step % cfg.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    break

            pbar.set_postfix({
                "loss": f"{loss_dict['loss'].item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        # Epoch stats
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_bce = epoch_bce / max(n_batches, 1)
        avg_train_dice = epoch_dice / max(n_batches, 1)

        # Validation
        val_loss_cracks, val_miou_cracks = validate(
            model, val_loader_cracks, loss_fn, device, processor
        )
        val_loss_tape, val_miou_tape = validate(
            model, val_loader_tape, loss_fn, device, processor
        )
        avg_val_loss = (val_loss_cracks + val_loss_tape) / 2
        avg_val_miou = (val_miou_cracks + val_miou_tape) / 2

        epoch_time = time.time() - epoch_start
        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0

        # Logging
        log_row = {
            "epoch": epoch,
            "train_loss": f"{avg_train_loss:.6f}",
            "train_bce": f"{avg_train_bce:.6f}",
            "train_dice": f"{avg_train_dice:.6f}",
            "val_loss": f"{avg_val_loss:.6f}",
            "val_miou_cracks": f"{val_miou_cracks:.6f}",
            "val_miou_tape": f"{val_miou_tape:.6f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "peak_vram_gb": f"{peak_vram:.2f}",
            "epoch_time_s": f"{epoch_time:.1f}",
        }
        _log_to_csv(cfg.log_path, log_row)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={avg_train_loss:.4f} (bce={avg_train_bce:.4f}, dice={avg_train_dice:.4f}) | "
            f"val_loss={avg_val_loss:.4f} | "
            f"mIoU cracks={val_miou_cracks:.4f} tape={val_miou_tape:.4f} mean={avg_val_miou:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{epoch_time:.1f}s | VRAM={peak_vram:.2f}GB"
        )

        # Checkpointing
        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            patience_counter = 0
            if cfg.save_best_only:
                save_path = Path(cfg.output_dir) / "best.pt"
            else:
                save_path = Path(cfg.output_dir) / f"epoch_{epoch}.pt"

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
                "config": cfg.__dict__,
            }, save_path)
            print(f"  -> Saved checkpoint: {save_path} (mIoU={best_miou:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{cfg.patience})")

        if patience_counter >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={cfg.patience})")
            break

    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")
    print(f"Checkpoint: {cfg.output_dir}/best.pt")
    print(f"Log: {cfg.log_path}")

    return best_miou
