#!/usr/bin/env python3
"""
Training entrypoint for CLIPSeg drywall segmentation.

Usage:
    python train.py
    python train.py --epochs 10 --batch_size 4 --lr 1e-4
"""

import argparse
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.training.config import TrainConfig
from src.training.train import train


def main():
    parser = argparse.ArgumentParser(description="Train CLIPSeg on drywall segmentation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--masks_root", type=str, default="masks")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        seed=args.seed,
        data_root=args.data_root,
        masks_root=args.masks_root,
        output_dir=args.output_dir,
        log_path=str(Path(args.output_dir).parent / "train_log.csv"),
        use_amp=not args.no_amp,
    )

    best_miou = train(cfg)
    print(f"\nDone. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
