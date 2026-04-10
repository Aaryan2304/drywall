"""
Inference pipeline: threshold sweep, mask export, post-processing.

Performs:
1. Load trained checkpoint
2. Sweep thresholds [0.2–0.7] on validation set (per-prompt)
3. Select optimal threshold per prompt (argmax mIoU)
4. Export test set masks with optimal thresholds
5. Optional morphological closing (if improves val mIoU)
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPSegProcessor

from src.data.dataset import build_dataset
from src.models.clipseg import load_model
from src.eval.metrics import compute_mIoU, compute_dice


THRESHOLD_RANGE = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Prompt text used in export filenames (canonical form)
PROMPT_EXPORT_NAMES = {
    "cracks": "segment_crack",
    "taping": "segment_taping_area",
}


def sweep_thresholds(
    model,
    val_loader: DataLoader,
    device: torch.device,
    thresholds: list[float] = THRESHOLD_RANGE,
) -> dict:
    """Sweep thresholds on validation set and find optimal per-prompt.

    Args:
        model: Trained CLIPSeg model.
        val_loader: Validation DataLoader.
        device: Compute device.
        thresholds: List of thresholds to evaluate.

    Returns:
        Dict with 'optimal_threshold', 'best_miou', 'threshold_results'.
    """
    model.eval()
    all_logits = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting logits", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            all_logits.append(logits.cpu())
            all_masks.append(batch["mask"])

    all_logits = torch.cat(all_logits, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_probs = torch.sigmoid(all_logits)

    results = {}
    best_miou = 0.0
    best_thresh = 0.5

    for t in thresholds:
        miou = compute_mIoU(all_probs, all_masks, threshold=t)
        dice = compute_dice(all_probs, all_masks, threshold=t)
        results[t] = {"miou": miou, "dice": dice}

        if miou > best_miou:
            best_miou = miou
            best_thresh = t

    return {
        "optimal_threshold": best_thresh,
        "best_miou": best_miou,
        "threshold_results": {str(k): v for k, v in results.items()},
    }


def test_morphological_closing(
    model,
    val_loader: DataLoader,
    device: torch.device,
    threshold: float,
    kernel_size: int = 3,
) -> dict:
    """Check if morphological closing improves val mIoU.

    Args:
        model: Trained model.
        val_loader: Validation DataLoader.
        device: Compute device.
        threshold: Binarization threshold.
        kernel_size: Morphological kernel size.

    Returns:
        Dict with 'improves' (bool), 'miou_before', 'miou_after'.
    """
    model.eval()
    all_probs = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Closing eval", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            all_probs.append(torch.sigmoid(logits).cpu())
            all_masks.append(batch["mask"])

    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    miou_before = compute_mIoU(all_probs, all_masks, threshold=threshold)

    # Apply morphological closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_probs = []
    for i in range(all_probs.size(0)):
        mask_np = (all_probs[i].numpy() > threshold).astype(np.uint8) * 255
        closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        closed_probs.append(torch.from_numpy(closed.astype(np.float32) / 255.0))
    closed_probs = torch.stack(closed_probs)

    miou_after = compute_mIoU(closed_probs, all_masks, threshold=0.5)

    return {
        "improves": miou_after > miou_before,
        "miou_before": miou_before,
        "miou_after": miou_after,
        "kernel_size": kernel_size,
    }


def export_masks(
    model,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    threshold: float,
    dataset_type: str,
    apply_closing: bool = False,
    closing_kernel: int = 3,
) -> dict:
    """Export binary masks for test set.

    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        device: Compute device.
        output_dir: Directory to write mask PNGs.
        threshold: Binarization threshold.
        dataset_type: "cracks" or "taping" (for filename).
        apply_closing: Whether to apply morphological closing.
        closing_kernel: Kernel size for closing.

    Returns:
        Dict with 'masks_exported', 'avg_inference_time_ms'.
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_name = PROMPT_EXPORT_NAMES[dataset_type]

    exported = 0
    total_time = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Exporting {dataset_type}", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_ids = batch["image_id"]

            start = time.perf_counter()
            logits = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            torch.cuda.synchronize() if device.type == "cuda" else None
            total_time += time.perf_counter() - start

            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(probs.shape[0]):
                # Upsample 352x352 → 640x640 before thresholding
                upsampled = cv2.resize(probs[i], (640, 640), interpolation=cv2.INTER_LINEAR)

                # Binarize
                mask = (upsampled > threshold).astype(np.uint8) * 255

                # Optional morphological closing
                if apply_closing:
                    kernel = np.ones((closing_kernel, closing_kernel), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                filename = f"{image_ids[i]}__{prompt_name}.png"
                cv2.imwrite(str(output_dir / filename), mask)
                exported += 1

    avg_time_ms = (total_time / max(exported, 1)) * 1000

    return {
        "masks_exported": exported,
        "avg_inference_time_ms": avg_time_ms,
        "output_dir": str(output_dir),
        "threshold": threshold,
        "closing_applied": apply_closing,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference: threshold sweep + mask export")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--masks_root", type=str, default="masks")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Load model
    model = load_model(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (best mIoU: {ckpt['best_miou']:.4f})")

    output_dir = Path(args.output_dir)
    threshold_results = {}

    for dataset_type in ["cracks", "taping"]:
        print(f"\n{'='*50}")
        print(f"Processing: {dataset_type}")
        print(f"{'='*50}")

        # Validation set for threshold sweep
        val_ds = build_dataset(
            dataset_type, "valid", args.data_root, args.masks_root,
            processor, augment=False, seed=args.seed,
        )
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

        # Threshold sweep
        print("Sweeping thresholds...")
        sweep_results = sweep_thresholds(model, val_loader, device)
        optimal_t = sweep_results["optimal_threshold"]
        print(f"Optimal threshold: {optimal_t} (mIoU={sweep_results['best_miou']:.4f})")
        for t, metrics in sweep_results["threshold_results"].items():
            print(f"  t={t}: mIoU={metrics['miou']:.4f}, Dice={metrics['dice']:.4f}")

        # Test morphological closing
        print("Testing morphological closing...")
        closing_results = test_morphological_closing(model, val_loader, device, optimal_t)
        apply_closing = closing_results["improves"]
        print(f"  Before: mIoU={closing_results['miou_before']:.4f}")
        print(f"  After:  mIoU={closing_results['miou_after']:.4f}")
        print(f"  Apply closing: {apply_closing}")

        # Export test masks
        test_ds = build_dataset(
            dataset_type, "test", args.data_root, args.masks_root,
            processor, augment=False, seed=args.seed,
        )
        test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=2)

        export_dir = output_dir / dataset_type
        export_results = export_masks(
            model, test_loader, device, export_dir,
            threshold=optimal_t,
            dataset_type=dataset_type,
            apply_closing=apply_closing,
        )
        print(f"Exported {export_results['masks_exported']} masks to {export_dir}")
        print(f"Avg inference time: {export_results['avg_inference_time_ms']:.1f} ms/image")

        threshold_results[dataset_type] = {
            **sweep_results,
            "closing": closing_results,
            "export": export_results,
        }

    # Save threshold results
    results_path = output_dir / "threshold_results.json"
    with open(results_path, "w") as f:
        json.dump(threshold_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
