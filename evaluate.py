"""
Evaluation: test-set metrics + consistency analysis.

Computes:
1. Per-prompt mIoU and Dice on test set
2. Consistency analysis across brightness / edge-density / defect-size strata
3. Summary tables for report
"""

import argparse
import json
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


def compute_image_statistics(image_path: str) -> dict:
    """Compute image statistics for consistency stratification.

    Args:
        image_path: Path to source image.

    Returns:
        Dict with 'brightness', 'edge_density'.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges) / 255.0)

    return {
        "brightness": brightness,
        "edge_density": edge_density,
    }


def compute_defect_size(mask: torch.Tensor) -> float:
    """Compute foreground ratio of mask.

    Args:
        mask: (H, W) binary mask tensor.

    Returns:
        Foreground pixel ratio.
    """
    return float(mask.sum() / mask.numel())


def assign_tertile(values: list[float]) -> list[str]:
    """Assign Low/Medium/High tertile labels.

    Args:
        values: List of numeric values.

    Returns:
        List of tertile labels.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    t1 = sorted_vals[n // 3]
    t2 = sorted_vals[2 * n // 3]

    labels = []
    for v in values:
        if v <= t1:
            labels.append("Low")
        elif v <= t2:
            labels.append("Medium")
        else:
            labels.append("High")
    return labels


def evaluate_test_set(
    model,
    test_loader: DataLoader,
    dataset_type: str,
    device: torch.device,
    threshold: float,
    dataset_samples: list[dict],
) -> dict:
    """Compute per-prompt metrics + consistency stratification.

    Args:
        model: Trained CLIPSeg model.
        test_loader: Test DataLoader.
        dataset_type: "cracks" or "taping".
        device: Compute device.
        threshold: Optimal threshold for this prompt class.
        dataset_samples: List of sample dicts with image_path and mask_path.

    Returns:
        Dict with overall metrics and per-stratum breakdown.
    """
    model.eval()

    all_probs = []
    all_masks = []
    sample_stats = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Eval {dataset_type}", leave=False)):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            masks = batch["mask"]
            image_ids = batch["image_id"]

            logits = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_masks.append(masks)

            # Per-sample stats
            for i in range(probs.shape[0]):
                img_id = image_ids[i]
                # Find sample info
                sample = None
                for s in dataset_samples:
                    if s["image_id"] == img_id:
                        sample = s
                        break

                if sample:
                    img_stats = compute_image_statistics(sample["image_path"])
                    defect_size = compute_defect_size(masks[i])
                    sample_stats.append({
                        "image_id": img_id,
                        "brightness": img_stats["brightness"],
                        "edge_density": img_stats["edge_density"],
                        "defect_size": defect_size,
                    })

    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Overall metrics
    overall_miou = compute_mIoU(all_probs, all_masks, threshold=threshold)
    overall_dice = compute_dice(all_probs, all_masks, threshold=threshold)

    # Consistency stratification
    brightness_vals = [s["brightness"] for s in sample_stats]
    edge_vals = [s["edge_density"] for s in sample_stats]
    size_vals = [s["defect_size"] for s in sample_stats]

    brightness_labels = assign_tertile(brightness_vals)
    edge_labels = assign_tertile(edge_vals)
    size_labels = assign_tertile(size_vals)

    for i, s in enumerate(sample_stats):
        s["brightness_stratum"] = brightness_labels[i]
        s["edge_stratum"] = edge_labels[i]
        s["size_stratum"] = size_labels[i]

    # Per-stratum mIoU
    strata = {}
    for stratum_name, label_key in [
        ("brightness", "brightness_stratum"),
        ("edge_density", "edge_stratum"),
        ("defect_size", "size_stratum"),
    ]:
        strata[stratum_name] = {}
        for level in ["Low", "Medium", "High"]:
            indices = [i for i, s in enumerate(sample_stats) if s[label_key] == level]
            if indices:
                stratum_probs = all_probs[indices]
                stratum_masks = all_masks[indices]
                stratum_miou = compute_mIoU(stratum_probs, stratum_masks, threshold=threshold)
                stratum_dice = compute_dice(stratum_probs, stratum_masks, threshold=threshold)
                strata[stratum_name][level] = {
                    "count": len(indices),
                    "miou": stratum_miou,
                    "dice": stratum_dice,
                }

    return {
        "dataset": dataset_type,
        "threshold": threshold,
        "n_samples": len(sample_stats),
        "overall": {
            "miou": overall_miou,
            "dice": overall_dice,
        },
        "strata": strata,
    }


def print_evaluation_report(results: dict):
    """Print formatted evaluation report to stdout."""
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {results['dataset'].upper()}")
    print(f"{'='*60}")
    print(f"Threshold: {results['threshold']}")
    print(f"Test samples: {results['n_samples']}")

    print(f"\nOverall Metrics:")
    print(f"  mIoU:  {results['overall']['miou']:.4f}")
    print(f"  Dice:  {results['overall']['dice']:.4f}")

    print(f"\nConsistency Analysis:")
    for stratum_name, levels in results["strata"].items():
        print(f"\n  {stratum_name}:")
        print(f"  {'Level':<10} {'Count':>6} {'mIoU':>8} {'Dice':>8}")
        print(f"  {'-'*34}")
        for level in ["Low", "Medium", "High"]:
            if level in levels:
                s = levels[level]
                print(f"  {level:<10} {s['count']:>6} {s['miou']:>8.4f} {s['dice']:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CLIPSeg model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--masks_root", type=str, default="masks")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (otherwise load from checkpoint/infer results)")
    parser.add_argument("--threshold_file", type=str, default="outputs/predictions/threshold_results.json",
                        help="Path to threshold_results.json from infer.py")
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Load model
    model = load_model(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load thresholds
    thresholds = {}
    if args.threshold is not None:
        thresholds = {"cracks": args.threshold, "taping": args.threshold}
    elif Path(args.threshold_file).exists():
        with open(args.threshold_file) as f:
            thresh_data = json.load(f)
        for ds in ["cracks", "taping"]:
            if ds in thresh_data:
                thresholds[ds] = thresh_data[ds]["optimal_threshold"]
    else:
        # Default fallback
        thresholds = {"cracks": 0.5, "taping": 0.5}
        print(f"Warning: No threshold file found, using default 0.5")

    all_results = {}

    for dataset_type in ["cracks", "taping"]:
        threshold = thresholds.get(dataset_type, 0.5)

        test_ds = build_dataset(
            dataset_type, "test", args.data_root, args.masks_root,
            processor, augment=False, seed=args.seed,
        )
        test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=2)

        # Get sample paths for stat computation
        dataset_samples = test_ds.samples if hasattr(test_ds, 'samples') else []

        results = evaluate_test_set(
            model, test_loader, dataset_type, device, threshold, dataset_samples,
        )
        all_results[dataset_type] = results
        print_evaluation_report(results)

    # Mean across prompts
    mean_miou = (all_results["cracks"]["overall"]["miou"] + all_results["taping"]["overall"]["miou"]) / 2
    mean_dice = (all_results["cracks"]["overall"]["dice"] + all_results["taping"]["overall"]["dice"]) / 2

    print(f"\n{'='*60}")
    print(f"SUMMARY (mean across prompts)")
    print(f"{'='*60}")
    print(f"  mIoU:  {mean_miou:.4f}")
    print(f"  Dice:  {mean_dice:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
