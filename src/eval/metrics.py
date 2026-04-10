"""Segmentation metrics — mIoU and Dice."""

import torch


def compute_mIoU(pred_probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute mean Intersection over Union (Jaccard Index).

    Args:
        pred_probs: (B, H, W) sigmoid probabilities.
        targets: (B, H, W) binary ground truth {0.0, 1.0}.
        threshold: Binarization threshold.

    Returns:
        Scalar mIoU averaged over batch.
    """
    pred_bin = (pred_probs > threshold).float()
    targets = targets.float()

    # Flatten per sample
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    targ_flat = targets.view(targets.size(0), -1)

    intersection = (pred_flat * targ_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + targ_flat.sum(dim=1) - intersection

    # Avoid division by zero for empty masks
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return iou.mean().item()


def compute_dice(pred_probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice coefficient.

    Args:
        pred_probs: (B, H, W) sigmoid probabilities.
        targets: (B, H, W) binary ground truth {0.0, 1.0}.
        threshold: Binarization threshold.

    Returns:
        Scalar Dice averaged over batch.
    """
    pred_bin = (pred_probs > threshold).float()
    targets = targets.float()

    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    targ_flat = targets.view(targets.size(0), -1)

    intersection = (pred_flat * targ_flat).sum(dim=1)
    total = pred_flat.sum(dim=1) + targ_flat.sum(dim=1)

    dice = torch.where(total > 0, (2.0 * intersection) / total, torch.ones_like(total))
    return dice.mean().item()
