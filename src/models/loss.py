"""
Combined BCE + Dice loss for binary segmentation.

BCEWithLogitsLoss: pixel-level supervision, numerically stable on raw logits.
DiceLoss: optimizes overlap directly (aligns with mIoU/Dice eval metrics),
          handles class imbalance in sparse masks (thin cracks).

Equal weighting (0.5 / 0.5) as default.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss operating on raw logits.

    Computes 1 - Dice coefficient via sigmoid to maintain differentiability.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, H, W) raw model output.
            targets: (B, H, W) binary ground truth {0.0, 1.0}.

        Returns:
            Scalar Dice loss.
        """
        probs = torch.sigmoid(logits)
        # Flatten spatial dims
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """BCE + Dice loss with configurable weighting.

    Args:
        bce_weight: Weight for BCE component.
        dice_weight: Weight for Dice component.
        bce_pos_weight: Optional positive class weight for BCE
                        (useful if foreground is rare).
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        bce_pos_weight: float | None = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        pos_weight = torch.tensor([bce_pos_weight]) if bce_pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Args:
            logits: (B, H, W) raw model output.
            targets: (B, H, W) binary ground truth {0.0, 1.0}.

        Returns:
            Dict with 'loss' (total), 'bce', 'dice' for logging.
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return {
            "loss": total,
            "bce": bce_loss.detach(),
            "dice": dice_loss.detach(),
        }
