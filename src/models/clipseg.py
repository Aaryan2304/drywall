"""
CLIPSeg model wrapper with selective layer freezing.

Architecture (from named_modules inspection):
  model.clip                    — CLIP backbone (vision + text encoders) → FROZEN
  model.decoder                 — Decoder (includes FiLM, reduces, layers, transposed_conv) → TRAINABLE
    decoder.film_mul            — FiLM multiplicative conditioning (text→mask)
    decoder.film_add            — FiLM additive conditioning (text→mask)
    decoder.reduces             — Feature projection layers
    decoder.layers              — Decoder transformer layers
    decoder.transposed_convolution — Upsampling convolutions

Strategy: freeze clip, unfreeze entire decoder. This gives full conditioning
capacity (FiLM + projection + decoder layers) within T4 VRAM budget.
"""

import torch
from transformers import CLIPSegForImageSegmentation


def load_model(device: torch.device = torch.device("cpu")) -> CLIPSegForImageSegmentation:
    """Load pretrained CLIPSeg and configure trainable layers.

    Args:
        device: Target device (cuda or cpu).

    Returns:
        Model with frozen backbone, trainable decoder (includes FiLM + projection).
    """
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Freeze backbone (vision + text encoders)
    for param in model.clip.parameters():
        param.requires_grad = False

    # Unfreeze entire decoder (includes FiLM, reduces, layers, transposed_conv)
    for param in model.decoder.parameters():
        param.requires_grad = True

    model.to(device)
    return model


def count_parameters(model: CLIPSegForImageSegmentation) -> dict:
    """Count trainable vs frozen parameters.

    Returns:
        Dict with trainable/frozen counts and total.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": trainable + frozen,
        "trainable_pct": 100 * trainable / (trainable + frozen),
    }
