"""
DrywallSegDataset — PyTorch Dataset for text-conditioned segmentation.

Handles image preprocessing via CLIPSegProcessor, mask binarization,
and random prompt variant sampling per __getitem__ call.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import CLIPSegProcessor

from .transforms import get_train_transforms


class DrywallSegDataset(Dataset):
    """Text-conditioned segmentation dataset.

    Args:
        image_dir: Directory containing source images (.jpg).
        mask_dir: Directory containing binary masks (.png, {0,255}).
        prompt_variants: List of prompt strings for this class
            (e.g. ["segment crack", "segment wall crack"]).
        processor: CLIPSegProcessor for image + text preprocessing.
        augment: Whether to apply training augmentations.
        seed: Random seed for reproducible prompt sampling.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        prompt_variants: list[str],
        processor: CLIPSegProcessor,
        augment: bool = False,
        seed: int = 42,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.prompt_variants = prompt_variants
        self.processor = processor
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        # Collect matched image/mask pairs
        self.samples = self._collect_samples()
        if len(self.samples) == 0:
            raise ValueError(
                f"No matched image/mask pairs found. "
                f"Image dir: {self.image_dir}, Mask dir: {self.mask_dir}"
            )

        self.transforms = get_train_transforms() if augment else None

    def _collect_samples(self) -> list[dict]:
        """Find images with corresponding masks."""
        samples = []
        mask_files = {p.stem: p for p in self.mask_dir.glob("*.png")}

        for img_path in sorted(self.image_dir.glob("*.jpg")):
            stem = img_path.stem
            if stem in mask_files:
                samples.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_files[stem]),
                    "image_id": stem,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and preprocess a single sample.

        Returns:
            Dict with keys:
                - pixel_values: (3, 352, 352) float tensor
                - input_ids: (seq_len,) long tensor
                - attention_mask: (seq_len,) long tensor
                - mask: (352, 352) float tensor, values {0.0, 1.0}
                - image_id: str
        """
        sample = self.samples[idx]

        # Load image (BGR → RGB)
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)

        # Apply augmentations (training only)
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Sample random prompt variant
        prompt_idx = self.rng.randint(0, len(self.prompt_variants))
        prompt = self.prompt_variants[prompt_idx]

        # CLIPSegProcessor: resize image to 352x352, normalize
        # Returns pixel_values as (3, 352, 352) tensor
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        # Resize mask to 352x352 to match CLIPSeg output resolution
        mask_resized = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)

        # Binarize: any nonzero pixel → 1.0
        mask_tensor = torch.from_numpy(mask_resized).float() / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # (3, 352, 352)
            "input_ids": inputs["input_ids"].squeeze(0),  # (seq_len,)
            "attention_mask": inputs["attention_mask"].squeeze(0),  # (seq_len,)
            "mask": mask_tensor,  # (352, 352)
            "image_id": sample["image_id"],
        }


def build_dataset(
    dataset_type: str,
    split: str,
    data_root: str | Path,
    masks_root: str | Path,
    processor: CLIPSegProcessor,
    augment: bool = False,
    seed: int = 42,
) -> DrywallSegDataset:
    """Factory for creating dataset splits.

    Args:
        dataset_type: "cracks" or "taping".
        split: "train", "valid", or "test".
        data_root: Root directory containing dataset folders.
        masks_root: Root directory containing generated masks.
        processor: CLIPSegProcessor instance.
        augment: Apply training augmentations.
        seed: Random seed for prompt sampling.

    Returns:
        Configured DrywallSegDataset.
    """
    data_root = Path(data_root)
    masks_root = Path(masks_root)

    prompt_map = {
        "cracks": ["segment crack", "segment wall crack"],
        "taping": ["segment taping area", "segment joint/tape", "segment drywall seam"],
    }

    if dataset_type not in prompt_map:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Expected 'cracks' or 'taping'.")

    image_dir = data_root / dataset_type / split
    mask_dir = masks_root / dataset_type / split

    return DrywallSegDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        prompt_variants=prompt_map[dataset_type],
        processor=processor,
        augment=augment,
        seed=seed,
    )
