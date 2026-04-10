"""
Albumentations transforms for training.

Applied only during training — validation and test use CLIPSegProcessor normalization only.
Conservative parameters chosen: thin cracks degrade under aggressive augmentation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms() -> A.Compose:
    """Training augmentation pipeline.

    Returns:
        Albumentations Compose with geometric + photometric transforms.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.GaussNoise(std_range=(10.0 / 255.0, 50.0 / 255.0), p=0.3),
            A.RandomScale(scale_limit=0.2, p=0.3),
        ],
        additional_targets={"mask": "mask"},
    )
