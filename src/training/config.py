"""Training configuration as a dataclass."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # Paths
    data_root: str = "."
    masks_root: str = "masks"
    output_dir: str = "outputs/checkpoints"
    log_path: str = "outputs/train_log.csv"

    # Training
    epochs: int = 20
    max_steps: int | None = None  # Override epochs if set (total optimizer steps)
    patience: int = 5  # Early stopping on val mIoU
    seed: int = 42

    # Optimizer
    lr: float = 5e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 100

    # Batch
    batch_size: int = 2
    grad_accum_steps: int = 2
    num_workers: int = 2

    # AMP
    use_amp: bool = True

    # LR schedule
    lr_scheduler: str = "cosine"  # cosine or none

    # Checkpointing
    save_best_only: bool = True

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
