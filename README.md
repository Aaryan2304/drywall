# Prompted Segmentation for Drywall QA

Text-conditioned binary segmentation for construction defect detection. Given an image and a natural-language prompt ("segment crack" or "segment taping area"), the model produces a binary segmentation mask.

## Setup

```bash
git clone https://github.com/Aaryan2304/drywall.git
cd drywall
pip install -r requirements.txt
```

### Environment

| Component | Version |
|-----------|---------|
| Python | 3.11+ |
| PyTorch | 2.x (CUDA 12.x) |
| transformers | 4.x |
| albumentations | 2.x |
| OpenCV | 4.x |
| torchmetrics | 1.x |

### Hardware

Tested on NVIDIA T4 (16 GB VRAM). Configuration:

| Parameter | Value |
|-----------|-------|
| Batch size | 2 |
| Gradient accumulation | 2 |
| Effective batch | 4 |
| Precision | FP16 (AMP) |
| Peak VRAM (training) | ~0.7 GB |
| Trainable parameters | 1.1M (0.7% of total) |

## Reproducibility

All random seeds fixed to `42`:
- PyTorch: `torch.manual_seed(42)`
- NumPy: used for augmentation and prompt sampling
- CUDA: `torch.cuda.manual_seed_all(42)`

## Data Preparation

Two datasets, both 640x640 resolution, COCO JSON format:

| Dataset | Images | Annotation | Source |
|---------|--------|------------|--------|
| Cracks | 5,369 | Polygon segmentation | [Roboflow](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36) |
| Taping | 1,022 | Bounding box only | [Roboflow](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) |

Pre-split by Roboflow into train/valid/test (~70/15/15). Used as-is.

### Mask Generation

COCO annotations are rasterized into binary masks (`{0, 255}`, single-channel PNG):

- **Cracks**: polygon segmentation → `cv2.fillPoly` (16 annotations with empty segmentation are skipped)
- **Taping**: bounding box → `cv2.rectangle` (geometric proxy for linear tape seams; no polygon masks exist in the dataset)

```bash
python -c "
from src.data.mask_conversion import generate_masks

# Cracks: polygon mode
for split in ['train', 'valid', 'test']:
    generate_masks(f'cracks/{split}/_annotations.coco.json', f'masks/cracks/{split}', mode='polygon')

# Taping: bbox mode
for split in ['train', 'valid', 'test']:
    generate_masks(f'taping/{split}/_annotations.coco.json', f'masks/taping/{split}', mode='bbox')
"
```

Output: `masks/{dataset}/{split}/{image_stem}.png`

## Model

[CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) (`CIDAS/clipseg-rd64-refined`) — ViT-B/16 backbone with lightweight decoder and FiLM conditioning for text-to-mask generation.

### Layer Freezing

| Component | Status | Parameters |
|-----------|--------|------------|
| CLIP backbone (vision + text) | Frozen | 149.6M |
| Decoder (FiLM, projection, transformer, conv) | Trainable | 1.1M |

Only the decoder is fine-tuned. The backbone preserves pretrained vision-language features.

### Prompt Variants

The model is trained with random prompt variant sampling per batch to ensure consistency across phrasings:

- Cracks: `"segment crack"`, `"segment wall crack"`
- Taping: `"segment taping area"`, `"segment joint/tape"`, `"segment drywall seam"`

## Training

```bash
# Default (20 epochs, lr=5e-5, batch=2)
python train.py

# Custom
python train.py --epochs 10 --batch_size 4 --lr 1e-4 --patience 3
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 1e-4 |
| LR schedule | Cosine with linear warmup (100 steps) |
| Loss | BCE + Dice (equal weight) |
| Epochs | 20 (early stopping, patience=5) |
| Checkpoint | Best val mIoU |

### Augmentation (train only)

| Transform | Parameters | Probability |
|-----------|-----------|-------------|
| HorizontalFlip | — | 0.5 |
| RandomBrightnessContrast | ±0.2 | 0.5 |
| GaussNoise | std 10-50 | 0.3 |
| RandomScale | ±20% | 0.3 |

Conservative parameters chosen — thin cracks degrade under aggressive augmentation.

### Output

- `outputs/checkpoints/best.pt` — model checkpoint
- `outputs/train_log.csv` — per-epoch metrics (loss, mIoU, VRAM, timing)

## Inference & Mask Export

```bash
# Coming in Phase 3
python infer.py --checkpoint outputs/checkpoints/best.pt --split test
```

### Threshold Selection

CLIPSeg logits are uncalibrated on construction imagery. The optimal binarization threshold is determined by sweeping `[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]` on the validation set, separately per prompt class, and selecting the value that maximizes mIoU.

### Export Specification

| Property | Value |
|----------|-------|
| Format | Single-channel PNG, uint8 |
| Values | `{0, 255}` |
| Resolution | 640x640 (logits upsampled from 352x352 before thresholding) |
| Filename | `{image_id}__segment_crack.png` / `{image_id}__segment_taping_area.png` |

## Evaluation

Metrics computed on held-out test set, per prompt class:

| Metric | Implementation |
|--------|---------------|
| mIoU (Jaccard) | `torchmetrics.JaccardIndex(task="binary")` |
| Dice | `2 * intersection / (pred + gt)` |

### Consistency Analysis

Performance stratified across image statistics to measure stability:

| Stratum | Proxy | Computation |
|---------|-------|-------------|
| Lighting | Mean brightness | `np.mean(grayscale)` |
| Scene complexity | Edge density | `np.mean(Canny(gray)) / 255` |
| Defect size | Foreground ratio | `mask.sum() / mask.numel()` |

Each stratum split into Low/Medium/High tertiles. mIoU reported per stratum.

## Project Structure

```
.
├── train.py                    # Training entrypoint
├── src/
│   ├── data/
│   │   ├── mask_conversion.py  # COCO → binary masks
│   │   ├── dataset.py          # DrywallSegDataset
│   │   └── transforms.py       # Albumentations pipeline
│   ├── models/
│   │   ├── clipseg.py          # Model wrapper + layer freezing
│   │   └── loss.py             # BCE + Dice loss
│   ├── training/
│   │   ├── config.py           # TrainConfig dataclass
│   │   └── train.py            # Training loop
│   └── eval/
│       └── metrics.py          # mIoU, Dice
├── masks/                      # Generated binary masks (gitignored)
├── outputs/                    # Checkpoints + logs (gitignored)
└── Implementation_Plan.md      # Detailed technical plan
```

## Engineering Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data split | Roboflow as-is | Pre-split ~70/15/15; avoids merge complexity |
| Taping GT | `cv2.rectangle` from bbox | No polygon annotations exist; geometrically reasonable for linear seams |
| Backbone | Frozen | VRAM constraint; pretrained features sufficient |
| Decoder | Full stack unfrozen | FiLM layers critical for text-mask alignment |
| Threshold | Per-prompt, swept on val | CLIPSeg logits uncalibrated on target domain |
| Loss | BCE + Dice | BCE for stability; Dice aligns with eval metric |
| Prompt sampling | Random variant per batch | Trains consistency across phrasings |
| Upsample order | Logits → upsample → threshold | Avoids aliasing artifacts |
