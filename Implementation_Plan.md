# Prompted Segmentation for Drywall QA — Implementation Plan

**Assignment:** Text-conditioned binary segmentation for two construction defect classes  
**Model:** CLIPSeg (`CIDAS/clipseg-rd64-refined`)  
**Deadline:** 2 days  
**Hardware:** RTX 3050 Ti (4GB VRAM) / Colab T4 (16GB)  
**Seed:** `42` (fixed globally across all splits, augmentation, and sampling)

---

## Table of Contents

1. [Data Acquisition & Formatting](#1-data-acquisition--formatting)
2. [Model Selection & Justification](#2-model-selection--justification)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Training Configuration](#4-training-configuration)
5. [Inference & Mask Export](#5-inference--mask-export)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Reporting & Documentation](#7-reporting--documentation)
8. [Hardware Constraints](#8-hardware-constraints)
9. [Engineering Decisions Log](#9-engineering-decisions-log)
10. [Final Verification Checklist](#10-final-verification-checklist)

---

## 1. Data Acquisition & Formatting

**Roboflow Export Configuration**
- Format: COCO JSON (segmentation polygons)
- Preprocessing applied at export: Auto-Orient only (EXIF correction, lossless)
- Augmentations at export: None (applied in training loop via albumentations)
- Split: 70/15/15, re-split in code with seed=42 via sklearn.train_test_split

### 1.1 Datasets

| Dataset | Roboflow Slug | Prompt Variants | Task |
|---------|--------------|-----------------|------|
| Drywall-Join-Detect | `objectdetect-pu6rn/drywall-join-detect` | `"segment taping area"`, `"segment joint/tape"`, `"segment drywall seam"` | Tape/joint segmentation |
| Cracks | `fyp-ny1jt/cracks-3ii36` | `"segment crack"`, `"segment wall crack"` | Crack segmentation |

Export format: **COCO Segmentation JSON** (preserves polygon masks; easier to convert to binary PNG than bounding-box formats).

### 1.2 Mask Conversion

Cracks: polygon segmentation verified (99.7% coverage) → rasterize via `cv2.fillPoly`. Skip 16 annotations where `segmentation=[]`.
Taping: bounding-box annotations only, `segmentation=[]` for all 997 train annotations → rasterize bbox `[x, y, w, h]` as filled rectangle via `cv2.rectangle`. Bbox is a reasonable geometric proxy for linear tape seams.

### 1.3 Data Split

Use Roboflow-exported splits as-is. The exports already provide a ~70/15/15 split per dataset, and re-splitting introduces unnecessary merge complexity without meaningful benefit.

`seed=42` is used **only** for augmentation randomness and prompt variant sampling — not for data splitting.

Both datasets: all images are uniform 640×640. Upsample target in inference is fixed — no per-image dynamic resizing required.

| Split | Dataset 1 (Taping) | Dataset 2 (Cracks) | Total |
|-------|--------------------|--------------------|-------|
| Train | 715 | 3,758 | 4,473 |
| Val   | 153 | 805 | 958 |
| Test  | 154 | 806 | 960 |
| **Total** | **1,022** | **5,369** | **6,391** |

### 1.4 Augmentation Strategy

Applied **only during training**, not validation or test.

| Transform | Parameters | Rationale |
|-----------|-----------|-----------|
| `HorizontalFlip` | p=0.5 | Orientation invariance |
| `RandomBrightnessContrast` | brightness=±0.2, contrast=±0.2, p=0.5 | Lighting variation on construction sites |
| `GaussNoise` | var_limit=(10, 50), p=0.3 | Sensor noise robustness |
| `RandomScale` | scale_limit=0.2, p=0.3 | Object size variation |

**No augmentation** applied to masks beyond what's geometrically co-applied with the image.  
Augmentation library: `albumentations`.  
Justification: Thin cracks are easily lost under aggressive augmentation — conservative parameters chosen deliberately.

---

## 2. Model Selection & Justification

### 2.1 Candidate Comparison

| Model | Architecture | Text-Conditioned | Fine-tunable | Inference Speed | Notes |
|-------|-------------|-----------------|--------------|-----------------|-------|
| **CLIPSeg** | ViT-B/16 + lightweight decoder | ✅ Native | ✅ Decoder-only feasible | Fast (~30ms/img) | Best fit for this task |
| SAM + CLIP routing | ViT-H encoder + mask decoder | Indirect (prompt engineering) | Complex | Moderate | Two-stage; no joint optimization |
| GroundingDINO + SAM | Transformer + ViT-H | ✅ Text grounding | Limited | Slow | Overkill; two-stage pipeline |
| LLaVA / BLIP-2 + decoder | LLM-scale | ✅ | Not feasible in 2 days | Slow | Out of scope for binary segmentation |

**Decision:** CLIPSeg (`CIDAS/clipseg-rd64-refined`) is selected because:
- Native text-to-mask architecture — no prompt engineering hacks
- Decoder-only fine-tuning fits within 4GB VRAM
- Pretrained on broad vision-language data; reasonable zero-shot baseline on construction imagery
- Single-model, single forward pass — clean evaluation

### 2.2 Actual Model Footprint

| Component | Precision | Size |
|-----------|-----------|------|
| CLIP ViT-B/16 backbone (frozen) | FP16 | ~170 MB |
| CLIPSeg decoder + projections (trainable) | FP32 | ~30 MB |
| **Total on disk** | — | **~400 MB** |

> **Correction from initial estimate:** The ~1.2GB figure applies to ViT-L/14 (larger CLIP variants). CLIPSeg uses ViT-B/16; FP16 backbone is ~170MB. Log observed values during training for the report.

---

## 3. Pipeline Architecture

### 3.1 Dataset Class

```
DrywallSegDataset(torch.utils.data.Dataset)
├── __init__(image_paths, mask_paths, prompt_variants, processor, augment)
├── __getitem__ → (pixel_values, input_ids, attention_mask, mask_tensor)
└── Prompt sampling: random.choice(prompt_variants) per __getitem__ call
```

- Image preprocessing: `CLIPSegProcessor` (handles resize to 352×352, normalization)
- Mask preprocessing: Resize to match logit output size (352×352), binarize to `{0,1}`
- Prompt sampling during training ensures the model learns consistency across prompt variants — directly addresses the 30-point consistency rubric criterion

### 3.2 Forward Pass

```
logits = model(pixel_values=img, input_ids=text_ids, attention_mask=attn_mask).logits
# logits shape: (B, 352, 352)
loss = BCE(sigmoid(logits), mask) + DiceLoss(sigmoid(logits), mask)
```

### 3.3 Loss Function

**Combined BCE + Dice loss:**

- `BCEWithLogitsLoss` — pixel-level supervision; numerically stable (operates on raw logits)
- `DiceLoss` — handles class imbalance; optimizes overlap metric directly (aligns with mIoU/Dice eval)
- Equal weighting (0.5 / 0.5) as default; can adjust if one class is severely imbalanced

---

## 4. Training Configuration

### 4.1 Frozen vs. Trainable Layers

| Component | Status | Rationale |
|-----------|--------|-----------|
| `model.clip.vision_model` | **Frozen** | Preserves visual features; VRAM constraint |
| `model.clip.text_model` | **Frozen** | Preserves language alignment |
| `model.decoder` | **Trainable** | Task-specific adaptation |
| `model.film_mul`, `model.film_add` | **Trainable** | FiLM conditioning layers — critical for text-mask alignment |
| `model.reduce` (projection) | **Trainable** | Feature projection to decoder |

### 4.2 Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | `5e-5` |
| Weight decay | `1e-4` |
| LR schedule | CosineAnnealingLR |
| Epochs | 20 (early stop on val mIoU, patience=5) |
| Warmup steps | 100 |

### 4.3 VRAM Configuration

| Hardware | Batch Size | Grad Accum Steps | Effective Batch | Precision | Trainable Components |
|----------|-----------|-----------------|-----------------|-----------|---------------------|
| Colab T4 (16GB) | 2 | 2 | 4 | FP16 (AMP) | Decoder + FiLM + projection |

AMP via `torch.cuda.amp.autocast()` + `GradScaler`.

> T4's 16GB VRAM allows unfreezing FiLM conditioning layers (`film_mul`, `film_add`) and the `reduce` projection layer in addition to the decoder. This provides stronger text-to-mask alignment than decoder-only fine-tuning.

### 4.4 Checkpointing

Save `model.state_dict()` at highest `val_mIoU` across both prompts combined. Log per-epoch metrics to CSV.

---

## 5. Inference & Mask Export

### 5.1 Threshold Selection

Threshold is **not fixed at 0.5**. CLIPSeg logits on out-of-distribution construction imagery are not calibrated — optimal threshold is determined empirically.

**Sweep on validation set:** `[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]`  
**Per-prompt thresholds** — crack prompts and taping prompts may require different values.  
**Metric for selection:** argmax mIoU on validation set.  
**Document selected thresholds** in report and README.

### 5.2 Post-processing (Optional, Documented)

If binary masks contain isolated noise pixels:
- `cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=3×3)` — fills small holes in crack predictions
- Applied only if it improves val mIoU; decision must be logged either way

### 5.3 Export Specification

| Requirement | Specification |
|-------------|--------------|
| Format | Single-channel PNG, `uint8` |
| Values | `{0, 255}` |
| Spatial size | Upsample CLIPSeg logits 352×352 → 640×640 via F.interpolate(mode='bilinear', align_corners=False) before thresholding. Target resolution is uniform across both datasets. |
| Filename | `{image_id}__segment_crack.png` / `{image_id}__segment_taping_area.png` |

> **Note:** CLIPSeg outputs logits at 352×352 regardless of input size. Upsample to source resolution **before** thresholding, not after.

---

## 6. Evaluation Protocol

### 6.1 Primary Metrics (Correctness — 50 pts)

Computed on held-out test set, separately per prompt class:

| Metric | Implementation |
|--------|---------------|
| mIoU | `torchmetrics.JaccardIndex(task="binary")` |
| Dice | `2 * (pred ∩ gt) / (pred + gt + ε)`, ε=1e-6 |

Report format:

| Prompt | mIoU | Dice |
|--------|------|------|
| segment crack | — | — |
| segment taping area | — | — |
| **Mean** | — | — |

### 6.2 Consistency Evaluation (30 pts)

The datasets do not include scene metadata. Stratify post-hoc using computable image statistics:

| Stratum | Proxy Metric | Computation |
|---------|-------------|-------------|
| Lighting | Mean pixel brightness | `np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))` |
| Scene complexity | Edge density | `np.mean(cv2.Canny(gray, 100, 200)) / 255` |
| Defect size | Foreground mask area ratio | `mask.sum() / mask.numel()` |

Split each stratum into Low / Medium / High tertiles. Report mIoU per stratum. Goal: demonstrate metric variance is low across strata.

### 6.3 Failure Analysis

Document systematic failure modes with visual examples:

- False negatives on thin cracks (< 5px width)
- False positives on textured backgrounds mimicking tape seams
- Low-contrast scenes (brightness stratum: Low)
- Occlusion / partial visibility
- Taping GT masks are bbox-derived rectangles — lower Dice scores on taping vs. cracks are expected due to GT mask imprecision at seam edges, not model failure.

---

## 7. Reporting & Documentation

### 7.1 Report Structure

1. **Goal Summary** — one paragraph, what the system does
2. **Approach & Model** — CLIPSeg selection rationale, alternatives considered (Section 2.1 table)
3. **Data Split Counts** — filled table from Section 1.3
4. **Metrics Table** — per-prompt mIoU and Dice (Section 6.1)
5. **Visual Examples** — 3–4 trios: `Original | GT Mask | Predicted Mask` (2 from cracks, 2 from taping; include one failure case)
6. **Failure Notes** — 1 paragraph per prompt class
7. **Runtime & Footprint** — table below

| Metric | Value |
|--------|-------|
| Total training time | — |
| Avg inference time / image | — ms |
| Model size (disk) | — MB |
| Peak VRAM (training) | — GB |
| Peak VRAM (inference) | — GB |

8. Taping dataset lacks polygon annotations; rectangular proxy masks used as GT. SAM-assisted bbox-prompted re-annotation identified as the primary improvement path.

### 7.2 README Structure

```
## Setup
## Environment (Python version, CUDA version, key package versions)
## Hardware
## Reproducibility (seed=42, documented everywhere)
## Data Preparation
## Training
## Inference & Mask Export
## Evaluation
## Results Summary
```

---

## 8. Hardware Constraints

| Hardware | Batch | Grad Accum | Precision | Expected Peak VRAM | Notes |
|----------|-------|-----------|-----------|-------------------|-------|
| Colab T4 (16GB) | 2 | 2 | FP16 | ~5–6 GB | Backbone frozen; decoder + FiLM + projection unfrozen |

> Exact VRAM consumption varies with input resolution and framework version. **Measure and log observed peak values** using `torch.cuda.max_memory_allocated()` — do not report estimates in the final submission.

---

## 9. Engineering Decisions Log

All choices below must be explicitly documented in the report and README.

| Decision | Choice | Justification |
|----------|--------|--------------|
| Train/Val/Test split | Roboflow splits as-is | Pre-split by Roboflow (~70/15/15); avoids merge complexity; proportions verified |
| Loss function | BCE + Dice (equal weight) | BCE for stability; Dice for overlap optimization aligned with eval metrics |
| Optimizer | AdamW, lr=5e-5 | Standard for fine-tuning transformer decoders |
| Threshold | Per-prompt, swept on val set over [0.2–0.7] | CLIPSeg logits not calibrated on construction domain |
| Augmentation | Flip, brightness/contrast, noise, scale (conservative) | Thin cracks degrade under aggressive transforms |
| Post-processing | Morphological closing (if improves val mIoU) | Documented regardless of application |
| Prompt sampling | Random variant per batch | Trains consistency across prompt phrasings |
| Backbone | Frozen (ViT-B/16) | VRAM constraint; sufficient pretrained features |
| Upsample order | Logits → upsample → threshold | Threshold before upsample introduces aliasing artifacts |
| Taping GT mask source | `cv2.rectangle` from bbox | No polygon segmentation exists in dataset; bbox is geometrically reasonable for linear seams |
| Empty segmentation handling | Skip 16 cracks annotations | `segmentation=[]`; no valid mask can be generated |
| Fine-tuning scope | Decoder + FiLM + projection (T4) | T4 VRAM allows full conditioning stack; stronger text-mask alignment |

---

## 10. Final Verification Checklist

### Data
- [ ] Both Roboflow datasets downloaded; COCO JSON exported
- [ ] Masks: single-channel PNG, `{0,255}`, exact spatial match to source images
- [ ] Split counts logged with seed=42

### Training
- [ ] Backbone frozen; decoder + FiLM layers trainable
- [ ] AMP enabled; batch size and grad accum documented
- [ ] Best checkpoint saved on val mIoU

### Inference
- [ ] Logits upsampled to source resolution before thresholding
- [ ] Threshold swept per-prompt on validation set; optimal value documented
- [ ] Filenames: `{id}__segment_crack.png` and `{id}__segment_taping_area.png`

### Evaluation
- [ ] mIoU and Dice computed on test set for both prompts
- [ ] Consistency evaluation across brightness / complexity / defect-size strata
- [ ] At least one failure case included in visual examples

### Documentation
- [ ] Report includes all 7 sections from Section 7.1
- [ ] Runtime and footprint table filled with **observed** (not estimated) values
- [ ] README covers setup through results with seed documented
- [ ] All engineering decisions from Section 9 explicitly stated in report
