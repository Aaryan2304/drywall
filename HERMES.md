# HERMES.md — Drywall QA Segmentation Project

## Project Overview

**Goal:** Text-conditioned binary segmentation for construction defect QA. Given an image + natural language prompt ("segment crack" or "segment taping area"), produce a binary mask ({0, 255} PNG).

**Model:** CLIPSeg (`CIDAS/clipseg-rd64-refined`) — ViT-B/16 backbone (frozen) + lightweight decoder with FiLM conditioning.

**Domain:** computer-vision, text-conditioned segmentation, construction QA

**Hardware:** Colab T4 (16GB VRAM), batch=2, FP16 AMP, decoder + FiLM + projection unfrozen.

**Deadline:** 2 days from 2026-04-10

## Datasets

| Dataset | Path | Images | Annotations | Seg Type | Size |
|---------|------|--------|-------------|----------|------|
| Cracks | `cracks/` | 5,369 (3758/805/806) | 5,868 | Polygon | 640x640 |
| Taping | `taping/` | 1,022 (715/153/154) | 997 | Bbox only | 640x640 |

- Cracks: 16 annotations with empty segmentation → skip
- Taping: ALL annotations are bbox-only → rasterize as `cv2.rectangle`
- Splits: Roboflow exports as-is (no re-splitting)
- Seed: `42` for augmentation + prompt sampling only

## Key Decisions

1. **Roboflow splits as-is** — no re-merge/resplit; avoids merge complexity
2. **Decoder + FiLM + projection unfrozen** — T4 VRAM allows full conditioning stack
3. **Per-prompt threshold sweep** — [0.2–0.7] on val set, argmax mIoU
4. **BCE + Dice loss** (equal weight) — aligned with eval metrics
5. **Prompt variant sampling** — random prompt variant per `__getitem__` call
6. **Logits → upsample → threshold** — upsample 352→640 before thresholding

## Grading (100 pts)

- 50 pts Correctness: mIoU + Dice on test set, both prompts
- 30 pts Consistency: stable across brightness/complexity/defect-size strata
- 20 pts Presentation: README, report, tables, visual examples, seeds

## File Structure

```
.
├── HERMES.md                  # This file
├── Implementation_Plan.md     # Detailed implementation plan
├── assignment.md              # Assignment specification
├── cracks/                    # Cracks dataset (COCO JSON)
│   ├── train/                 # 3758 images + _annotations.coco.json
│   ├── valid/                 # 805 images + _annotations.coco.json
│   └── test/                  # 806 images + _annotations.coco.json
├── taping/                    # Taping dataset (COCO JSON)
│   ├── train/                 # 715 images + _annotations.coco.json
│   ├── valid/                 # 153 images + _annotations.coco.json
│   └── test/                  # 154 images + _annotations.coco.json
├── src/                       # Source code (to be created)
│   ├── data/                  # Dataset, mask conversion
│   ├── models/                # CLIPSeg wrapper, loss
│   ├── training/              # Training loop, config
│   ├── inference/             # Inference, mask export
│   └── eval/                  # Metrics, consistency analysis
├── notebooks/                 # Colab notebooks
├── outputs/                   # Masks, checkpoints, logs
├── report/                    # Report assets (figures, tables)
└── README.md                  # Project README
```

## Obsidian Vault

- **Overview:** `~/Documents/Obsidian Vault/Drywall/Overview.md`
- **Tasks:** `~/Documents/Obsidian Vault/Drywall/Tasks.md`
- **Techniques:** `~/Documents/Obsidian Vault/Drywall/Techniques/`

## Workflow

1. Read this file + `Implementation_Plan.md` at session start
2. Follow plan sections sequentially
3. After each task: update Obsidian Tasks.md, commit progress
4. On completion: run `project-memory-update` skill

## Known Issues

- Taping GT masks are bbox rectangles — expect lower Dice on taping vs cracks
- Cracks may have sub-5px thin features — may be lost at 352x352 internal resolution
- Class imbalance: cracks (5369) >> taping (1022) — monitor per-class metrics
- CLIPSeg logits uncalibrated on construction domain — threshold sweep critical

## Notes

- Colab notebook should be self-contained with install cells
- All code seed-locked to `42`
- Model shouldn't look AI-generated — clean, minimal, professional
- Report must include runtime/footprint table with **observed** values, not estimates
