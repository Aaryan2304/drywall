"""
COCO JSON annotations → binary mask rasterization.

Cracks: polygon segmentation → cv2.fillPoly
Taping: bbox-only → cv2.rectangle (geometric proxy for linear tape seams)
"""

import json
from pathlib import Path

import cv2
import numpy as np


def _rasterize_polygon(segmentation: list, height: int, width: int) -> np.ndarray:
    """Rasterize COCO polygon segmentation into a binary mask.

    Args:
        segmentation: List of flat coordinate arrays [x1,y1,x2,y2,...].
        height: Image height.
        width: Image width.

    Returns:
        uint8 mask with values {0, 255}.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly_flat in segmentation:
        pts = np.array(poly_flat, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _rasterize_bbox(bbox: list, height: int, width: int) -> np.ndarray:
    """Rasterize COCO bbox [x, y, w, h] as a filled rectangle.

    Args:
        bbox: [x, y, w, h] in COCO format (float coordinates).
        height: Image height.
        width: Image width.

    Returns:
        uint8 mask with values {0, 255}.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y, w, h = bbox
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def generate_masks(
    annotation_path: str | Path,
    output_dir: str | Path,
    mode: str = "polygon",
) -> dict:
    """Rasterize COCO annotations into binary masks.

    Args:
        annotation_path: Path to _annotations.coco.json.
        output_dir: Directory to write mask PNGs.
        mode: "polygon" for polygon segmentation, "bbox" for bounding box.

    Returns:
        Dict with stats: total annotations, skipped, masks written.
    """
    annotation_path = Path(annotation_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    written = 0
    skipped = 0

    # Group annotations by image_id
    anns_by_image: dict[int, list] = {}
    for ann in annotations:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    for img_id, img_info in images.items():
        h, w = img_info["height"], img_info["width"]
        filename = Path(img_info["file_name"]).stem

        mask = np.zeros((h, w), dtype=np.uint8)
        img_anns = anns_by_image.get(img_id, [])

        if not img_anns:
            # Image with no annotations — write empty mask
            cv2.imwrite(str(output_dir / f"{filename}.png"), mask)
            written += 1
            continue

        for ann in img_anns:
            if mode == "polygon":
                seg = ann.get("segmentation", [])
                if not seg:
                    skipped += 1
                    continue
                ann_mask = _rasterize_polygon(seg, h, w)
                mask = np.maximum(mask, ann_mask)
            elif mode == "bbox":
                bbox = ann.get("bbox", [])
                if not bbox:
                    skipped += 1
                    continue
                ann_mask = _rasterize_bbox(bbox, h, w)
                mask = np.maximum(mask, ann_mask)

        cv2.imwrite(str(output_dir / f"{filename}.png"), mask)
        written += 1

    return {
        "total_annotations": len(annotations),
        "skipped": skipped,
        "masks_written": written,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="COCO → binary mask rasterization")
    parser.add_argument("annotation", help="Path to _annotations.coco.json")
    parser.add_argument("output", help="Output directory for masks")
    parser.add_argument(
        "--mode",
        choices=["polygon", "bbox"],
        default="polygon",
        help="Rasterization mode",
    )
    args = parser.parse_args()

    stats = generate_masks(args.annotation, args.output, args.mode)
    print(f"Done: {stats['masks_written']} masks written, {stats['skipped']} annotations skipped")
