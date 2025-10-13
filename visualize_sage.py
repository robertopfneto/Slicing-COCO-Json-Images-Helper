#!/usr/bin/env python3
"""
SAGE Visualization Tool (Adaptive Overlap)

Visualizes how the SAGE (Stride-Aligned Grid Extraction) would divide
a high-resolution image into stride-aligned tiles. Calculates the
ideal overlap automatically based on bounding box statistics
(median width and height) from a COCO dataset.
"""

import json
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # evita erro Qt headless
import matplotlib.pyplot as plt


# ===================== SAGE CORE ===================== #

def compute_sage_stride(image_shape, block_size=(640, 640), overlap=0.03, mode="align"):
    """Compute stride-aligned coordinates for SAGE tiling."""
    H, W = image_shape[:2]
    Bw, Bh = block_size

    # Initial stride
    Sx = Bw * (1 - overlap)
    Sy = Bh * (1 - overlap)

    # Number of tiles
    nx = int(np.ceil((W - Bw) / Sx)) + 1
    ny = int(np.ceil((H - Bh) / Sy)) + 1

    # Adjust stride to close exactly on borders
    if mode == "align" and nx > 1 and ny > 1:
        Sx = (W - Bw) / (nx - 1)
        Sy = (H - Bh) / (ny - 1)

    grid = []
    for j in range(ny):
        for i in range(nx):
            x1 = int(round(i * Sx))
            y1 = int(round(j * Sy))
            x2 = min(x1 + Bw, W)
            y2 = min(y1 + Bh, H)
            grid.append((x1, y1, x2, y2))

    return grid, (Sx, Sy), (nx, ny)


def draw_sage_grid(image, grid, color=(0, 255, 0), thickness=2):
    """Draw stride-aligned grid over the image."""
    img_copy = image.copy()
    for (x1, y1, x2, y2) in grid:
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy


# ===================== COCO UTILITIES ===================== #

def _roboflow_aliases(filename):
    """
    Generate possible original filenames for Roboflow-hashed assets like
    22_jpg.rf.<hash>.jpg → {22.jpg}. Returns aliases (may be empty).
    """
    name = os.path.basename(filename)
    if ".rf." not in name:
        return set()

    prefix, _ = name.split(".rf.", 1)
    if "_" not in prefix:
        return set()

    stem, ext_tag = prefix.rsplit("_", 1)
    ext_lower = ext_tag.lower()
    valid_exts = {"jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"}
    if ext_lower not in valid_exts:
        return set()

    candidates = {
        f"{stem}.{ext_lower}",
        f"{stem}.{ext_tag}",
    }
    return {candidate for candidate in candidates if candidate != name}


def _find_coco_image_entry(data, image_filename):
    """
    Locate the COCO image entry corresponding to image_filename, supporting
    Roboflow hashed names and the optional extra.name metadata.
    """
    basename = os.path.basename(image_filename)
    candidates = {basename} | _roboflow_aliases(basename)

    for entry in data.get("images", []):
        entry_names = {entry.get("file_name")}
        extra_name = entry.get("extra", {}).get("name")
        if extra_name:
            entry_names.add(os.path.basename(extra_name))

        names = {name for name in entry_names if name}
        if candidates & names:
            return entry

    return None


def get_box_medians(coco_json):
    """Compute median width and height of bounding boxes from COCO annotations."""
    with open(coco_json, 'r') as f:
        data = json.load(f)

    widths, heights = [], []
    for ann in data["annotations"]:
        _, _, w, h = ann["bbox"]
        widths.append(w)
        heights.append(h)

    if not widths or not heights:
        raise ValueError("No bounding boxes found in COCO JSON!")

    w_med = float(np.median(widths))
    h_med = float(np.median(heights))
    return w_med, h_med


def draw_bounding_boxes(image, coco_json, image_filename, color=(255, 0, 0)):
    """Draw all bounding boxes for one image."""
    with open(coco_json, 'r') as f:
        data = json.load(f)

    img_entry = _find_coco_image_entry(data, image_filename)
    if img_entry is None:
        print(f"⚠️ {os.path.basename(image_filename)} não encontrada nas anotações.")
        return image

    anns = [a for a in data["annotations"] if a["image_id"] == img_entry["id"]]
    img_copy = image.copy()
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(img_copy, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

    print(f"✅ {len(anns)} boxes desenhadas para {os.path.basename(image_filename)}")
    return img_copy


# ===================== MAIN SCRIPT ===================== #

def main():
    image_path = "dataset/train/22_jpg.rf.5eabd33caec9acd4fe43e26018b32ece.jpg"
    coco_json = "dataset/train/_annotations.coco.json"
    output_dir = "sage_vis"
    os.makedirs(output_dir, exist_ok=True)

    print("\n📸 SAGE Visualization (Adaptive Overlap)")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Annotations: {coco_json}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image {image_path}")
        return
    H, W = image.shape[:2]
    B = 640  # tile size (assuming square)
    print(f"\nImage size: {W} × {H}")
    print(f"Tile size: {B} × {B}")

    # === 1️⃣ Compute median box dimensions ===
    w_med, h_med = get_box_medians(coco_json)
    A_med = w_med * h_med
    d = np.sqrt(A_med)
    print(f"\nMedian box width = {w_med:.2f}, height = {h_med:.2f}")
    print(f"Equivalent side d = √(w×h) = {d:.2f}px")

    # === 2️⃣ Compute adaptive overlap ===
    r = d / B
    O_star = 0.5 * r
    print(f"Proportion r = d/B = {r:.4f}")
    print(f"Adaptive overlap O* = 0.5 × r = {O_star:.4f} ({O_star*100:.2f}%)")

    # === 3️⃣ Compute SAGE grid using adaptive overlap ===
    grid, stride, (nx, ny) = compute_sage_stride(image.shape, (B, B), O_star, "align")
    print(f"\nGrid: {nx} × {ny} tiles (stride={stride})")

    # === 4️⃣ Draw results ===
    image_boxes = draw_bounding_boxes(image, coco_json, image_path, color=(255, 0, 0))
    image_grid = draw_sage_grid(image_boxes, grid, color=(0, 255, 0), thickness=2)

    out_path = os.path.join(output_dir, "sage_adaptive_grid.jpg")
    cv2.imwrite(out_path, image_grid)

    print(f"\n✅ Visualization saved at: {out_path}")
    print("\nLegend:")
    print("🟥 = bounding boxes (objects)")
    print("🟩 = stride-aligned SAGE tiles\n")

    # Optional save preview
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_grid, cv2.COLOR_BGR2RGB))
    plt.title(
        f"SAGE Adaptive Grid (O*={O_star*100:.2f}%) - {nx}x{ny} tiles - tile={B}x{B}px"
    )
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "sage_adaptive_grid_preview.png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
