#!/usr/bin/env python3
"""
SAGE Visualization Tool (Adaptive Overlap + Context Padding)

Produces a complete preview of the SAGE pipeline:
  * Computes adaptive overlap (O*)
  * Draws the stride-aligned grid with context padding
  * Extracts tiles using the same engine logic (including padding)
  * Saves a sample tile and a contact sheet for quick inspection
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")  # keep headless environments happy

# Make sure we can import the project modules
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from src.config.settings import TilingConfig
from src.core.tiling.engine import GeneratedTile, TilingEngine
from src.core.tiling.sage import compute_sage_grid


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _roboflow_aliases(filename: str) -> set[str]:
    """Return possible original filenames for Roboflow-hashed assets."""
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


def _find_coco_image_entry(data: dict, image_filename: str) -> dict | None:
    """Locate the COCO image entry corresponding to image_filename."""
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


def get_box_medians(coco_json: str) -> Tuple[float, float]:
    """Compute median width and height of bounding boxes from COCO annotations."""
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    widths: List[float] = []
    heights: List[float] = []
    for ann in data["annotations"]:
        _, _, w, h = ann["bbox"]
        widths.append(w)
        heights.append(h)

    if not widths or not heights:
        raise ValueError("No bounding boxes found in COCO JSON.")

    w_med = float(np.median(widths))
    h_med = float(np.median(heights))
    return w_med, h_med


def draw_sage_grid(
    image: np.ndarray,
    grid: Iterable[Tuple[int, int, int, int]],
    context_pad: int = 0,
) -> np.ndarray:
    """Draw SAGE tiles and optional context padding over the image."""
    colors = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]
    grid_list = list(grid)
    img_copy = image.copy()
    height, width = image.shape[:2]
    overlay = np.zeros_like(image)

    for idx, (x1, y1, x2, y2) in enumerate(grid_list):
        color = colors[idx % len(colors)]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        if context_pad > 0:
            pad_x1 = max(0, x1 - context_pad)
            pad_y1 = max(0, y1 - context_pad)
            pad_x2 = min(width, x2 + context_pad)
            pad_y2 = min(height, y2 + context_pad)
            pad_color = tuple(int(c * 0.55) for c in color)
            # Fill overlay with the pad color and carve out the tile interior
            cv2.rectangle(overlay, (pad_x1, pad_y1), (pad_x2, pad_y2), pad_color, -1)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.rectangle(img_copy, (pad_x1, pad_y1), (pad_x2, pad_y2), pad_color, 1)

    if context_pad > 0 and grid_list:
        img_copy = cv2.addWeighted(overlay, 0.35, img_copy, 1.0, 0.0)
        x1, y1, x2, y2 = grid_list[0]
        pad_x1 = max(0, x1 - context_pad)
        pad_y1 = max(0, y1 - context_pad)
        label_pos = (pad_x1 + 10, max(20, pad_y1 - 10))
        arrow_end = (pad_x1 + 10, pad_y1 + 10)
        cv2.putText(
            img_copy,
            "Context padding area",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (250, 250, 250),
            2,
            cv2.LINE_AA,
        )
        cv2.arrowedLine(
            img_copy,
            (label_pos[0] + 10, label_pos[1] + 5),
            arrow_end,
            (250, 250, 250),
            2,
            cv2.LINE_AA,
            tipLength=0.1,
        )

    return img_copy


def draw_bounding_boxes(image: np.ndarray, coco_json: str, image_filename: str, color=(255, 0, 0)) -> np.ndarray:
    """Draw all bounding boxes for one image."""
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_entry = _find_coco_image_entry(data, image_filename)
    if img_entry is None:
        print(f"Warning: {os.path.basename(image_filename)} not found in annotations.")
        return image

    anns = [a for a in data["annotations"] if a["image_id"] == img_entry["id"]]
    img_copy = image.copy()
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(img_copy, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

    print(f"Info: drew {len(anns)} boxes for {os.path.basename(image_filename)}.")
    return img_copy


def extract_tiles_for_preview(
    image: np.ndarray,
    overlap_ratio: float,
    tile_size: Tuple[int, int],
    context_pad: int,
) -> List[GeneratedTile]:
    """Use the same engine logic to extract tiles with context padding."""
    tiling_config = TilingConfig(
        tile_size=tile_size,
        overlap=0,
        overlap_ratio=overlap_ratio,
        context_pad=context_pad,
        min_object_coverage=0.0,
        resize_output=None,
        mode="sage",
        keep_empty_tiles=True,
    )
    engine = TilingEngine(tiling_config)

    rgb_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tiles = list(engine.generate_tiles(rgb_image))
    print(f"Info: generated {len(tiles)} tiles for preview.")
    return tiles


def save_contact_sheet(tiles: List[GeneratedTile], output_path: Path, max_tiles: int = 9) -> None:
    """Create a contact sheet with the first N tiles."""
    if not tiles:
        return

    num_tiles = min(max_tiles, len(tiles))
    cols = 3
    rows = int(np.ceil(num_tiles / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.4))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes.flat[idx]
        if idx < num_tiles:
            tile = tiles[idx]
            ax.imshow(tile.image)
            origin = tile.grid_origin or tile.origin
            ax.set_title(f"Tile {idx}\norig={origin}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Info: contact sheet saved to {output_path}")


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main() -> None:
    image_path = Path("dataset/train/9.jpg")
    coco_json = Path("dataset/train/_annotations.coco.json")
    output_dir = Path("sage_vis")
    tile_size = (640, 640)
    context_pad = 64

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSAGE Visualization (Adaptive Overlap + Context Pad)")
    print("=" * 70)
    print(f"Image path: {image_path}")
    print(f"Annotations: {coco_json}")
    print(f"Tile size: {tile_size}")
    print(f"Context padding: {context_pad}px")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: could not load image {image_path}")
        return
    height, width = image.shape[:2]
    print(f"Image size: {width} x {height}")

    # Step 1: bounding box statistics
    w_med, h_med = get_box_medians(str(coco_json))
    A_med = w_med * h_med
    d = np.sqrt(A_med)
    print(f"\nBounding box median width={w_med:.2f}, height={h_med:.2f}")
    print(f"Equivalent side (sqrt(w*h)) = {d:.2f}px")

    # Step 2: adaptive overlap
    r = d / tile_size[0]
    overlap_ratio = 0.5 * r
    print(f"Ratio r = d/B = {r:.4f}")
    print(f"Adaptive overlap O* = {overlap_ratio:.4f} ({overlap_ratio*100:.2f}%)")

    # Step 3: compute SAGE grid
    grid_data = compute_sage_grid((height, width), tile_size, overlap_ratio)
    grid = grid_data.boxes
    stride = grid_data.stride
    nx, ny = grid_data.layout
    print(f"\nGrid layout: {nx} x {ny} tiles  |  stride = ({stride[0]:.2f}, {stride[1]:.2f})")

    # Step 4: draw grid with context padding
    image_with_boxes = draw_bounding_boxes(image, str(coco_json), str(image_path))
    image_grid = draw_sage_grid(image_with_boxes, grid, context_pad=context_pad)

    grid_path = output_dir / "sage_adaptive_grid.jpg"
    cv2.imwrite(str(grid_path), image_grid)
    print(f"Info: grid visualization saved to {grid_path}")

    preview_path = output_dir / "sage_adaptive_grid_preview.png"
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_grid, cv2.COLOR_BGR2RGB))
    plt.title(
        f"SAGE Adaptive Grid (O*={overlap_ratio*100:.2f}%)\n"
        f"Tiles: {nx}x{ny}  Tile={tile_size[0]}x{tile_size[1]}  "
        f"Pad={context_pad}px  Stride={stride[0]:.1f}x{stride[1]:.1f}"
    )
    plt.axis("off")
    plt.savefig(preview_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Info: preview figure saved to {preview_path}")

    # Step 5: extract tiles using the real engine
    tiles = extract_tiles_for_preview(image, overlap_ratio, tile_size, context_pad)

    if tiles:
        mid_tile = tiles[len(tiles) // 2]
        sample_tile_path = output_dir / "sample_tile_with_context.jpg"
        sample_bgr = cv2.cvtColor(np.array(mid_tile.image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(sample_tile_path), sample_bgr)
        print(
            f"Info: sample tile saved to {sample_tile_path} "
            f"(origin={mid_tile.origin}, grid_origin={mid_tile.grid_origin})"
        )

        contact_path = output_dir / "sage_tile_contact_sheet.png"
        save_contact_sheet(tiles, contact_path, max_tiles=9)
    else:
        print("Warning: no tiles were generated (check configuration).")

    print("\nDone.")


if __name__ == "__main__":
    main()
