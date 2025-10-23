#!/usr/bin/env python3
"""
SAHI Visualization Tool

Creates a visual preview of the SAHI tiling process used by the project:
  * Computes the slicing window layout (manual or auto) and overlap ratios
  * Draws the SAHI grid over an image alongside existing annotations
  * Extracts tiles through the real tiling engine for quick inspection
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

matplotlib.use("Agg")  # ensures compatibility with headless environments

# Make sure we can import the project modules
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from src.config.settings import TilingConfig
from src.core.tiling.engine import (
    TilingEngine,
    get_auto_slice_params,
    get_slice_bboxes,
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

TilePreview = Tuple[Image.Image, Tuple[int, int, int, int], float]


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


def draw_sahi_grid(
    image: np.ndarray,
    grid: Iterable[Tuple[int, int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
    """Draw SAHI tiles over the image."""
    colors = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]
    img_copy = image.copy()

    for idx, (x1, y1, x2, y2) in enumerate(grid):
        color = colors[idx % len(colors)]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            img_copy,
            f"{idx}",
            (x1 + 6, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return img_copy


def draw_bounding_boxes(
    image: np.ndarray,
    coco_json: str,
    image_filename: str,
    color=(255, 0, 0),
) -> np.ndarray:
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


def extract_tiles_for_preview(image: np.ndarray, engine: TilingEngine) -> List[TilePreview]:
    """Use the SAHI engine to extract tiles for inspection."""
    rgb_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tiles = list(engine.generate_tiles(rgb_image))
    print(f"Info: generated {len(tiles)} tiles for preview.")
    return tiles


def save_contact_sheet(tiles: List[TilePreview], output_path: Path, max_tiles: int = 9) -> None:
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
            tile_image, bbox, scale = tiles[idx]
            ax.imshow(tile_image)
            ax.set_title(
                f"Tile {idx}\norigin=({bbox[0]}, {bbox[1]})\nscale={scale:.3f}",
                fontsize=8,
            )
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Info: contact sheet saved to {output_path}")


def _infer_layout(grid: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    if not grid:
        return 0, 0
    sorted_grid = sorted(grid, key=lambda b: (b[1], b[0]))
    first_row_y = sorted_grid[0][1]
    cols = sum(1 for bbox in sorted_grid if bbox[1] == first_row_y)
    if cols == 0:
        return 0, 0
    rows = int(np.ceil(len(sorted_grid) / cols))
    return cols, rows


def _resolve_overlap_ratio(configured_ratio: float | None, pixel_overlap: int, tile_extent: int) -> float:
    if configured_ratio is not None:
        return max(0.0, min(configured_ratio, 1.0))
    if tile_extent <= 0 or pixel_overlap <= 0:
        return 0.0
    return max(0.0, min(pixel_overlap / tile_extent, 1.0))


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main() -> None:
    image_path = Path("dataset/train/30_jpg.rf.3ed47a5937372f408cfd0f5df94d0fad.jpg")
    coco_json = Path("dataset/train/_annotations.coco.json")
    output_dir = Path("sahi_vis")

    # Configure SAHI parameters (adjust as needed)
    use_auto_slice = True
    tile_size = (640, 640)
    overlap_pixels = 64

    tiling_config = TilingConfig(
        tile_size=tile_size,
        overlap=overlap_pixels,
        overlap_height_ratio=None,
        overlap_width_ratio=None,
        auto_slice_resolution=use_auto_slice,
        min_object_coverage=0.0,
        min_area_ratio=0.0,
        resize_output=None,
        ignore_negative_samples=True,
        verbose=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSAHI Visualization (Sliding Window)")
    print("=" * 70)
    print(f"Image path: {image_path}")
    print(f"Annotations: {coco_json}")
    print(f"Tile size (configured): {tile_size}")
    print(f"Overlap (configured pixels): {overlap_pixels}")
    print(f"Auto slice resolution: {use_auto_slice}")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: could not load image {image_path}")
        return
    height, width = image.shape[:2]
    print(f"Image size: {width} x {height}")

    # Bounding box statistics (optional but helpful for context)
    w_med, h_med = get_box_medians(str(coco_json))
    print(f"\nBounding box median width={w_med:.2f}, height={h_med:.2f}")

    # Compute SAHI slicing parameters
    if tiling_config.auto_slice_resolution:
        x_overlap_px, y_overlap_px, slice_width, slice_height = get_auto_slice_params(
            height=height, width=width
        )
        overlap_width_ratio = x_overlap_px / slice_width if slice_width else 0.0
        overlap_height_ratio = y_overlap_px / slice_height if slice_height else 0.0
    else:
        slice_width, slice_height = tiling_config.tile_size
        overlap_width_ratio = _resolve_overlap_ratio(
            tiling_config.overlap_width_ratio, tiling_config.overlap, slice_width
        )
        overlap_height_ratio = _resolve_overlap_ratio(
            tiling_config.overlap_height_ratio, tiling_config.overlap, slice_height
        )
        x_overlap_px = int(round(overlap_width_ratio * slice_width))
        y_overlap_px = int(round(overlap_height_ratio * slice_height))

    stride_x = max(1, slice_width - x_overlap_px)
    stride_y = max(1, slice_height - y_overlap_px)

    print("\nResolved SAHI slicing parameters:")
    print(f"  Slice size:      {slice_width} x {slice_height}")
    print(f"  Overlap (px):    x={x_overlap_px}, y={y_overlap_px}")
    print(f"  Overlap (ratio): x={overlap_width_ratio:.3f}, y={overlap_height_ratio:.3f}")
    print(f"  Effective stride: x={stride_x}, y={stride_y}")

    grid = get_slice_bboxes(
        image_height=height,
        image_width=width,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    nx, ny = _infer_layout(grid)
    print(f"\nGrid layout: {nx} columns x {ny} rows ({len(grid)} tiles)")

    # Draw grid alongside dataset boxes
    image_with_boxes = draw_bounding_boxes(image, str(coco_json), str(image_path))
    image_grid = draw_sahi_grid(image_with_boxes, grid, thickness=2)

    grid_path = output_dir / "sahi_grid_with_boxes.jpg"
    cv2.imwrite(str(grid_path), image_grid)
    print(f"Info: grid visualization saved to {grid_path}")

    preview_path = output_dir / "sahi_grid_preview.png"
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_grid, cv2.COLOR_BGR2RGB))
    plt.title(
        f"SAHI Grid\nTiles: {nx}x{ny}  Slice={slice_width}x{slice_height}  "
        f"Overlap={overlap_width_ratio*100:.1f}%/{overlap_height_ratio*100:.1f}%  "
        f"Stride={stride_x}x{stride_y}"
    )
    plt.axis("off")
    plt.savefig(preview_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Info: preview figure saved to {preview_path}")

    # Extract tiles using the SAHI engine
    engine = TilingEngine(tiling_config)
    tiles = extract_tiles_for_preview(image, engine)

    if tiles:
        mid_tile = tiles[len(tiles) // 2]
        tile_image, tile_bbox, _ = mid_tile
        sample_tile_path = output_dir / "sample_tile.jpg"
        sample_bgr = cv2.cvtColor(np.array(tile_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(sample_tile_path), sample_bgr)
        print(f"Info: sample tile saved to {sample_tile_path} (bbox={tile_bbox})")

        contact_path = output_dir / "sahi_tile_contact_sheet.png"
        save_contact_sheet(tiles, contact_path, max_tiles=9)
    else:
        print("Warning: no tiles were generated (check configuration).")

    print("\nDone.")


if __name__ == "__main__":
    main()
