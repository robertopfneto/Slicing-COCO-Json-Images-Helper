#!/usr/bin/env python3
"""
SAHI / ASAHI Visualization Tool

Creates a visual preview of the tiling process used by the project:
  * Computes the slicing plan (SAHI or ASAHI adaptive)
  * Draws the grid alongside existing annotations
  * Extracts tiles through the real tiling engine for inspection
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

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
# CLI helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SAHI / ASAHI tiling layout with annotations."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("dataset/train/30_jpg.rf.3ed47a5937372f408cfd0f5df94d0fad.jpg"),
        help="Path to the image to visualize.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("dataset/train/_annotations.coco.json"),
        help="COCO annotations file that contains the image entry.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sahi_vis"),
        help="Directory where visualization artifacts are written.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs=2,
        default=(640, 640),
        metavar=("WIDTH", "HEIGHT"),
        help="Manual SAHI tile size (ignored when --adaptive is used).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Manual SAHI overlap in pixels (ignored when --adaptive is used).",
    )
    parser.add_argument(
        "--auto-slice",
        action="store_true",
        help="Enable SAHI auto slice heuristics.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable ASAHI adaptive slicing mode.",
    )
    parser.add_argument(
        "--restrict-size",
        type=int,
        default=512,
        help="Base restrict size used for ASAHI LS calculation.",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.15,
        help="Overlap ratio (0-1) used in ASAHI mode.",
    )
    parser.add_argument(
        "--ls-threshold",
        type=float,
        help="Override ASAHI LS threshold (auto-computed when omitted).",
    )
    parser.add_argument(
        "--cluster-diou-nms",
        action="store_true",
        help="Flag ASAHI visualizations that use Cluster-DIoU-NMS.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=9,
        help="Maximum number of tiles in the contact sheet.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional debugging information.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TilingConfig:
    ls_threshold = (
        args.ls_threshold
        if args.ls_threshold is not None
        else args.restrict_size * (4 - 3 * args.overlap_ratio) + 1
    )

    return TilingConfig(
        tile_size=tuple(args.tile_size),
        overlap=args.overlap,
        overlap_height_ratio=None,
        overlap_width_ratio=None,
        auto_slice_resolution=args.auto_slice and not args.adaptive,
        min_object_coverage=0.0,
        min_area_ratio=0.0,
        output_format="COCO",
        resize_output=None,
        ignore_negative_samples=True,
        verbose=args.verbose or args.adaptive,
        adaptive_mode=args.adaptive,
        restrict_size=args.restrict_size,
        overlap_ratio=args.overlap_ratio,
        ls_threshold=ls_threshold,
        cluster_diou_nms=args.cluster_diou_nms,
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


def get_box_medians(coco_json: Path) -> Tuple[float, float]:
    """Compute median width and height of bounding boxes from COCO annotations."""
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    widths: List[float] = []
    heights: List[float] = []
    for ann in data.get("annotations", []):
        _, _, w, h = ann["bbox"]
        widths.append(w)
        heights.append(h)

    if not widths or not heights:
        raise ValueError("No bounding boxes found in COCO JSON.")

    w_med = float(np.median(widths))
    h_med = float(np.median(heights))
    return w_med, h_med


def draw_tiling_grid(
    image: np.ndarray,
    grid: Iterable[Tuple[int, int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
    """Draw tiles over the image."""
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
    coco_json: Path,
    image_filename: Path,
    color=(255, 0, 0),
) -> np.ndarray:
    """Draw all bounding boxes for one image."""
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_entry = _find_coco_image_entry(data, str(image_filename))
    if img_entry is None:
        print(f"Warning: {image_filename.name} not found in annotations.")
        return image

    anns = [a for a in data.get("annotations", []) if a["image_id"] == img_entry["id"]]
    img_copy = image.copy()
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(img_copy, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

    print(f"Info: drew {len(anns)} boxes for {image_filename.name}.")
    return img_copy


def extract_tiles_for_preview(image: np.ndarray, engine: TilingEngine) -> List[TilePreview]:
    """Use the tiling engine to extract tiles for inspection."""
    rgb_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tiles = list(engine.generate_tiles(rgb_image))
    print(f"Info: generated {len(tiles)} tiles for preview.")
    return tiles


def save_contact_sheet(
    tiles: List[TilePreview], output_path: Path, max_tiles: int = 9
) -> None:
    """Create a contact sheet with the first N tiles."""
    if not tiles:
        return

    num_tiles = min(max_tiles, len(tiles))
    cols = min(3, max(1, int(np.ceil(np.sqrt(num_tiles)))))
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


def infer_layout(tile_bboxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    """Infer grid layout (cols, rows) from tile bounding boxes."""
    if not tile_bboxes:
        return (0, 0)
    sorted_tiles = sorted(tile_bboxes, key=lambda b: (b[1], b[0]))
    first_row_y = sorted_tiles[0][1]
    cols = sum(1 for bbox in sorted_tiles if bbox[1] == first_row_y)
    if cols == 0:
        return (0, 0)
    rows = (len(sorted_tiles) + cols - 1) // cols
    return (cols, rows)


def compute_stride(tile_bboxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    """Estimate stride along x/y based on tile coordinates."""
    if not tile_bboxes:
        return (0, 0)

    xs = sorted({bbox[0] for bbox in tile_bboxes})
    ys = sorted({bbox[1] for bbox in tile_bboxes})

    stride_x = min((xs[i + 1] - xs[i]) for i in range(len(xs) - 1)) if len(xs) > 1 else 0
    stride_y = min((ys[i + 1] - ys[i]) for i in range(len(ys) - 1)) if len(ys) > 1 else 0
    return stride_x, stride_y


# ---------------------------------------------------------------------------
# Main visualization flow
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.annotations.exists():
        raise FileNotFoundError(f"Annotations not found: {args.annotations}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tiling_config = build_config(args)
    engine = TilingEngine(tiling_config)

    mode_label = "ASAHI adaptive" if tiling_config.adaptive_mode else "SAHI"
    print(f"\n{mode_label.upper()} Visualization")
    print("=" * 70)
    print(f"Image path: {args.image}")
    print(f"Annotations: {args.annotations}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Error: could not load image {args.image}")
    height, width = image.shape[:2]
    print(f"Image size: {width} x {height}")

    # Bounding box statistics
    w_med, h_med = get_box_medians(args.annotations)
    print(f"\nBounding box median width={w_med:.2f}, height={h_med:.2f}")

    # Generate tiles via the engine
    tiles = extract_tiles_for_preview(image, engine)
    tile_bboxes = [tuple(map(int, bbox)) for _, bbox, _ in tiles]
    layout_cols, layout_rows = infer_layout(tile_bboxes)
    stride_x, stride_y = compute_stride(tile_bboxes)

    plan_summary = engine.get_last_plan_summary()
    print("\nResolved tiling parameters:")
    if tiling_config.adaptive_mode and plan_summary:
        print(f"  Mode: ASAHI ({plan_summary.get('layout', 'unknown')})")
        print(f"  Tile count: {plan_summary.get('actual_total', 0)}")
        print(f"  Tile size: {plan_summary.get('tile_length', 0)} px square")
        print(f"  Overlap ratio: {plan_summary.get('overlap_ratio', 0.0):.3f}")
        print(f"  Overlap px: {plan_summary.get('overlap_px', 0)}")
        print(f"  Stride: {plan_summary.get('stride', 0)}")
        redundancy = plan_summary.get("redundancy_reduction", 0.0) * 100
        print(f"  Estimated redundancy reduction: {redundancy:.1f}%")
        print(f"  LS threshold: {plan_summary.get('ls_threshold', 0.0):.2f}")
    else:
        if tiling_config.auto_slice_resolution:
            x_overlap_px, y_overlap_px, slice_width, slice_height = get_auto_slice_params(
                height=height, width=width
            )
        else:
            slice_width, slice_height = tiling_config.tile_size
            overlap_width_ratio = (
                tiling_config.overlap / slice_width if slice_width else 0.0
            )
            overlap_height_ratio = (
                tiling_config.overlap / slice_height if slice_height else 0.0
            )
            x_overlap_px = int(round(overlap_width_ratio * slice_width))
            y_overlap_px = int(round(overlap_height_ratio * slice_height))

        stride_x = max(1, slice_width - x_overlap_px)
        stride_y = max(1, slice_height - y_overlap_px)

        print("  Mode: SAHI")
        print(f"  Slice size: {slice_width} x {slice_height}")
        print(f"  Overlap (px): x={x_overlap_px}, y={y_overlap_px}")
        print(
            "  Overlap (ratio): "
            f"x={x_overlap_px / slice_width if slice_width else 0:.3f}, "
            f"y={y_overlap_px / slice_height if slice_height else 0:.3f}"
        )
        print(f"  Effective stride: x={stride_x}, y={stride_y}")

    print(f"\nGrid layout: {layout_cols} columns x {layout_rows} rows ({len(tile_bboxes)} tiles)")

    # Draw grid and annotations
    image_with_boxes = draw_bounding_boxes(image, args.annotations, args.image)
    image_grid = draw_tiling_grid(image_with_boxes, tile_bboxes, thickness=2)

    grid_path = output_dir / "tiling_grid_with_boxes.jpg"
    cv2.imwrite(str(grid_path), image_grid)
    print(f"Info: grid visualization saved to {grid_path}")

    preview_path = output_dir / "tiling_grid_preview.png"
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_grid, cv2.COLOR_BGR2RGB))
    title_parts = [
        f"{mode_label.upper()} Grid",
        f"Tiles: {layout_cols}x{layout_rows}",
        f"Stride: {stride_x}x{stride_y}",
    ]
    if tiling_config.adaptive_mode and plan_summary:
        title_parts.append(
            f"Overlap: {plan_summary.get('overlap_ratio', 0.0)*100:.1f}%"
        )
        title_parts.append(
            f"Redundancy↓ {plan_summary.get('redundancy_reduction', 0.0)*100:.1f}%"
        )
    plt.title("  |  ".join(title_parts))
    plt.axis("off")
    plt.savefig(preview_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Info: preview figure saved to {preview_path}")

    # Contact sheet
    tiles_output = output_dir / "tiling_contact_sheet.png"
    save_contact_sheet(tiles, tiles_output, max_tiles=args.max_tiles)

    print("\nDone.")


if __name__ == "__main__":
    main()
