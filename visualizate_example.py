#!/usr/bin/env python3
"""
Tiling layout visualizer.

Draws tile borders and highlights redundancy (overlap >= 2) to audit the slicing
configuration for a single image.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from src.config.settings import TilingConfig
from src.core.tiling.engine import TilingEngine, get_auto_slice_params

BBox = Tuple[int, int, int, int]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_TILE_SIZE = (640, 640)
DEFAULT_OVERLAP = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize generated tiles with boundary boxes and overlap coverage."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Source image to inspect (optional; picks a random image when omitted).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path for the generated visualization (defaults next to the image).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Override tile dimensions when auto-slice is disabled.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="Override pixel overlap when auto-slice is disabled.",
    )
    parser.add_argument(
        "--auto-slice",
        action="store_true",
        help="Enable SAHI-style automatic slicing.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Do not render tile indices on top of the image.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory recursively searched for an image when none is provided (default: ./dataset).",
    )
    return parser.parse_args()


def build_engine(args: argparse.Namespace) -> TilingEngine:
    config = TilingConfig(
        tile_size=DEFAULT_TILE_SIZE,
        overlap=DEFAULT_OVERLAP,
    )
    if args.auto_slice:
        config.auto_slice_resolution = True
    if args.tile_size:
        config.tile_size = (max(args.tile_size[0], 1), max(args.tile_size[1], 1))
    if args.overlap is not None:
        config.overlap = max(args.overlap, 0)
        config.overlap_height_ratio = None
        config.overlap_width_ratio = None
    return TilingEngine(config)


def compute_coverage(bboxes: List[BBox], height: int, width: int) -> np.ndarray:
    coverage = np.zeros((height, width), dtype=np.uint16)
    for xmin, ymin, xmax, ymax in bboxes:
        coverage[ymin:ymax, xmin:xmax] += 1
    return coverage


def _resolve_overlap_ratio(configured_ratio: float | None, pixel_overlap: int, tile_extent: int) -> float:
    if configured_ratio is not None:
        return max(0.0, min(configured_ratio, 1.0))
    if tile_extent <= 0 or pixel_overlap <= 0:
        return 0.0
    return max(0.0, min(pixel_overlap / tile_extent, 1.0))


def resolve_slice_params(engine: TilingEngine, image_width: int, image_height: int) -> dict:
    config = engine.config
    if config.auto_slice_resolution:
        x_overlap_px, y_overlap_px, slice_width, slice_height = get_auto_slice_params(
            height=image_height, width=image_width
        )
        overlap_width_ratio = x_overlap_px / slice_width if slice_width else 0.0
        overlap_height_ratio = y_overlap_px / slice_height if slice_height else 0.0
    else:
        slice_width, slice_height = config.tile_size
        overlap_width_ratio = _resolve_overlap_ratio(config.overlap_width_ratio, config.overlap, slice_width)
        overlap_height_ratio = _resolve_overlap_ratio(config.overlap_height_ratio, config.overlap, slice_height)
        x_overlap_px = int(round(overlap_width_ratio * slice_width))
        y_overlap_px = int(round(overlap_height_ratio * slice_height))

    stride_x = max(1, slice_width - x_overlap_px)
    stride_y = max(1, slice_height - y_overlap_px)

    return {
        "slice_width": slice_width,
        "slice_height": slice_height,
        "overlap_x_px": x_overlap_px,
        "overlap_y_px": y_overlap_px,
        "overlap_x_ratio": overlap_width_ratio,
        "overlap_y_ratio": overlap_height_ratio,
        "stride_x": stride_x,
        "stride_y": stride_y,
    }


def infer_layout(bboxes: List[BBox]) -> Tuple[int, int]:
    if not bboxes:
        return 0, 0
    sorted_grid = sorted(bboxes, key=lambda b: (b[1], b[0]))
    first_row_y = sorted_grid[0][1]
    cols = sum(1 for bbox in sorted_grid if bbox[1] == first_row_y)
    if cols == 0:
        return 0, 0
    rows = (len(sorted_grid) + cols - 1) // cols
    return cols, rows


def compute_redundancy_stats(coverage: np.ndarray) -> Tuple[int, float, int]:
    redundant_pixels = int((coverage > 1).sum())
    total_pixels = coverage.size
    redundant_ratio = redundant_pixels / total_pixels if total_pixels else 0.0
    max_overlap = int(coverage.max()) if coverage.size else 0
    return redundant_pixels, redundant_ratio, max_overlap


def choose_random_image(source_dir: Path) -> Path:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    candidates = [
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No image files with extensions {sorted(SUPPORTED_EXTENSIONS)} found under {source_dir}"
        )

    selected = random.choice(candidates)
    print(f"No image provided. Selected random image: {selected}")
    return selected


def draw_visualization(
    image_array: np.ndarray,
    bboxes: List[BBox],
    coverage: np.ndarray,
    output_path: Path,
    show_labels: bool,
    metrics: dict,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].imshow(image_array)
    axes[0].set_title("Tiles with highlighted redundancy")
    axes[0].axis("off")

    redundant_mask = np.ma.masked_less(coverage, 2)
    axes[0].imshow(redundant_mask, cmap="magma", alpha=0.35, interpolation="nearest")

    for idx, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
        width = xmax - xmin
        height = ymax - ymin
        rect = Rectangle(
            (xmin, ymin), width, height, linewidth=1.5, edgecolor="#4CAF50", facecolor="none"
        )
        axes[0].add_patch(rect)
        if show_labels:
            axes[0].text(
                xmin + 4,
                ymin + 16,
                str(idx),
                color="#ffeb3b",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="center",
                bbox={"facecolor": "black", "alpha": 0.4, "pad": 2},
            )

    im = axes[1].imshow(coverage, cmap="viridis", interpolation="nearest")
    axes[1].set_title("Coverage heatmap (tiles per pixel)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="count")

    metrics_lines = [
        f"Tiles: {metrics['tile_count']} ({metrics['layout'][0]}x{metrics['layout'][1]})",
        (
            f"Tile: {metrics['tile_size'][0]}x{metrics['tile_size'][1]} | "
            f"Overlap: {metrics['overlap'][0]}px/{metrics['overlap'][1]}px | "
            f"Stride: {metrics['stride'][0]}x{metrics['stride'][1]}"
        ),
        f"Redundant pixels: {metrics['redundant_pixels']} ({metrics['redundant_ratio']:.2%})",
        f"Max per-pixel overlap: {metrics['max_overlap']}x",
    ]
    fig.text(0.5, 0.02, "\n".join(metrics_lines), ha="center", va="bottom", fontsize=10)

    fig.suptitle("Tiling diagnostics", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    image_path = args.image
    if image_path is None:
        image_path = choose_random_image(args.source_dir)
    elif not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    engine = build_engine(args)
    bboxes = engine._compute_slice_bboxes(image.width, image.height)
    coverage = compute_coverage(bboxes, image.height, image.width)

    params = resolve_slice_params(engine, image.width, image.height)
    layout = infer_layout(bboxes)
    redundant_pixels, redundant_ratio, max_overlap = compute_redundancy_stats(coverage)

    print(f"Generated tiles: {len(bboxes)}")
    print(f"Grid layout (cols x rows): {layout[0]} x {layout[1]}")
    print(f"Tile size: {params['slice_width']}x{params['slice_height']}")
    print(f"Overlap (px): x={params['overlap_x_px']}, y={params['overlap_y_px']}")
    print(f"Stride: x={params['stride_x']}, y={params['stride_y']}")
    print(f"Maximum overlap per pixel: {max_overlap}")
    print(f"Redundant pixels (>=2 tiles): {redundant_pixels} ({redundant_ratio:.2%} of the image)")

    output_path = args.output
    if output_path is None:
        output_path = Path("visualizations") / f"{image_path.stem}_tiling_overlaps.png"

    draw_visualization(
        np.array(image),
        bboxes,
        coverage,
        output_path,
        show_labels=not args.hide_labels,
        metrics={
            "tile_count": len(bboxes),
            "layout": layout,
            "tile_size": (params["slice_width"], params["slice_height"]),
            "overlap": (params["overlap_x_px"], params["overlap_y_px"]),
            "stride": (params["stride_x"], params["stride_y"]),
            "redundant_pixels": redundant_pixels,
            "redundant_ratio": redundant_ratio,
            "max_overlap": max_overlap,
        },
    )
    print(f"Visualization written to: {output_path}")


if __name__ == "__main__":
    main()
