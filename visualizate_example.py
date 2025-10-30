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
from src.core.tiling.engine import TilingEngine

BBox = Tuple[int, int, int, int]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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
    config = TilingConfig()
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

    max_overlap = int(coverage.max())
    redundant_pixels = int((coverage > 1).sum())
    total_pixels = coverage.size
    redundant_ratio = redundant_pixels / total_pixels if total_pixels else 0.0

    print(f"Generated tiles: {len(bboxes)}")
    print(f"Maximum overlap per pixel: {max_overlap}")
    print(
        f"Redundant pixels (>=2 tiles): {redundant_pixels} ({redundant_ratio:.2%} of the image)"
    )

    output_path = args.output
    if output_path is None:
        output_path = Path("visualizations") / f"{image_path.stem}_tiling_overlaps.png"

    draw_visualization(
        np.array(image), bboxes, coverage, output_path, show_labels=not args.hide_labels
    )
    print(f"Visualization written to: {output_path}")


if __name__ == "__main__":
    main()
