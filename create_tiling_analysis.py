#!/usr/bin/env python3
"""
SAGE tiling analysis tool.

Mirrors the step-by-step flow of visualize_sage.py, but produces annotated
overviews for one or more dataset images using the SAGE tiling strategy.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Ensure project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from src.config.settings import AppConfig, TilingConfig  # noqa: E402
from src.core.tiling.engine import TilingEngine  # noqa: E402
from src.core.tiling.sage import (  # noqa: E402
    SageGrid,
    compute_adaptive_overlap,
    compute_sage_grid,
)
from src.models.coco import CocoAnnotation, CocoDataset, CocoImage  # noqa: E402
from src.utils.visualization import BoundingBoxVisualizer  # noqa: E402


def _compute_dataset_box_medians(dataset: CocoDataset) -> Tuple[float, float]:
    widths: List[float] = []
    heights: List[float] = []
    for ann in dataset.annotations:
        x, y, w, h = ann.bbox
        if w > 0 and h > 0:
            widths.append(float(w))
            heights.append(float(h))

    if not widths or not heights:
        return 0.0, 0.0

    return float(np.median(widths)), float(np.median(heights))


def _resolve_overlap_ratio(
    config: TilingConfig,
    dataset: CocoDataset,
    image_filter: Optional[Sequence[int]],
    tile_size: Tuple[int, int],
) -> float:
    """Determine the SAGE overlap ratio following visualize_sage logic."""
    if config.overlap_ratio > 0.0:
        return max(0.0, min(config.overlap_ratio, 0.5))

    w_med, h_med = _compute_dataset_box_medians(dataset)
    if w_med <= 0.0 or h_med <= 0.0:
        return 0.0

    d = math.sqrt(w_med * h_med)
    base_edge = tile_size[0]
    if base_edge <= 0:
        return 0.0

    ratio = 0.5 * (d / base_edge)
    ratio = max(0.0, min(ratio, 0.5))
    return ratio or compute_adaptive_overlap(
        dataset.annotations,
        tile_size,
        annotation_filter=image_filter,
    )


def _build_info_text(
    image_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    overlap_ratio: float,
    grid: SageGrid,
) -> str:
    width, height = image_size
    stride_x, stride_y = grid.stride
    layout_x, layout_y = grid.layout
    ov_x_px = max(0.0, tile_size[0] - stride_x)
    ov_y_px = max(0.0, tile_size[1] - stride_y)
    ov_x_ratio = ov_x_px / tile_size[0] if tile_size[0] else 0.0
    ov_y_ratio = ov_y_px / tile_size[1] if tile_size[1] else 0.0

    return (
        f"Image: {width}x{height} | "
        f"Tile: {tile_size[0]}x{tile_size[1]} | "
        f"Tiles: {layout_x}x{layout_y} ({layout_x * layout_y}) | "
        f"Stride: {stride_x:.1f}x{stride_y:.1f} | "
        f"O*: {overlap_ratio * 100:.2f}% "
        f"(px overlap ~{ov_x_px:.1f}/{ov_y_px:.1f} "
        f"{ov_x_ratio * 100:.1f}%/{ov_y_ratio * 100:.1f}%)"
    )


def _select_images(dataset: CocoDataset, target_image: Optional[str], limit: int = 3) -> List[CocoImage]:
    if target_image:
        matches = [img for img in dataset.images if img.file_name == target_image]
        if not matches:
            raise FileNotFoundError(f"Image '{target_image}' not found in dataset.")
        return matches
    return dataset.images[:limit]


def _prepare_tiling_config(base_config: TilingConfig, overlap_ratio: float) -> TilingConfig:
    return TilingConfig(
        tile_size=base_config.tile_size,
        overlap=base_config.overlap,
        overlap_ratio=overlap_ratio,
        context_pad=base_config.context_pad,
        min_object_coverage=base_config.min_object_coverage,
        output_format=base_config.output_format,
        resize_output=base_config.resize_output,
        mode="sage",
        keep_empty_tiles=True,
    )


def create_tiling_analysis(dataset_path: str, output_dir: str, target_image: Optional[str] = None) -> None:
    print("SAGE Tiling Analysis Tool")
    print("=" * 60)

    annotations_path = os.path.join(dataset_path, "train", "_annotations.coco.json")
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    dataset = CocoDataset.from_json(annotations_path)
    print(f"Loaded dataset: {len(dataset.images)} images | {len(dataset.annotations)} annotations")

    categories = {cat.id: cat.name for cat in dataset.categories}
    os.makedirs(output_dir, exist_ok=True)

    selected_images = _select_images(dataset, target_image)
    print(f"\nAnalyzing {len(selected_images)} image(s)...")

    app_config = AppConfig.from_env()
    base_tiling_config = app_config.tiling
    tile_size = base_tiling_config.tile_size
    context_pad = base_tiling_config.context_pad

    overlap_ratio = _resolve_overlap_ratio(
        base_tiling_config,
        dataset,
        image_filter=[img.id for img in selected_images],
        tile_size=tile_size,
    )

    visualizer = BoundingBoxVisualizer()

    for index, coco_image in enumerate(selected_images, start=1):
        print(f"\n[{index}/{len(selected_images)}] Processing: {coco_image.file_name}")

        image_path = os.path.join(dataset_path, "train", coco_image.file_name)
        if not os.path.exists(image_path):
            print(f"  Warning: image file not found at {image_path}")
            continue

        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")

        width, height = image.size
        print(f"  Image size: {width} x {height}")

        image_annotations: List[CocoAnnotation] = [
            ann for ann in dataset.annotations if ann.image_id == coco_image.id
        ]
        print(f"  Annotations: {len(image_annotations)}")

        grid = compute_sage_grid((height, width), tile_size, overlap_ratio)
        stride_info = f"{grid.stride[0]:.2f}, {grid.stride[1]:.2f}"
        print(f"  SAGE layout: {grid.layout[0]} cols x {grid.layout[1]} rows | stride ~ ({stride_info})")
        print(f"  Tiles generated: {len(grid.boxes)} (context pad {context_pad}px)")

        info_text = _build_info_text((width, height), tile_size, overlap_ratio, grid)
        overview = visualizer.create_tiling_overview(
            image,
            image_annotations,
            categories,
            tile_size=tile_size,
            overlap=int(max(tile_size[0] - grid.stride[0], tile_size[1] - grid.stride[1])),
            max_width=1600,
            tile_bboxes=grid.boxes,
            info_text=info_text,
        )

        base_stem = Path(coco_image.file_name).stem
        overview_path = Path(output_dir) / f"sage_tiling_{base_stem}.jpg"
        overview.save(overview_path, quality=95)
        print(f"  Saved overview: {overview_path}")

        if width <= 2000:
            full_res_overview = visualizer.create_tiling_overview(
                image,
                image_annotations,
                categories,
                tile_size=tile_size,
                overlap=int(max(tile_size[0] - grid.stride[0], tile_size[1] - grid.stride[1])),
                max_width=width,
                tile_bboxes=grid.boxes,
                info_text=info_text,
            )
            full_res_path = Path(output_dir) / f"sage_tiling_full_{base_stem}.jpg"
            full_res_overview.save(full_res_path, quality=95)
            print(f"  Saved full resolution overview: {full_res_path}")

        tiling_engine = TilingEngine(_prepare_tiling_config(base_tiling_config, overlap_ratio))
        generated_tiles = list(tiling_engine.generate_tiles(image))
        print(f"  Engine preview: {len(generated_tiles)} tiles emitted")

        if generated_tiles:
            mid_tile = generated_tiles[len(generated_tiles) // 2]
            sample_tile_path = Path(output_dir) / f"sage_sample_tile_{base_stem}.jpg"
            mid_tile.image.save(sample_tile_path, quality=95)
            print(
                f"  Sample tile saved: {sample_tile_path} "
                f"(origin={mid_tile.origin}, grid_origin={mid_tile.grid_origin})"
            )

    print("\nDone. Review the generated overviews for SAGE tile coverage verification.")


if __name__ == "__main__":
    target = "2_jpg.rf.abab05c14fe2fb698fb39b966919cb6d.jpg"
    create_tiling_analysis("./dataset", "./tiling_analysis", target_image=target)
