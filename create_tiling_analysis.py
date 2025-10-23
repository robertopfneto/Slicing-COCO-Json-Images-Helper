#!/usr/bin/env python3
"""
SAHI tiling analysis script that overlays tile boundaries and annotations on images.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

from PIL import Image

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.config.settings import AppConfig, TilingConfig
from src.core.tiling.engine import TilingEngine, get_auto_slice_params
from src.models.coco import CocoAnnotation, CocoDataset, CocoImage
from src.utils.visualization import BoundingBoxVisualizer


def _resolve_overlap_ratio(configured_ratio: Optional[float], pixel_overlap: int, tile_extent: int) -> float:
    if configured_ratio is not None:
        return max(0.0, min(configured_ratio, 1.0))
    if tile_extent <= 0 or pixel_overlap <= 0:
        return 0.0
    return max(0.0, min(pixel_overlap / tile_extent, 1.0))


def _infer_layout(tile_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    if not tile_bboxes:
        return 0, 0
    sorted_tiles = sorted(tile_bboxes, key=lambda b: (b[1], b[0]))
    first_row_y = sorted_tiles[0][1]
    cols = sum(1 for bbox in sorted_tiles if bbox[1] == first_row_y)
    if cols == 0:
        return 0, 0
    rows = (len(sorted_tiles) + cols - 1) // cols
    return cols, rows


def _resolve_sahi_parameters(config: TilingConfig, image_width: int, image_height: int) -> Dict[str, float]:
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


def create_tiling_analysis(dataset_path: str, output_dir: str, target_image: Optional[str] = None) -> None:
    """Create SAHI tiling analysis visualizations."""

    print("SAHI Tiling Analysis Tool")
    print("=" * 50)

    annotations_path = os.path.join(dataset_path, "train", "_annotations.coco.json")
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found: {annotations_path}")
        return

    dataset = CocoDataset.from_json(annotations_path)
    print(f"Loaded dataset: {len(dataset.images)} images, {len(dataset.annotations)} annotations")

    os.makedirs(output_dir, exist_ok=True)
    categories = {cat.id: cat.name for cat in dataset.categories}
    visualizer = BoundingBoxVisualizer()

    app_config = AppConfig.from_env()
    tiling_config = app_config.tiling
    engine = TilingEngine(tiling_config)

    if target_image:
        selected: List[CocoImage] = [img for img in dataset.images if img.file_name == target_image]
        if not selected:
            print(f"Error: Image {target_image} not found in dataset")
            return
        images_to_analyze = selected
    else:
        images_to_analyze = dataset.images[:3]

    print(f"\nAnalyzing {len(images_to_analyze)} images...")

    for index, image_info in enumerate(images_to_analyze, start=1):
        print(f"\nProcessing {index}/{len(images_to_analyze)}: {image_info.file_name}")

        image_path = os.path.join(dataset_path, "train", image_info.file_name)
        if not os.path.exists(image_path):
            print(f"  Warning: Image file not found: {image_path}")
            continue

        try:
            with Image.open(image_path) as pil_image:
                pil_image.load()
                image = pil_image.copy()
            print(f"  Image size: {image.size}")

            image_annotations: List[CocoAnnotation] = [ann for ann in dataset.annotations if ann.image_id == image_info.id]
            print(f"  Annotations: {len(image_annotations)}")

            params = _resolve_sahi_parameters(tiling_config, image.width, image.height)
            tiles = list(engine.generate_tiles(image))
            tile_bboxes = [(bbox[0], bbox[1], bbox[2], bbox[3]) for _, bbox, _ in tiles]
            tile_count = len(tile_bboxes)
            cols, rows = _infer_layout(tile_bboxes)

            print(f"  Tiles generated: {tile_count} ({cols} cols x {rows} rows)")
            print(
                "  Slice: {slice_width}x{slice_height} | "
                "Overlap px (x/y): {overlap_x_px}/{overlap_y_px} | "
                "Overlap % (x/y): {overlap_x_ratio:.2f}/{overlap_y_ratio:.2f} | "
                "Stride: {stride_x}x{stride_y}".format(**params)
            )

            info_text = (
                f"Original: {image.width}x{image.height} | "
                f"Slice: {params['slice_width']}x{params['slice_height']} | "
                f"Overlap px: {params['overlap_x_px']}/{params['overlap_y_px']} | "
                f"Overlap %: {params['overlap_x_ratio']*100:.1f}/{params['overlap_y_ratio']*100:.1f} | "
                f"Stride: {params['stride_x']}x{params['stride_y']} | "
                f"Tiles: {tile_count}"
            )

            overview = visualizer.create_tiling_overview(
                image,
                image_annotations,
                categories,
                tile_size=(params["slice_width"], params["slice_height"]),
                overlap=max(params["overlap_x_px"], params["overlap_y_px"]),
                max_width=1600,
                tile_bboxes=tile_bboxes,
                info_text=info_text,
            )

            base_name = os.path.splitext(image_info.file_name)[0]
            overview_path = os.path.join(output_dir, f"tiling_analysis_{base_name}.jpg")
            overview.save(overview_path, quality=95)
            print(f"  Saved tiling analysis: {overview_path}")

            if image.width <= 2000:
                full_res_overview = visualizer.create_tiling_overview(
                    image,
                    image_annotations,
                    categories,
                    tile_size=(params["slice_width"], params["slice_height"]),
                    overlap=max(params["overlap_x_px"], params["overlap_y_px"]),
                    max_width=image.width,
                    tile_bboxes=tile_bboxes,
                    info_text=info_text,
                )

                full_res_path = os.path.join(output_dir, f"full_res_analysis_{base_name}.jpg")
                full_res_overview.save(full_res_path, quality=95)
                print(f"  Saved full resolution analysis: {full_res_path}")

        except Exception as exc:
            print(f"  Error processing {image_info.file_name}: {exc}")

    print(f"\nDone. Tiling analysis complete! Check the '{output_dir}' directory.")
    print("\nThe visualizations show:")
    print("  - Cyan rectangles: SAHI tile boundaries")
    print("  - Colored boxes: Annotation bounding boxes")
    print("  - T1, T2, etc.: Tile indices in traversal order")
    print("\nThis helps identify:")
    print("  - Annotations spanning multiple tiles")
    print("  - Tile overlap coverage")
    print("  - Edge tile positioning")


if __name__ == "__main__":
    target_image = "30_jpg.rf.3ed47a5937372f408cfd0f5df94d0fad.jpg"

    create_tiling_analysis(
        dataset_path="./dataset",
        output_dir="./tiling_analysis",
        target_image=target_image,
    )
