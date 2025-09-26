#!/usr/bin/env python3
"""Generate a visual explanation of the tiling process for a single image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from src.config.settings import TilingConfig
from src.core.tiling.engine import TilingEngine
from src.models.coco import CocoAnnotation, CocoDataset, CocoImage
from src.utils.visualization import BoundingBoxVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize how a high-resolution image is sliced into tiles.")

    parser.add_argument(
        "--input",
        type=str,
        default="./dataset/all/train",
        help="Directory containing the original images and _annotations.coco.json",
    )
    parser.add_argument(
        "--image-file",
        type=str,
        help="File name of the image to visualize (must exist inside the input directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./visualizations/tiling_examples",
        help="Directory where the visualization and JSON summary will be written",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("WIDTH", "HEIGHT"),
        help="Tile size (width height) in pixels",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Tile overlap in pixels (stride = tile_size - overlap)",
    )
    parser.add_argument(
        "--min-ioa",
        type=float,
        default=0.30,
        help="Minimum intersection-over-area (IoA) for keeping a bounding box",
    )

    return parser.parse_args()


def choose_image(dataset: CocoDataset, target_file: Optional[str]) -> CocoImage:
    if target_file:
        match = next((img for img in dataset.images if img.file_name == target_file), None)
        if not match:
            raise FileNotFoundError(f"Image {target_file} not found in dataset annotations")
        return match

    # Prefer an image that has annotations so the example is informative
    annotated = [img for img in dataset.images]
    annotated.sort(key=lambda img: img.file_name)

    with_annotations = [img for img in annotated if any(ann.image_id == img.id for ann in dataset.annotations)]
    if with_annotations:
        return with_annotations[0]

    if annotated:
        return annotated[0]

    raise RuntimeError("Dataset does not contain any images")


def summarise_tile(tile_annotations: List[CocoAnnotation], categories: Dict[int, str]) -> List[Dict[str, object]]:
    summary = []
    for ann in tile_annotations:
        extra = ann.extra or {}
        summary.append(
            {
                "annotation_id": ann.id,
                "category_id": ann.category_id,
                "category_name": categories.get(ann.category_id, f"cat_{ann.category_id}"),
                "bbox": [round(v, 2) for v in ann.bbox],
                "area": round(ann.area, 2),
                "ioa": round(extra.get("intersection_over_area", 0.0), 3),
                "source_annotation_id": extra.get("source_annotation_id"),
            }
        )
    return summary


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    annotations_path = input_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    print("📥 Loading dataset...")
    dataset = CocoDataset.from_json(str(annotations_path))
    categories = {cat.id: cat.name for cat in dataset.categories}

    image_info = choose_image(dataset, args.image_file)
    image_path = input_dir / image_info.file_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found on disk: {image_path}")

    with Image.open(image_path) as img:
        original_image = img.convert("RGB")

    image_annotations = [ann for ann in dataset.annotations if ann.image_id == image_info.id]
    print(f"   Image: {image_info.file_name} ({original_image.width}x{original_image.height})")
    print(f"   Annotations: {len(image_annotations)}")

    tiling_config = TilingConfig(
        tile_size=(args.tile_size[0], args.tile_size[1]),
        overlap=args.overlap,
        min_object_coverage=args.min_ioa,
    )
    engine = TilingEngine(tiling_config)

    tiles: List[Dict[str, object]] = []
    for idx, (tile_img, offset, scale) in enumerate(engine.generate_tiles(original_image), start=1):
        tile_annotations = engine.transform_annotations(image_annotations, offset, scale)
        tile_name = f"{Path(image_info.file_name).stem}_tile_{offset[0]}_{offset[1]}.jpg"
        tiles.append(
            {
                "index": idx,
                "offset": offset,
                "image": tile_img,
                "annotations": tile_annotations,
                "file_name": tile_name,
            }
        )

    if not tiles:
        raise RuntimeError("No tiles were generated for the selected image")

    positives = [tile for tile in tiles if tile["annotations"]]
    if positives:
        chosen_tile = max(positives, key=lambda t: len(t["annotations"]))
    else:
        chosen_tile = tiles[len(tiles) // 2]

    print(
        "   Tiles generated: "
        f"{len(tiles)} (positives: {len(positives)} | negatives: {len(tiles) - len(positives)})"
    )
    print(
        f"   Highlighting tile #{chosen_tile['index']} at offset {chosen_tile['offset']} "
        f"with {len(chosen_tile['annotations'])} annotations"
    )

    visualizer = BoundingBoxVisualizer()
    comparison = visualizer.create_comparison_view(
        original_img=original_image,
        tiled_img=chosen_tile["image"],
        original_annotations=image_annotations,
        tiled_annotations=chosen_tile["annotations"],
        categories=categories,
        tile_offset=chosen_tile["offset"],
        tile_size=tiling_config.tile_size,
        overlap=args.overlap,
        tile_label=chosen_tile["file_name"],
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / f"tiling_example_{Path(image_info.file_name).stem}.jpg"
    comparison.save(figure_path, quality=95)

    tile_summary = summarise_tile(chosen_tile["annotations"], categories)
    summary_payload = {
        "image": {
            "id": image_info.id,
            "file_name": image_info.file_name,
            "size": [original_image.width, original_image.height],
            "annotation_count": len(image_annotations),
        },
        "tiling": {
            "tile_size": list(tiling_config.tile_size),
            "overlap": args.overlap,
            "min_ioa": args.min_ioa,
            "tiles_total": len(tiles),
            "tiles_positive": len(positives),
            "tiles_negative": len(tiles) - len(positives),
        },
        "highlighted_tile": {
            "index": chosen_tile["index"],
            "offset": list(chosen_tile["offset"]),
            "file_name": chosen_tile["file_name"],
            "annotation_count": len(chosen_tile["annotations"]),
            "annotations": tile_summary,
        },
    }

    summary_path = output_dir / f"tiling_example_{Path(image_info.file_name).stem}.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print("📤 Saved visualization:", figure_path)
    print("📄 Saved summary:", summary_path)

    print("\nExplanation:")
    print(
        f" - The original image ({original_image.width}x{original_image.height}) is tiled into "
        f"{len(tiles)} regions of {tiling_config.tile_size[0]}x{tiling_config.tile_size[1]} pixels."
    )
    print(
        f" - Border tiles are anchored at the edges so the full image is covered without resizing." 
        ""
    )
    print(
        f" - Bounding boxes are projected onto each tile and kept when at least {args.min_ioa:.2f} "
        f"of the original area remains visible."
    )
    if tile_summary:
        print(" - Highlighted tile annotations:")
        for ann in tile_summary:
            print(
                f"    • {ann['category_name']} (source #{ann['source_annotation_id']}) "
                f"→ bbox {ann['bbox']} | IoA={ann['ioa']}"
            )
    else:
        print(" - The highlighted tile contains no annotations (hard negative example).")


if __name__ == "__main__":
    main()
