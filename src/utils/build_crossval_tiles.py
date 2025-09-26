#!/usr/bin/env python3
"""Cross-validation aware tiling pipeline for high-resolution insect dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupKFold

from src.config.settings import TilingConfig
from src.core.tiling.engine import TilingEngine
from src.models.coco import CocoAnnotation, CocoDataset, CocoImage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice high-resolution images into tiles and build cross-validation folds.")

    parser.add_argument(
        "--input",
        type=str,
        default="./dataset/all/train",
        help="Path to directory containing original images and _annotations.coco.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output_crossval",
        help="Directory where tiled dataset and folds will be written",
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
        help="Minimum intersection-over-area (IoA) required to keep a projected bounding box",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (GroupKFold)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible image ordering before GroupKFold",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality used when saving tile images",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )

    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {path} already exists. Use --overwrite to regenerate the dataset."
            )
        print(f"⚠️  Removing existing output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def create_symlink_or_copy(src: Path, dst: Path, tile_files: Iterable[Path]) -> None:
    """Create a symlink to src at dst/images, or copy files if symlink fails."""
    images_path = dst / "images"
    if images_path.exists():
        return

    relative_target = os.path.relpath(src, dst)

    try:
        os.symlink(relative_target, images_path, target_is_directory=True)
        return
    except OSError as exc:
        print(f"⚠️  Could not create symlink ({exc}). Falling back to copying tiles for {dst.name}.")

    images_path.mkdir(parents=True, exist_ok=True)
    for tile_file in tile_files:
        destination = images_path / tile_file.name
        if not destination.exists():
            shutil.copy2(tile_file, destination)


def build_manifest_entry(
    tile_id: int,
    tile_file: Path,
    source_image: CocoImage,
    offset: Tuple[int, int],
    tile_size: Tuple[int, int],
    annotations: List[CocoAnnotation],
) -> Dict:
    entry = {
        "tile_id": tile_id,
        "file_name": f"images/{tile_file.name}",
        "source_image": {
            "id": source_image.id,
            "file_name": source_image.file_name,
            "width": source_image.width,
            "height": source_image.height,
        },
        "offset": {"x": offset[0], "y": offset[1]},
        "tile_size": {"width": tile_size[0], "height": tile_size[1]},
        "is_positive": bool(annotations),
        "annotations": [],
    }

    for ann in annotations:
        extra = ann.extra or {}
        entry["annotations"].append(
            {
                "id": ann.id,
                "category_id": ann.category_id,
                "bbox": ann.bbox,
                "area": ann.area,
                "iscrowd": ann.iscrowd,
                "source_annotation_id": extra.get("source_annotation_id"),
                "intersection_over_area": extra.get("intersection_over_area"),
                "original_bbox": extra.get("original_bbox"),
            }
        )

    return entry


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    annotations_path = input_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {annotations_path}")

    output_dir = Path(args.output)
    ensure_output_dir(output_dir, args.overwrite)

    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Loading COCO dataset...")
    dataset = CocoDataset.from_json(str(annotations_path))
    print(f"   Images: {len(dataset.images)} | Annotations: {len(dataset.annotations)} | Categories: {len(dataset.categories)}")

    annotations_by_image: Dict[int, List[CocoAnnotation]] = defaultdict(list)
    for ann in dataset.annotations:
        annotations_by_image[ann.image_id].append(ann)

    image_order = list(dataset.images)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(image_order)

    tiling_config = TilingConfig(
        tile_size=(args.tile_size[0], args.tile_size[1]),
        overlap=args.overlap,
        min_object_coverage=args.min_ioa,
    )
    tiling_engine = TilingEngine(tiling_config)

    tile_entries: Dict[int, Dict] = {}
    tiles_by_image: Dict[int, List[int]] = defaultdict(list)
    manifest_entries: List[Dict] = []

    tile_id_counter = 1
    annotation_id_counter = 1

    print("🧩 Generating tiles...")
    for idx, coco_image in enumerate(image_order, start=1):
        image_path = input_dir / coco_image.file_name
        if not image_path.exists():
            print(f"   ⚠️  Skipping missing image: {coco_image.file_name}")
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_annotations = annotations_by_image.get(coco_image.id, [])

            tile_count = 0
            for tile, offset, scale in tiling_engine.generate_tiles(image):
                tile_filename = f"{coco_image.id}_tile_{offset[0]}_{offset[1]}.jpg"
                tile_path = tiles_dir / tile_filename
                tile.save(tile_path, quality=args.quality)

                transformed_annotations = tiling_engine.transform_annotations(
                    image_annotations, offset, scale
                )

                new_annotations: List[CocoAnnotation] = []
                for ann in transformed_annotations:
                    copied = CocoAnnotation(
                        id=annotation_id_counter,
                        image_id=tile_id_counter,
                        category_id=ann.category_id,
                        segmentation=ann.segmentation,
                        area=ann.area,
                        bbox=ann.bbox,
                        iscrowd=ann.iscrowd,
                        extra=ann.extra,
                    )
                    new_annotations.append(copied)
                    annotation_id_counter += 1

                tile_image = CocoImage(
                    id=tile_id_counter,
                    width=tile.width,
                    height=tile.height,
                    file_name=f"images/{tile_filename}",
                )

                metadata = build_manifest_entry(
                    tile_id_counter,
                    tile_path,
                    coco_image,
                    offset,
                    (tile.width, tile.height),
                    new_annotations,
                )

                tile_entries[tile_id_counter] = {
                    "image": tile_image,
                    "annotations": new_annotations,
                    "metadata": metadata,
                    "file_path": tile_path,
                }

                tiles_by_image[coco_image.id].append(tile_id_counter)
                manifest_entries.append(metadata)

                tile_id_counter += 1
                tile_count += 1

        print(f"   [{idx:4d}/{len(image_order)}] {coco_image.file_name} → {tile_count} tiles")

    total_tiles = len(tile_entries)
    positives = sum(1 for entry in tile_entries.values() if entry["metadata"]["is_positive"])
    negatives = total_tiles - positives
    print(f"✅ Generated {total_tiles} tiles ({positives} positives | {negatives} negatives)")

    # Prepare data for GroupKFold
    image_ids_ordered = [img.id for img in image_order]
    image_targets = [1 if annotations_by_image.get(img_id) else 0 for img_id in image_ids_ordered]

    gkf = GroupKFold(n_splits=args.folds)
    fold_assignments: Dict[int, Dict[str, List[int]]] = {}

    print("🔀 Creating GroupKFold splits...")
    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(np.zeros(len(image_ids_ordered)), image_targets, groups=image_ids_ordered)
    ):
        train_image_ids = [image_ids_ordered[i] for i in train_idx]
        val_image_ids = [image_ids_ordered[i] for i in val_idx]
        fold_assignments[fold_idx] = {
            "train_images": train_image_ids,
            "val_images": val_image_ids,
            "test_images": val_image_ids,  # same as val for reconstruction/evaluation
        }
        print(
            f"   Fold {fold_idx}: train={len(train_image_ids)} images | val/test={len(val_image_ids)} images"
        )

    # Build datasets per fold
    print("💾 Saving fold datasets...")
    fold_summaries: Dict[int, Dict[str, Dict[str, int]]] = {}

    for fold_idx, splits in fold_assignments.items():
        fold_dir = folds_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        split_stats: Dict[str, Dict[str, int]] = {}

        for split_name in ["train", "val", "test"]:
            image_ids = splits[f"{split_name}_images"]
            tile_ids = [tile_id for image_id in image_ids for tile_id in tiles_by_image.get(image_id, [])]

            split_dir = fold_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            dataset_split = CocoDataset(
                info=dataset.info,
                licenses=dataset.licenses,
                images=[tile_entries[tile_id]["image"] for tile_id in tile_ids],
                annotations=[
                    ann
                    for tile_id in tile_ids
                    for ann in tile_entries[tile_id]["annotations"]
                ],
                categories=dataset.categories,
            )
            dataset_path = split_dir / "_annotations.coco.json"
            dataset_split.save_json(str(dataset_path))

            positive_tiles = [tile_id for tile_id in tile_ids if tile_entries[tile_id]["metadata"]["is_positive"]]
            negative_tiles = [tile_id for tile_id in tile_ids if not tile_entries[tile_id]["metadata"]["is_positive"]]

            with open(split_dir / "tiles.json", "w") as f:
                json.dump(
                    {
                        "tile_ids": tile_ids,
                        "positive_tiles": positive_tiles,
                        "negative_tiles": negative_tiles,
                    },
                    f,
                    indent=2,
                )

            split_total = len(tile_ids)
            split_stats[split_name] = {
                "tiles": split_total,
                "positives": len(positive_tiles),
                "negatives": len(negative_tiles),
                "positive_ratio": round(len(positive_tiles) / split_total, 4) if split_total else 0.0,
            }

        fold_summaries[fold_idx] = split_stats

        # Create shared images directory (symlink or copies)
        tile_ids_for_fold = {
            tile_id
            for split_name in ["train", "val", "test"]
            for image_id in splits[f"{split_name}_images"]
            for tile_id in tiles_by_image.get(image_id, [])
        }
        required_tile_files = {tile_entries[tile_id]["file_path"] for tile_id in tile_ids_for_fold}
        create_symlink_or_copy(tiles_dir, fold_dir, required_tile_files)

        with open(fold_dir / "summary.json", "w") as f:
            json.dump(split_stats, f, indent=2)

    # Save global manifests
    with open(manifests_dir / "tiles_manifest.json", "w") as f:
        json.dump(manifest_entries, f, indent=2)

    with open(manifests_dir / "fold_assignments.json", "w") as f:
        json.dump(fold_assignments, f, indent=2)

    image_manifest = []
    for image in dataset.images:
        tile_ids = tiles_by_image.get(image.id, [])
        image_manifest.append(
            {
                "image_id": image.id,
                "file_name": image.file_name,
                "tiles": tile_ids,
                "positive_tiles": [tile_id for tile_id in tile_ids if tile_entries[tile_id]["metadata"]["is_positive"]],
                "negative_tiles": [tile_id for tile_id in tile_ids if not tile_entries[tile_id]["metadata"]["is_positive"]],
            }
        )

    with open(manifests_dir / "images_manifest.json", "w") as f:
        json.dump(image_manifest, f, indent=2)

    summary = {
        "config": {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "min_ioa": args.min_ioa,
            "folds": args.folds,
            "seed": args.seed,
        },
        "counts": {
            "images": len(dataset.images),
            "annotations": len(dataset.annotations),
            "tiles": total_tiles,
            "positive_tiles": positives,
            "negative_tiles": negatives,
        },
        "folds": fold_summaries,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("🎉 Cross-validation tiling completed!")
    print(f"   Output directory: {output_dir}")
    print(f"   Total tiles: {total_tiles} | Positives: {positives} | Negatives: {negatives}")
    print("   Folds:"
          + "".join(
              f"\n      Fold {idx}: train={fold['train']['tiles']} tiles, val={fold['val']['tiles']} tiles, test={fold['test']['tiles']} tiles"
              for idx, fold in fold_summaries.items()
          ))


if __name__ == "__main__":
    main()
