#!/usr/bin/env python3
"""
Quick verification script to check if tiling preserved annotations correctly.
Supports both the legacy SAHI workflow and the new ASAHI adaptive tiling output.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Tuple

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.coco import CocoDataset

VALID_MODES = {"sahi", "asahi"}


def _resolve_annotation_path(tiled_root: str, mode: str, fold: int, split: str) -> str:
    if mode == "sahi":
        return os.path.join(tiled_root, split, "_annotations.coco.json")
    return os.path.join(tiled_root, f"fold_{fold}", split, "_annotations.coco.json")


def _resolve_image_dir(tiled_root: str, mode: str, fold: int, split: str) -> str:
    if mode == "sahi":
        return os.path.join(tiled_root, split)
    return os.path.join(tiled_root, f"fold_{fold}", split)


def _load_datasets(original_path: str, tiled_path: str) -> Tuple[CocoDataset, CocoDataset]:
    original_dataset = CocoDataset.from_json(original_path)
    tiled_dataset = CocoDataset.from_json(tiled_path)
    return original_dataset, tiled_dataset


def _category_counts(dataset: CocoDataset) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for ann in dataset.annotations:
        counts[ann.category_id] = counts.get(ann.category_id, 0) + 1
    return counts


def verify_tiling_process(
    original_path: str,
    tiled_path: str,
    mode: str = "asahi",
    fold: int = 1,
    split: str = "train",
) -> bool:
    """Verify that the tiling process worked correctly."""

    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {VALID_MODES}.")

    print("Dataset Tiling Verification")
    print("=" * 40)
    print(f"Mode: {mode.upper()}")
    if mode == "asahi":
        print(f"Fold: {fold} | Split: {split}")
    print()

    original_annotations = os.path.join(original_path, "_annotations.coco.json")
    tiled_annotations = _resolve_annotation_path(tiled_path, mode, fold, split)

    if not os.path.exists(original_annotations):
        print(f"[X] Original annotations not found: {original_annotations}")
        return False

    if not os.path.exists(tiled_annotations):
        print(f"[X] Tiled annotations not found: {tiled_annotations}")
        return False

    try:
        print("Loading datasets...")
        original_dataset, tiled_dataset = _load_datasets(original_annotations, tiled_annotations)
        print("[✓] Successfully loaded both datasets")
    except Exception as exc:  # noqa: BLE001
        print(f"[X] Error loading datasets: {exc}")
        return False

    print("\nDataset Statistics")
    print("-" * 20)
    print(f"Original Images: {len(original_dataset.images)}")
    print(f"Tiled Images ({split}): {len(tiled_dataset.images)}")
    print(f"Original Annotations: {len(original_dataset.annotations)}")
    print(f"Tiled Annotations ({split}): {len(tiled_dataset.annotations)}")

    if original_dataset.images:
        expansion_ratio = len(tiled_dataset.images) / len(original_dataset.images)
        ratio_label = "per split" if mode == "asahi" else ""
        print(f"Image Expansion Ratio {ratio_label}: {expansion_ratio:.2f}x")
    if original_dataset.annotations:
        retention_rate = (len(tiled_dataset.annotations) / len(original_dataset.annotations)) * 100
        print(f"Annotation Retention ({split}): {retention_rate:.1f}%")

    print("\nCategories")
    print("-" * 20)
    original_categories = {cat.id: cat.name for cat in original_dataset.categories}
    tiled_categories = {cat.id: cat.name for cat in tiled_dataset.categories}
    categories_match = original_categories == tiled_categories
    print(f"Categories preserved: {'[✓] Yes' if categories_match else '[X] No'}")

    original_counts = _category_counts(original_dataset)
    tiled_counts = _category_counts(tiled_dataset)
    for cat_id, cat_name in original_categories.items():
        print(
            f"  {cat_name} (ID {cat_id}): "
            f"{original_counts.get(cat_id, 0)} -> {tiled_counts.get(cat_id, 0)}"
        )

    print("\nSample Verification")
    print("-" * 20)
    sample_size = min(5, len(original_dataset.images))
    issues_found = 0

    for idx in range(sample_size):
        original_img = original_dataset.images[idx]
        original_img_anns = [ann for ann in original_dataset.annotations if ann.image_id == original_img.id]
        base_name = Path(original_img.file_name).stem
        corresponding_tiles = [img for img in tiled_dataset.images if base_name in Path(img.file_name).stem]

        total_tile_anns = sum(
            1
            for tile_img in corresponding_tiles
            for ann in tiled_dataset.annotations
            if ann.image_id == tile_img.id
        )

        print(f"  {original_img.file_name}: {len(original_img_anns)} annotations")
        print(f"    -> {len(corresponding_tiles)} tiles with {total_tile_anns} total annotations")

        if original_img_anns:
            retention = total_tile_anns / len(original_img_anns)
            if retention < 0.5:
                issues_found += 1
                print(f"    [!] Low annotation retention: {retention:.1%}")

    print("\nFile Verification")
    print("-" * 20)
    tile_dir = _resolve_image_dir(tiled_path, mode, fold, split)
    sample_tiles = tiled_dataset.images[:10]
    missing_files = sum(
        1 for tile_img in sample_tiles if not os.path.exists(os.path.join(tile_dir, tile_img.file_name))
    )

    if missing_files == 0:
        print("[✓] All sampled tile files exist")
    else:
        print(f"[X] {missing_files} out of {len(sample_tiles)} sampled files missing")

    print("\nOverall Assessment")
    print("=" * 20)
    success = True

    if len(tiled_dataset.images) <= len(original_dataset.images):
        print("[X] No image expansion detected - tiling may have failed")
        success = False
    else:
        print("[✓] Images were successfully tiled")

    if not categories_match:
        print("[X] Categories were not preserved correctly")
        success = False
    else:
        print("[✓] Categories preserved correctly")

    if issues_found > sample_size // 2:
        print("[X] Significant annotation loss detected")
        success = False
    elif tiled_dataset.annotations:
        print("[✓] Annotations appear to be preserved")

    if missing_files > 0:
        print("[X] Some tiled image files are missing")
        success = False
    else:
        print("[✓] Tiled image files exist")

    if success:
        print("\n[✓] Tiling process appears to have worked correctly!")
        print("    Run ./visualize_results.sh to see visual confirmation.")
    else:
        print("\n[!] Issues detected - please review the tiling process.")

    return success


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify dataset tiling process")
    parser.add_argument("--original", default="./dataset", help="Original dataset path")
    parser.add_argument("--tiled", default="./output", help="Tiled dataset path")
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default="asahi",
        help="Tiling mode used to generate the dataset.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="ASAHI fold index to verify (ignored in SAHI mode).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to verify (train/val/test).",
    )

    args = parser.parse_args()

    success = verify_tiling_process(
        args.original,
        args.tiled,
        mode=args.mode,
        fold=args.fold,
        split=args.split,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
