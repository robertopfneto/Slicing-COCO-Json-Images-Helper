#!/usr/bin/env python3
"""
K-Fold SAGE Tiling Driver

This script orchestrates adaptive stride-aligned tiling (SAGE) for every fold and
split of a cross-validation experiment. It expects COCO annotations that define
the original images for each fold/split and produces tiled datasets that keep
every patch inside the same fold it originated from.

Output layout:
    dataset/tiles/sage/fold_<k>/<split>/
        ├── _annotations.coco.json
        ├── metadata.json
        ├── summary.json
        └── summary.txt
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import AppConfig, DatasetConfig, ProcessingConfig, TilingConfig
from src.models.coco import CocoDataset
from src.services.dataset.processor import DatasetProcessor
from src.utils.helpers import calculate_split_indices

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_FOLDS = 5
SPLITS = ["train", "val", "test"]
TILE_SIZE = (640, 640)
MIN_OBJECT_COVERAGE = 0.3
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "tiles" / "sage"


def _get_fallback_split_ratios() -> Tuple[float, float, float]:
    """Resolve fallback split ratios from percentage-based configuration."""
    train_pct = float(os.getenv("TRAIN_IMR_PROP", "80"))
    val_pct = float(os.getenv("VAL_IMR_PROP", "10"))
    test_pct = float(os.getenv("TEST_IMR_PROP", "10"))

    if any(value < 0 for value in (train_pct, val_pct, test_pct)):
        raise ValueError("Split percentages must be non-negative.")

    total_pct = train_pct + val_pct + test_pct
    if total_pct <= 0:
        raise ValueError("Split percentages must sum to a positive value.")

    if abs(total_pct - 100.0) > 1e-3:
        raise ValueError(
            f"Split percentages must sum to 100.0 (got {total_pct}). "
            "Adjust TRAIN_IMR_PROP, VAL_IMR_PROP, and TEST_IMR_PROP."
        )

    return train_pct / 100.0, val_pct / 100.0, test_pct / 100.0


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


SOURCE_IMAGES_DIR = _first_existing(
    [
        PROJECT_ROOT / "dataset" / "all" / "train",
        PROJECT_ROOT / "dataset" / "train",
    ]
)

BASE_DATASET_ANNOTATIONS = _first_existing(
    [
        PROJECT_ROOT / "dataset" / "all" / "train" / "_annotations.coco.json",
        PROJECT_ROOT / "dataset" / "train" / "_annotations.coco.json",
    ]
)

FOLD_JSON_DIR = _first_existing(
    [
        PROJECT_ROOT / "dataset" / "all" / "filesJSON",
        PROJECT_ROOT / "dataset" / "folds",
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_git_commit(project_root: Path) -> str:
    """Return the current git commit hash (first 8 chars) or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def generate_metadata(
    fold: int,
    split: str,
    config: TilingConfig,
    annotations_path: Path,
    overlap_ratio: float,
) -> Dict:
    """Build a metadata dictionary for the processed split."""
    return {
        "fold": fold,
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(PROJECT_ROOT),
        "source_annotations": str(annotations_path),
        "source_images_dir": str(SOURCE_IMAGES_DIR),
        "tiling_config": {
            "mode": config.mode,
            "tile_size": config.tile_size,
            "min_object_coverage": config.min_object_coverage,
            "keep_empty_tiles": config.keep_empty_tiles,
            "resize_output": config.resize_output,
            "overlap_ratio": overlap_ratio,
        },
    }


def generate_summary_report(output_dir: Path, split: str) -> Dict:
    """Generate summary statistics for a processed split."""
    annotations_path = output_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        return {"error": f"Annotations file not found at {annotations_path}"}

    try:
        dataset = CocoDataset.from_json(str(annotations_path))

        category_counts: Dict[int, int] = {}
        for ann in dataset.annotations:
            category_counts[ann.category_id] = category_counts.get(ann.category_id, 0) + 1

        category_map = {cat.id: cat.name for cat in dataset.categories}
        category_stats = {
            category_map.get(cat_id, f"category_{cat_id}"): count
            for cat_id, count in category_counts.items()
        }

        images_without_annotations = sum(
            1
            for image in dataset.images
            if not any(ann.image_id == image.id for ann in dataset.annotations)
        )

        return {
            "split": split,
            "num_images": len(dataset.images),
            "num_annotations": len(dataset.annotations),
            "num_categories": len(dataset.categories),
            "category_counts": category_stats,
            "images_without_annotations": images_without_annotations,
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {"error": str(exc)}


def build_fallback_splits(annotations_path: Path) -> Dict[int, Dict[str, List[int]]]:
    """Create k-fold train/val/test splits directly from a base annotations file."""
    dataset = CocoDataset.from_json(str(annotations_path))
    image_ids = [image.id for image in dataset.images]

    if not image_ids:
        raise ValueError("No images found to build fallback folds.")

    train_ratio, val_ratio, test_ratio = _get_fallback_split_ratios()

    fold_splits: Dict[int, Dict[str, List[int]]] = {}
    for fold_idx in range(NUM_FOLDS):
        rng = random.Random(42 + fold_idx)
        shuffled_ids = image_ids[:]
        rng.shuffle(shuffled_ids)

        # Use deterministic shuffles per fold so that ratios can be honoured exactly.
        split_ranges = calculate_split_indices(
            total_count=len(shuffled_ids),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        fold_splits[fold_idx + 1] = {
            "train": [shuffled_ids[i] for i in split_ranges["train"]],
            "val": [shuffled_ids[i] for i in split_ranges["val"]],
            "test": [shuffled_ids[i] for i in split_ranges["test"]],
       }

    return fold_splits


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_fold_split(
    fold: int,
    split: str,
    app_config: AppConfig,
    annotations_path: Path,
    output_dir: Path,
    image_filter: Optional[List[int]] = None,
) -> bool:
    """Process a single (fold, split) combination."""

    print("\n" + "=" * 80)
    print(f"[SAGE] Processing Fold {fold} - {split.upper()}")
    print("=" * 80)

    if not annotations_path.exists():
        print(f"[ERROR] Missing annotations file: {annotations_path}")
        return False
    if not SOURCE_IMAGES_DIR or not SOURCE_IMAGES_DIR.exists():
        print(f"[ERROR] Missing source images directory: {SOURCE_IMAGES_DIR}")
        return False
    if image_filter is not None and not image_filter:
        print(f"[WARN] No images assigned to fold {fold} split {split}; skipping.")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    processor = DatasetProcessor(app_config)
    summary = processor.process_dataset(
        annotations_path=str(annotations_path),
        images_dir=str(SOURCE_IMAGES_DIR),
        split_name=split,
        output_dir=str(output_dir),
        mode="sage",
        keep_empty_tiles=(split == "test"),
        image_filter=image_filter,
    )

    print("\n[check] Validating output...")
    if not processor.validate_output(split_name=split, output_dir=str(output_dir)):
        print("[ERROR] Validation failed")
        return False
    print("[ok] Validation successful")

    metadata = generate_metadata(
        fold=fold,
        split=split,
        config=app_config.tiling,
        annotations_path=annotations_path,
        overlap_ratio=summary.get("overlap_ratio", 0.0),
    )
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)
    print(f"[info] Metadata saved to {metadata_path}")

    summary_report = generate_summary_report(output_dir, split)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary_report, summary_file, indent=2)

    summary_txt_path = output_dir / "summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"Summary Report - Fold {fold} - {split.upper()}\n")
        txt_file.write("=" * 60 + "\n\n")
        txt_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if "error" in summary_report:
            txt_file.write(f"Error: {summary_report['error']}\n")
        else:
            txt_file.write(f"Number of images: {summary_report['num_images']}\n")
            txt_file.write(f"Number of annotations: {summary_report['num_annotations']}\n")
            txt_file.write(
                f"Images without annotations: {summary_report['images_without_annotations']}\n"
            )
            txt_file.write(f"Number of categories: {summary_report['num_categories']}\n\n")
            txt_file.write("Annotations per category:\n")
            for name, count in summary_report["category_counts"].items():
                txt_file.write(f"  - {name}: {count}\n")

    print(f"[info] Summary saved to {summary_path}")
    print(f"[info] Text report saved to {summary_txt_path}")
    print(f"[done] Fold {fold} - {split.upper()} completed")
    return True


def build_app_config(output_dir: Path) -> AppConfig:
    """Create the application configuration for a fold/split run."""

    tiling_config = TilingConfig(
        tile_size=TILE_SIZE,
        overlap=0,
        overlap_ratio=0.0,
        min_object_coverage=MIN_OBJECT_COVERAGE,
        keep_empty_tiles=True,  # Will be overridden per split
        resize_output=None,
        mode="sage",
    )

    dataset_config = DatasetConfig(
        input_path=str(SOURCE_IMAGES_DIR.parent),
        output_path=str(output_dir.parent),
    )

    processing_config = ProcessingConfig()

    return AppConfig(
        tiling=tiling_config,
        dataset=dataset_config,
        processing=processing_config,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if SOURCE_IMAGES_DIR is None or not SOURCE_IMAGES_DIR.exists():
        raise FileNotFoundError(
            "Could not locate the source images directory. "
            "Expected either dataset/all/train or dataset/train."
        )

    fold_dir = FOLD_JSON_DIR if FOLD_JSON_DIR and FOLD_JSON_DIR.exists() else None
    fallback_splits: Optional[Dict[int, Dict[str, List[int]]]] = None

    if fold_dir is None:
        if BASE_DATASET_ANNOTATIONS is None or not BASE_DATASET_ANNOTATIONS.exists():
            raise FileNotFoundError(
                "No fold definitions found and base annotations file is missing. "
                "Provide fold JSONs under dataset/all/filesJSON (or dataset/folds) "
                "or ensure dataset/train/_annotations.coco.json exists."
            )
        print(
            "[info] Fold JSON directory not found. "
            "Building cross-validation splits from dataset/train/_annotations.coco.json."
        )
        train_ratio, val_ratio, test_ratio = _get_fallback_split_ratios()
        print(
            "    Using fallback split configuration: "
            f"train={train_ratio * 100:.1f}% | "
            f"val={val_ratio * 100:.1f}% | "
            f"test={test_ratio * 100:.1f}%"
        )
        fallback_splits = build_fallback_splits(BASE_DATASET_ANNOTATIONS)
        for fold_idx, split_map in fallback_splits.items():
            train_len = len(split_map.get("train", []))
            val_len = len(split_map.get("val", []))
            test_len = len(split_map.get("test", []))
            print(
                f"    Fold {fold_idx}: "
                f"train={train_len} images | val={val_len} | test={test_len}"
            )

    print("=" * 80)
    print("K-Fold SAGE Tiling Pipeline")
    print("=" * 80)
    print(f"Folds: {NUM_FOLDS}")
    print(f"Splits: {', '.join(SPLITS)}")
    print(f"Tile size: {TILE_SIZE}")
    print(f"Min object coverage: {MIN_OBJECT_COVERAGE}")
    print(f"Source images: {SOURCE_IMAGES_DIR}")
    print(f"Fold JSONs: {fold_dir if fold_dir else 'generated from base annotations'}")
    print(f"Output root: {OUTPUT_ROOT}")
    print("=" * 80)

    response = input("\nProceed with SAGE tiling? [y/N]: ").strip().lower()
    if response != "y":
        print("Cancelled.")
        return

    start_time = time.time()
    results: List[Dict[str, object]] = []

    for fold in range(1, NUM_FOLDS + 1):
        for split in SPLITS:
            if fold_dir:
                annotations_path = fold_dir / f"fold_{fold}_{split}.json"
                image_filter = None
            else:
                annotations_path = BASE_DATASET_ANNOTATIONS
                image_filter = fallback_splits.get(fold, {}).get(split, []) if fallback_splits else None

            output_dir = OUTPUT_ROOT / f"fold_{fold}" / split

            app_config = build_app_config(output_dir)
            app_config.tiling.keep_empty_tiles = split == "test"

            success = process_fold_split(
                fold=fold,
                split=split,
                app_config=app_config,
                annotations_path=annotations_path,
                output_dir=output_dir,
                image_filter=image_filter,
            )
            results.append({"fold": fold, "split": split, "success": success})

    elapsed = time.time() - start_time
    elapsed_mins = int(elapsed // 60)
    elapsed_secs = int(elapsed % 60)

    print("\n" + "=" * 80)
    print("SAGE tiling complete!")
    print("=" * 80)

    successful = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"\nResults: {successful}/{total} successful")
    print(f"Time elapsed: {elapsed_mins}m {elapsed_secs}s")

    if successful < total:
        print("\nSome fold/split combinations failed:")
        for r in results:
            if not r["success"]:
                print(f"  - Fold {r['fold']} | {r['split']}")
        sys.exit(1)

    print(f"\nOutputs written to: {OUTPUT_ROOT}")
    print(
        "\nNext steps:\n"
        "1. Inspect the summary reports for distribution checks.\n"
        "2. Review metadata.json for SAGE overlap ratios per split.\n"
        "3. Point your training pipeline to the generated fold directories."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
