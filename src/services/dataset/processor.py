import os
import shutil
import time
import random
import statistics
import json
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from PIL import Image

from src.config.settings import AppConfig
from src.models.coco import CocoAnnotation, CocoDataset, CocoImage
from src.core.tiling.engine import TilingEngine
from src.services.annotation.manager import AnnotationManager
from src.services.image.handler import ImageHandler


class DatasetProcessor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tiling_engine = TilingEngine(config.tiling)
        self.image_handler = ImageHandler()
        self.annotation_manager = AnnotationManager()
        self.rng = random.Random(config.processing.fold_seed)
        self.adaptive_context: Optional[Dict[str, Any]] = None
        self.adaptive_metrics: Dict[str, float] = {
            "images": 0.0,
            "baseline_tiles": 0.0,
            "adaptive_tiles": 0.0,
        }
        self._tiling_plan_written = False

        # Ensure base output directory exists
        os.makedirs(config.dataset.output_path, exist_ok=True)

    def process_dataset(self) -> None:
        """Process the dataset, tile images, and export K-fold cross-validation splits."""
        start_time = time.time()
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Loading dataset...")

        annotations_path = os.path.join(self.config.dataset.input_path, "train", "_annotations.coco.json")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

        original_dataset = CocoDataset.from_json(annotations_path)
        total_images = len(original_dataset.images)
        total_annotations = len(original_dataset.annotations)

        print(f"Dataset loaded: {total_images} images, {total_annotations} annotations")
        print(f"Tile size: {self.config.tiling.tile_size}")
        print(f"Overlap: {self.config.tiling.overlap} pixels")
        print(f"Min coverage: {self.config.tiling.min_object_coverage}")
        if self.config.tiling.resize_output:
            print(f"Resize output: {self.config.tiling.resize_output}")
        print()

        if self.config.tiling.adaptive_mode:
            self.adaptive_context = self._build_adaptive_context(original_dataset)
            self.tiling_engine.set_dataset_context(self.adaptive_context)
            self._log_adaptive_context(self.adaptive_context)

        tile_root = self._prepare_output_root()
        num_folds = max(1, self.config.processing.num_folds)

        image_ids = [img.id for img in original_dataset.images]
        if self.config.processing.shuffle_folds and len(image_ids) > 1:
            self.rng.shuffle(image_ids)
        role_lookup = self._build_role_lookup(image_ids, num_folds)

        fold_storage = self._initialise_fold_storage(num_folds)

        processed_images = 0
        skipped_images = 0
        skipped_train_tiles = 0
        negative_tiles_by_split = {"val": 0, "test": 0}
        unique_tiles = 0
        unique_annotations = 0
        total_tile_area = 0
        total_original_area = 0

        print("Starting image processing...")
        print("=" * 60)

        for idx, original_image in enumerate(original_dataset.images, 1):
            progress_pct = (idx / total_images) * 100 if total_images else 100
            print(f"[{idx:4d}/{total_images}] ({progress_pct:5.1f}%) Processing: {original_image.file_name}")

            image_path = os.path.join(self.config.dataset.input_path, "train", original_image.file_name)
            if not os.path.exists(image_path):
                print(f"  Warning: image file not found: {image_path}")
                continue

            image_annotations = [
                ann for ann in original_dataset.annotations if ann.image_id == original_image.id
            ]
            print(f"  Annotations: {len(image_annotations)}")

            skip_train_negative = (
                self.config.tiling.ignore_negative_samples and not image_annotations
            )
            if skip_train_negative:
                print("  No annotations; skipping tiles for train splits only")
                skipped_images += 1

            image_tile_count = 0
            with Image.open(image_path) as pil_image:
                image_width, image_height = pil_image.size
                total_original_area += image_width * image_height
                for tile, slice_bbox, scale_factor in self.tiling_engine.generate_tiles(pil_image):
                    image_tile_count += 1
                    unique_tiles += 1

                    tile_offset = (slice_bbox[0], slice_bbox[1])
                    tile_filename = f"{Path(original_image.file_name).stem}_tile_{tile_offset[0]}_{tile_offset[1]}.jpg"
                    tile_width, tile_height = tile.size
                    total_tile_area += tile_width * tile_height

                    tile_annotations = self.tiling_engine.transform_annotations(
                        image_annotations, slice_bbox, scale_factor
                    )
                    unique_annotations += len(tile_annotations)

                    tile_bytes = self._encode_tile(tile, tile_filename)

                    for fold_idx in range(1, num_folds + 1):
                        split_name = role_lookup[fold_idx][original_image.id]
                        if skip_train_negative and split_name == "train":
                            skipped_train_tiles += 1
                            continue
                        if skip_train_negative and split_name in negative_tiles_by_split:
                            negative_tiles_by_split[split_name] += 1
                        split_dir = self._ensure_split_dir(tile_root, fold_idx, split_name)
                        split_store = fold_storage[fold_idx][split_name]

                        image_id = split_store["next_image_id"]
                        split_store["next_image_id"] += 1

                        split_store["images"].append(
                            CocoImage(
                                id=image_id,
                                width=tile_width,
                                height=tile_height,
                                file_name=tile_filename,
                            )
                        )

                        for ann in tile_annotations:
                            ann_id = split_store["next_annotation_id"]
                            split_store["next_annotation_id"] += 1
                            split_store["annotations"].append(
                                self._clone_annotation(ann, ann_id, image_id)
                            )

                        split_store["tile_count"] += 1
                        split_store["annotation_count"] += len(tile_annotations)

                        destination = os.path.join(split_dir, tile_filename)
                        with open(destination, "wb") as output_file:
                            output_file.write(tile_bytes)

                    tile.close()

            if self.config.tiling.adaptive_mode:
                plan_summary = self.tiling_engine.get_last_plan_summary()
                self._accumulate_adaptive_metrics(plan_summary)
                self._maybe_write_tiling_plan(plan_summary)

            print(f"  Generated {image_tile_count} tiles")
            processed_images += 1

            if idx % 50 == 0 or idx == total_images:
                print()
                print(f"Progress summary (after {idx} images):")
                print(f"  Processed images: {processed_images}")
                print(f"  Generated tiles: {unique_tiles}")
                print(f"  Transformed annotations: {unique_annotations}")
                print("=" * 60)

        print()
        print("Saving cross-validation folds...")

        total_fold_tiles = 0
        total_fold_annotations = 0

        for fold_idx in sorted(fold_storage):
            split_map = fold_storage[fold_idx]
            for split_name, split_data in split_map.items():
                split_dir = self._ensure_split_dir(tile_root, fold_idx, split_name)
                dataset = CocoDataset(
                    info=original_dataset.info,
                    licenses=original_dataset.licenses,
                    images=split_data["images"],
                    annotations=split_data["annotations"],
                    categories=original_dataset.categories,
                )

                annotations_output = os.path.join(split_dir, "_annotations.coco.json")
                dataset.save_json(annotations_output)
                print(
                    f"  Fold {fold_idx} [{split_name}] -> "
                    f"{len(split_data['images'])} tiles, {len(split_data['annotations'])} annotations"
                )

                total_fold_tiles += split_data["tile_count"]
                total_fold_annotations += split_data["annotation_count"]

        elapsed_time = time.time() - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)

        print()
        print("Dataset processing complete!")
        print("=" * 60)
        print("Final summary:")
        print(f"  Original images: {total_images}")
        print(f"  Unique tiles generated: {unique_tiles}")
        print(f"  Unique transformed annotations: {unique_annotations}")
        print(f"  Total tile copies across folds: {total_fold_tiles}")
        print(f"  Total annotation copies across folds: {total_fold_annotations}")
        if total_tile_area > 0:
            redundant_area = max(0, total_tile_area - total_original_area)
            redundancy_pct = (redundant_area / total_tile_area) * 100
            print(f"  Total original area (px): {total_original_area}")
            print(f"  Total tile area (px): {total_tile_area}")
            print(f"  Redundant area (px): {redundant_area}")
            print(f"  Redundant area (% of tiles): {redundancy_pct:.2f}%")
        print(f"  Images without annotations (train skipped): {skipped_images}")
        print(f"  Train tiles skipped (no annotations): {skipped_train_tiles}")
        print(
            "  Negative tiles kept (val/test): "
            f"{negative_tiles_by_split['val']} val, {negative_tiles_by_split['test']} test"
        )
        print(f"  Cross-validation folds: {num_folds}")
        print(f"  Processing time: {elapsed_mins}m {elapsed_secs}s")
        if self.config.tiling.adaptive_mode:
            self._log_adaptive_summary()
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    def validate_output(self) -> bool:
        """Validate the generated cross-validation datasets."""
        tile_root = os.path.join(self.config.dataset.output_path, "tile")
        if not os.path.isdir(tile_root):
            print(f"Missing tile directory: {tile_root}")
            return False

        fold_dirs = [name for name in os.listdir(tile_root) if name.startswith("fold_")]
        if not fold_dirs:
            print("No fold directories found in tile output.")
            return False

        for fold_name in fold_dirs:
            for split_name in ("train", "val", "test"):
                split_dir = os.path.join(tile_root, fold_name, split_name)
                if not os.path.isdir(split_dir):
                    print(f"Missing split directory: {split_dir}")
                    return False

                annotations_path = os.path.join(split_dir, "_annotations.coco.json")
                if not os.path.exists(annotations_path):
                    print(f"Missing annotations file: {annotations_path}")
                    return False

                dataset = CocoDataset.from_json(annotations_path)
                for image in dataset.images:
                    image_path = os.path.join(split_dir, image.file_name)
                    if not os.path.exists(image_path):
                        print(f"Missing image file: {image_path}")
                        return False

        return True

    def clean_fold_train_without_annotations(self) -> None:
        """Remove train tiles without annotations from existing fold outputs."""
        tile_root = os.path.join(self.config.dataset.output_path, "tile")
        if not os.path.isdir(tile_root):
            print(f"No tile directory found at: {tile_root}")
            return

        fold_dirs = sorted(name for name in os.listdir(tile_root) if name.startswith("fold_"))
        if not fold_dirs:
            print("No fold directories found to clean.")
            return

        total_removed = 0
        for fold_name in fold_dirs:
            train_dir = os.path.join(tile_root, fold_name, "train")
            annotations_path = os.path.join(train_dir, "_annotations.coco.json")

            if not os.path.isdir(train_dir) or not os.path.exists(annotations_path):
                continue

            dataset = CocoDataset.from_json(annotations_path)
            cleaned_dataset, removed_images = self._filter_unannotated_images(dataset)

            if not removed_images:
                print(f" Fold {fold_name}: no empty tiles detected.")
                continue

            for image in removed_images:
                image_path = os.path.join(train_dir, image.file_name)
                if os.path.exists(image_path):
                    os.remove(image_path)

            cleaned_dataset.save_json(annotations_path)
            print(f" Fold {fold_name}: removed {len(removed_images)} tiles without annotations.")
            total_removed += len(removed_images)

        if total_removed == 0:
            print("\nNo tiles without annotations were found.")
        else:
            print(f"\nTotal tiles removed without annotations: {total_removed}")

    def _build_adaptive_context(self, dataset: CocoDataset) -> Dict[str, Any]:
        valid_dims: List[Tuple[int, int]] = [
            (image.width, image.height)
            for image in dataset.images
            if image.width and image.height
        ]
        if not valid_dims:
            return {
                "count": 0,
                "mean_width": 0.0,
                "mean_height": 0.0,
                "median_width": 0.0,
                "median_height": 0.0,
                "mean_resolution": 0.0,
                "median_resolution": 0.0,
                "max_side": 0.0,
                "ratio_over_ls": 0.0,
                "ls_threshold": self.config.tiling.ls_threshold,
            }

        widths = [width for width, _ in valid_dims]
        heights = [height for _, height in valid_dims]
        resolutions = [width * height for width, height in valid_dims]
        mean_width = statistics.mean(widths)
        mean_height = statistics.mean(heights)
        median_width = statistics.median(widths)
        median_height = statistics.median(heights)
        mean_resolution = statistics.mean(resolutions)
        median_resolution = statistics.median(resolutions)
        max_side = max(max(widths), max(heights))
        ls_threshold = self.config.tiling.ls_threshold
        over_ls = sum(1 for width, height in valid_dims if max(width, height) > ls_threshold)
        ratio_over_ls = over_ls / len(valid_dims) if valid_dims else 0.0

        return {
            "count": len(valid_dims),
            "mean_width": float(mean_width),
            "mean_height": float(mean_height),
            "median_width": float(median_width),
            "median_height": float(median_height),
            "mean_resolution": float(mean_resolution),
            "median_resolution": float(median_resolution),
            "max_side": float(max_side),
            "ratio_over_ls": float(ratio_over_ls),
            "ls_threshold": float(ls_threshold),
        }

    @staticmethod
    def _log_adaptive_context(context: Dict[str, Any]) -> None:
        print("ASAHI adaptive dataset context")
        print("-" * 40)
        print(f"  Samples analysed: {context.get('count', 0)}")
        print(
            "  Mean size: "
            f"{context.get('mean_width', 0.0):.1f} x {context.get('mean_height', 0.0):.1f}"
        )
        print(
            "  Median size: "
            f"{context.get('median_width', 0.0):.1f} x {context.get('median_height', 0.0):.1f}"
        )
        print(f"  Max side length: {context.get('max_side', 0.0):.1f}")
        print(
            f"  LS threshold: {context.get('ls_threshold', 0.0):.2f} "
            f"(>{context.get('ratio_over_ls', 0.0)*100:.1f}% images)"
        )
        print(
            "  Mean/median resolution: "
            f"{context.get('mean_resolution', 0.0):.1f} / {context.get('median_resolution', 0.0):.1f}"
        )
        print()

    def _accumulate_adaptive_metrics(self, plan_summary: Dict[str, Any]) -> None:
        if not plan_summary or plan_summary.get("mode") != "ASAHI":
            return

        self.adaptive_metrics["images"] += 1
        self.adaptive_metrics["baseline_tiles"] += float(plan_summary.get("baseline_total", 0.0) or 0.0)
        self.adaptive_metrics["adaptive_tiles"] += float(plan_summary.get("actual_total", 0.0) or 0.0)

    def _log_adaptive_summary(self) -> None:
        images = int(self.adaptive_metrics.get("images", 0))
        if images <= 0:
            return

        baseline_tiles = self.adaptive_metrics.get("baseline_tiles", 0.0)
        adaptive_tiles = self.adaptive_metrics.get("adaptive_tiles", 0.0)
        avg_baseline = baseline_tiles / images if images else 0.0
        avg_adaptive = adaptive_tiles / images if images else 0.0
        redundancy = 0.0
        if baseline_tiles > 0:
            redundancy = max(0.0, 1.0 - (adaptive_tiles / baseline_tiles))

        print("ASAHI tiling summary")
        print("-" * 40)
        print(f"  Images processed (adaptive): {images}")
        print(f"  Avg baseline tiles per image: {avg_baseline:.2f}")
        print(f"  Avg ASAHI tiles per image:   {avg_adaptive:.2f}")
        print(f"  Estimated redundancy drop:   {redundancy * 100:.1f}%")
        print()

    def _maybe_write_tiling_plan(self, plan_summary: Dict[str, Any]) -> None:
        """Persist tiling plan metadata for later reconstruction if available."""
        if self._tiling_plan_written:
            return
        if not plan_summary or plan_summary.get("mode") != "ASAHI-AutoOverlap":
            return

        target_size = plan_summary.get("target_reconstruction_size", (0, 0))
        if isinstance(target_size, tuple):
            target_size = list(target_size)

        plan_payload = {
            "tile_size": int(plan_summary.get("tile_size", 0) or 0),
            "stride_x": int(plan_summary.get("stride_x", 0) or 0),
            "stride_y": int(plan_summary.get("stride_y", 0) or 0),
            "overlap_ratio_x": float(plan_summary.get("overlap_ratio_x", 0.0) or 0.0),
            "overlap_ratio_y": float(plan_summary.get("overlap_ratio_y", 0.0) or 0.0),
            "target_reconstruction_size": target_size,
        }

        output_path = os.path.join(self.config.dataset.output_path, "tiling_plan.json")
        try:
            with open(output_path, "w", encoding="utf-8") as fp:
                json.dump(plan_payload, fp, indent=2)
            self._tiling_plan_written = True
            if self.config.tiling.verbose:
                print(f"[ASAHI-AutoOverlap] Saved tiling plan metadata to {output_path}")
        except OSError:
            if self.config.tiling.verbose:
                print(f"[ASAHI-AutoOverlap] Failed to save tiling plan at {output_path}")

    def _prepare_output_root(self) -> str:
        tile_root = os.path.join(self.config.dataset.output_path, "tile")
        if os.path.exists(tile_root):
            shutil.rmtree(tile_root)
        os.makedirs(tile_root, exist_ok=True)
        return tile_root

    def _build_role_lookup(self, ordered_ids: List[int], num_folds: int) -> Dict[int, Dict[int, str]]:
        total = len(ordered_ids)
        if total == 0:
            return {fold_idx: {} for fold_idx in range(1, num_folds + 1)}

        val_ratio = max(self.config.dataset.val_split, 0.0)
        test_ratio = max(self.config.dataset.test_split, 0.0)
        train_ratio = max(self.config.dataset.train_split, 0.0)

        ratio_sum = val_ratio + test_ratio + train_ratio
        if ratio_sum == 0:
            val_ratio = 0.1
            test_ratio = 0.1
            train_ratio = 0.8
            ratio_sum = 1.0

        val_ratio /= ratio_sum
        test_ratio /= ratio_sum

        val_count = int(round(total * val_ratio))
        test_count = int(round(total * test_ratio))

        if val_count + test_count > total:
            overflow = val_count + test_count - total
            if test_count >= overflow:
                test_count -= overflow
            else:
                val_count = max(0, val_count - (overflow - test_count))
                test_count = 0

        if val_count == 0 and val_ratio > 0:
            val_count = 1
        if test_count == 0 and test_ratio > 0:
            test_count = 1 if total - val_count > 0 else 0

        lookup: Dict[int, Dict[int, str]] = {}
        for fold_idx in range(1, num_folds + 1):
            val_start = ((fold_idx - 1) * val_count) % total if total else 0
            val_ids = set(self._rotating_slice(ordered_ids, val_start, val_count))

            test_start = (val_start + val_count) % total if total else 0
            test_ids = set(self._rotating_slice(ordered_ids, test_start, test_count, skip=val_ids))

            fold_lookup: Dict[int, str] = {}
            for image_id in ordered_ids:
                if image_id in val_ids:
                    fold_lookup[image_id] = "val"
                elif image_id in test_ids:
                    fold_lookup[image_id] = "test"
                else:
                    fold_lookup[image_id] = "train"
            lookup[fold_idx] = fold_lookup

        return lookup

    @staticmethod
    def _rotating_slice(items: List[int], start: int, length: int, skip: Optional[set] = None) -> List[int]:
        if length <= 0 or not items:
            return []

        result: List[int] = []
        total = len(items)
        index = start % total
        visited = 0
        skip = skip or set()

        while len(result) < length and visited < total:
            candidate = items[index]
            if candidate not in skip and candidate not in result:
                result.append(candidate)
            index = (index + 1) % total
            visited += 1
            if visited >= total and len(result) < length:
                break

        return result

    @staticmethod
    def _initialise_fold_storage(num_folds: int) -> Dict[int, Dict[str, Dict[str, object]]]:
        storage: Dict[int, Dict[str, Dict[str, object]]] = {}
        for fold_idx in range(1, num_folds + 1):
            storage[fold_idx] = {}
            for split_name in ("train", "val", "test"):
                storage[fold_idx][split_name] = {
                    "images": [],
                    "annotations": [],
                    "next_image_id": 1,
                    "next_annotation_id": 1,
                    "tile_count": 0,
                    "annotation_count": 0,
                }
        return storage

    @staticmethod
    def _filter_unannotated_images(dataset: CocoDataset) -> Tuple[CocoDataset, List[CocoImage]]:
        annotations_by_image: Dict[int, List[CocoAnnotation]] = {}
        for annotation in dataset.annotations:
            annotations_by_image.setdefault(annotation.image_id, []).append(annotation)

        filtered_images: List[CocoImage] = []
        filtered_annotations: List[CocoAnnotation] = []
        removed_images: List[CocoImage] = []

        for image in dataset.images:
            image_annotations = annotations_by_image.get(image.id, [])
            if image_annotations:
                filtered_images.append(image)
                filtered_annotations.extend(image_annotations)
            else:
                removed_images.append(image)

        if not removed_images:
            return dataset, []

        cleaned_dataset = CocoDataset(
            info=dataset.info,
            licenses=dataset.licenses,
            images=filtered_images,
            annotations=filtered_annotations,
            categories=dataset.categories,
        )

        return cleaned_dataset, removed_images

    @staticmethod
    def _ensure_split_dir(tile_root: str, fold_idx: int, split_name: str) -> str:
        split_dir = os.path.join(tile_root, f"fold_{fold_idx}", split_name)
        os.makedirs(split_dir, exist_ok=True)
        return split_dir

    @staticmethod
    def _encode_tile(tile: Image.Image, filename: str) -> bytes:
        suffix = Path(filename).suffix.lower()
        format_map = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".tif": "TIFF",
            ".tiff": "TIFF",
        }
        pil_format = format_map.get(suffix, suffix.replace(".", "").upper() or "JPEG")
        buffer = BytesIO()
        tile.save(buffer, format=pil_format)
        return buffer.getvalue()

    @staticmethod
    def _clone_annotation(source: CocoAnnotation, annotation_id: int, image_id: int) -> CocoAnnotation:
        return CocoAnnotation(
            id=annotation_id,
            image_id=image_id,
            category_id=source.category_id,
            segmentation=[[float(coord) for coord in segment] for segment in source.segmentation],
            area=float(source.area),
            bbox=[float(coord) for coord in source.bbox],
            iscrowd=source.iscrowd,
        )
