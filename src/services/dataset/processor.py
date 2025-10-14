import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image

from src.config.settings import AppConfig
from src.core.tiling.engine import GeneratedTile, TilingEngine
from src.core.tiling.sage import compute_adaptive_overlap, compute_sage_grid
from src.models.coco import CocoAnnotation, CocoDataset, CocoImage
from src.services.annotation.manager import AnnotationManager
from src.services.image.handler import ImageHandler


class DatasetProcessor:
    """High-level orchestration for tiling and annotation remapping."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.tiling_engine = TilingEngine(config.tiling)
        self.image_handler = ImageHandler()
        self.annotation_manager = AnnotationManager()
        os.makedirs(self.config.dataset.output_path, exist_ok=True)

    def process_dataset(
        self,
        annotations_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        split_name: str = "train",
        output_dir: Optional[str] = None,
        *,
        mode: Optional[str] = None,
        overlap_ratio: Optional[float] = None,
        keep_empty_tiles: Optional[bool] = None,
        image_filter: Optional[Sequence[int]] = None,
    ) -> Dict[str, float]:
        """Tile a dataset split and rewrite annotations."""

        annotations_path = annotations_path or os.path.join(
            self.config.dataset.input_path, split_name, "_annotations.coco.json"
        )
        images_dir = images_dir or os.path.join(self.config.dataset.input_path, split_name)
        output_dir = output_dir or os.path.join(self.config.dataset.output_path, split_name)
        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()
        print(f"[{split_name}] Loading annotations from {annotations_path}")

        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        dataset = CocoDataset.from_json(annotations_path)

        image_filter_set = set(image_filter) if image_filter else None
        images_to_process = [
            image for image in dataset.images if not image_filter_set or image.id in image_filter_set
        ]

        annotations_by_image: Dict[int, List[CocoAnnotation]] = {}
        for ann in dataset.annotations:
            if image_filter_set and ann.image_id not in image_filter_set:
                continue
            annotations_by_image.setdefault(ann.image_id, []).append(ann)

        effective_mode = (mode or self.config.tiling.mode).lower()
        requested_overlap = (
            overlap_ratio if overlap_ratio is not None else self.config.tiling.overlap_ratio
        )
        if keep_empty_tiles is None:
            keep_empty_tiles = self.config.tiling.keep_empty_tiles

        original_mode = self.config.tiling.mode
        original_overlap = self.config.tiling.overlap_ratio
        original_keep_empty = self.config.tiling.keep_empty_tiles
        original_context_pad = self.config.tiling.context_pad

        effective_overlap_ratio = requested_overlap or 0.0

        try:
            self.config.tiling.mode = effective_mode
            self.config.tiling.keep_empty_tiles = keep_empty_tiles

            if effective_mode == "sage":
                if requested_overlap is None or requested_overlap <= 0.0:
                    requested_overlap = compute_adaptive_overlap(
                        dataset.annotations,
                        self.config.tiling.tile_size,
                        annotation_filter=image_filter,
                    )
                self.config.tiling.overlap_ratio = requested_overlap
                effective_overlap_ratio = self.config.tiling.overlap_ratio
                print(
                    f"[{split_name}] SAGE overlap ratio set to {self.config.tiling.overlap_ratio:.4f} "
                    f"({self.config.tiling.overlap_ratio * 100:.2f}%)"
                )
            else:
                self.config.tiling.overlap_ratio = requested_overlap or 0.0
                effective_overlap_ratio = self.config.tiling.overlap_ratio

            summary = self._tile_split(
                dataset=dataset,
                images=images_to_process,
                annotations_by_image=annotations_by_image,
                images_dir=images_dir,
                output_dir=output_dir,
                split_name=split_name,
                keep_empty_tiles=keep_empty_tiles,
            )
        finally:
            self.config.tiling.mode = original_mode
            self.config.tiling.overlap_ratio = original_overlap
            self.config.tiling.keep_empty_tiles = original_keep_empty
            self.config.tiling.context_pad = original_context_pad

        elapsed = time.time() - start_time
        print(
            f"[{split_name}] Done in {elapsed:.1f}s "
            f"({summary['tiles']} tiles, {summary['annotations']} annotations)"
        )
        summary["overlap_ratio"] = effective_overlap_ratio
        return summary

    def _tile_split(
        self,
        dataset: CocoDataset,
        images: List[CocoImage],
        annotations_by_image: Dict[int, List[CocoAnnotation]],
        images_dir: str,
        output_dir: str,
        split_name: str,
        keep_empty_tiles: bool,
    ) -> Dict[str, int]:
        """Internal helper that performs the tiling loop and writes outputs."""

        tile_size = self.config.tiling.tile_size
        min_coverage = self.config.tiling.min_object_coverage
        resize_output = self.config.tiling.resize_output
        context_pad = self.config.tiling.context_pad

        print(f"[{split_name}] Processing {len(images)} images")
        print(f"[{split_name}] Tile size: {tile_size} | Min coverage: {min_coverage}")
        if context_pad:
            print(f"[{split_name}] Context pad: {context_pad}px")
        if self.config.tiling.mode == "sage":
            print(
                f"[{split_name}] Mode: SAGE | stride overlap {self.config.tiling.overlap_ratio * 100:.2f}%"
            )
        else:
            print(f"[{split_name}] Mode: standard | overlap {self.config.tiling.overlap}px")
        if resize_output:
            print(f"[{split_name}] Resize output to: {resize_output}")
        print(f"[{split_name}] Keep empty tiles: {keep_empty_tiles}")

        new_images: List[CocoImage] = []
        new_annotations: List[CocoAnnotation] = []
        new_image_id = 1
        new_annotation_id = 1
        saved_tiles = 0
        sage_info_printed = False

        for index, original_image in enumerate(images, start=1):
            image_path = os.path.join(images_dir, original_image.file_name)
            if not os.path.exists(image_path):
                print(f"[{split_name}] Warning: missing file {image_path}")
                continue

            with Image.open(image_path) as img:
                if self.config.tiling.mode == "sage" and not sage_info_printed:
                    grid_preview = compute_sage_grid(
                        image_shape=(img.height, img.width),
                        tile_size=tile_size,
                        overlap=self.config.tiling.overlap_ratio,
                    )
                    overlap_px_x = max(0.0, tile_size[0] - grid_preview.stride[0])
                    overlap_px_y = max(0.0, tile_size[1] - grid_preview.stride[1])
                    layout_x, layout_y = grid_preview.layout
                    stride_x, stride_y = grid_preview.stride
                    closed = grid_preview.boxes and grid_preview.boxes[-1][2] == img.width and grid_preview.boxes[-1][3] == img.height
                    print(f"[{split_name}] SAGE adaptive mode")
                    print(
                        f"[{split_name}] Tile size: {tile_size} | overlap: "
                        f"{overlap_px_x:.1f}px x {overlap_px_y:.1f}px "
                        f"({self.config.tiling.overlap_ratio * 100:.2f}%)"
                    )
                    print(
                        f"[{split_name}] Grid: {layout_x} x {layout_y} tiles | "
                        f"stride ~= ({stride_x:.2f}, {stride_y:.2f}) to close at ({img.width}x{img.height})"
                    )
                    print(
                        f"[{split_name}] Corner alignment: "
                        f"{'yes' if closed else 'no'} bottom-right edge closed"
                    )
                    sage_info_printed = True

                image_annotations = annotations_by_image.get(original_image.id, [])
                tile_counter = 0

                for generated_tile in self.tiling_engine.generate_tiles(img):
                    tile_annotations = self.tiling_engine.transform_annotations(
                        image_annotations,
                        tile_offset=generated_tile.origin,
                        tile_size=generated_tile.original_size,
                        scale_factor=generated_tile.scale_factor,
                    )

                    if not tile_annotations and not keep_empty_tiles:
                        continue

                    stem = Path(original_image.file_name).stem
                    grid_origin = generated_tile.grid_origin or generated_tile.origin
                    tile_filename = f"{stem}_tile_{grid_origin[0]}_{grid_origin[1]}.jpg"
                    tile_output_path = os.path.join(output_dir, tile_filename)
                    os.makedirs(os.path.dirname(tile_output_path), exist_ok=True)
                    generated_tile.image.save(tile_output_path)

                    new_image = CocoImage(
                        id=new_image_id,
                        width=generated_tile.image.width,
                        height=generated_tile.image.height,
                        file_name=tile_filename,
                    )
                    new_images.append(new_image)

                    for ann in tile_annotations:
                        ann.id = new_annotation_id
                        ann.image_id = new_image_id
                        new_annotations.append(ann)
                        new_annotation_id += 1

                    new_image_id += 1
                    tile_counter += 1
                    saved_tiles += 1

                print(
                    f"[{split_name}] {index:04d}/{len(images):04d} | "
                    f"{original_image.file_name} -> {tile_counter} tiles"
                )

        output_annotations_path = os.path.join(output_dir, "_annotations.coco.json")
        tiled_dataset = CocoDataset(
            info=dataset.info,
            licenses=dataset.licenses,
            images=new_images,
            annotations=new_annotations,
            categories=dataset.categories,
        )
        tiled_dataset.save_json(output_annotations_path)

        return {
            "tiles": saved_tiles,
            "annotations": len(new_annotations),
            "output_annotations": output_annotations_path,
        }

    def validate_output(self, split_name: str = "train", output_dir: Optional[str] = None) -> bool:
        """Validate that tiled images referenced in the COCO JSON exist on disk."""

        output_dir = output_dir or os.path.join(self.config.dataset.output_path, split_name)
        annotations_path = os.path.join(output_dir, "_annotations.coco.json")

        if not os.path.exists(annotations_path):
            print(f"[{split_name}] Missing annotations file at {annotations_path}")
            return False

        try:
            dataset = CocoDataset.from_json(annotations_path)
            for image in dataset.images:
                image_path = os.path.join(output_dir, image.file_name)
                if not os.path.exists(image_path):
                    print(f"[{split_name}] Missing tile image {image_path}")
                    return False
            return True
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[{split_name}] Validation error: {exc}")
            return False
