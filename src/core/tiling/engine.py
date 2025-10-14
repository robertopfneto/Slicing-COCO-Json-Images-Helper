from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

from PIL import Image

from src.config.settings import TilingConfig
from src.core.tiling.sage import iter_sage_boxes
from src.models.coco import CocoAnnotation


@dataclass
class GeneratedTile:
    """Container for a generated tile and its metadata."""

    image: Image.Image
    origin: Tuple[int, int]  # Top-left corner used to crop the tile (after padding/clamp)
    original_size: Tuple[int, int]  # Width/height before optional resize
    scale_factor: float
    grid_origin: Optional[Tuple[int, int]] = None  # SAGE stride origin without padding


class TilingEngine:
    def __init__(self, config: TilingConfig):
        self.config = config
    
    def generate_tiles(self, image: Image.Image) -> Generator[GeneratedTile, None, None]:
        """Generate tiles from an image according to the configured strategy."""
        if self.config.mode == "sage":
            yield from self._generate_tiles_sage(image)
        else:
            yield from self._generate_tiles_standard(image)
    
    def _generate_tiles_standard(self, image: Image.Image) -> Generator[GeneratedTile, None, None]:
        """Generate tiles using the legacy sliding-window strategy."""
        img_width, img_height = image.size
        tile_width, tile_height = self.config.tile_size
        overlap = self.config.overlap
        
        step_x = max(1, tile_width - overlap)
        step_y = max(1, tile_height - overlap)
        
        for y in range(0, max(1, img_height - tile_height + 1), step_y):
            for x in range(0, max(1, img_width - tile_width + 1), step_x):
                yield self._crop_tile(image, x, y, tile_width, tile_height)
        
        # Handle edge cases - tiles that don't fit perfectly
        if img_width % step_x != 0 and img_width > tile_width:
            x = img_width - tile_width
            for y in range(0, max(1, img_height - tile_height + 1), step_y):
                yield self._crop_tile(image, x, y, tile_width, tile_height)
        
        if img_height % step_y != 0 and img_height > tile_height:
            y = img_height - tile_height
            for x in range(0, max(1, img_width - tile_width + 1), step_x):
                yield self._crop_tile(image, x, y, tile_width, tile_height)
        
        # Corner tile if needed
        if (
            img_width % step_x != 0
            and img_height % step_y != 0
            and img_width > tile_width
            and img_height > tile_height
        ):
            x = img_width - tile_width
            y = img_height - tile_height
            yield self._crop_tile(image, x, y, tile_width, tile_height)
    
    def _generate_tiles_sage(self, image: Image.Image) -> Generator[GeneratedTile, None, None]:
        """Generate tiles using the SAGE stride-aligned strategy with optional context padding."""
        img_width, img_height = image.size
        tile_width, tile_height = self.config.tile_size
        overlap_ratio = max(0.0, min(self.config.overlap_ratio, 0.5))
        pad = max(0, self.config.context_pad)
        
        for (x1, y1, x2, y2) in iter_sage_boxes((img_height, img_width), (tile_width, tile_height), overlap_ratio):
            left = max(0, x1 - pad)
            top = max(0, y1 - pad)
            right = min(img_width, x2 + pad)
            bottom = min(img_height, y2 + pad)
            yield self._crop_tile(
                image=image,
                x=left,
                y=top,
                box_width=right - left,
                box_height=bottom - top,
                grid_origin=(x1, y1),
            )
    
    def _crop_tile(
        self,
        image: Image.Image,
        x: int,
        y: int,
        box_width: int,
        box_height: int,
        grid_origin: Optional[Tuple[int, int]] = None,
    ) -> GeneratedTile:
        """Crop a tile, optionally resize it, and package metadata."""
        img_width, img_height = image.size
        right = min(img_width, x + box_width)
        bottom = min(img_height, y + box_height)
        left = max(0, x)
        top = max(0, y)

        original_width = max(1, right - left)
        original_height = max(1, bottom - top)

        tile = image.crop((left, top, left + original_width, top + original_height))

        scale_factor = 1.0
        if self.config.resize_output:
            resize_width, resize_height = self.config.resize_output
            scale_factor = min(resize_width / original_width, resize_height / original_height)
            tile = tile.resize(self.config.resize_output, Image.LANCZOS)

        return GeneratedTile(
            image=tile,
            origin=(left, top),
            original_size=(original_width, original_height),
            scale_factor=scale_factor,
            grid_origin=grid_origin,
        )
    
    def transform_annotations(
        self,
        annotations: List[CocoAnnotation],
        tile_offset: Tuple[int, int],
        tile_size: Tuple[int, int],
        scale_factor: float = 1.0,
    ) -> List[CocoAnnotation]:
        """Transform annotations for a specific tile with optional scaling."""
        tile_x, tile_y = tile_offset
        tile_width, tile_height = tile_size
        tile_x2 = tile_x + tile_width
        tile_y2 = tile_y + tile_height
        transformed_annotations: List[CocoAnnotation] = []
        
        for ann in annotations:
            x, y, width, height = ann.bbox  # [x, y, width, height]
            
            # Check if annotation intersects with tile
            if (x + width <= tile_x or x >= tile_x2 or y + height <= tile_y or y >= tile_y2):
                continue
            
            # Calculate intersection area for coverage check
            inter_x1 = max(x, tile_x)
            inter_y1 = max(y, tile_y)
            inter_x2 = min(x + width, tile_x2)
            inter_y2 = min(y + height, tile_y2)
            
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            original_area = max(1e-6, width * height)
            
            # Check if enough of the object is visible
            if inter_area / original_area < self.config.min_object_coverage:
                continue
            
            # Transform coordinates to tile space and apply scaling if needed.
            local_x1 = max(0.0, inter_x1 - tile_x)
            local_y1 = max(0.0, inter_y1 - tile_y)
            local_x2 = min(float(tile_width), inter_x2 - tile_x)
            local_y2 = min(float(tile_height), inter_y2 - tile_y)

            clipped_width = max(0.0, local_x2 - local_x1)
            clipped_height = max(0.0, local_y2 - local_y1)
            if clipped_width <= 0.0 or clipped_height <= 0.0:
                continue

            new_x = local_x1 * scale_factor
            new_y = local_y1 * scale_factor
            new_width = clipped_width * scale_factor
            new_height = clipped_height * scale_factor
            new_area = new_width * new_height
            
            # Create new annotation with bbox clipped to the tile bounds.
            new_annotation = CocoAnnotation(
                id=ann.id,  # Will be reassigned later
                image_id=ann.image_id,  # Will be reassigned later
                category_id=ann.category_id,
                segmentation=self._transform_segmentation(ann.segmentation, tile_offset, scale_factor),
                area=new_area,
                bbox=[new_x, new_y, new_width, new_height],
                iscrowd=ann.iscrowd
            )
            transformed_annotations.append(new_annotation)
        
        return transformed_annotations
    
    def _transform_segmentation(
        self,
        segmentation: List[List[float]],
        tile_offset: Tuple[int, int],
        scale_factor: float = 1.0,
    ) -> List[List[float]]:
        """Transform segmentation coordinates to tile space with optional scaling."""
        tile_x, tile_y = tile_offset
        transformed_segmentation: List[List[float]] = []
        
        for segment in segmentation:
            transformed_segment: List[float] = []
            for i in range(0, len(segment), 2):
                x = (segment[i] - tile_x) * scale_factor
                y = (segment[i + 1] - tile_y) * scale_factor
                transformed_segment.extend([x, y])
            transformed_segmentation.append(transformed_segment)
        
        return transformed_segmentation
