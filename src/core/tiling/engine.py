from typing import List, Tuple, Generator
import numpy as np
from PIL import Image

from src.config.settings import TilingConfig
from src.models.coco import CocoAnnotation


class TilingEngine:
    def __init__(self, config: TilingConfig):
        self.config = config
    
    def generate_tiles(self, image: Image.Image) -> Generator[Tuple[Image.Image, Tuple[int, int], float], None, None]:
        """Generate tiles from an image with optional overlap and resizing."""
        img_width, img_height = image.size
        tile_width, tile_height = self.config.tile_size
        overlap = self.config.overlap
        
        step_x = tile_width - overlap
        step_y = tile_height - overlap
        
        # Calculate scaling factor if resize is enabled
        scale_factor = 1.0
        if self.config.resize_output:
            resize_width, resize_height = self.config.resize_output
            scale_factor = min(resize_width / tile_width, resize_height / tile_height)
        
        for y in range(0, img_height - tile_height + 1, step_y):
            for x in range(0, img_width - tile_width + 1, step_x):
                tile = image.crop((x, y, x + tile_width, y + tile_height))
                
                # Resize tile if requested
                if self.config.resize_output:
                    tile = tile.resize(self.config.resize_output, Image.LANCZOS)
                
                yield tile, (x, y), scale_factor
        
        # Handle edge cases - tiles that don't fit perfectly
        if img_width % step_x != 0:
            x = img_width - tile_width
            for y in range(0, img_height - tile_height + 1, step_y):
                tile = image.crop((x, y, x + tile_width, y + tile_height))
                if self.config.resize_output:
                    tile = tile.resize(self.config.resize_output, Image.LANCZOS)
                yield tile, (x, y), scale_factor
        
        if img_height % step_y != 0:
            y = img_height - tile_height
            for x in range(0, img_width - tile_width + 1, step_x):
                tile = image.crop((x, y, x + tile_width, y + tile_height))
                if self.config.resize_output:
                    tile = tile.resize(self.config.resize_output, Image.LANCZOS)
                yield tile, (x, y), scale_factor
        
        # Corner tile if needed
        if img_width % step_x != 0 and img_height % step_y != 0:
            x = img_width - tile_width
            y = img_height - tile_height
            tile = image.crop((x, y, x + tile_width, y + tile_height))
            if self.config.resize_output:
                tile = tile.resize(self.config.resize_output, Image.LANCZOS)
            yield tile, (x, y), scale_factor
    
    def transform_annotations(self, annotations: List[CocoAnnotation], 
                            tile_offset: Tuple[int, int], 
                            scale_factor: float = 1.0) -> List[CocoAnnotation]:
        """Transform annotations for a specific tile with optional scaling."""
        tile_x, tile_y = tile_offset
        tile_width, tile_height = self.config.tile_size
        transformed_annotations = []
        
        for ann in annotations:
            bbox = ann.bbox  # [x, y, width, height]
            x, y, width, height = bbox
            
            # Check if annotation intersects with tile
            if (x + width < tile_x or x > tile_x + tile_width or
                y + height < tile_y or y > tile_y + tile_height):
                continue
            
            # Calculate intersection area for coverage check
            inter_x1 = max(x, tile_x)
            inter_y1 = max(y, tile_y)
            inter_x2 = min(x + width, tile_x + tile_width)
            inter_y2 = min(y + height, tile_y + tile_height)
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            original_area = width * height
            
            # Check if enough of the object is visible
            if inter_area / original_area < self.config.min_object_coverage:
                continue
            
            # Transform coordinates to tile space and apply scaling if needed
            new_x = (x - tile_x) * scale_factor
            new_y = (y - tile_y) * scale_factor
            new_width = width * scale_factor
            new_height = height * scale_factor
            
            # Create new annotation with original dimensions preserved
            new_annotation = CocoAnnotation(
                id=ann.id,  # Will be reassigned later
                image_id=ann.image_id,  # Will be reassigned later
                category_id=ann.category_id,
                segmentation=self._transform_segmentation(ann.segmentation, tile_offset, scale_factor),
                area=original_area * (scale_factor ** 2),  # Scale area by square of scale factor
                bbox=[new_x, new_y, new_width, new_height],
                iscrowd=ann.iscrowd
            )
            transformed_annotations.append(new_annotation)
        
        return transformed_annotations
    
    def _transform_segmentation(self, segmentation: List[List[float]], 
                               tile_offset: Tuple[int, int],
                               scale_factor: float = 1.0) -> List[List[float]]:
        """Transform segmentation coordinates to tile space with optional scaling."""
        tile_x, tile_y = tile_offset
        transformed_segmentation = []
        
        for segment in segmentation:
            transformed_segment = []
            for i in range(0, len(segment), 2):
                x = (segment[i] - tile_x) * scale_factor
                y = (segment[i + 1] - tile_y) * scale_factor
                transformed_segment.extend([x, y])
            transformed_segmentation.append(transformed_segment)
        
        return transformed_segmentation