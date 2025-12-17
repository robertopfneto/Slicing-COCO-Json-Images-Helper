from typing import List, Tuple, Generator
import numpy as np
from PIL import Image

from src.config.settings import TilingConfig
from src.models.coco import CocoAnnotation


class TilingEngine:
    def __init__(self, config: TilingConfig):
        self.config = config
        self.last_grid_info = None
    
    def _axis_positions(self, img_dim: int, tile_dim: int) -> Tuple[List[int], int, int, int]:
        """Return tile start positions along one axis plus effective stride/overlap."""
        stride = max(1, tile_dim - self.config.overlap)

        positions = list(range(0, max(img_dim - tile_dim + 1, 1), stride))
        if positions and positions[-1] != img_dim - tile_dim:
            positions.append(img_dim - tile_dim)

        positions = sorted(set(positions))
        stride_effective = stride if len(positions) > 1 else tile_dim
        overlap_effective = max(0, tile_dim - stride_effective)
        redundancy = 0
        if positions:
            redundancy = max(
                0, (len(positions) - 1) * stride_effective + tile_dim - img_dim
            )

        return positions, stride_effective, overlap_effective, redundancy

    def generate_tiles(self, image: Image.Image) -> Generator[Tuple[Image.Image, Tuple[int, int], float], None, None]:
        """Generate tiles from an image with optional overlap and resizing."""
        img_width, img_height = image.size
        tile_width, tile_height = self.config.tile_size
        x_positions, step_x, overlap_x, redundancy_x = self._axis_positions(img_width, tile_width)
        y_positions, step_y, overlap_y, redundancy_y = self._axis_positions(img_height, tile_height)

        self.last_grid_info = {
            "cols": len(x_positions),
            "rows": len(y_positions),
            "stride_x": step_x,
            "stride_y": step_y,
            "overlap_x": overlap_x,
            "overlap_y": overlap_y,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "image_width": img_width,
            "image_height": img_height,
            "redundancy_x": redundancy_x,
            "redundancy_y": redundancy_y,
        }

        # Calculate scaling factor if resize is enabled
        scale_factor = 1.0
        if self.config.resize_output:
            resize_width, resize_height = self.config.resize_output
            scale_factor = min(resize_width / tile_width, resize_height / tile_height)

        for y in y_positions:
            for x in x_positions:
                tile = image.crop((x, y, x + tile_width, y + tile_height))
                
                # Resize tile if requested
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

            tile_x1 = tile_x
            tile_y1 = tile_y
            tile_x2 = tile_x + tile_width
            tile_y2 = tile_y + tile_height

            # Check if annotation intersects with tile
            if (x + width <= tile_x1 or x >= tile_x2 or
                y + height <= tile_y1 or y >= tile_y2):
                continue
            
            # Calculate intersection area for coverage check
            inter_x1 = max(x, tile_x1)
            inter_y1 = max(y, tile_y1)
            inter_x2 = min(x + width, tile_x2)
            inter_y2 = min(y + height, tile_y2)

            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            original_area = width * height
            
            # Check if enough of the object is visible
            if inter_area / original_area < self.config.min_object_coverage:
                continue
            
            # Clip bbox to tile bounds and transform to tile space
            clipped_x1 = inter_x1
            clipped_y1 = inter_y1
            clipped_x2 = inter_x2
            clipped_y2 = inter_y2

            new_x = (clipped_x1 - tile_x1) * scale_factor
            new_y = (clipped_y1 - tile_y1) * scale_factor
            new_width = (clipped_x2 - clipped_x1) * scale_factor
            new_height = (clipped_y2 - clipped_y1) * scale_factor

            if new_width <= 0 or new_height <= 0:
                continue
            
            # Create new annotation with original dimensions preserved
            new_annotation = CocoAnnotation(
                id=ann.id,  # Will be reassigned later
                image_id=ann.image_id,  # Will be reassigned later
                category_id=ann.category_id,
                segmentation=self._transform_segmentation(ann.segmentation, tile_offset, scale_factor),
                area=(new_width * new_height),
                bbox=[new_x, new_y, new_width, new_height],
                iscrowd=ann.iscrowd,
                extra={
                    "source_annotation_id": ann.id,
                    "intersection_over_area": inter_area / original_area,
                    "original_bbox": bbox,
                    "tile_offset": {
                        "x": tile_x,
                        "y": tile_y
                    }
                }
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
