from typing import List, Dict, Any, Tuple
import json
import os

from src.models.coco import CocoDataset, CocoAnnotation, CocoImage


class AnnotationManager:
    """Manages annotation operations for COCO format datasets."""
    
    @staticmethod
    def load_annotations(annotations_path: str) -> CocoDataset:
        """Load COCO annotations from JSON file."""
        return CocoDataset.from_json(annotations_path)
    
    @staticmethod
    def save_annotations(dataset: CocoDataset, output_path: str) -> bool:
        """Save COCO dataset to JSON file."""
        try:
            dataset.save_json(output_path)
            return True
        except Exception as e:
            print(f"Error saving annotations to {output_path}: {e}")
            return False
    
    @staticmethod
    def get_annotations_for_image(dataset: CocoDataset, image_id: int) -> List[CocoAnnotation]:
        """Get all annotations for a specific image."""
        return [ann for ann in dataset.annotations if ann.image_id == image_id]
    
    @staticmethod
    def filter_annotations_by_category(annotations: List[CocoAnnotation], 
                                     category_ids: List[int]) -> List[CocoAnnotation]:
        """Filter annotations by category IDs."""
        return [ann for ann in annotations if ann.category_id in category_ids]
    
    @staticmethod
    def validate_annotation_bbox(annotation: CocoAnnotation, 
                                image_width: int, image_height: int) -> bool:
        """Validate if annotation bbox is within image boundaries."""
        x, y, width, height = annotation.bbox
        return (x >= 0 and y >= 0 and 
                x + width <= image_width and 
                y + height <= image_height and
                width > 0 and height > 0)
    
    @staticmethod
    def calculate_bbox_area(bbox: List[float]) -> float:
        """Calculate area of a bounding box [x, y, width, height]."""
        return bbox[2] * bbox[3]
    
    @staticmethod
    def bbox_intersection(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate intersection area between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection coordinates
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)
        
        # Check if there's an intersection
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        return 0.0
    
    @staticmethod
    def merge_datasets(datasets: List[CocoDataset]) -> CocoDataset:
        """Merge multiple COCO datasets into one."""
        if not datasets:
            raise ValueError("No datasets provided")
        
        base_dataset = datasets[0]
        merged_images = list(base_dataset.images)
        merged_annotations = list(base_dataset.annotations)
        merged_categories = list(base_dataset.categories)
        
        # Track used IDs to avoid conflicts
        max_image_id = max(img.id for img in merged_images) if merged_images else 0
        max_annotation_id = max(ann.id for ann in merged_annotations) if merged_annotations else 0
        max_category_id = max(cat.id for cat in merged_categories) if merged_categories else 0
        
        for dataset in datasets[1:]:
            # Create ID mappings
            image_id_mapping = {}
            category_id_mapping = {}
            
            # Merge categories
            for category in dataset.categories:
                existing_cat = next((cat for cat in merged_categories if cat.name == category.name), None)
                if existing_cat:
                    category_id_mapping[category.id] = existing_cat.id
                else:
                    max_category_id += 1
                    new_category = category
                    new_category.id = max_category_id
                    merged_categories.append(new_category)
                    category_id_mapping[category.id] = max_category_id
            
            # Merge images
            for image in dataset.images:
                max_image_id += 1
                image_id_mapping[image.id] = max_image_id
                new_image = image
                new_image.id = max_image_id
                merged_images.append(new_image)
            
            # Merge annotations
            for annotation in dataset.annotations:
                max_annotation_id += 1
                new_annotation = annotation
                new_annotation.id = max_annotation_id
                new_annotation.image_id = image_id_mapping[annotation.image_id]
                new_annotation.category_id = category_id_mapping[annotation.category_id]
                merged_annotations.append(new_annotation)
        
        return CocoDataset(
            info=base_dataset.info,
            licenses=base_dataset.licenses,
            images=merged_images,
            annotations=merged_annotations,
            categories=merged_categories
        )