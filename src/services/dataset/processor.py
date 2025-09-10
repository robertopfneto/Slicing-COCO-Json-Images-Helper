import os
import shutil
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image

from src.config.settings import AppConfig
from src.models.coco import CocoDataset, CocoImage, CocoAnnotation
from src.core.tiling.engine import TilingEngine
from src.services.image.handler import ImageHandler
from src.services.annotation.manager import AnnotationManager


class DatasetProcessor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tiling_engine = TilingEngine(config.tiling)
        self.image_handler = ImageHandler()
        self.annotation_manager = AnnotationManager()
        
        # Ensure output directory exists
        os.makedirs(config.dataset.output_path, exist_ok=True)
    
    def process_dataset(self) -> None:
        """Process the entire dataset by tiling images and updating annotations."""
        start_time = time.time()
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔄 Loading dataset...")
        
        # Load original COCO dataset
        annotations_path = os.path.join(self.config.dataset.input_path, "train", "_annotations.coco.json")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        
        original_dataset = CocoDataset.from_json(annotations_path)
        
        total_images = len(original_dataset.images)
        total_annotations = len(original_dataset.annotations)
        
        print(f"📊 Dataset loaded: {total_images} images, {total_annotations} annotations")
        print(f"🎯 Tile size: {self.config.tiling.tile_size}")
        print(f"📐 Overlap: {self.config.tiling.overlap} pixels")
        print(f"📏 Min coverage: {self.config.tiling.min_object_coverage}")
        if self.config.tiling.resize_output:
            print(f"🔄 Resize output: {self.config.tiling.resize_output}")
        print()
        
        # Initialize counters for new IDs
        new_image_id = 1
        new_annotation_id = 1
        
        # Storage for new dataset
        new_images = []
        new_annotations = []
        
        # Initialize progress tracking
        processed_images = 0
        generated_tiles = 0
        processed_annotations = 0
        
        print("🚀 Starting image processing...")
        print("=" * 60)
        
        # Process each image
        for i, original_image in enumerate(original_dataset.images, 1):
            # Progress indicator
            progress_pct = (i / total_images) * 100
            print(f"📸 [{i:4d}/{total_images}] ({progress_pct:5.1f}%) Processing: {original_image.file_name}")
            
            image_path = os.path.join(self.config.dataset.input_path, "train", original_image.file_name)
            
            if not os.path.exists(image_path):
                print(f"   ⚠️  Warning: Image file not found: {image_path}")
                continue
            
            # Load image
            image = Image.open(image_path)
            print(f"   📏 Image size: {image.size}")
            
            # Get annotations for this image
            image_annotations = [ann for ann in original_dataset.annotations 
                               if ann.image_id == original_image.id]
            print(f"   🏷️  Annotations: {len(image_annotations)}")
            
            # Track tiles for this image
            image_tile_count = 0
            
            # Generate tiles
            for tile, tile_offset, scale_factor in self.tiling_engine.generate_tiles(image):
                image_tile_count += 1
                # Create new image entry
                tile_filename = f"{Path(original_image.file_name).stem}_tile_{tile_offset[0]}_{tile_offset[1]}.jpg"
                
                new_image = CocoImage(
                    id=new_image_id,
                    width=tile.width,
                    height=tile.height,
                    file_name=tile_filename
                )
                new_images.append(new_image)
                
                # Save tile image
                tile_output_path = os.path.join(self.config.dataset.output_path, "train", tile_filename)
                os.makedirs(os.path.dirname(tile_output_path), exist_ok=True)
                tile.save(tile_output_path)
                
                # Transform annotations for this tile
                tile_annotations = self.tiling_engine.transform_annotations(
                    image_annotations, tile_offset, scale_factor
                )
                
                # Update annotation IDs and image references
                for ann in tile_annotations:
                    ann.id = new_annotation_id
                    ann.image_id = new_image_id
                    new_annotations.append(ann)
                    new_annotation_id += 1
                
                generated_tiles += 1
                processed_annotations += len(tile_annotations)
                new_image_id += 1
            
            # Summary for this image
            print(f"   ✅ Generated {image_tile_count} tiles")
            processed_images += 1
            
            # Show periodic summary
            if i % 50 == 0 or i == total_images:
                print()
                print(f"📈 Progress Summary (after {i} images):")
                print(f"   🖼️  Processed images: {processed_images}")
                print(f"   🧩 Generated tiles: {generated_tiles}")
                print(f"   🏷️  Processed annotations: {processed_annotations}")
                print("=" * 60)
        
        print()
        print("💾 Saving dataset...")
        
        # Create new dataset
        new_dataset = CocoDataset(
            info=original_dataset.info,
            licenses=original_dataset.licenses,
            images=new_images,
            annotations=new_annotations,
            categories=original_dataset.categories
        )
        
        # Save new annotations
        output_annotations_path = os.path.join(self.config.dataset.output_path, "train", "_annotations.coco.json")
        new_dataset.save_json(output_annotations_path)
        print(f"   📄 Annotations saved: {output_annotations_path}")
        
        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)
        
        print()
        print("🎉 Dataset processing complete!")
        print("=" * 60)
        print(f"📊 Final Summary:")
        print(f"   📥 Original images: {len(original_dataset.images)}")
        print(f"   📤 Generated tiles: {len(new_images)}")
        print(f"   📥 Original annotations: {len(original_dataset.annotations)}")
        print(f"   📤 Transformed annotations: {len(new_annotations)}")
        
        if len(new_annotations) > len(original_dataset.annotations):
            duplicates = len(new_annotations) - len(original_dataset.annotations)
            print(f"   🔄 Annotations spanning tiles: {duplicates}")
        
        print(f"⏱️  Processing time: {elapsed_mins}m {elapsed_secs}s")
        print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def validate_output(self) -> bool:
        """Validate the generated dataset."""
        annotations_path = os.path.join(self.config.dataset.output_path, "train", "_annotations.coco.json")
        
        if not os.path.exists(annotations_path):
            return False
        
        try:
            dataset = CocoDataset.from_json(annotations_path)
            
            # Check if all referenced images exist
            for image in dataset.images:
                image_path = os.path.join(self.config.dataset.output_path, "train", image.file_name)
                if not os.path.exists(image_path):
                    print(f"Missing image file: {image_path}")
                    return False
            
            return True
        except Exception as e:
            print(f"Validation error: {e}")
            return False