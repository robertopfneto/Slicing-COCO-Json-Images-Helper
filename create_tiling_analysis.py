#!/usr/bin/env python3
"""
Enhanced tiling analysis script that shows tile boundaries on original images
"""

import sys
import os
from PIL import Image

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.models.coco import CocoDataset
from src.utils.visualization import BoundingBoxVisualizer


def create_tiling_analysis(dataset_path: str, output_dir: str, target_image: str = None):
    """Create enhanced tiling analysis with tile boundaries visible."""
    
    print("Enhanced Tiling Analysis Tool")
    print("=" * 50)
    
    # Load dataset
    annotations_path = os.path.join(dataset_path, "_annotations.coco.json")
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found: {annotations_path}")
        return
    
    dataset = CocoDataset.from_json(annotations_path)
    print(f"Loaded dataset: {len(dataset.images)} images, {len(dataset.annotations)} annotations")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create category lookup
    categories = {cat.id: cat.name for cat in dataset.categories}
    
    # Initialize visualizer
    visualizer = BoundingBoxVisualizer()
    
    # Select images to analyze
    if target_image:
        # Analyze specific image
        target_img = None
        for img in dataset.images:
            if img.file_name == target_image:
                target_img = img
                break
        
        if not target_img:
            print(f"Error: Image {target_image} not found in dataset")
            return
        
        images_to_analyze = [target_img]
    else:
        # Analyze first few images
        images_to_analyze = dataset.images[:3]
    
    print(f"\nAnalyzing {len(images_to_analyze)} images...")
    
    for i, image_info in enumerate(images_to_analyze):
        print(f"\nProcessing {i+1}/{len(images_to_analyze)}: {image_info.file_name}")
        
        # Load image
        image_path = os.path.join(dataset_path, image_info.file_name)
        if not os.path.exists(image_path):
            print(f"  Warning: Image file not found: {image_path}")
            continue
        
        try:
            image = Image.open(image_path)
            print(f"  Image size: {image.size}")
            
            # Get annotations for this image
            image_annotations = [ann for ann in dataset.annotations 
                               if ann.image_id == image_info.id]
            print(f"  Annotations: {len(image_annotations)}")
            
            # Create tiling overview
            overview = visualizer.create_tiling_overview(
                image, image_annotations, categories, 
                tile_size=(512, 512), overlap=0, max_width=1600
            )
            
            # Save overview
            base_name = image_info.file_name.replace('.jpg', '')
            overview_path = os.path.join(output_dir, f"tiling_analysis_{base_name}.jpg")
            overview.save(overview_path, quality=95)
            print(f"  Saved tiling analysis: {overview_path}")
            
            # Also create a high-resolution version for detailed inspection
            if image.width <= 2000:  # Only for reasonably sized images
                full_res_overview = visualizer.create_tiling_overview(
                    image, image_annotations, categories, 
                    tile_size=(512, 512), overlap=0, max_width=image.width
                )
                
                full_res_path = os.path.join(output_dir, f"full_res_analysis_{base_name}.jpg")
                full_res_overview.save(full_res_path, quality=95)
                print(f"  Saved full resolution analysis: {full_res_path}")
            
        except Exception as e:
            print(f"  Error processing {image_info.file_name}: {e}")
    
    print(f"\n✅ Tiling analysis complete! Check the '{output_dir}' directory.")
    print("\nThe visualizations show:")
    print("  • Cyan rectangles: Tile boundaries")
    print("  • Red rectangles: Individual tile areas")
    print("  • Colored boxes: Annotation bounding boxes")
    print("  • T1, T2, etc.: Tile numbers")
    print("\nThis helps identify:")
    print("  • Which annotations span multiple tiles")
    print("  • Tile boundary overlaps")
    print("  • Edge tile positioning")


if __name__ == "__main__":
    # Test with the problematic image
    target_image = "112.jpg"
    
    create_tiling_analysis(
        dataset_path="./dataset/all/train",
        output_dir="./tiling_analysis",
        target_image=target_image
    )
