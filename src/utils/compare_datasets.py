#!/usr/bin/env python3
"""
Dataset Comparison Visualization Tool

Creates side-by-side comparisons of original images and their corresponding tiles
to verify that bounding box annotations are preserved correctly during tiling.
"""

import argparse
import sys
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.coco import CocoDataset
from src.utils.visualization import BoundingBoxVisualizer


class DatasetComparator:
    """Compares original and tiled datasets with visualizations."""
    
    def __init__(self):
        self.visualizer = BoundingBoxVisualizer()
    
    def create_side_by_side_comparison(self, original_path: str, tiled_path: str, 
                                     output_dir: str, num_comparisons: int = 10):
        """Create side-by-side comparisons of original vs tiled images."""
        
        print(f"Creating side-by-side comparisons...")
        print(f"Original dataset: {original_path}")
        print(f"Tiled dataset: {tiled_path}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Load datasets
        original_annotations = os.path.join(original_path, "train", "_annotations.coco.json")
        tiled_annotations = os.path.join(tiled_path, "train", "_annotations.coco.json")
        
        if not os.path.exists(original_annotations):
            raise FileNotFoundError(f"Original annotations not found: {original_annotations}")
        if not os.path.exists(tiled_annotations):
            raise FileNotFoundError(f"Tiled annotations not found: {tiled_annotations}")
        
        original_dataset = CocoDataset.from_json(original_annotations)
        tiled_dataset = CocoDataset.from_json(tiled_annotations)
        
        # Create category lookup
        original_categories = {cat.id: cat.name for cat in original_dataset.categories}
        tiled_categories = {cat.id: cat.name for cat in tiled_dataset.categories}
        
        print(f"Loaded datasets:")
        print(f"  Original: {len(original_dataset.images)} images, {len(original_dataset.annotations)} annotations")
        print(f"  Tiled: {len(tiled_dataset.images)} images, {len(tiled_dataset.annotations)} annotations")
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample random original images
        sample_originals = random.sample(original_dataset.images, min(num_comparisons, len(original_dataset.images)))
        
        for i, original_img in enumerate(sample_originals):
            print(f"Processing comparison {i+1}/{len(sample_originals)}: {original_img.file_name}")
            
            try:
                # Load original image
                original_img_path = os.path.join(original_path, "train", original_img.file_name)
                if not os.path.exists(original_img_path):
                    print(f"  Warning: Original image not found: {original_img_path}")
                    continue
                
                original_image = Image.open(original_img_path)
                
                # Get original annotations
                original_anns = [ann for ann in original_dataset.annotations 
                               if ann.image_id == original_img.id]
                
                # Find corresponding tiles
                base_name = original_img.file_name.split('.')[0]  # Remove extension
                corresponding_tiles = [img for img in tiled_dataset.images 
                                     if base_name in img.file_name and img.file_name.startswith(base_name)]
                
                print(f"  Found {len(corresponding_tiles)} corresponding tiles")
                print(f"  Original has {len(original_anns)} annotations")
                
                if not corresponding_tiles:
                    print(f"  Warning: No corresponding tiles found for {original_img.file_name}")
                    continue
                
                # Create comparison for each tile (limit to first 6 tiles for manageable output)
                tiles_to_show = corresponding_tiles[:6]
                
                for j, tile_img in enumerate(tiles_to_show):
                    # Load tile image
                    tile_img_path = os.path.join(tiled_path, "train", tile_img.file_name)
                    if not os.path.exists(tile_img_path):
                        print(f"    Warning: Tile image not found: {tile_img_path}")
                        continue
                    
                    tile_image = Image.open(tile_img_path)
                    
                    # Get tile annotations
                    tile_anns = [ann for ann in tiled_dataset.annotations 
                               if ann.image_id == tile_img.id]
                    
                    # Extract tile offset from filename
                    # Format: basename_tile_x_y.jpg
                    filename_parts = tile_img.file_name.split('_tile_')
                    if len(filename_parts) >= 2:
                        coords = filename_parts[1].replace('.jpg', '').split('_')
                        if len(coords) >= 2:
                            tile_offset = (int(coords[0]), int(coords[1]))
                        else:
                            tile_offset = (0, 0)
                    else:
                        tile_offset = (0, 0)
                    
                    print(f"    Tile {j+1}: {tile_img.file_name} - {len(tile_anns)} annotations")
                    
                    # Create comparison
                    comparison = self.create_single_comparison(
                        original_image, tile_image, 
                        original_anns, tile_anns, 
                        original_categories, tile_offset,
                        original_img.file_name, tile_img.file_name
                    )
                    
                    # Save comparison with high quality
                    comparison_filename = f"comparison_{i+1:02d}_{j+1}__{original_img.file_name}_vs_{tile_img.file_name}"
                    comparison_path = os.path.join(output_dir, comparison_filename)
                    comparison.save(comparison_path, quality=95, optimize=True)
                
            except Exception as e:
                print(f"  Error processing {original_img.file_name}: {e}")
        
        print(f"\nComparisons saved to: {output_dir}")
    
    def create_single_comparison(self, original_img: Image.Image, tile_img: Image.Image,
                               original_anns, tile_anns, categories, tile_offset,
                               original_name: str, tile_name: str) -> Image.Image:
        """Create a single side-by-side comparison image."""
        
        # Draw tile boundaries on original image first, then bounding boxes
        orig_with_tiles = self.visualizer.draw_tile_boundaries(original_img, (512, 512), 0, tile_offset)
        orig_with_boxes = self.visualizer.draw_bounding_boxes(orig_with_tiles, original_anns, categories)
        
        # Draw only bounding boxes on tile image
        tile_with_boxes = self.visualizer.draw_bounding_boxes(tile_img, tile_anns, categories)
        
        # Use much higher resolution - max 1200px width each to keep details visible
        max_width = 1200
        
        # For original images, maintain aspect ratio but ensure good visibility
        if orig_with_boxes.width > max_width:
            ratio = max_width / orig_with_boxes.width
            new_height = int(orig_with_boxes.height * ratio)
            orig_with_boxes = orig_with_boxes.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # For tile images, scale them up to match the original's scale for better comparison
        # Calculate the scale factor between original and tile
        original_scale = min(max_width / original_img.width, max_width / original_img.height)
        tile_scale = min(max_width / tile_img.width, max_width / tile_img.height)
        
        # Use the larger scale to make tiles more visible
        target_scale = max(original_scale * 2, tile_scale)  # Make tiles at least 2x larger than original scale
        
        new_tile_width = int(tile_img.width * target_scale)
        new_tile_height = int(tile_img.height * target_scale)
        
        # Limit maximum size to prevent huge images
        if new_tile_width > max_width:
            ratio = max_width / new_tile_width
            new_tile_width = max_width
            new_tile_height = int(new_tile_height * ratio)
        
        tile_with_boxes = tile_with_boxes.resize((new_tile_width, new_tile_height), Image.Resampling.LANCZOS)
        
        # Create comparison canvas
        margin = 20
        title_height = 80
        info_height = 60
        canvas_width = orig_with_boxes.width + tile_with_boxes.width + (3 * margin)
        canvas_height = max(orig_with_boxes.height, tile_with_boxes.height) + title_height + info_height + margin
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        draw = ImageDraw.Draw(canvas)
        
        # Try to get a nice font
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            info_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()
        
        # Add titles
        draw.text((margin, 10), "Original Image", fill='black', font=title_font)
        draw.text((orig_with_boxes.width + 2*margin, 10), 
                 f"Tiled Image (Offset: {tile_offset})", fill='black', font=title_font)
        
        # Add image info
        draw.text((margin, 35), f"File: {original_name}", fill='gray', font=info_font)
        draw.text((margin, 50), f"Size: {original_img.size}", fill='gray', font=info_font)
        
        draw.text((orig_with_boxes.width + 2*margin, 35), f"File: {tile_name}", fill='gray', font=info_font)
        draw.text((orig_with_boxes.width + 2*margin, 50), f"Size: {tile_img.size}", fill='gray', font=info_font)
        
        # Paste images
        y_offset = title_height + margin
        canvas.paste(orig_with_boxes, (margin, y_offset))
        canvas.paste(tile_with_boxes, (orig_with_boxes.width + 2*margin, y_offset))
        
        # Add annotation counts at bottom
        info_y = y_offset + max(orig_with_boxes.height, tile_with_boxes.height) + 10
        draw.text((margin, info_y), 
                 f"Annotations: {len(original_anns)}", fill='blue', font=info_font)
        draw.text((orig_with_boxes.width + 2*margin, info_y), 
                 f"Annotations: {len(tile_anns)}", fill='blue', font=info_font)
        
        return canvas
    
    def create_overview_grid(self, original_path: str, tiled_path: str, output_dir: str):
        """Create an overview grid showing multiple examples."""
        
        print("Creating overview grid...")
        
        # Load datasets
        original_dataset = CocoDataset.from_json(os.path.join(original_path, "train", "_annotations.coco.json"))
        tiled_dataset = CocoDataset.from_json(os.path.join(tiled_path, "train", "_annotations.coco.json"))
        categories = {cat.id: cat.name for cat in original_dataset.categories}
        
        # Sample 4 original images for the grid
        sample_images = random.sample(original_dataset.images, min(4, len(original_dataset.images)))
        
        grid_images = []
        
        for original_img in sample_images:
            try:
                # Load original image
                original_img_path = os.path.join(original_path, "train", original_img.file_name)
                original_image = Image.open(original_img_path)
                
                # Get original annotations
                original_anns = [ann for ann in original_dataset.annotations if ann.image_id == original_img.id]
                
                # Draw bounding boxes and resize for grid
                orig_with_boxes = self.visualizer.draw_bounding_boxes(original_image, original_anns, categories)
                orig_with_boxes = orig_with_boxes.resize((400, 300), Image.Resampling.LANCZOS)
                
                # Find one representative tile
                base_name = original_img.file_name.split('.')[0]
                tiles = [img for img in tiled_dataset.images if base_name in img.file_name]
                
                if tiles:
                    # Pick middle tile or first one
                    representative_tile = tiles[len(tiles)//2] if len(tiles) > 1 else tiles[0]
                    
                    tile_img_path = os.path.join(tiled_path, "train", representative_tile.file_name)
                    tile_image = Image.open(tile_img_path)
                    
                    tile_anns = [ann for ann in tiled_dataset.annotations if ann.image_id == representative_tile.id]
                    tile_with_boxes = self.visualizer.draw_bounding_boxes(tile_image, tile_anns, categories)
                    tile_with_boxes = tile_with_boxes.resize((400, 300), Image.Resampling.LANCZOS)
                    
                    # Create pair
                    pair = Image.new('RGB', (820, 300), 'white')
                    pair.paste(orig_with_boxes, (0, 0))
                    pair.paste(tile_with_boxes, (410, 0))
                    
                    # Add labels
                    draw = ImageDraw.Draw(pair)
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                    
                    draw.text((10, 10), "Original", fill='red', font=font)
                    draw.text((420, 10), "Tiled", fill='blue', font=font)
                    
                    grid_images.append(pair)
                
            except Exception as e:
                print(f"Error processing {original_img.file_name}: {e}")
        
        if grid_images:
            # Create final grid
            grid_width = 820
            grid_height = len(grid_images) * 320  # 300 + 20 margin
            
            final_grid = Image.new('RGB', (grid_width, grid_height + 100), 'white')
            draw = ImageDraw.Draw(final_grid)
            
            # Add title
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                title_font = ImageFont.load_default()
            
            draw.text((grid_width//2 - 200, 20), "Dataset Comparison Overview", fill='black', font=title_font)
            draw.text((grid_width//2 - 150, 50), "Original Images vs Tiled Images", fill='gray', font=title_font)
            
            # Paste grid images
            y_offset = 80
            for grid_img in grid_images:
                final_grid.paste(grid_img, (0, y_offset))
                y_offset += 320
            
            # Save overview
            overview_path = os.path.join(output_dir, "dataset_comparison_overview.jpg")
            final_grid.save(overview_path)
            print(f"Overview grid saved: {overview_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare original and tiled datasets")
    parser.add_argument("--original", default="./dataset", help="Original dataset path")
    parser.add_argument("--tiled", default="./output", help="Tiled dataset path") 
    parser.add_argument("--output", default="./comparison_visualizations", help="Output directory")
    parser.add_argument("--samples", type=int, default=5, help="Number of original images to compare")
    parser.add_argument("--overview", action="store_true", help="Create overview grid")
    
    args = parser.parse_args()
    
    print("Dataset Comparison Tool")
    print("=" * 40)
    
    # Check paths
    if not os.path.exists(args.original):
        print(f"Error: Original dataset not found: {args.original}")
        sys.exit(1)
    if not os.path.exists(args.tiled):
        print(f"Error: Tiled dataset not found: {args.tiled}")
        sys.exit(1)
    
    try:
        comparator = DatasetComparator()
        
        # Create side-by-side comparisons
        comparator.create_side_by_side_comparison(
            args.original, args.tiled, args.output, args.samples
        )
        
        # Create overview if requested
        if args.overview:
            comparator.create_overview_grid(args.original, args.tiled, args.output)
        
        print(f"\n🎉 Comparison visualizations complete!")
        print(f"Check the '{args.output}' directory to see:")
        print("  - Side-by-side comparisons of original vs tiled images")
        print("  - Bounding boxes drawn on both versions")
        print("  - Verification that annotations are preserved correctly")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()