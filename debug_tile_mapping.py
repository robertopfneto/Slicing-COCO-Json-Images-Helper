#!/usr/bin/env python3
"""
Debug script to verify tile-annotation mapping
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.models.coco import CocoDataset


def debug_tile_mapping(original_dataset_path: str, tiled_dataset_path: str, target_image: str):
    """Debug tile mapping by creating detailed verification images."""
    
    print(f"🔍 Debugging tile mapping for: {target_image}")
    print("=" * 60)
    
    # Load datasets
    original = CocoDataset.from_json(f'{original_dataset_path}/train/_annotations.coco.json')
    tiled = CocoDataset.from_json(f'{tiled_dataset_path}/train/_annotations.coco.json')
    
    # Find original image
    orig_img_info = next((img for img in original.images if img.file_name == target_image), None)
    if not orig_img_info:
        print(f"❌ Original image {target_image} not found")
        return
    
    # Load original image
    orig_path = f'{original_dataset_path}/train/{target_image}'
    orig_img = Image.open(orig_path)
    
    print(f"✅ Original image loaded: {orig_img.size}")
    
    # Get original annotations
    orig_annotations = [ann for ann in original.annotations if ann.image_id == orig_img_info.id]
    print(f"📋 Original annotations: {len(orig_annotations)}")
    
    # Find corresponding tiles
    base_name = target_image.replace('.jpg', '')
    tile_images = [img for img in tiled.images if base_name in img.file_name]
    print(f"🧩 Found {len(tile_images)} tiles")
    
    # Create output directory
    debug_dir = "./debug_tile_mapping"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Check specific problematic tiles
    problem_tiles = [
        '112_jpg.rf.7ad9edc92c4e2368a5710d1e7c8a6ab9_tile_2048_1536.jpg',
        '112_jpg.rf.7ad9edc92c4e2368a5710d1e7c8a6ab9_tile_1024_512.jpg'
    ]
    
    for tile_filename in problem_tiles:
        print(f"\n🔬 Analyzing: {tile_filename}")
        
        # Find tile info
        tile_info = next((img for img in tile_images if img.file_name == tile_filename), None)
        if not tile_info:
            print(f"❌ Tile not found in dataset")
            continue
        
        # Parse offset from filename
        parts = tile_filename.split('_tile_')
        offset_str = parts[1].replace('.jpg', '')
        tile_x, tile_y = map(int, offset_str.split('_'))
        
        print(f"📍 Tile offset: ({tile_x}, {tile_y})")
        
        # Load actual tile file
        tile_path = f'{tiled_dataset_path}/train/{tile_filename}'
        if not os.path.exists(tile_path):
            print(f"❌ Tile file not found: {tile_path}")
            continue
        
        actual_tile = Image.open(tile_path)
        print(f"🖼️  Tile size: {actual_tile.size}")
        
        # Extract expected region from original
        expected_tile = orig_img.crop((tile_x, tile_y, tile_x + 512, tile_y + 512))
        
        # Get tile annotations from dataset
        tile_annotations = [ann for ann in tiled.annotations if ann.image_id == tile_info.id]
        print(f"🏷️  Tile annotations: {len(tile_annotations)}")
        
        # Create debug visualization
        debug_width = 512 * 3 + 40  # 3 tiles side by side with margins
        debug_height = 512 + 100    # space for labels
        
        debug_img = Image.new('RGB', (debug_width, debug_height), 'white')
        draw = ImageDraw.Draw(debug_img)
        
        # Try to load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Paste images
        debug_img.paste(expected_tile, (10, 80))
        debug_img.paste(actual_tile, (532, 80))
        
        # Create difference image
        diff_img = Image.new('RGB', (512, 512), 'black')
        diff_pixels = []
        
        for y in range(512):
            for x in range(512):
                expected_px = expected_tile.getpixel((x, y))
                actual_px = actual_tile.getpixel((x, y))
                
                if expected_px == actual_px:
                    diff_img.putpixel((x, y), (0, 128, 0))  # Green for match
                else:
                    diff_img.putpixel((x, y), (255, 0, 0))  # Red for mismatch
                    diff_pixels.append((x, y, expected_px, actual_px))
        
        debug_img.paste(diff_img, (1054, 80))
        
        # Add labels
        draw.text((10, 10), "Expected (from original)", fill='black', font=font)
        draw.text((532, 10), "Actual (tile file)", fill='black', font=font)
        draw.text((1054, 10), "Difference (Red=mismatch)", fill='black', font=font)
        
        draw.text((10, 30), f"Offset: ({tile_x}, {tile_y})", fill='blue', font=font)
        draw.text((532, 30), f"Annotations: {len(tile_annotations)}", fill='blue', font=font)
        draw.text((1054, 30), f"Mismatches: {len(diff_pixels)}", fill='red' if diff_pixels else 'green', font=font)
        
        # Draw annotations on expected and actual
        for ann in tile_annotations:
            bbox = ann.bbox
            x, y, w, h = bbox
            
            # Draw on expected tile
            draw.rectangle([10 + x, 80 + y, 10 + x + w, 80 + y + h], 
                         outline='red', width=2)
            
            # Draw on actual tile  
            draw.rectangle([532 + x, 80 + y, 532 + x + w, 80 + y + h], 
                         outline='red', width=2)
        
        # Save debug image
        debug_path = f"{debug_dir}/debug_{tile_filename}"
        debug_img.save(debug_path)
        print(f"💾 Debug image saved: {debug_path}")
        
        # Print pixel difference summary
        if diff_pixels:
            print(f"⚠️  Found {len(diff_pixels)} pixel differences")
            if len(diff_pixels) <= 10:
                for x, y, exp, act in diff_pixels[:5]:
                    print(f"   ({x},{y}): expected {exp} vs actual {act}")
        else:
            print("✅ Images match perfectly")
    
    print(f"\n🎯 Debug complete! Check {debug_dir}/ for detailed visualizations")


if __name__ == "__main__":
    debug_tile_mapping(
        "./dataset", 
        "./output", 
        "112_jpg.rf.7ad9edc92c4e2368a5710d1e7c8a6ab9.jpg"
    )