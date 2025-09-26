#!/usr/bin/env python3
"""
Quick verification script to check if tiling preserved annotations correctly
"""

import sys
import os
import json
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.coco import CocoDataset


def verify_tiling_process(original_path: str, tiled_path: str):
    """Verify that the tiling process worked correctly."""
    
    print("Dataset Tiling Verification")
    print("=" * 40)
    
    # Load datasets
    try:
        print("Loading datasets...")
        original_annotations = os.path.join(original_path, "train", "_annotations.coco.json")
        tiled_annotations = os.path.join(tiled_path, "train", "_annotations.coco.json")
        
        if not os.path.exists(original_annotations):
            print(f"❌ Original annotations not found: {original_annotations}")
            return False
            
        if not os.path.exists(tiled_annotations):
            print(f"❌ Tiled annotations not found: {tiled_annotations}")
            return False
        
        original_dataset = CocoDataset.from_json(original_annotations)
        tiled_dataset = CocoDataset.from_json(tiled_annotations)
        
        print("✅ Successfully loaded both datasets")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return False
    
    # Basic statistics
    print("\nDataset Statistics:")
    print("-" * 20)
    print(f"Original Images: {len(original_dataset.images)}")
    print(f"Tiled Images: {len(tiled_dataset.images)}")
    print(f"Original Annotations: {len(original_dataset.annotations)}")
    print(f"Tiled Annotations: {len(tiled_dataset.annotations)}")
    
    # Expansion ratio
    if len(original_dataset.images) > 0:
        expansion_ratio = len(tiled_dataset.images) / len(original_dataset.images)
        print(f"Image Expansion Ratio: {expansion_ratio:.2f}x")
    
    # Annotation retention
    if len(original_dataset.annotations) > 0:
        retention_rate = (len(tiled_dataset.annotations) / len(original_dataset.annotations)) * 100
        print(f"Annotation Retention: {retention_rate:.1f}%")
    
    # Category consistency
    original_categories = {cat.id: cat.name for cat in original_dataset.categories}
    tiled_categories = {cat.id: cat.name for cat in tiled_dataset.categories}
    
    print(f"\nCategories:")
    print("-" * 20)
    categories_match = original_categories == tiled_categories
    print(f"Categories preserved: {'✅ Yes' if categories_match else '❌ No'}")
    
    for cat_id, cat_name in original_categories.items():
        original_count = sum(1 for ann in original_dataset.annotations if ann.category_id == cat_id)
        tiled_count = sum(1 for ann in tiled_dataset.annotations if ann.category_id == cat_id)
        print(f"  {cat_name} (ID {cat_id}): {original_count} → {tiled_count}")
    
    # Sample verification - check a few images
    print(f"\nSample Verification:")
    print("-" * 20)
    
    sample_size = min(5, len(original_dataset.images))
    issues_found = 0
    
    for i in range(sample_size):
        original_img = original_dataset.images[i]
        original_img_anns = [ann for ann in original_dataset.annotations if ann.image_id == original_img.id]
        
        # Find corresponding tiles
        base_name = original_img.file_name.split('.')[0]
        corresponding_tiles = [img for img in tiled_dataset.images if base_name in img.file_name]
        
        total_tile_anns = 0
        for tile_img in corresponding_tiles:
            tile_anns = [ann for ann in tiled_dataset.annotations if ann.image_id == tile_img.id]
            total_tile_anns += len(tile_anns)
        
        print(f"  {original_img.file_name}: {len(original_img_anns)} annotations")
        print(f"    → {len(corresponding_tiles)} tiles with {total_tile_anns} total annotations")
        
        # Check if we lost significant annotations (some loss is expected due to min coverage)
        if len(original_img_anns) > 0:
            retention = total_tile_anns / len(original_img_anns)
            if retention < 0.5:  # Lost more than 50% of annotations
                issues_found += 1
                print(f"    ⚠️  Low annotation retention: {retention:.1%}")
    
    # File existence check
    print(f"\nFile Verification:")
    print("-" * 20)
    
    missing_files = 0
    sample_tiles = tiled_dataset.images[:10]  # Check first 10 tiles
    
    for tile_img in sample_tiles:
        tile_path = os.path.join(tiled_path, "train", tile_img.file_name)
        if not os.path.exists(tile_path):
            missing_files += 1
    
    if missing_files == 0:
        print(f"✅ All sampled tile files exist")
    else:
        print(f"❌ {missing_files} out of {len(sample_tiles)} sampled files missing")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    print("=" * 20)
    
    success = True
    
    if len(tiled_dataset.images) <= len(original_dataset.images):
        print("❌ No image expansion detected - tiling may have failed")
        success = False
    else:
        print("✅ Images were successfully tiled")
    
    if not categories_match:
        print("❌ Categories were not preserved correctly")
        success = False
    else:
        print("✅ Categories preserved correctly")
    
    if issues_found > sample_size // 2:
        print("❌ Significant annotation loss detected")
        success = False
    elif len(tiled_dataset.annotations) > 0:
        print("✅ Annotations appear to be preserved")
    
    if missing_files > 0:
        print("❌ Some tiled image files are missing")
        success = False
    else:
        print("✅ Tiled image files exist")
    
    if success:
        print("\n🎉 Tiling process appears to have worked correctly!")
        print("   Run ./visualize_results.sh to see visual confirmation.")
    else:
        print("\n⚠️  Issues detected - please review the tiling process.")
    
    return success


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify dataset tiling process")
    parser.add_argument("--original", default="./dataset", help="Original dataset path")
    parser.add_argument("--tiled", default="./output", help="Tiled dataset path")
    
    args = parser.parse_args()
    
    success = verify_tiling_process(args.original, args.tiled)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()