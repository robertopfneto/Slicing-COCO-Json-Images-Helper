#!/usr/bin/env python3
"""
COCO Dataset Merger

Merges multiple COCO format datasets into a single dataset, handling:
- Filename conflicts by adding dataset prefixes
- ID remapping for images and annotations
- Category consolidation and validation
- Unified annotations.coco.json generation

Usage:
    python3 merge_datasets.py --datasets dataset1 dataset2 dataset3 --output merged_dataset
"""

import argparse
import sys
import os
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.coco import CocoDataset, CocoImage, CocoAnnotation, CocoCategory, CocoInfo, CocoLicense


class DatasetMerger:
    """Merges multiple COCO datasets into a single unified dataset."""
    
    def __init__(self):
        self.merged_images = []
        self.merged_annotations = []
        self.merged_categories = []
        self.category_id_mapping = {}  # old_id -> new_id
        self.next_image_id = 1
        self.next_annotation_id = 1
        self.next_category_id = 1
        
    def merge_datasets(self, dataset_paths: List[str], output_path: str) -> None:
        """Merge multiple datasets into a single output dataset."""
        
        print("=" * 60)
        print("           COCO Dataset Merger")
        print("=" * 60)
        print()
        
        # Validate input datasets
        print("🔍 Validating input datasets...")
        validated_datasets = self._validate_datasets(dataset_paths)
        
        print(f"✅ Found {len(validated_datasets)} valid datasets to merge")
        print()
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        train_output_dir = os.path.join(output_path, "train")
        os.makedirs(train_output_dir, exist_ok=True)
        
        # Process each dataset
        print("🔄 Processing datasets...")
        print("-" * 60)
        
        for i, (dataset_path, dataset_name) in enumerate(validated_datasets, 1):
            print(f"📁 [{i}/{len(validated_datasets)}] Processing: {dataset_name}")
            self._process_dataset(dataset_path, dataset_name, train_output_dir)
            print()
        
        # Create merged dataset
        print("📦 Creating unified dataset...")
        merged_dataset = self._create_merged_dataset(validated_datasets)
        
        # Save merged annotations
        output_annotations = os.path.join(output_path, "train", "_annotations.coco.json")
        merged_dataset.save_json(output_annotations)
        
        # Generate summary
        self._print_summary(validated_datasets, output_path, merged_dataset)
        
    def _validate_datasets(self, dataset_paths: List[str]) -> List[tuple]:
        """Validate that all dataset paths exist and contain required files."""
        validated_datasets = []
        
        for dataset_path in dataset_paths:
            if not os.path.exists(dataset_path):
                print(f"❌ Warning: Dataset path does not exist: {dataset_path}")
                continue
                
            train_path = os.path.join(dataset_path, "train")
            if not os.path.exists(train_path):
                print(f"❌ Warning: Train directory not found: {train_path}")
                continue
                
            annotations_path = os.path.join(train_path, "_annotations.coco.json")
            if not os.path.exists(annotations_path):
                print(f"❌ Warning: Annotations file not found: {annotations_path}")
                continue
            
            # Generate dataset name from path
            dataset_name = Path(dataset_path).name
            if not dataset_name:  # Handle trailing slash
                dataset_name = Path(dataset_path).parent.name
                
            validated_datasets.append((dataset_path, dataset_name))
            print(f"   ✅ {dataset_name}: {annotations_path}")
            
        if not validated_datasets:
            raise ValueError("No valid datasets found to merge")
            
        return validated_datasets
    
    def _process_dataset(self, dataset_path: str, dataset_name: str, output_dir: str) -> None:
        """Process a single dataset and add to merged collections."""
        
        # Load dataset
        annotations_path = os.path.join(dataset_path, "train", "_annotations.coco.json")
        dataset = CocoDataset.from_json(annotations_path)
        
        print(f"   📊 Original: {len(dataset.images)} images, {len(dataset.annotations)} annotations")
        
        # Process categories first (needed for ID mapping)
        category_mapping = self._process_categories(dataset.categories, dataset_name)
        
        # Process images
        image_id_mapping = self._process_images(
            dataset.images, dataset_path, dataset_name, output_dir
        )
        
        # Process annotations
        self._process_annotations(
            dataset.annotations, image_id_mapping, category_mapping, dataset_name
        )
        
        print(f"   ✅ Processed: {len(image_id_mapping)} images, {len(dataset.annotations)} annotations")
    
    def _process_categories(self, categories: List[CocoCategory], dataset_name: str) -> Dict[int, int]:
        """Process categories and return old_id -> new_id mapping."""
        category_mapping = {}
        
        for category in categories:
            # Check if this category already exists (by name)
            existing_category = None
            for merged_cat in self.merged_categories:
                if merged_cat.name == category.name:
                    existing_category = merged_cat
                    break
            
            if existing_category:
                # Use existing category ID
                category_mapping[category.id] = existing_category.id
                print(f"   🔗 Category '{category.name}': reused ID {existing_category.id}")
            else:
                # Create new category with new ID
                new_category = CocoCategory(
                    id=self.next_category_id,
                    name=category.name,
                    supercategory=category.supercategory
                )
                self.merged_categories.append(new_category)
                category_mapping[category.id] = self.next_category_id
                print(f"   ➕ Category '{category.name}': new ID {self.next_category_id}")
                self.next_category_id += 1
        
        return category_mapping
    
    def _process_images(self, images: List[CocoImage], dataset_path: str, 
                       dataset_name: str, output_dir: str) -> Dict[int, int]:
        """Process images and return old_id -> new_id mapping."""
        image_id_mapping = {}
        
        for image in images:
            # Create new filename with dataset prefix
            original_filename = image.file_name
            name_parts = os.path.splitext(original_filename)
            new_filename = f"{dataset_name}_{name_parts[0]}{name_parts[1]}"
            
            # Copy image file
            source_path = os.path.join(dataset_path, "train", original_filename)
            dest_path = os.path.join(output_dir, new_filename)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
            else:
                print(f"   ⚠️  Warning: Image file not found: {source_path}")
            
            # Create new image entry
            new_image = CocoImage(
                id=self.next_image_id,
                width=image.width,
                height=image.height,
                file_name=new_filename
            )
            
            self.merged_images.append(new_image)
            image_id_mapping[image.id] = self.next_image_id
            self.next_image_id += 1
        
        return image_id_mapping
    
    def _process_annotations(self, annotations: List[CocoAnnotation], 
                           image_id_mapping: Dict[int, int],
                           category_mapping: Dict[int, int],
                           dataset_name: str) -> None:
        """Process annotations with ID remapping."""
        
        for annotation in annotations:
            # Remap IDs
            new_image_id = image_id_mapping.get(annotation.image_id)
            new_category_id = category_mapping.get(annotation.category_id)
            
            if new_image_id is None:
                print(f"   ⚠️  Warning: Image ID {annotation.image_id} not found in mapping")
                continue
                
            if new_category_id is None:
                print(f"   ⚠️  Warning: Category ID {annotation.category_id} not found in mapping")
                continue
            
            # Create new annotation
            new_annotation = CocoAnnotation(
                id=self.next_annotation_id,
                image_id=new_image_id,
                category_id=new_category_id,
                segmentation=annotation.segmentation,
                area=annotation.area,
                bbox=annotation.bbox,
                iscrowd=annotation.iscrowd
            )
            
            self.merged_annotations.append(new_annotation)
            self.next_annotation_id += 1
    
    def _create_merged_dataset(self, validated_datasets: List[tuple]) -> CocoDataset:
        """Create the final merged dataset object."""
        
        # Create merged info as proper CocoInfo object
        info = CocoInfo(
            description=f"Merged dataset from {len(validated_datasets)} source datasets",
            version="1.0",
            year=datetime.now().year,
            contributor="Dataset Merger Tool",
            url="",  # Required field, empty string as default
            date_created=datetime.now().isoformat()
        )
        
        # Create merged dataset
        merged_dataset = CocoDataset(
            info=info,
            licenses=[],  # Empty list of CocoLicense objects
            images=self.merged_images,
            annotations=self.merged_annotations,
            categories=self.merged_categories
        )
        
        return merged_dataset
    
    def _print_summary(self, validated_datasets: List[tuple], output_path: str, 
                      merged_dataset: CocoDataset) -> None:
        """Print merge summary statistics."""
        
        print("=" * 60)
        print("           Merge Complete!")
        print("=" * 60)
        print()
        print(f"📁 Output location: {output_path}")
        print()
        print("📊 Merge Summary:")
        print(f"   📂 Source datasets: {len(validated_datasets)}")
        print(f"   🖼️  Total images: {len(merged_dataset.images)}")
        print(f"   🏷️  Total annotations: {len(merged_dataset.annotations)}")
        print(f"   🏪 Total categories: {len(merged_dataset.categories)}")
        print()
        
        print("📂 Source datasets:")
        for dataset_path, dataset_name in validated_datasets:
            print(f"   • {dataset_name} ({dataset_path})")
        print()
        
        print("🏪 Merged categories:")
        for category in merged_dataset.categories:
            print(f"   • ID {category.id}: {category.name}")
        print()
        
        print("✅ Dataset merging completed successfully!")
        print(f"📄 Annotations saved: {os.path.join(output_path, 'train', '_annotations.coco.json')}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Merge multiple COCO datasets into one")
    parser.add_argument("--datasets", nargs="+", required=True,
                       help="List of dataset paths to merge")
    parser.add_argument("--output", required=True,
                       help="Output path for merged dataset")
    parser.add_argument("--validate", action="store_true",
                       help="Validate merged dataset after creation")
    
    args = parser.parse_args()
    
    if len(args.datasets) < 2:
        print("❌ Error: At least 2 datasets are required for merging")
        sys.exit(1)
    
    try:
        merger = DatasetMerger()
        merger.merge_datasets(args.datasets, args.output)
        
        if args.validate:
            print("🔍 Validating merged dataset...")
            # Basic validation
            annotations_path = os.path.join(args.output, "train", "_annotations.coco.json")
            if os.path.exists(annotations_path):
                dataset = CocoDataset.from_json(annotations_path)
                print(f"   ✅ Validation successful: {len(dataset.images)} images, {len(dataset.annotations)} annotations")
            else:
                print("   ❌ Validation failed: annotations file not found")
                sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error during dataset merging: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()