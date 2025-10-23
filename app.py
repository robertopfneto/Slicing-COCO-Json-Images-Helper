#!/usr/bin/env python3
"""
Dataset Tiling Application

This application tiles images in a Roboflow COCO dataset while preserving 
bounding box annotations, following Configuration-Driven Architecture principles.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import AppConfig
from src.services.dataset.processor import DatasetProcessor


def main():
    parser = argparse.ArgumentParser(description="Tile dataset images with annotation preservation")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--input", type=str, default="./dataset", help="Input dataset path")
    parser.add_argument("--output", type=str, default="./output", help="Output dataset path")
    parser.add_argument("--tile-size", type=int, nargs=2, default=[640, 640], 
                       help="Tile size (width height)")
    parser.add_argument("--overlap", type=int, default=40, help="Tile overlap in pixels")
    parser.add_argument("--min-coverage", type=float, default=0.3, 
                       help="Minimum object coverage to keep annotation")
    parser.add_argument("--resize-output", type=int, nargs=2, 
                       help="Resize output tiles to this size (width height)")
    parser.add_argument("--auto-slice", action="store_true",
                       help="Enable SAHI auto slice resolution heuristics")
    parser.add_argument("--overlap-height-ratio", type=float,
                       help="Fractional overlap ratio applied along the vertical axis")
    parser.add_argument("--overlap-width-ratio", type=float,
                       help="Fractional overlap ratio applied along the horizontal axis")
    parser.add_argument("--min-area-ratio", type=float,
                       help="Minimum retained annotation area ratio after slicing")
    parser.add_argument("--ignore-negative-samples", action="store_true",
                       help="Skip images without annotations when slicing")
    parser.add_argument("--folds", type=int,
                       help="Number of cross-validation folds to generate")
    parser.add_argument("--fold-seed", type=int,
                       help="Random seed used for fold shuffling")
    parser.add_argument("--no-fold-shuffle", action="store_true",
                       help="Disable shuffling before assigning images to folds")
    parser.add_argument("--clean-output", action="store_true",
                       help="Remove tiles without annotations from each fold's train split")
    parser.add_argument("--validate", action="store_true", 
                       help="Validate output after processing")
    
    args = parser.parse_args()
    
    # Create configuration
    config = AppConfig.from_env()
    
    # Override with command line arguments
    if args.input:
        config.dataset.input_path = args.input
    if args.output:
        config.dataset.output_path = args.output
    if args.tile_size:
        config.tiling.tile_size = tuple(args.tile_size)
    if args.overlap is not None:
        config.tiling.overlap = args.overlap
    if args.min_coverage is not None:
        config.tiling.min_object_coverage = args.min_coverage
    if args.resize_output:
        config.tiling.resize_output = tuple(args.resize_output)
    if args.auto_slice:
        config.tiling.auto_slice_resolution = True
    if args.overlap_height_ratio is not None:
        config.tiling.overlap_height_ratio = args.overlap_height_ratio
    if args.overlap_width_ratio is not None:
        config.tiling.overlap_width_ratio = args.overlap_width_ratio
    if args.min_area_ratio is not None:
        config.tiling.min_area_ratio = args.min_area_ratio
    if args.ignore_negative_samples:
        config.tiling.ignore_negative_samples = True
    if args.folds is not None:
        config.processing.num_folds = max(1, args.folds)
    if args.fold_seed is not None:
        config.processing.fold_seed = args.fold_seed
    if args.no_fold_shuffle:
        config.processing.shuffle_folds = False
    
    print("Dataset Tiling Application")
    print("=" * 40)
    print(f"Input path: {config.dataset.input_path}")
    print(f"Output path: {config.dataset.output_path}")
    print(f"Tile size: {config.tiling.tile_size}")
    print(f"Overlap: {config.tiling.overlap}")
    if config.tiling.auto_slice_resolution:
        print("Auto slice resolution: enabled")
    overlap_height_ratio = config.tiling.overlap_height_ratio
    overlap_width_ratio = config.tiling.overlap_width_ratio
    if overlap_height_ratio is None and config.tiling.tile_size[1]:
        overlap_height_ratio = config.tiling.overlap / config.tiling.tile_size[1]
    if overlap_width_ratio is None and config.tiling.tile_size[0]:
        overlap_width_ratio = config.tiling.overlap / config.tiling.tile_size[0]
    print(
        f"Overlap ratios (height, width): "
        f"{overlap_height_ratio if overlap_height_ratio is not None else 0:.3f}, "
        f"{overlap_width_ratio if overlap_width_ratio is not None else 0:.3f}"
    )
    print(f"Min coverage: {config.tiling.min_object_coverage}")
    print(f"Min area ratio: {config.tiling.min_area_ratio}")
    if config.tiling.ignore_negative_samples:
        print("Ignore negative samples: enabled")
    if config.tiling.resize_output:
        print(f"Resize output: {config.tiling.resize_output}")
    print(
        "Cross-validation folds: "
        f"{config.processing.num_folds} "
        f"(shuffle={'yes' if config.processing.shuffle_folds else 'no'}, "
        f"seed={config.processing.fold_seed})"
    )
    print("=" * 40)
    
    try:
        processor = DatasetProcessor(config)
        
        if args.clean_output:
            processor.clean_fold_train_without_annotations()
            print()
            print(" Output cleanup completed successfully!")
            return
        
        # Validate input
        if not os.path.exists(config.dataset.input_path):
            print(f"Error: Input path does not exist: {config.dataset.input_path}")
            sys.exit(1)
        
        annotations_path = os.path.join(config.dataset.input_path, "train", "_annotations.coco.json")
        if not os.path.exists(annotations_path):
            print(f"Error: Annotations file not found: {annotations_path}")
            sys.exit(1)
        
        # Process dataset
        processor.process_dataset()
        
        # Validate if requested
        if args.validate:
            print("🔍 Validating output...")
            if processor.validate_output():
                print("   ✅ Output validation successful")
            else:
                print("   ❌ Output validation failed")
                sys.exit(1)
        
        print()
        print("🏁 Dataset processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        print("   Partial results may be available in the output directory")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("   Check the logs above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
