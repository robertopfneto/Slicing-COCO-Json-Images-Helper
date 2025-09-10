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
    parser.add_argument("--tile-size", type=int, nargs=2, default=[512, 512], 
                       help="Tile size (width height)")
    parser.add_argument("--overlap", type=int, default=0, help="Tile overlap in pixels")
    parser.add_argument("--min-coverage", type=float, default=0.3, 
                       help="Minimum object coverage to keep annotation")
    parser.add_argument("--resize-output", type=int, nargs=2, 
                       help="Resize output tiles to this size (width height)")
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
    
    print("Dataset Tiling Application")
    print("=" * 40)
    print(f"Input path: {config.dataset.input_path}")
    print(f"Output path: {config.dataset.output_path}")
    print(f"Tile size: {config.tiling.tile_size}")
    print(f"Overlap: {config.tiling.overlap}")
    print(f"Min coverage: {config.tiling.min_object_coverage}")
    if config.tiling.resize_output:
        print(f"Resize output: {config.tiling.resize_output}")
    print("=" * 40)
    
    # Validate input
    if not os.path.exists(config.dataset.input_path):
        print(f"Error: Input path does not exist: {config.dataset.input_path}")
        sys.exit(1)
    
    annotations_path = os.path.join(config.dataset.input_path, "train", "_annotations.coco.json")
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found: {annotations_path}")
        sys.exit(1)
    
    try:
        # Process dataset
        processor = DatasetProcessor(config)
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