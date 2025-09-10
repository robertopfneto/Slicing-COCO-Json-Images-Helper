#!/usr/bin/env python3
"""
Dataset Visualization Tool

Creates visualizations of images with bounding boxes to verify that
the dataset processing is working correctly.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.visualization import BoundingBoxVisualizer


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset with bounding boxes")
    parser.add_argument("--input", type=str, default="./dataset", 
                       help="Input dataset directory (default: ./dataset)")
    parser.add_argument("--output", type=str, default="./visualizations", 
                       help="Output directory for visualizations (default: ./visualizations)")
    parser.add_argument("--samples", type=int, default=10, 
                       help="Number of images to visualize (default: 10)")
    parser.add_argument("--tiled-input", type=str, default="./output", 
                       help="Tiled dataset directory for comparison (default: ./output)")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare original vs tiled datasets")
    parser.add_argument("--summary", action="store_true", 
                       help="Create dataset summary report")
    
    args = parser.parse_args()
    
    print("Dataset Visualization Tool")
    print("=" * 40)
    print(f"Input dataset: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Samples to visualize: {args.samples}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)
    
    # Check if annotations file exists
    annotations_file = os.path.join(args.input, "train", "_annotations.coco.json")
    if not os.path.exists(annotations_file):
        print(f"Error: Annotations file not found: {annotations_file}")
        sys.exit(1)
    
    try:
        visualizer = BoundingBoxVisualizer()
        
        # Create main visualizations
        print("Creating visualizations...")
        visualizer.visualize_dataset(
            dataset_path=args.input,
            output_dir=args.output,
            max_samples=args.samples
        )
        
        # Create summary report if requested
        if args.summary:
            print("\nCreating summary report...")
            visualizer.create_summary_report(args.input, args.output)
        
        # Compare datasets if requested and tiled dataset exists
        if args.compare and os.path.exists(args.tiled_input):
            tiled_annotations = os.path.join(args.tiled_input, "train", "_annotations.coco.json")
            if os.path.exists(tiled_annotations):
                print(f"\nCreating tiled dataset visualizations...")
                tiled_output = os.path.join(args.output, "tiled_dataset")
                visualizer.visualize_dataset(
                    dataset_path=args.tiled_input,
                    output_dir=tiled_output,
                    max_samples=args.samples
                )
                
                # Create comparison summary
                print("Creating comparison summary...")
                visualizer.create_summary_report(args.tiled_input, tiled_output)
            else:
                print(f"Warning: Tiled dataset annotations not found: {tiled_annotations}")
        
        print(f"\nVisualization complete!")
        print(f"Check the '{args.output}' directory for results.")
        
        if args.compare and os.path.exists(args.tiled_input):
            print(f"Tiled dataset visualizations are in: {os.path.join(args.output, 'tiled_dataset')}")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()