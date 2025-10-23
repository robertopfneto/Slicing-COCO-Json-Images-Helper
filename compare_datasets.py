#!/usr/bin/env python3
"""
Dataset Comparison Visualization Tool

Creates side-by-side comparisons of original images and their corresponding tiles
to verify that bounding box annotations are preserved correctly during tiling.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.coco import CocoDataset, CocoImage
from src.utils.visualization import BoundingBoxVisualizer


def _resolve_split_paths(base_path: str, split: Optional[str]) -> Tuple[str, str]:
    """Return (images_dir, annotations_path) for a dataset split."""
    if split:
        images_dir = os.path.join(base_path, split)
        annotations_path = os.path.join(images_dir, "_annotations.coco.json")
    else:
        images_dir = os.path.join(base_path, "train")
        annotations_path = os.path.join(images_dir, "_annotations.coco.json")
    return images_dir, annotations_path


def _parse_tile_offset(filename: str) -> Optional[Tuple[int, int]]:
    """Extract tile origin from filenames formatted as *_tile_X_Y.ext."""
    if "_tile_" not in filename:
        return None
    try:
        suffix = filename.rsplit("_tile_", 1)[1]
        coord_part = suffix.rsplit(".", 1)[0]
        x_str, y_str = coord_part.split("_", 1)
        return int(x_str), int(y_str)
    except (ValueError, IndexError):
        return None


class DatasetComparator:
    """Compares original and tiled datasets with visualizations."""

    def __init__(self):
        self.visualizer = BoundingBoxVisualizer()

    def create_side_by_side_comparison(
        self,
        original_path: str,
        tiled_path: str,
        output_dir: str,
        num_comparisons: int = 10,
        tiled_split: Optional[str] = None,
    ) -> None:
        """Create side-by-side comparisons of original vs tiled images."""
        
        original_images_dir, original_annotations = _resolve_split_paths(original_path, "train")
        tiled_images_dir, tiled_annotations = _resolve_split_paths(tiled_path, tiled_split)

        print("Creating side-by-side comparisons...")
        print(f"Original dataset: {original_images_dir}")
        print(f"Tiled dataset:   {tiled_images_dir}")
        print(f"Output directory: {output_dir}")
        print()
        
        if not os.path.exists(original_annotations):
            raise FileNotFoundError(f"Original annotations not found: {original_annotations}")
        if not os.path.exists(tiled_annotations):
            raise FileNotFoundError(f"Tiled annotations not found: {tiled_annotations}")

        original_dataset = CocoDataset.from_json(original_annotations)
        tiled_dataset = CocoDataset.from_json(tiled_annotations)

        original_categories = {cat.id: cat.name for cat in original_dataset.categories}
        tiled_categories = {cat.id: cat.name for cat in tiled_dataset.categories}

        print("Loaded datasets:")
        print(
            f"  Original: {len(original_dataset.images)} images, "
            f"{len(original_dataset.annotations)} annotations"
        )
        print(
            f"  Tiled:    {len(tiled_dataset.images)} tiles, "
            f"{len(tiled_dataset.annotations)} annotations"
        )
        print()

        os.makedirs(output_dir, exist_ok=True)

        # Build lookup of tiles per original base name
        tile_lookup: Dict[str, List[CocoImage]] = {}
        tile_bbox_lookup: Dict[str, List[Tuple[int, int, int, int]]] = {}
        for tile_img in tiled_dataset.images:
            base_name = tile_img.file_name.split("_tile_")[0]
            tile_lookup.setdefault(base_name, []).append(tile_img)

            offset = _parse_tile_offset(tile_img.file_name)
            if offset:
                x_off, y_off = offset
                tile_bbox_lookup.setdefault(base_name, []).append(
                    (x_off, y_off, x_off + tile_img.width, y_off + tile_img.height)
                )

        eligible_originals = [
            img for img in original_dataset.images if Path(img.file_name).stem in tile_lookup
        ]
        if not eligible_originals:
            print("Warning: no matching images found between original and tiled datasets.")
            return

        sample_originals = random.sample(
            eligible_originals,
            min(num_comparisons, len(eligible_originals)),
        )
        
        for i, original_img in enumerate(sample_originals):
            print(f"Processing comparison {i+1}/{len(sample_originals)}: {original_img.file_name}")
            
            try:
                # Load original image
                original_img_path = os.path.join(original_images_dir, original_img.file_name)
                if not os.path.exists(original_img_path):
                    print(f"  Warning: Original image not found: {original_img_path}")
                    continue
                
                original_image = Image.open(original_img_path)
                
                # Get original annotations
                original_anns = [ann for ann in original_dataset.annotations 
                               if ann.image_id == original_img.id]
                
                # Find corresponding tiles
                base_name = Path(original_img.file_name).stem
                corresponding_tiles = tile_lookup.get(base_name, [])

                print(f"  Found {len(corresponding_tiles)} corresponding tiles")
                print(f"  Original has {len(original_anns)} annotations")
                
                if not corresponding_tiles:
                    print(f"  Warning: No corresponding tiles found for {original_img.file_name}")
                    continue
                
                # Create comparison for each tile (limit to first 6 tiles for manageable output)
                tiles_to_show = corresponding_tiles[:6]
                
                for j, tile_img in enumerate(tiles_to_show):
                    # Load tile image
                    tile_img_path = os.path.join(tiled_images_dir, tile_img.file_name)
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
                        original_img.file_name, tile_img.file_name,
                        tile_bboxes=tile_bbox_lookup.get(base_name, []),
                    )
                    
                    # Save comparison with high quality
                    comparison_filename = f"comparison_{i+1:02d}_{j+1}__{original_img.file_name}_vs_{tile_img.file_name}"
                    comparison_path = os.path.join(output_dir, comparison_filename)
                    comparison.save(comparison_path, quality=95, optimize=True)
                
            except Exception as e:
                print(f"  Error processing {original_img.file_name}: {e}")
        
        print(f"\nComparisons saved to: {output_dir}")
    
    def create_single_comparison(
        self,
        original_img: Image.Image,
        tile_img: Image.Image,
        original_anns,
        tile_anns,
        categories,
        tile_offset,
        original_name: str,
        tile_name: str,
        tile_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Image.Image:
        """Create a single side-by-side comparison image."""
        
        # Draw tile boundaries on original image first, then bounding boxes
        orig_with_tiles = self.visualizer.draw_tile_boundaries(
            original_img,
            tile_img.size,
            0,
            highlight_tile=tile_offset,
            tile_bboxes=tile_bboxes,
        )
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
    
    def create_overview_grid(
        self,
        original_path: str,
        tiled_path: str,
        output_dir: str,
        tiled_split: Optional[str] = None,
    ) -> None:
        """Create an overview grid showing multiple examples."""
        
        print("Creating overview grid...")
        
        original_images_dir, original_annotations = _resolve_split_paths(original_path, "train")
        tiled_images_dir, tiled_annotations = _resolve_split_paths(tiled_path, tiled_split)

        original_dataset = CocoDataset.from_json(original_annotations)
        tiled_dataset = CocoDataset.from_json(tiled_annotations)
        categories = {cat.id: cat.name for cat in original_dataset.categories}

        tile_base_names = {img.file_name.split("_tile_")[0] for img in tiled_dataset.images}
        eligible_originals = [img for img in original_dataset.images if Path(img.file_name).stem in tile_base_names]

        if not eligible_originals:
            print("Warning: unable to build overview grid (no overlapping images).")
            return

        sample_images = random.sample(eligible_originals, min(4, len(eligible_originals)))
        
        grid_images = []
        
        for original_img in sample_images:
            try:
                # Load original image
                original_img_path = os.path.join(original_images_dir, original_img.file_name)
                original_image = Image.open(original_img_path)
                
                # Get original annotations
                original_anns = [ann for ann in original_dataset.annotations if ann.image_id == original_img.id]
                
                # Draw bounding boxes and resize for grid
                orig_with_boxes = self.visualizer.draw_bounding_boxes(original_image, original_anns, categories)
                orig_with_boxes = orig_with_boxes.resize((400, 300), Image.Resampling.LANCZOS)
                
                # Find one representative tile
                base_name = Path(original_img.file_name).stem
                tiles = [img for img in tiled_dataset.images if img.file_name.startswith(base_name)]
                
                if tiles:
                    # Pick middle tile or first one
                    representative_tile = tiles[len(tiles)//2] if len(tiles) > 1 else tiles[0]
                    
                    tile_img_path = os.path.join(tiled_images_dir, representative_tile.file_name)
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
    parser.add_argument(
        "--tiled-split",
        default=None,
        help="Specific split inside the tiled dataset (e.g., train, val, test, fold_1/train)",
    )
    parser.add_argument("--output", default="./comparison_visualizations", help="Output directory")
    parser.add_argument("--samples", type=int, default=5, help="Number of original images to compare")
    parser.add_argument("--overview", action="store_true", help="Create overview grid")

    args = parser.parse_args()

    print("Dataset Comparison Tool")
    print("=" * 40)

    if not os.path.exists(args.original):
        print(f"Error: Original dataset not found: {args.original}")
        sys.exit(1)
    if not os.path.exists(args.tiled):
        print(f"Error: Tiled dataset not found: {args.tiled}")
        sys.exit(1)

    try:
        comparator = DatasetComparator()

        comparator.create_side_by_side_comparison(
            args.original,
            args.tiled,
            args.output,
            args.samples,
            tiled_split=args.tiled_split,
        )

        if args.overview:
            comparator.create_overview_grid(
                args.original,
                args.tiled,
                args.output,
                tiled_split=args.tiled_split,
            )

        print()
        print("Comparison visualizations complete!")
        print(f"Artifacts written to: {args.output}")
        print("Inspect the outputs to confirm annotation fidelity across tiling.")

    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
