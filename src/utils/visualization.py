import os
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import random

from src.models.coco import CocoDataset, CocoAnnotation, CocoImage


class BoundingBoxVisualizer:
    """Visualizes images with bounding boxes for verification purposes."""
    
    def __init__(self):
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
            '#FFA500', '#800080', '#008000', '#FFC0CB', '#A52A2A', '#808080'
        ]
        
    def draw_bounding_boxes(self, image: Image.Image, annotations: List[CocoAnnotation], 
                           categories: dict, show_labels: bool = True) -> Image.Image:
        """Draw bounding boxes on an image."""
        # Create a copy to avoid modifying the original
        img_with_boxes = image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        # Scale font and line width based on image size for better visibility
        image_size = max(image.width, image.height)
        font_size = max(16, int(image_size / 150))  # Dynamic font size
        line_width = max(3, int(image_size / 800))   # Dynamic line width
        
        # Try to use a better font if available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        for i, ann in enumerate(annotations):
            # Get bounding box coordinates [x, y, width, height]
            x, y, width, height = ann.bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + width), int(y + height)
            
            # Choose color based on category
            color = self.colors[ann.category_id % len(self.colors)]
            
            # Draw bounding box with dynamic width
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            if show_labels:
                # Get category name
                category_name = categories.get(ann.category_id, f"Category {ann.category_id}")
                
                # Draw label background
                label = f"{category_name} (ID: {ann.id})"
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                label_height = bbox[3] - bbox[1]
                
                # Position label
                label_x = max(0, x1)
                label_y = max(0, y1 - label_height - 5)
                
                # Draw label background
                draw.rectangle([label_x, label_y, label_x + label_width + 4, 
                              label_y + label_height + 4], fill=color, outline=color)
                
                # Draw label text
                draw.text((label_x + 2, label_y + 2), label, fill='white', font=font)
        
        return img_with_boxes
    
    def draw_tile_boundaries(self, image: Image.Image, tile_size: Tuple[int, int], 
                            overlap: int = 0, highlight_tile: Optional[Tuple[int, int]] = None) -> Image.Image:
        """Draw tile boundaries on an image to show how it would be sliced."""
        img_with_tiles = image.copy()
        draw = ImageDraw.Draw(img_with_tiles)
        
        img_width, img_height = image.size
        tile_width, tile_height = tile_size
        step_x = tile_width - overlap
        step_y = tile_height - overlap
        
        # Generate tile positions (same logic as TilingEngine)
        tile_positions = []
        
        # Regular grid
        for y in range(0, img_height - tile_height + 1, step_y):
            for x in range(0, img_width - tile_width + 1, step_x):
                tile_positions.append((x, y))
        
        # Edge tiles
        if img_width % step_x != 0:
            x = img_width - tile_width
            for y in range(0, img_height - tile_height + 1, step_y):
                tile_positions.append((x, y))
        
        if img_height % step_y != 0:
            y = img_height - tile_height
            for x in range(0, img_width - tile_width + 1, step_x):
                tile_positions.append((x, y))
        
        # Corner tile
        if img_width % step_x != 0 and img_height % step_y != 0:
            x = img_width - tile_width
            y = img_height - tile_height
            tile_positions.append((x, y))
        
        # Draw tile boundaries
        line_width = max(2, int(max(img_width, img_height) / 1000))  # Dynamic line width
        
        for i, (x, y) in enumerate(tile_positions):
            # Choose color and width for highlighting
            if highlight_tile and (x, y) == highlight_tile:
                color = '#FF0000'  # Red for highlighted tile
                width = line_width * 3
            else:
                color = '#00FFFF'  # Cyan for regular tile boundaries
                width = line_width
            
            # Draw tile rectangle
            draw.rectangle([x, y, x + tile_width, y + tile_height], 
                          outline=color, width=width)
            
            # Add tile index
            font_size = max(12, int(max(img_width, img_height) / 200))
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Draw tile number with background
            tile_label = f"T{i+1}"
            bbox = draw.textbbox((0, 0), tile_label, font=font)
            label_width = bbox[2] - bbox[0]
            label_height = bbox[3] - bbox[1]
            
            label_x = x + 5
            label_y = y + 5
            draw.rectangle([label_x - 2, label_y - 2, label_x + label_width + 2, 
                          label_y + label_height + 2], fill='white', outline='black')
            draw.text((label_x, label_y), tile_label, fill='black', font=font)
        
        return img_with_tiles

    def create_comparison_view(
        self,
        original_img: Image.Image,
        tiled_img: Image.Image,
        original_annotations: List[CocoAnnotation],
        tiled_annotations: List[CocoAnnotation],
        categories: dict,
        tile_offset: Tuple[int, int],
        tile_size: Tuple[int, int],
        overlap: int = 0,
        tile_label: Optional[str] = None,
    ) -> Image.Image:
        """Create a side-by-side comparison of original and tiled images with boxes."""

        # Draw bounding boxes and tile boundaries on original image
        # First draw tile boundaries, then bounding boxes on top
        orig_with_tiles = self.draw_tile_boundaries(original_img, tile_size, overlap, tile_offset)
        orig_with_boxes = self.draw_bounding_boxes(orig_with_tiles, original_annotations, categories)
        
        # Draw only bounding boxes on tiled image
        tiled_with_boxes = self.draw_bounding_boxes(tiled_img, tiled_annotations, categories)
        
        # Create comparison image
        total_width = orig_with_boxes.width + tiled_with_boxes.width + 20
        total_height = max(orig_with_boxes.height, tiled_with_boxes.height) + 100
        
        comparison = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(comparison)
        
        # Try to use a better font for titles
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            title_font = ImageFont.load_default()
        
        # Add titles
        tile_info = f"Offset ({tile_offset[0]}, {tile_offset[1]})"
        tile_dims = f"Tile {tile_size[0]}x{tile_size[1]}"
        right_title = f"Tiled Image | {tile_info} | {tile_dims}"
        if tile_label:
            right_title = f"{right_title} | {tile_label}"

        draw.text((10, 10), "Original Image", fill='black', font=title_font)
        draw.text((orig_with_boxes.width + 30, 10), right_title, fill='black', font=title_font)
        
        # Paste images
        comparison.paste(orig_with_boxes, (10, 50))
        comparison.paste(tiled_with_boxes, (orig_with_boxes.width + 20, 50))
        
        # Add annotation counts
        info_y = 50 + max(orig_with_boxes.height, tiled_with_boxes.height) + 10
        draw.text((10, info_y), 
                 f"Original annotations: {len(original_annotations)}", 
                 fill='blue', font=title_font)
        draw.text((orig_with_boxes.width + 30, info_y), 
                 f"Tiled annotations: {len(tiled_annotations)}", 
                 fill='blue', font=title_font)
        
        return comparison
    
    def create_tiling_overview(self, image: Image.Image, annotations: List[CocoAnnotation], 
                              categories: dict, tile_size: Tuple[int, int] = (512, 512), 
                              overlap: int = 0, max_width: int = 1600) -> Image.Image:
        """Create an overview showing the complete tiling grid with annotations."""
        
        # Scale image if too large for overview
        img_width, img_height = image.size
        if img_width > max_width:
            scale_factor = max_width / img_width
            new_width = max_width
            new_height = int(img_height * scale_factor)
            scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Scale tile size and annotations accordingly
            scaled_tile_size = (int(tile_size[0] * scale_factor), int(tile_size[1] * scale_factor))
            scaled_overlap = int(overlap * scale_factor)
            
            # Scale annotations
            scaled_annotations = []
            for ann in annotations:
                scaled_bbox = [coord * scale_factor for coord in ann.bbox]
                scaled_ann = CocoAnnotation(
                    id=ann.id,
                    image_id=ann.image_id,
                    category_id=ann.category_id,
                    segmentation=ann.segmentation,
                    area=ann.area * (scale_factor ** 2),
                    bbox=scaled_bbox,
                    iscrowd=ann.iscrowd
                )
                scaled_annotations.append(scaled_ann)
        else:
            scaled_image = image
            scaled_tile_size = tile_size
            scaled_overlap = overlap
            scaled_annotations = annotations
            scale_factor = 1.0
        
        # Draw tile boundaries first
        img_with_tiles = self.draw_tile_boundaries(scaled_image, scaled_tile_size, scaled_overlap)
        
        # Draw annotations on top
        final_image = self.draw_bounding_boxes(img_with_tiles, scaled_annotations, categories)
        
        # Add title and info
        info_height = 80
        final_width = final_image.width
        final_height = final_image.height + info_height
        
        overview = Image.new('RGB', (final_width, final_height), 'white')
        draw = ImageDraw.Draw(overview)
        
        # Add title
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            info_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()
        
        title = "Tiling Overview with Annotations"
        draw.text((10, 10), title, fill='black', font=title_font)
        
        info_text = f"Original: {img_width}x{img_height} | Tile: {tile_size[0]}x{tile_size[1]} | Overlap: {overlap}px | Scale: {scale_factor:.2f}"
        draw.text((10, 40), info_text, fill='blue', font=info_font)
        
        # Paste the main image
        overview.paste(final_image, (0, info_height))
        
        return overview
    
    def visualize_dataset(self, dataset_path: str, output_dir: str, 
                         max_samples: int = 10, show_comparisons: bool = False) -> None:
        """Visualize a dataset with bounding boxes."""
        
        # Load dataset
        annotations_path = os.path.join(dataset_path, "train", "_annotations.coco.json")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        
        dataset = CocoDataset.from_json(annotations_path)
        
        # Create category lookup
        categories = {cat.id: cat.name for cat in dataset.categories}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample images to visualize
        sample_images = random.sample(dataset.images, min(max_samples, len(dataset.images)))
        
        print(f"Visualizing {len(sample_images)} images from dataset...")
        print(f"Total categories: {len(categories)}")
        for cat_id, cat_name in categories.items():
            print(f"  - {cat_name} (ID: {cat_id})")
        print()
        
        for i, image_info in enumerate(sample_images):
            print(f"Processing image {i+1}/{len(sample_images)}: {image_info.file_name}")
            
            # Load image
            image_path = os.path.join(dataset_path, "train", image_info.file_name)
            if not os.path.exists(image_path):
                print(f"  Warning: Image file not found: {image_path}")
                continue
            
            try:
                image = Image.open(image_path)
                
                # Get annotations for this image
                image_annotations = [ann for ann in dataset.annotations 
                                   if ann.image_id == image_info.id]
                
                print(f"  Found {len(image_annotations)} annotations")
                
                # Draw bounding boxes
                img_with_boxes = self.draw_bounding_boxes(image, image_annotations, categories)
                
                # Save visualization
                output_filename = f"visualization_{i+1:03d}_{image_info.file_name}"
                output_path = os.path.join(output_dir, output_filename)
                img_with_boxes.save(output_path)
                
                print(f"  Saved: {output_path}")
                
            except Exception as e:
                print(f"  Error processing {image_info.file_name}: {e}")
        
        print(f"\nVisualization complete! Check the '{output_dir}' folder.")
    
    def create_summary_report(self, dataset_path: str, output_dir: str) -> None:
        """Create a summary report of the dataset."""
        
        annotations_path = os.path.join(dataset_path, "train", "_annotations.coco.json")
        dataset = CocoDataset.from_json(annotations_path)
        
        # Calculate statistics
        total_images = len(dataset.images)
        total_annotations = len(dataset.annotations)
        categories = {cat.id: cat.name for cat in dataset.categories}
        
        # Count annotations per category
        category_counts = {}
        for ann in dataset.annotations:
            cat_name = categories.get(ann.category_id, f"Unknown_{ann.category_id}")
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        # Calculate average annotations per image
        avg_annotations = total_annotations / total_images if total_images > 0 else 0
        
        # Create report
        report_path = os.path.join(output_dir, "dataset_summary.txt")
        with open(report_path, 'w') as f:
            f.write("Dataset Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Images: {total_images}\n")
            f.write(f"Total Annotations: {total_annotations}\n")
            f.write(f"Average Annotations per Image: {avg_annotations:.2f}\n")
            f.write(f"Number of Categories: {len(categories)}\n\n")
            
            f.write("Category Breakdown:\n")
            f.write("-" * 20 + "\n")
            for cat_name, count in sorted(category_counts.items()):
                percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
                f.write(f"{cat_name}: {count} annotations ({percentage:.1f}%)\n")
            
            # Image size analysis
            if dataset.images:
                widths = [img.width for img in dataset.images]
                heights = [img.height for img in dataset.images]
                f.write(f"\nImage Size Analysis:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Width range: {min(widths)} - {max(widths)} pixels\n")
                f.write(f"Height range: {min(heights)} - {max(heights)} pixels\n")
                f.write(f"Most common size: {max(set(zip(widths, heights)), key=zip(widths, heights).count)}\n")
        
        print(f"Summary report saved: {report_path}")
