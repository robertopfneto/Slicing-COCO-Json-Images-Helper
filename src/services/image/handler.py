import os
from typing import Tuple, Optional
from PIL import Image
import numpy as np


class ImageHandler:
    """Handles image loading, processing, and saving operations."""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[Image.Image]:
        """Load an image from the given path."""
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def save_image(image: Image.Image, output_path: str, quality: int = 95) -> bool:
        """Save an image to the given path."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path, quality=quality)
            return True
        except Exception as e:
            print(f"Error saving image {output_path}: {e}")
            return False
    
    @staticmethod
    def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
        """Get image dimensions without fully loading the image."""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception as e:
            print(f"Error getting dimensions for {image_path}: {e}")
            return None
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """Validate if the image file is readable and valid."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def resize_image(image: Image.Image, size: Tuple[int, int], 
                    maintain_aspect_ratio: bool = False) -> Image.Image:
        """Resize an image to the specified size."""
        if maintain_aspect_ratio:
            image.thumbnail(size, Image.Resampling.LANCZOS)
            return image
        else:
            return image.resize(size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def crop_image(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """Crop an image using the given bounding box (x1, y1, x2, y2)."""
        return image.crop(bbox)
    
    @staticmethod
    def convert_format(image: Image.Image, format: str = "RGB") -> Image.Image:
        """Convert image to specified format."""
        return image.convert(format)