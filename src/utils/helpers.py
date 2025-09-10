import os
import json
from typing import Any, Dict, List
from pathlib import Path


def ensure_directory(path: str) -> None:
    """Ensure that a directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def get_file_extension(file_path: str) -> str:
    """Get file extension from path."""
    return Path(file_path).suffix.lower()


def is_image_file(file_path: str) -> bool:
    """Check if file is a supported image format."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return get_file_extension(file_path) in image_extensions


def get_files_with_extension(directory: str, extension: str) -> List[str]:
    """Get all files with specified extension in directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(extension.lower()):
                files.append(os.path.join(root, filename))
    return files


def generate_unique_filename(base_path: str, filename: str) -> str:
    """Generate a unique filename by adding numbers if file exists."""
    file_path = os.path.join(base_path, filename)
    if not os.path.exists(file_path):
        return filename
    
    name, ext = os.path.splitext(filename)
    counter = 1
    
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_path = os.path.join(base_path, new_filename)
        if not os.path.exists(new_path):
            return new_filename
        counter += 1


def calculate_split_indices(total_count: int, train_ratio: float, 
                          val_ratio: float, test_ratio: float) -> Dict[str, range]:
    """Calculate indices for train/val/test splits."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count
    
    return {
        'train': range(0, train_count),
        'val': range(train_count, train_count + val_count),
        'test': range(train_count + val_count, total_count)
    }