# Dataset Tiling Application

A Configuration-Driven Architecture application for tiling images in Roboflow COCO datasets while preserving bounding box annotations with SAHI.

## Features

- Tiles large images into smaller, manageable pieces
- Preserves and transforms COCO JSON annotations
- Configurable tile size and overlap
- Minimum object coverage filtering
- Configuration-driven design for easy customization

## Architecture

The application follows Configuration-Driven Architecture principles:

```
src/
├── config/           # Configuration management
├── core/             # Core business logic
│   ├── tiling/       # Tiling algorithms
│   └── validation/   # Validation logic
├── services/         # Service layer
│   ├── dataset/      # Dataset processing
│   ├── image/        # Image handling
│   └── annotation/   # Annotation management
├── models/           # Data models
└── utils/            # Utility functions
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python app.py --input ./dataset --output ./output
```

### Advanced Usage

```bash
python app.py \
  --input ./dataset \
  --output ./tiled_dataset \
  --tile-size 640 640 \
  --overlap 40 \
  --min-coverage 0.3 \
  --validate
```

### Configuration via Environment Variables

```bash
export TILE_WIDTH=640
export TILE_HEIGHT=640
export TILE_OVERLAP=40
export MIN_OBJECT_COVERAGE=0.3
export INPUT_PATH=./dataset
export OUTPUT_PATH=./output

python app.py
```

## Configuration Options

- `TILE_WIDTH`, `TILE_HEIGHT`: Tile dimensions (default: 640x640)
- `TILE_OVERLAP`: Overlap between tiles in pixels (default: 40)
- `MIN_OBJECT_COVERAGE`: Minimum fraction of object that must be visible (default: 0.3)
- `INPUT_PATH`: Input dataset directory (default: ./dataset)
- `OUTPUT_PATH`: Output directory (default: ./output)

## Input Format

The application expects a Roboflow COCO dataset structure:

```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
└── ...
```

## Output Format

The application now generates K-fold cross-validation splits while keeping all tiles from the same image together within a fold:

```
output/
└── tile/
    ├── fold_0/
    │   ├── train/
    │   │   ├── image1_tile_0_0.jpg
    │   │   └── ...
    │   ├── val/
    │   │   └── _annotations.coco.json
    │   └── test/
    │       └── _annotations.coco.json
    └── fold_1/
        └── ...
```

Each split directory contains its own tiled images and `_annotations.coco.json` describing only that fold and split.
