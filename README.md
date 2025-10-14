# Dataset Tiling Application

A Configuration-Driven Architecture application for tiling images in Roboflow COCO datasets while preserving bounding box annotations.

## Features

- Tiles large images into smaller, manageable pieces
- Preserves and transforms COCO JSON annotations
- Configurable tile size and overlap
- Minimum object coverage filtering
- SAGE-based k-fold tiling pipeline that keeps tiles from the same image inside the same fold
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
  --overlap 50 \
  --min-coverage 0.3 \
  --validate
```

### Configuration via Environment Variables

```bash
export TILE_WIDTH=640
export TILE_HEIGHT=640
export TILE_OVERLAP=50
export MIN_OBJECT_COVERAGE=0.3
export INPUT_PATH=./dataset
export OUTPUT_PATH=./output

python app.py
```

### K-Fold SAGE Tiling

Use `generate_kfold_tiles.py` to produce SAGE-tilled cross-validation splits where every tile created from an original image remains in the same fold:

```bash
python generate_kfold_tiles.py
```

The script loops through all folds and splits (`train`, `val`, `test`) and invokes the adaptive SAGE tiler under the hood. It first looks for fold definition JSON files (e.g. `dataset/all/filesJSON/fold_1_train.json`) and, if they are absent, builds deterministic fallback splits directly from `dataset/train/_annotations.coco.json`. The tiled datasets are written to `dataset/tiles/sage/fold_<k>/<split>/` with COCO annotations, summaries, and metadata describing the exact tiling parameters that were used.

## Configuration Options

- `TILE_WIDTH`, `TILE_HEIGHT`: Tile dimensions (default: 640x640)
- `TILE_OVERLAP`: Overlap between tiles in pixels (default: 0)
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

The application generates a tiled dataset with the same structure:

```
output/
└── train/
    ├── image1_tile_0_0.jpg
    ├── image1_tile_640_0.jpg
    ├── image2_tile_0_0.jpg
    └── _annotations.coco.json
```
