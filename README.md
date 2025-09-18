# Dataset Tiling Application

A Configuration-Driven Architecture application for tiling images in Roboflow COCO datasets while preserving bounding box annotations.

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
  --tile-size 512 512 \
  --overlap 50 \
  --min-coverage 0.3 \
  --validate
```

### Cross-validation tiling for large images

For high-resolution datasets (e.g., 4032×2268 insect imagery) that require tiling **and** image-level cross-validation, use the dedicated pipeline:

```bash
python build_crossval_tiles.py \
  --input ./dataset/all/train \
  --output ./output_crossval \
  --tile-size 640 640 \
  --overlap 0 \
  --min-ioa 0.30 \
  --folds 5 \
  --seed 42 \
  --overwrite
```

This command will:

- Slice every source image into 640×640 tiles, anchoring border tiles to preserve the full field of view.
- Keep only bounding boxes whose Intersection-over-Area (IoA) with the tile is at least 0.30, with coordinates clipped to tile bounds.
- Share the tile images across folds (via symlinks or copies) to avoid duplicating disk usage.
- Build GroupKFold (k=5) splits **grouped by original image id**, guaranteeing that tiles from one image never leak into other folds.
- Save per-fold COCO annotations under `output_crossval/folds/fold_*/{train,val,test}/_annotations.coco.json` along with tile id lists for hard-negative mining.
- Produce manifests in `output_crossval/manifests/` that map every tile back to its source image and original annotations, enabling reconstruction and quality checks.

### Visualizing a tiling example

Need a quick illustration similar to the screenshot in the issue description? Generate a side-by-side figure that overlays the tile grid on the original image, highlights a chosen tile, and shows its cropped view with projected boxes:

```bash
python generate_tiling_example.py \
  --input ./dataset/all/train \
  --image-file 100_jpg.rf.6abfdba36004e71c82c893d2028804c9.jpg \
  --tile-size 640 640 \
  --overlap 0 \
  --min-ioa 0.30 \
  --output ./visualizations/tiling_examples
```

The script saves a JPG figure plus a JSON summary describing the highlighted tile, including IoA values for each retained annotation. Leave `--image-file` unset to automatically pick the first annotated image.

### Configuration via Environment Variables

```bash
export TILE_WIDTH=512
export TILE_HEIGHT=512
export TILE_OVERLAP=50
export MIN_OBJECT_COVERAGE=0.3
export INPUT_PATH=./dataset
export OUTPUT_PATH=./output

python app.py
```

## Configuration Options

- `TILE_WIDTH`, `TILE_HEIGHT`: Tile dimensions (default: 512x512)
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
    ├── image1_tile_512_0.jpg
    ├── image2_tile_0_0.jpg
    └── _annotations.coco.json
```
