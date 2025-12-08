# Dataset Tiling Application

A Configuration-Driven Architecture application for tiling images in Roboflow COCO datasets while preserving bounding box annotations with SAHI.

## Features

- Tiles large images into smaller, manageable pieces
- Preserves and transforms COCO JSON annotations
- Configurable tile size and overlap (static SAHI) or adaptive ASAHI auto-overlap
- Generates K-fold splits keeping all tiles of an image in the same fold/split
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

### Clean Existing Tiles

```bash
python app.py --output ./output --clean-output
```

### Adaptive auto-overlap (ASAHI-AutoOverlap)

By default, the adaptive mode now computes the overlap automatically so the tile grid fully covers the original image with tile size capped at 640px:

```bash
python app.py \
  --input ./dataset \
  --output ./output \
  --adaptive-mode \
  --validate
```

Environment toggle:
- `ASAHI_AUTO_OVERLAP=0` to fall back to the previous adaptive tiler.
- `ASAHI_OVERLAP_RATIO`/`ASAHI_LS_THRESHOLD` still apply to the legacy adaptive mode when auto-overlap is disabled.

The plan summary logs with `SAHI_VERBOSE=1` and the metadata is written once to `./output/tiling_plan.json` for later reconstruction.

### Configuration via Environment Variables

```bash
export TILE_WIDTH=640
export TILE_HEIGHT=640
export TILE_OVERLAP=40
export MIN_OBJECT_COVERAGE=0.3
export INPUT_PATH=./dataset
export OUTPUT_PATH=./output
export ASAHI_AUTO_OVERLAP=1  # enable adaptive auto-overlap (default)

python app.py
```

## Configuration Options

- `TILE_WIDTH`, `TILE_HEIGHT`: Tile dimensions (default: 640x640)
- `TILE_OVERLAP`: Overlap between tiles in pixels (default: 40)
- `MIN_OBJECT_COVERAGE`: Minimum fraction of object that must be visible (default: 0.3)
- `IGNORE_NEGATIVE_SAMPLES`: Skip images without annotations when tiling (default: true)
- `INPUT_PATH`: Input dataset directory (default: ./dataset)
- `OUTPUT_PATH`: Output directory (default: ./output)
- `ASAHI_AUTO_OVERLAP`: Enable adaptive auto-overlap grid (default: true)
- `ASAHI_ADAPTIVE_MODE`: Toggle ASAHI adaptive tiler (default: true)

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

## Verifying reconstruction from tiles

A helper script can reassemble a random image from its tiles for sanity checks:

```bash
python reconstruct_image.py
```

- Reads `./output/tiling_plan.json` when available to size the canvas.
- Picks a random stem in `./output/tile/fold_1/train`, pastes all tiles by their encoded offsets, and writes `./output/reconstructed_<stem>.jpg`.
