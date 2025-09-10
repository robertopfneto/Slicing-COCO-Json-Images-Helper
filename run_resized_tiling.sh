#!/bin/bash

# Dataset Tiling Script with Resizing (1024x1024 -> 512x512)
# Creates 1024x1024 tiles and resizes them to 512x512 with proper annotation scaling

echo "=========================================="
echo "  Dataset Tiling with Resizing"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
INPUT_DATASET="${SCRIPT_DIR}/dataset"
OUTPUT_DATASET="${SCRIPT_DIR}/output_resized_1024"

# Tiling parameters for 1024x1024 tiles resized to 512x512
TILE_SIZE_WIDTH=1024
TILE_SIZE_HEIGHT=1024
RESIZE_WIDTH=512
RESIZE_HEIGHT=512
OVERLAP=0
MIN_COVERAGE=0.3

echo "Configuration:"
echo "  Input dataset: ${INPUT_DATASET}"
echo "  Output dataset: ${OUTPUT_DATASET}"
echo "  Tile size: ${TILE_SIZE_WIDTH}x${TILE_SIZE_HEIGHT}"
echo "  Resize to: ${RESIZE_WIDTH}x${RESIZE_HEIGHT}"
echo "  Overlap: ${OVERLAP} pixels"
echo "  Min coverage: ${MIN_COVERAGE}"
echo ""

# Activate conda environment if available
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  if conda env list | grep -q "create-dataset"; then
    echo "Activating conda environment 'create-dataset'..."
    conda activate create-dataset
    echo ""
  fi
fi

# Check if input dataset exists
if [ ! -d "${INPUT_DATASET}" ]; then
  echo "❌ Error: Input dataset not found: ${INPUT_DATASET}"
  exit 1
fi

# Check if annotations file exists
ANNOTATIONS_FILE="${INPUT_DATASET}/train/_annotations.coco.json"
if [ ! -f "${ANNOTATIONS_FILE}" ]; then
  echo "❌ Error: Annotations file not found: ${ANNOTATIONS_FILE}"
  exit 1
fi

# Remove output directory if it exists
if [ -d "${OUTPUT_DATASET}" ]; then
  echo "🗑️  Removing existing output directory..."
  rm -rf "${OUTPUT_DATASET}"
fi

echo "🚀 Starting dataset tiling with resizing..."
echo "This process will:"
echo "  1. Create ${TILE_SIZE_WIDTH}x${TILE_SIZE_HEIGHT} tiles from original images"
echo "  2. Resize each tile to ${RESIZE_WIDTH}x${RESIZE_HEIGHT}"
echo "  3. Scale annotation coordinates appropriately"
echo "  4. Preserve bounding box accuracy"
echo ""

# Run the tiling with resize parameters
python3 "${SCRIPT_DIR}/app.py" \
  --input "${INPUT_DATASET}" \
  --output "${OUTPUT_DATASET}" \
  --tile-size ${TILE_SIZE_WIDTH} ${TILE_SIZE_HEIGHT} \
  --resize-output ${RESIZE_WIDTH} ${RESIZE_HEIGHT} \
  --overlap ${OVERLAP} \
  --min-coverage ${MIN_COVERAGE} \
  --validate

TILING_EXIT_CODE=$?

echo ""
if [ ${TILING_EXIT_CODE} -eq 0 ]; then
  echo "=========================================="
  echo "   Resized Tiling Complete!"
  echo "=========================================="
  echo ""
  echo "✅ Dataset tiling with resizing completed successfully!"
  echo ""
  echo "📁 Output location: ${OUTPUT_DATASET}/"
  echo ""

  # Show some basic statistics
  ORIGINAL_COUNT=$(find "${INPUT_DATASET}/train" -name "*.jpg" | wc -l)
  TILED_COUNT=$(find "${OUTPUT_DATASET}/train" -name "*.jpg" | wc -l)

  echo "📊 Statistics:"
  echo "  📸 Original images: ${ORIGINAL_COUNT}"
  echo "  🧩 Generated tiles: ${TILED_COUNT}"

  if [ ${TILED_COUNT} -gt ${ORIGINAL_COUNT} ]; then
    RATIO=$((TILED_COUNT / ORIGINAL_COUNT))
    echo "  📈 Expansion ratio: ~${RATIO}x"
  fi

  echo ""
  echo "🎯 Features applied:"
  echo "  ✅ ${TILE_SIZE_WIDTH}x${TILE_SIZE_HEIGHT} tile extraction"
  echo "  ✅ ${RESIZE_WIDTH}x${RESIZE_HEIGHT} output resizing"
  echo "  ✅ Coordinate scaling for annotations"
  echo "  ✅ Bounding box preservation"
  echo "  ✅ Validation completed"
  echo ""

  echo "🔍 Next steps:"
  echo "  1. Run ./create_resized_analysis.sh to analyze the results"
  echo "  2. Check the output directory for resized tiles"
  echo "  3. Verify annotation accuracy in the analysis"
  echo ""

else
  echo "=========================================="
  echo "   Tiling Failed!"
  echo "=========================================="
  echo ""
  echo "❌ Dataset tiling with resizing failed with exit code: ${TILING_EXIT_CODE}"
  echo ""
  echo "💡 Troubleshooting:"
  echo "  - Check that the conda environment 'create-dataset' is properly set up"
  echo "  - Verify input dataset path and structure"
  echo "  - Check annotations file format"
  echo "  - Review error messages above"
  echo ""
  exit 1
fi

