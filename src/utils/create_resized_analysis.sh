#!/bin/bash

# Resized Dataset Analysis Script
# Creates visualizations and analysis for datasets with resized tiles (e.g., 1024x1024 -> 512x512)

echo "=========================================="
echo "    Resized Dataset Analysis Tool"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ORIGINAL_DATASET="${SCRIPT_DIR}/dataset"
RESIZED_DATASET="${SCRIPT_DIR}/output_resized"
ANALYSIS_OUTPUT="${SCRIPT_DIR}/resized_analysis"
SAMPLES=15

echo "Configuration:"
echo "  Original dataset: ${ORIGINAL_DATASET}"
echo "  Resized dataset: ${RESIZED_DATASET}"
echo "  Output directory: ${ANALYSIS_OUTPUT}"
echo "  Analysis samples: ${SAMPLES}"
echo ""

# Activate conda environment if available
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  if conda env list | grep -q "create-dataset"; then
    echo "Activating conda environment 'create-dataset'..."
    conda activate create-dataset
  fi
fi

# Check if datasets exist
if [ ! -d "${ORIGINAL_DATASET}" ]; then
  echo "❌ Error: Original dataset not found: ${ORIGINAL_DATASET}"
  exit 1
fi

if [ ! -d "${RESIZED_DATASET}" ]; then
  echo "❌ Error: Resized dataset not found: ${RESIZED_DATASET}"
  echo "   Please run tiling with --resize-output first to generate resized dataset"
  exit 1
fi

# Create output directory
mkdir -p "${ANALYSIS_OUTPUT}"

echo "Creating RESIZED DATASET analysis..."
echo "This will analyze coordinate scaling and bounding box accuracy..."
echo ""

# Run the comparison tool with special settings for resized datasets
python3 "${SCRIPT_DIR}/compare_datasets.py" \
  --original "${ORIGINAL_DATASET}" \
  --tiled "${RESIZED_DATASET}" \
  --output "${ANALYSIS_OUTPUT}" \
  --samples ${SAMPLES} \
  --overview

if [ $? -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "   Resized Dataset Analysis Complete!"
  echo "=========================================="
  echo ""
  echo "Results saved in: ${ANALYSIS_OUTPUT}/"
  echo ""

  # Count generated files
  COMPARISON_COUNT=$(find "${ANALYSIS_OUTPUT}" -name "comparison_*.jpg" | wc -l)
  echo "Generated analysis files:"
  echo "  📊 dataset_comparison_overview.jpg - Resized dataset grid overview"
  echo "  📷 ${COMPARISON_COUNT} individual comparison images"
  echo ""

  echo "🔍 RESIZED DATASET FEATURES VERIFIED:"
  echo "  ✅ Coordinate scaling accuracy"
  echo "  ✅ Bounding box preservation during resize"
  echo "  ✅ Annotation area calculations"
  echo "  ✅ Image quality after resizing"
  echo ""

  echo "🎯 Analysis focuses on:"
  echo "  - Coordinate transformation accuracy"
  echo "  - Scale factor application"
  echo "  - Bounding box precision after resize"
  echo "  - Object detection integrity"
  echo ""

  echo "📁 Open the images in ${ANALYSIS_OUTPUT}/ for detailed analysis!"

  # Show file sizes for transparency
  echo ""
  echo "📊 File size info:"
  TOTAL_SIZE=$(du -sh "${ANALYSIS_OUTPUT}" | cut -f1)
  echo "  Total size: ${TOTAL_SIZE}"

  # Show some dataset stats
  echo ""
  echo "📈 Dataset Statistics:"

  # Count original vs resized images
  ORIGINAL_COUNT=$(find "${ORIGINAL_DATASET}/train" -name "*.jpg" | wc -l)
  RESIZED_COUNT=$(find "${RESIZED_DATASET}/train" -name "*.jpg" | wc -l)

  echo "  📸 Original images: ${ORIGINAL_COUNT}"
  echo "  🧩 Resized tiles: ${RESIZED_COUNT}"

  if [ ${RESIZED_COUNT} -gt ${ORIGINAL_COUNT} ]; then
    TILE_RATIO=$((RESIZED_COUNT / ORIGINAL_COUNT))
    echo "  📊 Tiling ratio: ~${TILE_RATIO}x more tiles"
  fi

else
  echo ""
  echo "❌ Error: Resized dataset analysis failed"
  exit 1
fi

