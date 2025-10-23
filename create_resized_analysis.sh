#!/bin/bash

# Resized Dataset Analysis Script (SAHI / ASAHI aware)
# Creates visualizations and analysis for datasets with resized tiles

echo "=========================================="
echo "    Resized Dataset Analysis Tool"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODE="${MODE:-sahi}"                # sahi | asahi | asahi_cluster
ASAHI_FOLD="${ASAHI_FOLD:-1}"       # fold index used when MODE is ASAHI-like
DEFAULT_ORIGINAL="${SCRIPT_DIR}/dataset"
DEFAULT_RESIZED="${SCRIPT_DIR}/output_resized"
DEFAULT_ANALYSIS="${SCRIPT_DIR}/resized_analysis"

if [[ "${MODE}" == "asahi" || "${MODE}" == "asahi_cluster" ]]; then
  DEFAULT_RESIZED="${SCRIPT_DIR}/output/tile/fold_${ASAHI_FOLD}"
  DEFAULT_ANALYSIS="${SCRIPT_DIR}/resized_analysis_asahi_fold${ASAHI_FOLD}"
fi

ORIGINAL_DATASET="${ORIGINAL_DATASET:-${DEFAULT_ORIGINAL}}"
RESIZED_DATASET="${RESIZED_DATASET:-${DEFAULT_RESIZED}}"
ANALYSIS_OUTPUT="${ANALYSIS_OUTPUT:-${DEFAULT_ANALYSIS}}"
SAMPLES="${SAMPLES:-15}"

echo "Configuration:"
echo "  Mode: ${MODE}"
echo "  Original dataset: ${ORIGINAL_DATASET}"
echo "  Resized dataset: ${RESIZED_DATASET}"
echo "  Output directory: ${ANALYSIS_OUTPUT}"
echo "  Analysis samples: ${SAMPLES}"
if [[ "${MODE}" == "asahi" || "${MODE}" == "asahi_cluster" ]]; then
  echo "  ASAHI fold: ${ASAHI_FOLD}"
fi
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
  echo "[X] Error: Original dataset not found: ${ORIGINAL_DATASET}"
  exit 1
fi

if [ ! -d "${RESIZED_DATASET}" ]; then
  echo "[X] Error: Resized dataset not found: ${RESIZED_DATASET}"
  if [[ "${MODE}" == "asahi" || "${MODE}" == "asahi_cluster" ]]; then
    echo "    Expected ASAHI tiles under output/tile/fold_${ASAHI_FOLD}."
    echo "    Override RESIZED_DATASET or ASAHI_FOLD to point to the desired fold."
  else
    echo "    Run tiling with --resize-output to generate resized tiles."
  fi
  exit 1
fi

# Create output directory
mkdir -p "${ANALYSIS_OUTPUT}"

echo "Creating dataset analysis..."
echo "This inspects coordinate scaling, bounding boxes and resize fidelity."
echo ""

# Run the comparison tool with special settings for resized datasets
python3 "${SCRIPT_DIR}/compare_datasets.py" \
  --original "${ORIGINAL_DATASET}" \
  --tiled "${RESIZED_DATASET}" \
  --output "${ANALYSIS_OUTPUT}" \
  --samples "${SAMPLES}" \
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
  COMPARISON_COUNT=$(find "${ANALYSIS_OUTPUT}" -name "comparison_*.jpg" 2>/dev/null | wc -l)
  echo "Generated analysis files:"
  echo "  - dataset_comparison_overview.jpg (grid overview)"
  echo "  - ${COMPARISON_COUNT} individual comparison images"
  echo ""

  echo "Checks performed:"
  echo "  - Coordinate scaling accuracy"
  echo "  - Bounding box preservation during resize"
  echo "  - Annotation area calculations"
  echo "  - Image quality after resizing"
  echo ""

  echo "Analysis focus:"
  echo "  - Coordinate transformation accuracy"
  echo "  - Scale factor application"
  echo "  - Bounding box precision after resize"
  echo "  - Object detection integrity"
  echo ""

  echo "Open the images in ${ANALYSIS_OUTPUT}/ for detailed review."

  # Show file sizes for transparency
  echo ""
  echo "File size info:"
  TOTAL_SIZE=$(du -sh "${ANALYSIS_OUTPUT}" 2>/dev/null | cut -f1)
  echo "  Total size: ${TOTAL_SIZE}"

  # Show some dataset stats
  echo ""
  echo "Dataset statistics:"

  # Count original vs resized images
  ORIGINAL_COUNT=$(find "${ORIGINAL_DATASET}/train" -name "*.jpg" 2>/dev/null | wc -l)
  RESIZED_COUNT=$(find "${RESIZED_DATASET}/train" -name "*.jpg" 2>/dev/null | wc -l)

  echo "  - Original images: ${ORIGINAL_COUNT}"
  echo "  - Resized tiles: ${RESIZED_COUNT}"

  if [ "${ORIGINAL_COUNT}" -gt 0 ] && [ "${RESIZED_COUNT}" -gt "${ORIGINAL_COUNT}" ]; then
    TILE_RATIO=$((RESIZED_COUNT / ORIGINAL_COUNT))
    echo "  - Tiling ratio: ~${TILE_RATIO}x more tiles"
  fi

else
  echo ""
  echo "[X] Error: Resized dataset analysis failed"
  exit 1
fi
