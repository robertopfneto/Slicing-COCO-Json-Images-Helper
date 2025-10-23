#!/bin/bash

# SAGE Dataset Analysis Script
# Generates visual comparisons between the original dataset and a SAGE tiled split.

set -euo pipefail

echo "=========================================="
echo "        SAGE Dataset Analysis Tool"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ORIGINAL_DATASET="${SCRIPT_DIR}/dataset"

# Default location for SAGE outputs produced by generate_kfold_tiles.py
DEFAULT_SAGE_ROOT="${SCRIPT_DIR}/output/tile"
ALT_SAGE_ROOT="${SCRIPT_DIR}/dataset/tiles/sage"

FOLD_NAME="${FOLD_NAME:-fold_1}"
SPLIT_NAME="${SPLIT_NAME:-train}"
SAMPLES="${SAMPLES:-12}"

if [ -d "${DEFAULT_SAGE_ROOT}" ]; then
  SAGE_ROOT="${DEFAULT_SAGE_ROOT}"
elif [ -d "${ALT_SAGE_ROOT}" ]; then
  SAGE_ROOT="${ALT_SAGE_ROOT}"
else
  echo "ERROR: Could not locate a SAGE output directory."
  echo "Checked:"
  echo "  - ${DEFAULT_SAGE_ROOT}"
  echo "  - ${ALT_SAGE_ROOT}"
  echo "Run your SAGE tiling pipeline before executing this analysis."
  exit 1
fi

TILED_DATASET="${SAGE_ROOT}/${FOLD_NAME}/${SPLIT_NAME}"
ANALYSIS_ROOT="${SCRIPT_DIR}/sage_analysis"
ANALYSIS_OUTPUT="${ANALYSIS_ROOT}/${FOLD_NAME}_${SPLIT_NAME}"

echo "Configuration:"
echo "  Original dataset : ${ORIGINAL_DATASET}"
echo "  SAGE dataset     : ${TILED_DATASET}"
echo "  Output directory : ${ANALYSIS_OUTPUT}"
echo "  Fold / split     : ${FOLD_NAME} / ${SPLIT_NAME}"
echo "  Sample pairs     : ${SAMPLES}"
echo ""

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  if conda env list | grep -q "create-dataset"; then
    echo "Activating conda environment 'create-dataset'..."
    conda activate create-dataset
    echo ""
  fi
fi

# -----------------------------------------------------------------------------
# Validations
# -----------------------------------------------------------------------------
if [ ! -d "${ORIGINAL_DATASET}" ]; then
  echo "ERROR: Original dataset not found: ${ORIGINAL_DATASET}"
  exit 1
fi

if [ ! -f "${ORIGINAL_DATASET}/train/_annotations.coco.json" ]; then
  echo "ERROR: Original annotations not found at ${ORIGINAL_DATASET}/train/_annotations.coco.json"
  exit 1
fi

if [ ! -d "${TILED_DATASET}" ]; then
  echo "ERROR: SAGE split not found: ${TILED_DATASET}"
  echo "Make sure the requested fold/split exists."
  exit 1
fi

if [ ! -f "${TILED_DATASET}/_annotations.coco.json" ]; then
  echo "ERROR: SAGE annotations not found at ${TILED_DATASET}/_annotations.coco.json"
  exit 1
fi

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
mkdir -p "${ANALYSIS_OUTPUT}"

echo "Generating SAGE comparison visualizations..."
echo "This will focus on the selected fold/split to verify alignment and annotation fidelity."
echo ""

python3 "${SCRIPT_DIR}/compare_datasets.py" \
  --original "${ORIGINAL_DATASET}" \
  --tiled "${TILED_DATASET}" \
  --output "${ANALYSIS_OUTPUT}" \
  --samples "${SAMPLES}" \
  --overview

STATUS=$?

echo ""
if [ ${STATUS} -eq 0 ]; then
  echo "=========================================="
  echo "      SAGE Dataset Analysis Complete"
  echo "=========================================="
  echo ""
  echo "Results saved in: ${ANALYSIS_OUTPUT}/"
  echo ""

  COMPARISON_COUNT=$(find "${ANALYSIS_OUTPUT}" -name "comparison_*.jpg" | wc -l)
  OVERVIEW_PATH="${ANALYSIS_OUTPUT}/dataset_comparison_overview.jpg"

  echo "Generated artifacts:"
  if [ -f "${OVERVIEW_PATH}" ]; then
    echo "  - dataset_comparison_overview.jpg (grid summary of sampled tiles)"
  else
    echo "  - Overview grid not generated (check logs for details)"
  fi
  echo "  - ${COMPARISON_COUNT} individual comparison images"
  echo ""

  echo "Checks to perform:"
  echo "  - Tiles align with the stride-aligned SAGE grid."
  echo "  - Bounding boxes remain consistent after slicing."
  echo "  - Objects spanning multiple tiles keep correct offsets."
  echo "  - Category labels match between original and tiled views."
  echo ""

  echo "Tip: Set FOLD_NAME and SPLIT_NAME to audit other splits, e.g."
  echo "  FOLD_NAME=fold_2 SPLIT_NAME=val ./create_resized_analysis.sh"
  echo ""
else
  echo "ERROR: Analysis failed (exit code ${STATUS}). Review the messages above."
  exit ${STATUS}
fi
