#!/bin/bash

# Dataset Comparison Script
# Creates side-by-side visualizations of original vs tiled datasets

echo "=========================================="
echo "    Dataset Comparison Visualization"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ORIGINAL_DATASET="${SCRIPT_DIR}/dataset"
TILED_DATASET="${SCRIPT_DIR}/output"
COMPARISON_OUTPUT="${SCRIPT_DIR}/comparison_visualizations"
SAMPLES=10

echo "Configuration:"
echo "  Original dataset: ${ORIGINAL_DATASET}"
echo "  Tiled dataset: ${TILED_DATASET}"
echo "  Output directory: ${COMPARISON_OUTPUT}"
echo "  Sample comparisons: ${SAMPLES}"
echo ""

# Activate conda environment if available
if command -v conda &> /dev/null; then
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

if [ ! -f "${ORIGINAL_DATASET}/train/_annotations.coco.json" ]; then
    echo "❌ Error: Original annotations not found"
    exit 1
fi

if [ ! -d "${TILED_DATASET}" ]; then
    echo "❌ Error: Tiled dataset not found: ${TILED_DATASET}"
    echo "   Please run ./run_tiling.sh first to generate tiled dataset"
    exit 1
fi

if [ ! -f "${TILED_DATASET}/train/_annotations.coco.json" ]; then
    echo "❌ Error: Tiled annotations not found"
    echo "   Please run ./run_tiling.sh first to generate tiled dataset"
    exit 1
fi

# Create output directory
mkdir -p "${COMPARISON_OUTPUT}"

echo "Creating dataset comparisons..."
echo ""

# Run the comparison tool
python3 "${SCRIPT_DIR}/compare_datasets.py" \
    --original "${ORIGINAL_DATASET}" \
    --tiled "${TILED_DATASET}" \
    --output "${COMPARISON_OUTPUT}" \
    --samples ${SAMPLES} \
    --overview

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "     Comparison Visualization Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved in: ${COMPARISON_OUTPUT}/"
    echo ""
    
    # Count generated files
    COMPARISON_COUNT=$(find "${COMPARISON_OUTPUT}" -name "comparison_*.jpg" | wc -l)
    echo "Generated files:"
    echo "  📊 dataset_comparison_overview.jpg - Grid overview of multiple examples"
    echo "  📷 ${COMPARISON_COUNT} individual comparison images"
    echo ""
    
    echo "What to look for:"
    echo "  ✅ Original images on the left with bounding boxes"
    echo "  ✅ Corresponding tiled images on the right with preserved annotations"
    echo "  ✅ Bounding box coordinates adjusted correctly for tile offset"
    echo "  ✅ No missing or incorrectly positioned annotations"
    echo "  ✅ Category labels consistent between original and tiled versions"
    echo ""
    
    echo "💡 Tips for verification:"
    echo "  - Check that objects split across tile boundaries are handled correctly"
    echo "  - Verify that small objects aren't lost due to minimum coverage settings"
    echo "  - Look for consistency in annotation quality between versions"
    echo ""
    
    echo "Open the images in ${COMPARISON_OUTPUT}/ to verify your tiling worked correctly!"
    
else
    echo ""
    echo "❌ Error: Comparison generation failed"
    echo "Please check the error messages above"
    exit 1
fi