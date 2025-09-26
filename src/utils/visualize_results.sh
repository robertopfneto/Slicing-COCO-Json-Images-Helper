#!/bin/bash

# Dataset Visualization Script
# Visualizes both original and tiled datasets for verification

echo "=========================================="
echo "     Dataset Visualization Tool"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ORIGINAL_DATASET="${SCRIPT_DIR}/dataset"
TILED_DATASET="${SCRIPT_DIR}/output"
VIZ_OUTPUT="${SCRIPT_DIR}/visualizations"
SAMPLES=15

echo "Configuration:"
echo "  Original dataset: ${ORIGINAL_DATASET}"
echo "  Tiled dataset: ${TILED_DATASET}"
echo "  Output directory: ${VIZ_OUTPUT}"
echo "  Sample images: ${SAMPLES}"
echo ""

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "create-dataset"; then
        echo "Activating conda environment 'create-dataset'..."
        conda activate create-dataset
    fi
fi

# Check if original dataset exists
if [ ! -d "${ORIGINAL_DATASET}" ]; then
    echo "Error: Original dataset not found: ${ORIGINAL_DATASET}"
    exit 1
fi

if [ ! -f "${ORIGINAL_DATASET}/train/_annotations.coco.json" ]; then
    echo "Error: Original annotations not found: ${ORIGINAL_DATASET}/train/_annotations.coco.json"
    exit 1
fi

# Create output directory
mkdir -p "${VIZ_OUTPUT}"

echo "Starting visualization process..."
echo ""

# Visualize original dataset
echo "1. Visualizing original dataset..."
python3 "${SCRIPT_DIR}/visualize_dataset.py" \
    --input "${ORIGINAL_DATASET}" \
    --output "${VIZ_OUTPUT}/original" \
    --samples ${SAMPLES} \
    --summary

if [ $? -ne 0 ]; then
    echo "Error: Failed to visualize original dataset"
    exit 1
fi

echo ""

# Check if tiled dataset exists and visualize it
if [ -d "${TILED_DATASET}" ] && [ -f "${TILED_DATASET}/train/_annotations.coco.json" ]; then
    echo "2. Visualizing tiled dataset..."
    python3 "${SCRIPT_DIR}/visualize_dataset.py" \
        --input "${TILED_DATASET}" \
        --output "${VIZ_OUTPUT}/tiled" \
        --samples ${SAMPLES} \
        --summary
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "3. Creating comparison analysis..."
        
        # Create a simple comparison report
        python3 -c "
import json
import os

original_path = '${ORIGINAL_DATASET}/train/_annotations.coco.json'
tiled_path = '${TILED_DATASET}/train/_annotations.coco.json'
output_path = '${VIZ_OUTPUT}/comparison_report.txt'

try:
    with open(original_path, 'r') as f:
        original = json.load(f)
    with open(tiled_path, 'r') as f:
        tiled = json.load(f)
    
    with open(output_path, 'w') as f:
        f.write('Dataset Comparison Report\\n')
        f.write('=' * 50 + '\\n\\n')
        f.write(f'Original Dataset:\\n')
        f.write(f'  Images: {len(original[\"images\"])}\\n')
        f.write(f'  Annotations: {len(original[\"annotations\"])}\\n')
        f.write(f'  Categories: {len(original[\"categories\"])}\\n\\n')
        f.write(f'Tiled Dataset:\\n')
        f.write(f'  Images: {len(tiled[\"images\"])}\\n')
        f.write(f'  Annotations: {len(tiled[\"annotations\"])}\\n')
        f.write(f'  Categories: {len(tiled[\"categories\"])}\\n\\n')
        
        if len(original['images']) > 0:
            expansion_ratio = len(tiled['images']) / len(original['images'])
            f.write(f'Expansion Ratio: {expansion_ratio:.2f}x more images\\n')
        
        if len(original['annotations']) > 0:
            annotation_retention = (len(tiled['annotations']) / len(original['annotations'])) * 100
            f.write(f'Annotation Retention: {annotation_retention:.1f}%\\n')
    
    print('✓ Comparison report created')
except Exception as e:
    print(f'Warning: Could not create comparison report: {e}')
"
        
        TILES_CREATED=true
    else
        echo "Error: Failed to visualize tiled dataset"
        TILES_CREATED=false
    fi
else
    echo "2. Tiled dataset not found - run the tiling process first with ./run_tiling.sh"
    TILES_CREATED=false
fi

echo ""
echo "=========================================="
echo "     Visualization Complete!"
echo "=========================================="
echo ""
echo "Results saved in: ${VIZ_OUTPUT}/"
echo ""
echo "Generated files:"
echo "  📁 original/               - Original dataset visualizations"
echo "  📄 original/dataset_summary.txt - Original dataset statistics"

if [ "$TILES_CREATED" = true ]; then
    echo "  📁 tiled/                  - Tiled dataset visualizations"  
    echo "  📄 tiled/dataset_summary.txt - Tiled dataset statistics"
    echo "  📄 comparison_report.txt   - Side-by-side comparison"
fi

echo ""
echo "What to check:"
echo "  1. Open the visualization images to see bounding boxes"
echo "  2. Verify that objects are correctly detected and labeled"
echo "  3. Check that tiled images maintain annotation accuracy"
echo "  4. Review the summary reports for dataset statistics"
echo ""

if [ "$TILES_CREATED" = true ]; then
    echo "💡 Tip: Compare images in 'original/' vs 'tiled/' folders"
    echo "   to verify the tiling process preserved annotations correctly."
else
    echo "💡 Tip: Run ./run_tiling.sh first, then run this script again"
    echo "   to compare original vs tiled datasets."
fi