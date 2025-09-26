#!/bin/bash

# High Resolution Dataset Comparison Script
# Creates high-quality side-by-side visualizations for detailed analysis

echo "=========================================="
echo "  High Resolution Dataset Comparison"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ORIGINAL_DATASET="${SCRIPT_DIR}/dataset"
TILED_DATASET="${SCRIPT_DIR}/output"
COMPARISON_OUTPUT="${SCRIPT_DIR}/high_res_comparisons"
SAMPLES=15

echo "Configuration:"
echo "  Original dataset: ${ORIGINAL_DATASET}"
echo "  Tiled dataset: ${TILED_DATASET}"
echo "  Output directory: ${COMPARISON_OUTPUT}"
echo "  High resolution samples: ${SAMPLES}"
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

if [ ! -d "${TILED_DATASET}" ]; then
    echo "❌ Error: Tiled dataset not found: ${TILED_DATASET}"
    echo "   Please run ./run_tiling.sh first to generate tiled dataset"
    exit 1
fi

# Create output directory
mkdir -p "${COMPARISON_OUTPUT}"

echo "Creating HIGH RESOLUTION dataset comparisons..."
echo "This may take a few minutes due to image processing..."
echo ""

# Run the comparison tool with high-res settings
python3 "${SCRIPT_DIR}/compare_datasets.py" \
    --original "${ORIGINAL_DATASET}" \
    --tiled "${TILED_DATASET}" \
    --output "${COMPARISON_OUTPUT}" \
    --samples ${SAMPLES} \
    --overview

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  HIGH RESOLUTION Comparison Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved in: ${COMPARISON_OUTPUT}/"
    echo ""
    
    # Count generated files
    COMPARISON_COUNT=$(find "${COMPARISON_OUTPUT}" -name "comparison_*.jpg" | wc -l)
    echo "Generated HIGH RESOLUTION files:"
    echo "  📊 dataset_comparison_overview.jpg - High-res grid overview"
    echo "  📷 ${COMPARISON_COUNT} individual high-resolution comparison images"
    echo ""
    
    echo "🔍 HIGH RESOLUTION FEATURES:"
    echo "  ✅ 1200px max width for better detail visibility"
    echo "  ✅ Dynamic bounding box line thickness"
    echo "  ✅ Scaled font sizes based on image size"
    echo "  ✅ Quality=95 JPEG compression for crisp details"
    echo "  ✅ Tiled images scaled up for better comparison"
    echo ""
    
    echo "🎯 Perfect for detailed verification:"
    echo "  - Zoom in to see bounding box precision"
    echo "  - Check annotation coordinate accuracy"
    echo "  - Verify small object detection"
    echo "  - Compare annotation quality between versions"
    echo ""
    
    echo "📁 Open the images in ${COMPARISON_OUTPUT}/ for detailed analysis!"
    
    # Show file sizes for transparency
    echo ""
    echo "📊 File size info:"
    TOTAL_SIZE=$(du -sh "${COMPARISON_OUTPUT}" | cut -f1)
    echo "  Total size: ${TOTAL_SIZE}"
    echo "  (Higher quality = larger files for better analysis)"
    
else
    echo ""
    echo "❌ Error: High resolution comparison generation failed"
    exit 1
fi