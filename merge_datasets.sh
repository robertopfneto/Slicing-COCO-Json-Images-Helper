#!/bin/bash

# Dataset Merger Script
# Merges multiple COCO datasets into a single unified dataset

echo "=========================================="
echo "        COCO Dataset Merger"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Default datasets (can be overridden)
DEFAULT_DATASETS=(
    "${SCRIPT_DIR}/output/train"
    "${SCRIPT_DIR}/output_resized/train"
)

OUTPUT_DATASET="${SCRIPT_DIR}/merged_dataset"

# Parse command line arguments
DATASETS=()
CUSTOM_OUTPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASETS+=("$1")
                shift
            done
            ;;
        --output)
            CUSTOM_OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --datasets PATH1 PATH2 ...  List of dataset paths to merge"
            echo "  --output PATH               Output path for merged dataset"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --datasets ./output ./output_resized ./custom_dataset"
            echo "  $0 --output ./my_merged_dataset"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use custom output if provided
if [ -n "$CUSTOM_OUTPUT" ]; then
    OUTPUT_DATASET="$CUSTOM_OUTPUT"
fi

# Use default datasets if none provided
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("${DEFAULT_DATASETS[@]}")
    echo "Using default datasets:"
else
    echo "Using custom datasets:"
fi

# Display configuration
for i in "${!DATASETS[@]}"; do
    echo "  $((i+1)). ${DATASETS[$i]}"
done
echo ""
echo "Output dataset: ${OUTPUT_DATASET}"
echo ""

# Validate minimum dataset count
if [ ${#DATASETS[@]} -lt 2 ]; then
    echo "❌ Error: At least 2 datasets are required for merging"
    echo "   Current datasets: ${#DATASETS[@]}"
    exit 1
fi

# Activate conda environment if available
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "create-dataset"; then
        echo "Activating conda environment 'create-dataset'..."
        conda activate create-dataset
        echo ""
    fi
fi

# Check if datasets exist
echo "🔍 Validating datasets..."
VALID_DATASETS=()
for dataset in "${DATASETS[@]}"; do
    # Handle both full paths and train-only paths
    if [[ "$dataset" == */train ]]; then
        DATASET_PATH="${dataset%/train}"
        TRAIN_PATH="$dataset"
    else
        DATASET_PATH="$dataset"
        TRAIN_PATH="$dataset/train"
    fi
    
    if [ ! -d "$TRAIN_PATH" ]; then
        echo "   ⚠️  Warning: Train directory not found: $TRAIN_PATH"
        continue
    fi
    
    ANNOTATIONS_FILE="$TRAIN_PATH/_annotations.coco.json"
    if [ ! -f "$ANNOTATIONS_FILE" ]; then
        echo "   ⚠️  Warning: Annotations file not found: $ANNOTATIONS_FILE"
        continue
    fi
    
    VALID_DATASETS+=("$DATASET_PATH")
    echo "   ✅ Valid: $DATASET_PATH"
done

echo ""

if [ ${#VALID_DATASETS[@]} -lt 2 ]; then
    echo "❌ Error: Not enough valid datasets found (need at least 2)"
    echo "   Valid datasets: ${#VALID_DATASETS[@]}"
    exit 1
fi

# Remove output directory if it exists
if [ -d "$OUTPUT_DATASET" ]; then
    echo "🗑️  Removing existing output directory..."
    rm -rf "$OUTPUT_DATASET"
fi

echo "🚀 Starting dataset merging..."
echo "This process will:"
echo "  1. Copy all images with dataset prefixes to avoid conflicts"
echo "  2. Remap image and annotation IDs to prevent duplicates"
echo "  3. Consolidate categories across datasets"
echo "  4. Create unified _annotations.coco.json file"
echo ""

# Run the merger
python3 "${SCRIPT_DIR}/merge_datasets.py" \
    --datasets "${VALID_DATASETS[@]}" \
    --output "$OUTPUT_DATASET" \
    --validate

MERGE_EXIT_CODE=$?

echo ""
if [ ${MERGE_EXIT_CODE} -eq 0 ]; then
    echo "=========================================="
    echo "       Dataset Merging Complete!"
    echo "=========================================="
    echo ""
    echo "✅ Datasets merged successfully!"
    echo ""
    echo "📁 Output location: ${OUTPUT_DATASET}/"
    echo ""
    
    # Show some basic statistics
    MERGED_IMAGES=$(find "${OUTPUT_DATASET}/train" -name "*.jpg" | wc -l 2>/dev/null || echo "0")
    
    echo "📊 Merged Dataset Statistics:"
    echo "  📂 Source datasets: ${#VALID_DATASETS[@]}"
    echo "  🖼️  Total images: ${MERGED_IMAGES}"
    
    # Show merged annotations info if available
    ANNOTATIONS_FILE="${OUTPUT_DATASET}/train/_annotations.coco.json"
    if [ -f "$ANNOTATIONS_FILE" ]; then
        ANNOTATION_COUNT=$(python3 -c "
import json
try:
    with open('$ANNOTATIONS_FILE', 'r') as f:
        data = json.load(f)
    print(len(data.get('annotations', [])))
except:
    print('Unknown')
" 2>/dev/null || echo "Unknown")
        echo "  🏷️  Total annotations: ${ANNOTATION_COUNT}"
    fi
    
    echo ""
    echo "🎯 Features applied:"
    echo "  ✅ Filename conflict resolution (dataset prefixes)"
    echo "  ✅ ID remapping for images and annotations"
    echo "  ✅ Category consolidation"
    echo "  ✅ Unified COCO JSON format"
    echo "  ✅ Validation completed"
    echo ""
    
    echo "🔍 Next steps:"
    echo "  1. Check ${OUTPUT_DATASET}/train/ for merged images"
    echo "  2. Verify ${OUTPUT_DATASET}/train/_annotations.coco.json"
    echo "  3. Use the merged dataset for training or analysis"
    echo ""
    
else
    echo "=========================================="
    echo "       Dataset Merging Failed!"
    echo "=========================================="
    echo ""
    echo "❌ Dataset merging failed with exit code: ${MERGE_EXIT_CODE}"
    echo ""
    echo "💡 Troubleshooting:"
    echo "  - Check that all dataset paths exist and contain train/ directories"
    echo "  - Verify _annotations.coco.json files are valid"
    echo "  - Ensure sufficient disk space for copying images"
    echo "  - Check conda environment is properly set up"
    echo ""
    exit 1
fi