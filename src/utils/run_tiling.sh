#!/bin/bash

# Dataset Tiling Script
# Tiles images to 512x512 while preserving COCO bounding box annotations

echo "=========================================="
echo "     Dataset Tiling Application"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
INPUT_DIR="${SCRIPT_DIR}/dataset"
OUTPUT_DIR="${SCRIPT_DIR}/output"
TILE_SIZE=512
OVERLAP=0
MIN_COVERAGE=0.3

echo "Configuration:"
echo "  Input directory: ${INPUT_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Tile size: ${TILE_SIZE}x${TILE_SIZE}"
echo "  Overlap: ${OVERLAP} pixels"
echo "  Minimum object coverage: ${MIN_COVERAGE}"
echo ""

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory does not exist: ${INPUT_DIR}"
    echo "Please ensure the dataset folder is present in the project root."
    exit 1
fi

# Check if annotations file exists
ANNOTATIONS_FILE="${INPUT_DIR}/train/_annotations.coco.json"
if [ ! -f "${ANNOTATIONS_FILE}" ]; then
    echo "Error: COCO annotations file not found: ${ANNOTATIONS_FILE}"
    echo "Please ensure the dataset has the correct Roboflow structure."
    exit 1
fi

# Activate conda environment
echo "Activating conda environment 'create-dataset'..."
if command -v conda &> /dev/null; then
    # Initialize conda for bash (if not already done)
    eval "$(conda shell.bash hook)"
    
    # Check if environment exists, create if it doesn't
    if ! conda env list | grep -q "create-dataset"; then
        echo "Creating conda environment 'create-dataset'..."
        conda env create -f environment.yml
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create conda environment"
            echo "Please create it manually with: conda env create -f environment.yml"
            exit 1
        fi
    fi
    
    # Activate the environment
    conda activate create-dataset
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate conda environment 'create-dataset'"
        echo "Please activate it manually with: conda activate create-dataset"
        exit 1
    fi
    
    echo "Using conda environment: $(conda info --envs | grep '*')"
else
    echo "Warning: conda not found, using system Python"
    # Fallback to pip installation
    echo "Checking dependencies..."
    if ! python3 -c "import PIL, numpy" 2>/dev/null; then
        echo "Installing required dependencies..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install dependencies. Please install manually:"
            echo "  pip install -r requirements.txt"
            exit 1
        fi
    fi
fi

# Verify dependencies are available
echo "Verifying dependencies..."
if ! python3 -c "import PIL, numpy" 2>/dev/null; then
    echo "Error: Required dependencies not available. Please check your environment."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

echo "Starting dataset tiling process..."
echo ""

# Run the tiling application
python3 "${SCRIPT_DIR}/app.py" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --tile-size ${TILE_SIZE} ${TILE_SIZE} \
    --overlap ${OVERLAP} \
    --min-coverage ${MIN_COVERAGE} \
    --validate

# Check if the process was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "     Tiling Process Completed Successfully!"
    echo "=========================================="
    echo ""
    echo "Results:"
    echo "  Output directory: ${OUTPUT_DIR}"
    
    # Count generated files
    if [ -d "${OUTPUT_DIR}/train" ]; then
        TILE_COUNT=$(find "${OUTPUT_DIR}/train" -name "*.jpg" | wc -l)
        echo "  Generated tiles: ${TILE_COUNT}"
        
        if [ -f "${OUTPUT_DIR}/train/_annotations.coco.json" ]; then
            echo "  Annotations: Updated and saved"
        fi
    fi
    
    echo ""
    echo "You can now use the tiled dataset in: ${OUTPUT_DIR}"
    
else
    echo ""
    echo "=========================================="
    echo "     Error: Tiling Process Failed"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above and ensure:"
    echo "  1. The input dataset has the correct structure"
    echo "  2. All dependencies are installed"
    echo "  3. You have write permissions to the output directory"
    exit 1
fi