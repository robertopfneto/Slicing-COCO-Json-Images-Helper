#!/bin/bash

# Advanced Dataset Tiling Script with Custom Options
# Provides more configuration options for power users

echo "=========================================="
echo "  Advanced Dataset Tiling Application"
echo "=========================================="
echo ""

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
INPUT_DIR="${SCRIPT_DIR}/dataset"
OUTPUT_DIR="${SCRIPT_DIR}/output"
TILE_SIZE=512
OVERLAP=0
MIN_COVERAGE=0.3
VALIDATE=true

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --input DIR        Input dataset directory (default: ./dataset)"
    echo "  -o, --output DIR       Output directory (default: ./output)"
    echo "  -s, --size SIZE        Tile size in pixels (default: 512)"
    echo "  -v, --overlap PIXELS   Overlap between tiles (default: 0)"
    echo "  -c, --coverage RATIO   Minimum object coverage ratio (default: 0.3)"
    echo "  --no-validate          Skip output validation"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default settings"
    echo "  $0 -s 256 -v 50                     # 256x256 tiles with 50px overlap"
    echo "  $0 -i /path/to/dataset -o /path/to/output  # Custom paths"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--size)
            TILE_SIZE="$2"
            shift 2
            ;;
        -v|--overlap)
            OVERLAP="$2"
            shift 2
            ;;
        -c|--coverage)
            MIN_COVERAGE="$2"
            shift 2
            ;;
        --no-validate)
            VALIDATE=false
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Input directory: ${INPUT_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Tile size: ${TILE_SIZE}x${TILE_SIZE}"
echo "  Overlap: ${OVERLAP} pixels"
echo "  Minimum object coverage: ${MIN_COVERAGE}"
echo "  Validation: ${VALIDATE}"
echo ""

# Validation
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory does not exist: ${INPUT_DIR}"
    exit 1
fi

ANNOTATIONS_FILE="${INPUT_DIR}/train/_annotations.coco.json"
if [ ! -f "${ANNOTATIONS_FILE}" ]; then
    echo "Error: COCO annotations file not found: ${ANNOTATIONS_FILE}"
    exit 1
fi

# Validate numeric parameters
if ! [[ "${TILE_SIZE}" =~ ^[0-9]+$ ]] || [ "${TILE_SIZE}" -lt 64 ]; then
    echo "Error: Tile size must be a positive integer >= 64"
    exit 1
fi

if ! [[ "${OVERLAP}" =~ ^[0-9]+$ ]] || [ "${OVERLAP}" -ge "${TILE_SIZE}" ]; then
    echo "Error: Overlap must be a non-negative integer < tile size"
    exit 1
fi

if ! [[ "${MIN_COVERAGE}" =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "${MIN_COVERAGE} < 0 || ${MIN_COVERAGE} > 1" | bc -l) )); then
    echo "Error: Minimum coverage must be a number between 0 and 1"
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
            exit 1
        fi
    fi
    
    # Activate the environment
    conda activate create-dataset
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate conda environment 'create-dataset'"
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
            echo "Error: Failed to install dependencies"
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

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Starting dataset tiling process..."
echo ""

# Build command arguments
ARGS=(
    --input "${INPUT_DIR}"
    --output "${OUTPUT_DIR}"
    --tile-size ${TILE_SIZE} ${TILE_SIZE}
    --overlap ${OVERLAP}
    --min-coverage ${MIN_COVERAGE}
)

if [ "${VALIDATE}" = true ]; then
    ARGS+=(--validate)
fi

# Run the application
python3 "${SCRIPT_DIR}/app.py" "${ARGS[@]}"

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "     Process Completed Successfully!"
    echo "=========================================="
    echo ""
    
    if [ -d "${OUTPUT_DIR}/train" ]; then
        ORIGINAL_COUNT=$(find "${INPUT_DIR}/train" -name "*.jpg" 2>/dev/null | wc -l)
        TILE_COUNT=$(find "${OUTPUT_DIR}/train" -name "*.jpg" 2>/dev/null | wc -l)
        
        echo "Statistics:"
        echo "  Original images: ${ORIGINAL_COUNT}"
        echo "  Generated tiles: ${TILE_COUNT}"
        
        if [ "${TILE_COUNT}" -gt 0 ] && [ "${ORIGINAL_COUNT}" -gt 0 ]; then
            RATIO=$(echo "scale=1; ${TILE_COUNT} / ${ORIGINAL_COUNT}" | bc -l)
            echo "  Tiles per image: ${RATIO}"
        fi
    fi
    
else
    echo ""
    echo "Error: Process failed. Check the error messages above."
    exit 1
fi