# Setup Instructions

## Quick Start with Conda (Recommended)

The application automatically creates and manages the `create-dataset` conda environment.

### Option 1: Automatic Setup (Easiest)
```bash
# Just run the script - it will handle everything
./run_tiling.sh
```

### Option 2: Manual Conda Environment Setup
```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate create-dataset

# Run the application
python app.py --input ./dataset --output ./output
```

## Alternative Setup with pip

If you don't have conda, you can use pip:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py --input ./dataset --output ./output
```

## Environment Details

The `create-dataset` conda environment includes:
- Python 3.9
- Pillow >= 9.0.0 (for image processing)
- NumPy >= 1.21.0 (for numerical operations)

## Troubleshooting

### Error: "CocoImage got unexpected keyword argument 'extra'"
This error occurred with Roboflow-exported COCO datasets that include extra metadata fields. **This has been fixed** in the current version by updating the COCO models to handle additional fields gracefully.

### Conda Environment Issues
If you encounter conda environment issues:
```bash
# Remove existing environment
conda env remove -n create-dataset

# Recreate from environment file
conda env create -f environment.yml
```

### Permission Issues
Make sure the scripts are executable:
```bash
chmod +x run_tiling.sh run_tiling_advanced.sh
```