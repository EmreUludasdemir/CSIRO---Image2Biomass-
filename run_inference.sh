#!/bin/bash
# Inference script for CSIRO Image2Biomass Competition

echo "=========================================="
echo "CSIRO Image2Biomass - Inference Pipeline"
echo "=========================================="
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if test data exists
if [ ! -f "data/test.csv" ]; then
    echo "Error: data/test.csv not found!"
    echo "Please download competition data first."
    exit 1
fi

# Check if models exist
if [ ! -d "models/checkpoints" ] || [ -z "$(ls -A models/checkpoints)" ]; then
    echo "Error: No trained models found in models/checkpoints/"
    echo "Please train models first using run_training.sh"
    exit 1
fi

# Create submission directory
mkdir -p submissions

# Set CUDA devices (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# Run inference
echo ""
echo "Starting inference..."
echo ""

python src/inference.py \
    --config configs/config.yaml \
    --test-csv data/test.csv \
    --output submissions/submission_$(date +%Y%m%d_%H%M%S).csv

echo ""
echo "=========================================="
echo "Inference complete!"
echo "Submission saved to submissions/"
echo "=========================================="
