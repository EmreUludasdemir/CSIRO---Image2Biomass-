#!/bin/bash
# Training script for CSIRO Image2Biomass Competition

echo "=========================================="
echo "CSIRO Image2Biomass - Training Pipeline"
echo "=========================================="
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if data exists
if [ ! -f "data/train.csv" ]; then
    echo "Error: data/train.csv not found!"
    echo "Please download competition data first."
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p models/checkpoints
mkdir -p logs
mkdir -p submissions

# Set CUDA devices (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# Run training
echo ""
echo "Starting training..."
echo ""

python src/train.py \
    --config configs/config.yaml

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
