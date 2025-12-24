#!/bin/bash
# Train sparse SDF VQVAE
# Usage: bash scripts/train_vqvae.sh <resolution> <data_dir> <output_dir> [num_gpus]

if [ $# -lt 3 ]; then
    echo "Usage: bash scripts/train_vqvae.sh <resolution> <data_dir> <output_dir> [num_gpus]"
    echo ""
    echo "Arguments:"
    echo "  resolution  : Resolution to train (64, 512, or 1024)"
    echo "  data_dir    : Directory containing preprocessed data"
    echo "  output_dir  : Output directory for checkpoints and logs"
    echo "  num_gpus    : Number of GPUs to use (default: all available)"
    echo ""
    echo "Examples:"
    echo "  # Train at 512 resolution on single GPU"
    echo "  bash scripts/train_vqvae.sh 512 ./data/medical ./outputs/vqvae_512 1"
    echo ""
    echo "  # Train at 1024 resolution on 4 GPUs"
    echo "  bash scripts/train_vqvae.sh 1024 ./data/medical ./outputs/vqvae_1024 4"
    exit 1
fi

RESOLUTION=$1
DATA_DIR=$2
OUTPUT_DIR=$3
NUM_GPUS=${4:--1}

# Validate resolution
if [ "$RESOLUTION" != "64" ] && [ "$RESOLUTION" != "512" ] && [ "$RESOLUTION" != "1024" ]; then
    echo "Error: Resolution must be 64, 512, or 1024"
    exit 1
fi

CONFIG="configs/vae/sparse_sdf_vqvae_${RESOLUTION}.json"

echo "=== Training Sparse SDF VQVAE ==="
echo "Resolution: $RESOLUTION"
echo "Config: $CONFIG"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Num GPUs: $NUM_GPUS"
echo ""

# Check if config exists
if [ ! -f "TRELLIS-main/$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Check if data dir exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Check if metadata exists
if [ ! -f "$DATA_DIR/metadata.csv" ]; then
    echo "Error: metadata.csv not found in $DATA_DIR"
    echo "Please run prepare_medical_dataset.sh first"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start training
cd TRELLIS-main

echo "Starting training..."
echo ""

python train.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --num_gpus $NUM_GPUS \
    --auto_retry 3

TRAIN_STATUS=$?

cd ..

if [ $TRAIN_STATUS -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "Training completed successfully! ✓"
    echo "==================================="
    echo ""
    echo "Checkpoints saved to: $OUTPUT_DIR/ckpts"
    echo "Logs saved to: $OUTPUT_DIR/tb_logs"
    echo "Samples saved to: $OUTPUT_DIR/samples"
else
    echo ""
    echo "==================================="
    echo "Training failed with error code $TRAIN_STATUS"
    echo "==================================="
    exit $TRAIN_STATUS
fi

