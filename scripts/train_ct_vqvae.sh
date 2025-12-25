#!/bin/bash
# CT VQVAE Training Script
# 
# Usage:
#   bash scripts/train_ct_vqvae.sh <stage> <window_type> <data_dir> <output_dir> [pretrained_vae_path]
#
# Arguments:
#   stage: 1 or 2 (1=freeze VAE, train codebook; 2=joint training)
#   window_type: lung, bone, soft_tissue, or brain
#   data_dir: Path to processed CT dataset
#   output_dir: Output directory for checkpoints and logs
#   pretrained_vae_path: (Optional) Path to pretrained VAE weights (for stage 1 only)
#
# Examples:
#   # Stage 1 without pretrained weights
#   bash scripts/train_ct_vqvae.sh 1 lung ./processed_dataset ./outputs/ct_vqvae_lung_stage1
#
#   # Stage 1 with pretrained weights
#   bash scripts/train_ct_vqvae.sh 1 lung ./processed_dataset ./outputs/ct_vqvae_lung_stage1 ./pretrained_weights/direct3d_vae.pth
#
#   # Stage 2 (load from stage 1)
#   bash scripts/train_ct_vqvae.sh 2 lung ./processed_dataset ./outputs/ct_vqvae_lung_stage2 ./outputs/ct_vqvae_lung_stage1

set -e

# Parse arguments
STAGE=$1
WINDOW_TYPE=$2
DATA_DIR=$3
OUTPUT_DIR=$4
PRETRAINED_VAE_PATH=${5:-"null"}

# Validate arguments
if [ -z "$STAGE" ] || [ -z "$WINDOW_TYPE" ] || [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: bash scripts/train_ct_vqvae.sh <stage> <window_type> <data_dir> <output_dir> [pretrained_vae_path]"
    exit 1
fi

if [ "$STAGE" != "1" ] && [ "$STAGE" != "2" ]; then
    echo "Error: Stage must be 1 or 2"
    exit 1
fi

if [ "$WINDOW_TYPE" != "lung" ] && [ "$WINDOW_TYPE" != "bone" ] && [ "$WINDOW_TYPE" != "soft_tissue" ] && [ "$WINDOW_TYPE" != "brain" ]; then
    echo "Error: Window type must be one of: lung, bone, soft_tissue, brain"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Set configuration file
CONFIG_FILE="configs/vae/ct_vqvae_stage${STAGE}.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "CT VQVAE Training"
echo "=========================================="
echo "Stage:           $STAGE"
echo "Window Type:     $WINDOW_TYPE"
echo "Data Dir:        $DATA_DIR"
echo "Output Dir:      $OUTPUT_DIR"
echo "Config File:     $CONFIG_FILE"

# Create temporary config with updated parameters
TMP_CONFIG="/tmp/ct_vqvae_stage${STAGE}_${WINDOW_TYPE}_$$.json"

# Update config file with window_type
python -c "
import json
import sys

with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

# Update window type
config['dataset']['args']['window_type'] = '$WINDOW_TYPE'

# For stage 1, set pretrained VAE path
if '$STAGE' == '1' and '$PRETRAINED_VAE_PATH' != 'null':
    config['trainer']['args']['pretrained_vae_path'] = '$PRETRAINED_VAE_PATH'
    print('Pretrained VAE:  $PRETRAINED_VAE_PATH')

with open('$TMP_CONFIG', 'w') as f:
    json.dump(config, f, indent=4)
"

echo "=========================================="
echo ""

# Training command
if [ "$STAGE" == "1" ]; then
    # Stage 1: Train from scratch or with pretrained weights
    python train.py \
        --config "$TMP_CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --data_dir "$DATA_DIR" \
        --num_gpus 1
else
    # Stage 2: Load from stage 1
    if [ "$PRETRAINED_VAE_PATH" == "null" ]; then
        echo "Error: For stage 2, you must provide the stage 1 output directory as the 5th argument"
        rm -f "$TMP_CONFIG"
        exit 1
    fi
    
    LOAD_DIR="$PRETRAINED_VAE_PATH"
    
    if [ ! -d "$LOAD_DIR" ]; then
        echo "Error: Stage 1 output directory not found: $LOAD_DIR"
        rm -f "$TMP_CONFIG"
        exit 1
    fi
    
    echo "Loading from:    $LOAD_DIR"
    echo "=========================================="
    echo ""
    
    python train.py \
        --config "$TMP_CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --load_dir "$LOAD_DIR" \
        --ckpt latest \
        --data_dir "$DATA_DIR" \
        --num_gpus 1
fi

# Cleanup
rm -f "$TMP_CONFIG"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

