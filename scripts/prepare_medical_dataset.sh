#!/bin/bash
# Prepare medical dataset for training
# Usage: bash scripts/prepare_medical_dataset.sh <data_root> <output_dir>

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/prepare_medical_dataset.sh <data_root> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  data_root   : Root directory containing medical mesh/voxel files"
    echo "  output_dir  : Output directory for processed data"
    echo ""
    echo "Example:"
    echo "  bash scripts/prepare_medical_dataset.sh /data/medical_meshes ./data/medical"
    exit 1
fi

DATA_ROOT=$1
OUTPUT_DIR=$2
RESOLUTIONS="64,512,1024"

echo "=== Preparing Medical Dataset ==="
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Resolutions: $RESOLUTIONS"
echo ""

# Step 1: Build metadata
echo "Step 1: Building metadata..."
cd TRELLIS-main
python dataset_toolkits/datasets/MedicalData.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "✗ Failed to build metadata"
    exit 1
fi
echo "✓ Metadata built"
echo ""

# Step 2: Render meshes (if needed)
# Note: This step is optional for sparse SDF training
# Uncomment if you need rendered views for visualization
# echo "Step 2: Rendering meshes..."
# python dataset_toolkits/render.py MedicalData \
#     --output_dir "$OUTPUT_DIR" \
#     --data_root "$DATA_ROOT" \
#     --num_views 50

# Step 3: Compute sparse SDF
echo "Step 2: Computing sparse SDF at multiple resolutions..."
python dataset_toolkits/compute_sparse_sdf.py \
    --output_dir "$OUTPUT_DIR" \
    --resolutions "$RESOLUTIONS" \
    --input_type mesh \
    --max_workers 8

if [ $? -ne 0 ]; then
    echo "✗ Failed to compute sparse SDF"
    exit 1
fi
echo "✓ Sparse SDF computed"
echo ""

# Step 4: Merge results (if using distributed processing)
echo "Step 3: Merging results..."
python -c "
import pandas as pd
import glob
import os

output_dir = '$OUTPUT_DIR'
csv_files = glob.glob(os.path.join(output_dir, 'sparse_sdf_computed_*.csv'))

if len(csv_files) == 0:
    print('No sparse_sdf_computed CSV files found')
    exit(1)

# Merge all CSV files
dfs = [pd.read_csv(f) for f in csv_files]
merged = pd.concat(dfs, ignore_index=True)

# Load original metadata
metadata = pd.read_csv(os.path.join(output_dir, 'metadata.csv'))

# Merge with original metadata
metadata = metadata.merge(merged, on='sha256', how='left')

# Save updated metadata
metadata.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)

print(f'✓ Merged {len(csv_files)} result files')
print(f'Total instances: {len(metadata)}')
print(f'Sparse SDF computed: {metadata[\"sparse_sdf_computed\"].sum()}')
"

if [ $? -ne 0 ]; then
    echo "✗ Failed to merge results"
    exit 1
fi
echo "✓ Results merged"
echo ""

cd ..

echo "==================================="
echo "Dataset preparation complete! ✓"
echo ""
echo "Dataset statistics:"
python -c "
import pandas as pd
metadata = pd.read_csv('$OUTPUT_DIR/metadata.csv')
print(f'  Total instances: {len(metadata)}')
print(f'  Sparse SDF computed: {metadata[\"sparse_sdf_computed\"].sum()}')
for res in ['64', '512', '1024']:
    col = f'r{res}_num_points'
    if col in metadata.columns:
        valid = metadata[col].notna()
        if valid.sum() > 0:
            avg = metadata.loc[valid, col].mean()
            print(f'  Resolution {res}: {valid.sum()} instances, avg {avg:.0f} points')
"
echo ""
echo "Next steps:"
echo "  1. Start training with:"
echo "     cd TRELLIS-main"
echo "     python train.py \\"
echo "         --config configs/vae/sparse_sdf_vqvae_512.json \\"
echo "         --output_dir ./outputs/vqvae_512 \\"
echo "         --data_dir $OUTPUT_DIR"
echo ""

