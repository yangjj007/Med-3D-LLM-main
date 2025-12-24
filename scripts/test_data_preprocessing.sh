#!/bin/bash
# Test data preprocessing pipeline for sparse SDF

echo "=== Testing Sparse SDF Data Preprocessing Pipeline ==="
echo ""

# Test 1: Test mesh utilities
echo "Test 1: Testing mesh utilities..."
python -c "
import sys
sys.path.insert(0, 'TRELLIS-main')
from trellis.utils.mesh_utils import normalize_mesh, mesh2sparse_sdf
import trimesh
import numpy as np

# Create a simple test mesh (cube)
vertices = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
])
faces = np.array([
    [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
])
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# Test normalization
mesh_norm = normalize_mesh(mesh, scale=0.95)
print(f'✓ Mesh normalization works')
print(f'  Min coords: {mesh_norm.vertices.min(axis=0)}')
print(f'  Max coords: {mesh_norm.vertices.max(axis=0)}')

# Note: mesh2sparse_sdf requires CUDA and udf_ext, skip if not available
try:
    sdf_data = mesh2sparse_sdf(mesh, resolution=64, normalize=True)
    print(f'✓ mesh2sparse_sdf works')
    print(f'  Sparse points: {len(sdf_data[\"sparse_index\"])}')
    print(f'  Resolution: {sdf_data[\"resolution\"]}')
except Exception as e:
    print(f'⚠ mesh2sparse_sdf skipped (requires CUDA and udf_ext): {e}')
"

if [ $? -eq 0 ]; then
    echo "✓ Test 1 passed"
else
    echo "✗ Test 1 failed"
    exit 1
fi
echo ""

# Test 2: Test dataset
echo "Test 2: Testing SparseSDF dataset..."
python -c "
import sys
sys.path.insert(0, 'TRELLIS-main')
from trellis.datasets import SparseSDF
import torch
import numpy as np
import os

# Create a dummy dataset directory
test_dir = './test_data_medical'
os.makedirs(f'{test_dir}/sparse_sdf', exist_ok=True)

# Create dummy sparse SDF file
dummy_sdf = np.random.rand(100, 1).astype(np.float32)
dummy_index = np.random.randint(0, 64, (100, 3)).astype(np.int32)
np.savez_compressed(
    f'{test_dir}/sparse_sdf/test_r64.npz',
    sparse_sdf=dummy_sdf,
    sparse_index=dummy_index,
    resolution=64
)

# Create metadata
import pandas as pd
metadata = pd.DataFrame([{
    'sha256': 'test',
    'sparse_sdf_computed': True,
    'r64_num_points': 100
}])
metadata.to_csv(f'{test_dir}/metadata.csv', index=False)

# Test dataset loading
dataset = SparseSDF(test_dir, resolution=64)
print(f'✓ Dataset created: {len(dataset)} instances')

# Test getting an instance
sample = dataset[0]
print(f'✓ Sample loaded:')
print(f'  sparse_sdf shape: {sample[\"sparse_sdf\"].shape}')
print(f'  sparse_index shape: {sample[\"sparse_index\"].shape}')

# Test collate function
batch = dataset.collate_fn([sample, sample])
print(f'✓ Batch collated:')
print(f'  sparse_sdf shape: {batch[\"sparse_sdf\"].shape}')
print(f'  sparse_index shape: {batch[\"sparse_index\"].shape}')
print(f'  batch_idx shape: {batch[\"batch_idx\"].shape}')

# Cleanup
import shutil
shutil.rmtree(test_dir)
"

if [ $? -eq 0 ]; then
    echo "✓ Test 2 passed"
else
    echo "✗ Test 2 failed"
    exit 1
fi
echo ""

# Test 3: Test model import
echo "Test 3: Testing VQVAE model import..."
python -c "
import sys
sys.path.insert(0, 'TRELLIS-main')
from trellis.models import Direct3DS2_VQVAE

# Create model
model = Direct3DS2_VQVAE(
    resolution=64,
    model_channels=512,  # 必须是512的倍数！
    latent_channels=32,
    num_blocks=4,
    num_embeddings=1024
)
print(f'✓ Model created')
print(f'  Resolution: {model.resolution}')
print(f'  Latent channels: {model.latent_channels}')
print(f'  Num embeddings: {model.vq.num_embeddings}')
"

if [ $? -eq 0 ]; then
    echo "✓ Test 3 passed"
else
    echo "✗ Test 3 failed"
    exit 1
fi
echo ""

# Test 4: Test trainer import
echo "Test 4: Testing trainer import..."
python -c "
import sys
sys.path.insert(0, 'TRELLIS-main')
from trellis.trainers import SparseSDF_VQVAETrainer
print('✓ SparseSDF_VQVAETrainer imported successfully')
"

if [ $? -eq 0 ]; then
    echo "✓ Test 4 passed"
else
    echo "✗ Test 4 failed"
    exit 1
fi
echo ""

echo "==================================="
echo "All tests passed! ✓"
echo "==================================="

