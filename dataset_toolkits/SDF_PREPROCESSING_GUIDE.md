# SDF Preprocessing Guide

This guide explains how to use `sdf_voxelize.py` to preprocess mesh files into sparse SDF format for VQVAE training.

## Overview

The `sdf_voxelize.py` script converts GLB mesh files to sparse Signed Distance Field (SDF) representations at configurable resolutions. It supports two data formats:

1. **TRELLIS-500K format**: ObjaverseXL data with object-paths.json
2. **Custom labeled format**: Your own 3D models with metadata.csv

## Prerequisites

- CUDA-capable GPU (required for SDF computation)
- PyTorch with CUDA support
- Trimesh library
- TRELLIS utils installed

## Usage Examples

### 1. Process TRELLIS-500K Data

```bash
python dataset_toolkits/sdf_voxelize.py \
    --format trellis500k \
    --input_dir ./TRELLIS-500K/raw/hf-objaverse-v1 \
    --output_dir ./train_sdf_dataset \
    --resolutions 64,512 \
    --max_workers 4
```

**With aesthetic score filtering:**

```bash
python dataset_toolkits/sdf_voxelize.py \
    --format trellis500k \
    --input_dir ./TRELLIS-500K/raw/hf-objaverse-v1 \
    --output_dir ./train_sdf_dataset \
    --resolutions 512 \
    --filter_aesthetic_score 6.0 \
    --max_workers 4
```

### 2. Process Custom Labeled Data

```bash
python dataset_toolkits/sdf_voxelize.py \
    --format custom \
    --input_dir ./my_3d_dataset \
    --output_dir ./train_sdf_dataset \
    --resolutions 512 \
    --max_workers 4
```

## Command-Line Arguments

### Required Arguments

- `--format`: Dataset format type
  - `trellis500k`: TRELLIS-500K format with object-paths.json
  - `custom`: Custom format with metadata.csv
  
- `--input_dir`: Input directory containing mesh files and metadata

- `--output_dir`: Output directory for SDF files

### Optional Arguments

- `--resolutions`: Comma-separated list of resolutions (default: `512`)
  - Example: `64,512,1024` for multiple resolutions
  - Higher resolutions = more detail but slower and larger files
  
- `--threshold_factor`: UDF threshold factor for sparse extraction (default: `4.0`)
  - Controls how many points are kept near the surface
  - Higher values = more points, larger files
  
- `--max_workers`: Maximum number of parallel workers
  - Default: CPU count (max 4)
  - Adjust based on available GPU memory
  
- `--no_skip`: Reprocess all files even if they already exist
  - By default, already processed files are skipped
  
- `--filter_aesthetic_score`: Filter by aesthetic score (TRELLIS-500K only)
  - Only process objects with aesthetic_score >= this value

## Input Data Formats

### TRELLIS-500K Format

Expected directory structure:
```
TRELLIS-500K/
├── raw/
│   └── hf-objaverse-v1/
│       ├── glbs/
│       │   └── 000-023/
│       │       └── {sha256}.glb
│       └── object-paths.json
├── metadata.csv (optional, for captions)
└── ...
```

The `object-paths.json` file maps sha256 IDs to GLB file paths:
```json
{
  "8476c4170df24cf5bbe6967222d1a42d": "glbs/000-023/8476c4170df24cf5bbe6967222d1a42d.glb",
  ...
}
```

### Custom Labeled Format

Expected directory structure:
```
my_dataset/
├── metadata.csv
├── {sha256_1}.glb
├── {sha256_2}.glb
└── ...
```

The `metadata.csv` should have these columns:
- `sha256`: Unique identifier
- `glb_file`: GLB filename (can be just filename or relative path)
- `overall_label`: Overall description (optional)
- `materials_captions`: Part-level descriptions (optional)

## Output Format

The script generates:

### 1. SDF Files (NPZ format)

```
output_dir/
├── {sha256}_r64.npz
├── {sha256}_r512.npz
├── {sha256}_r1024.npz
└── ...
```

Each NPZ file contains:
- `sparse_sdf`: SDF values [N, 1] (float32)
- `sparse_index`: 3D coordinates [N, 3] (int32)
- `resolution`: Grid resolution (int)

### 2. Metadata CSV

Located at `{parent_of_output_dir}/metadata.csv`:
- Original metadata columns preserved
- Added columns:
  - `sdf_computed`: Boolean indicating success
  - `r{resolution}_num_points`: Number of points for each resolution
  - `r{resolution}_error`: Error message if processing failed (optional)

## Performance Tips

### GPU Memory Management

- **Multiple workers**: More workers = more GPU memory usage
  - Start with `--max_workers 2` and increase if you have enough VRAM
  - Each worker needs ~2-4GB VRAM depending on resolution
  
- **Resolution**: Higher resolution requires more memory
  - Resolution 64: Very fast, low memory
  - Resolution 512: Moderate (recommended)
  - Resolution 1024: Slow, high memory

### Processing Speed

Typical processing times (with RTX 3090):
- Resolution 64: ~0.5-1 second per mesh
- Resolution 512: ~2-5 seconds per mesh
- Resolution 1024: ~10-20 seconds per mesh

### Resuming Interrupted Processing

The script automatically skips already processed files (unless `--no_skip` is used). If processing is interrupted:

1. Simply re-run the same command
2. The script will detect existing NPZ files and skip them
3. Only unprocessed files will be processed

## Troubleshooting

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `--max_workers` (try 1 or 2)
- Lower `--resolutions` (use 256 or 512 instead of 1024)
- Process in smaller batches

### Missing Mesh Files

**Problem**: "mesh file not found" errors

**Solutions**:
- Check that GLB files exist at the specified paths
- For TRELLIS-500K: Some files may not be downloaded yet
- Files with errors will be skipped, metadata will show which ones failed

### Empty or Invalid Meshes

**Problem**: "empty mesh" or "invalid mesh format" errors

**Solutions**:
- These files will be skipped automatically
- Check the metadata CSV output to see which files failed
- Failed files will have `sdf_computed=False` and an error message

### Import Errors

**Problem**: `ImportError: No module named 'trellis'`

**Solutions**:
```bash
# Make sure you're in the project root directory
cd /path/to/Med-3D-LLM-main

# Install TRELLIS dependencies
pip install -r requirements.txt

# Compile CUDA extensions
cd third_party/voxelize
pip install -e . --no-build-isolation
cd ../..
```

## Example Workflow

### Complete TRELLIS-500K Processing Pipeline

```bash
# 1. Download TRELLIS-500K metadata
git clone https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K

# 2. Download ObjaverseXL metadata
python dataset_toolkits/build_metadata.py ObjaverseXL \
    --source sketchfab \
    --output_dir TRELLIS-500K

# 3. Download GLB files
python dataset_toolkits/download.py ObjaverseXL \
    --output_dir TRELLIS-500K \
    --rank 0 --world_size 1

# 4. Convert to SDF format (this script!)
python dataset_toolkits/sdf_voxelize.py \
    --format trellis500k \
    --input_dir ./TRELLIS-500K/raw/hf-objaverse-v1 \
    --output_dir ./train_sdf_dataset \
    --resolutions 64,512 \
    --filter_aesthetic_score 6.0 \
    --max_workers 4

# 5. Check results
ls -lh train_sdf_dataset/*.npz | wc -l
head metadata.csv
```

## Integration with Training Pipeline

After preprocessing, the SDF files can be used with the VQVAE training pipeline:

1. **Location**: Place processed SDF files in `train_sdf_dataset/`
2. **Metadata**: Update your training config to point to the generated `metadata.csv`
3. **Dataset class**: Use the SDF dataset loader that reads NPZ files

See `CT_VQVAE_TRAINING_README.md` for more details on training setup.

## Questions?

If you encounter issues not covered here:
1. Check the error messages in the console output
2. Review the `metadata.csv` to see which files failed
3. Try processing a small subset first (10-20 files) to diagnose issues

