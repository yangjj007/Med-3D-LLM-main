"""
SDF Voxelization Preprocessing Script

Convert mesh files (GLB format) to sparse SDF format for VQVAE training.
Supports three data formats:
1. TRELLIS-500K format with ObjaverseXL (with object-paths.json)
2. TRELLIS-500K format with HSSD (file_identifier as direct path)
3. Custom labeled format (with metadata.csv)

Usage:
    # TRELLIS-500K format - ObjaverseXL (multi-GPU: default --gpu_nums -1 uses all GPUs)
    python dataset_toolkits/sdf_voxelize.py \
        --format trellis500k \
        --input_dir ./TRELLIS-500K/raw/hf-objaverse-v1 \
        --output_dir ./train_sdf_dataset \
        --resolutions 64,512 \
        --max_workers 1
    
    # TRELLIS-500K format - HSSD
    python dataset_toolkits/sdf_voxelize.py \
        --format trellis500k \
        --input_dir ./TRELLIS-500K/HSSD/raw/objects \
        --output_dir ./train_sdf_dataset \
        --resolutions 512 \
        --filter_aesthetic_score 6.0 \
        --max_workers 1
    
    # Custom labeled format
    python dataset_toolkits/sdf_voxelize.py \
        --format custom \
        --input_dir ./my_dataset \
        --output_dir ./train_sdf_dataset \
        --resolutions 512 \
        --max_workers 4
"""

import os
import sys
import json
import copy
import glob
import argparse
import subprocess
from pathlib import Path
from functools import partial
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import trimesh
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from easydict import EasyDict as edict

# Add parent directory to path to import trellis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trellis.utils.mesh_utils import mesh2sparse_sdf


def _preflight_udf_ext() -> None:
    """在批量处理前确认当前解释器能加载与仓库代码匹配的 ``udf_ext``。

    硬性要求: compute_valid_sdf（有符号SDF 计算）
    软性建议: compute_sharp_mask（锐利边缘检测）– 缺失时仅打印警告，不中断。
    """
    import trellis.utils.mesh_utils as _mesh_utils

    mod = _mesh_utils._import_udf_ext()
    if not hasattr(mod, "compute_valid_sdf"):
        raise RuntimeError(
            "udf_ext 已导入但缺少 compute_valid_sdf（多为未重新编译的旧扩展）。\n"
            f"  扩展路径: {getattr(mod, '__file__', 'unknown')}\n"
            "  请执行:\n"
            "    cd third_party/voxelize && python -m pip uninstall -y udf_ext && "
            "python -m pip install -v -e . --no-build-isolation\n"
            f"  当前 Python: {sys.executable}"
        )
    if not hasattr(mod, "compute_sharp_mask"):
        print(
            "[_preflight_udf_ext] ⚠️  udf_ext.compute_sharp_mask 未找到。\n"
            "   edge_mask（锐利边缘检测）将退化为全 False。\n"
            "   如需启用，请重新编译:\n"
            "     cd third_party/voxelize && "
            "pip install -v -e . --no-build-isolation",
            flush=True,
        )


def _process_mesh_to_sdf(
    mesh_path: str,
    sha256: str,
    output_dir: str,
    resolutions: List[int] = [512],
    threshold_factor: float = 4.0,
    watertight: bool = False,
    compute_edge_mask: bool = True,
    sharp_grad_dev_thresh: float = 0.5,
) -> Dict[str, Any]:
    """
    Process a single mesh file and convert to sparse SDF at multiple resolutions.
    
    Args:
        mesh_path: Path to the mesh file (GLB format)
        sha256: SHA256 hash identifier for the mesh
        output_dir: Output directory for SDF files
        resolutions: List of resolutions to compute
        threshold_factor: Sparse shell half-width in voxels.  Set >= 4.0 so the
            GT SDF band is wide enough to provide supervision for decoder-predicted
            "extra" voxels beyond the input sparse set.
        watertight: If True, run pymeshfix-based watertight repair after normalize.
        compute_edge_mask: If True, compute and store the sharp-edge boolean mask
            alongside sparse_sdf using the GPU gradient-magnitude kernel.
        sharp_grad_dev_thresh: Gradient-deviation threshold for edge detection;
            voxels with |1 - |∇SDF|| > this are flagged as sharp (default 0.3).
    
    Returns:
        Dictionary with processing results including sha256, status, and point counts
    """
    result = {'sha256': sha256, 'sdf_computed': False}
    
    try:
        # Check if mesh file exists
        if not os.path.exists(mesh_path):
            result['error'] = 'mesh file not found'
            return result
        
        # Load mesh using trimesh
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
        except Exception as e:
            result['error'] = f'mesh loading failed: {str(e)}'
            return result
        
        # Ensure mesh is valid
        if not isinstance(mesh, trimesh.Trimesh):
            # Handle Scene objects - extract first geometry
            if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                result['error'] = 'invalid mesh format'
                return result
        
        # Check if mesh has vertices and faces
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            result['error'] = 'empty mesh'
            return result
        
        # Compute sparse SDF for each resolution
        result['sdf_computed'] = True
        
        for resolution in resolutions:
            try:
                # Convert mesh to sparse SDF (now also returns edge_mask)
                sdf_data = mesh2sparse_sdf(
                    mesh,
                    resolution=resolution,
                    threshold_factor=threshold_factor,
                    normalize=True,
                    scale=0.95,
                    watertight=watertight,
                    compute_edge_mask=compute_edge_mask,
                    sharp_grad_dev_thresh=sharp_grad_dev_thresh,
                )
                
                # Save to npz file - include edge_mask and extra_band_factor metadata
                output_path = os.path.join(
                    output_dir,
                    f'{sha256}_r{resolution}.npz'
                )
                np.savez_compressed(
                    output_path,
                    sparse_sdf=sdf_data['sparse_sdf'],
                    sparse_index=sdf_data['sparse_index'],
                    edge_mask=sdf_data['edge_mask'],
                    extra_band_factor=np.array(sdf_data['extra_band_factor'], dtype=np.float32),
                    resolution=sdf_data['resolution'],
                )
                
                # Record number of points and edge-mask stats
                n_pts = len(sdf_data['sparse_index'])
                n_sharp = int(sdf_data['edge_mask'].sum())
                result[f'r{resolution}_num_points'] = n_pts
                result[f'r{resolution}_sharp_points'] = n_sharp
                print(
                    f"[sdf_voxelize] sha256={sha256[:8]}.. res={resolution}: "
                    f"N={n_pts}, sharp={n_sharp} ({100.0*n_sharp/max(n_pts,1):.1f}%), "
                    f"band={threshold_factor} voxels",
                    flush=True,
                )
                
            except Exception as e:
                result[f'r{resolution}_error'] = str(e)
                result['sdf_computed'] = False
                print(
                    f"[sdf_voxelize] SDF failed: sha256={sha256}, "
                    f"resolution={resolution}, mesh_path={mesh_path}, error={e}",
                    flush=True,
                )
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        result['sdf_computed'] = False
        return result


def load_trellis500k_metadata(input_dir: str, metadata_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Load metadata for TRELLIS-500K format.
    
    Supports two dataset structures:
    
    1. ObjaverseXL structure:
        input_dir/
        ├── glbs/
        │   └── 000-023/
        │       └── {sha256}.glb
        └── object-paths.json
        
        Metadata at: input_dir/../../metadata.csv
    
    2. HSSD structure:
        input_dir/
        ├── 0/, 1/, ..., f/
        │   └── {hash}.glb
        ├── openings/, x/
        
        Metadata at: input_dir/../metadata.csv
        (file_identifier contains direct path like "objects/3/abc123.glb")
    
    Args:
        input_dir: Path to TRELLIS-500K raw data directory
        metadata_csv: Optional explicit path to metadata.csv (when auto-discovery fails or layout is non-standard)
    
    Returns:
        DataFrame with columns: sha256, glb_path, captions (optional), aesthetic_score (optional)
    """
    print(f"Loading TRELLIS-500K metadata from: {input_dir}")
    
    # Check for object-paths.json to determine dataset type
    object_paths_file = os.path.join(input_dir, 'object-paths.json')
    
    # Resolve metadata.csv
    if metadata_csv:
        metadata_csv_resolved = os.path.abspath(metadata_csv)
        if not os.path.isfile(metadata_csv_resolved):
            raise FileNotFoundError(f"metadata_csv not found: {metadata_csv_resolved}")
        metadata_csv = metadata_csv_resolved
        print(f"  Using explicit metadata_csv: {metadata_csv}")
    else:
        # Same dir as input_dir, then walk up ancestors (ObjaverseXL: .../raw/hf-objaverse-v1 -> .../ObjaverseXL/metadata.csv)
        root_abs = os.path.abspath(input_dir)
        tried_paths: List[str] = []
        metadata_csv = None
        cand_same = os.path.join(root_abs, 'metadata.csv')
        tried_paths.append(cand_same)
        if os.path.isfile(cand_same):
            metadata_csv = cand_same
        else:
            cur = root_abs
            for _ in range(12):
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent
                cand = os.path.join(cur, 'metadata.csv')
                tried_paths.append(cand)
                if os.path.isfile(cand):
                    metadata_csv = cand
                    break
        
        if not metadata_csv:
            raise FileNotFoundError(
                "metadata.csv not found under input_dir or any ancestor (up to 12 levels). Tried:\n  - "
                + "\n  - ".join(tried_paths[:8])
                + ("\n  - ..." if len(tried_paths) > 8 else "")
                + "\n  Pass metadata_csv=... explicitly, or fix --input_dir (e.g. use ./TRELLIS-500K/ObjaverseXL/raw/hf-objaverse-v1 if metadata lives under repo)."
            )
    
    print(f"  Loading metadata from: {metadata_csv}")
    metadata_df = pd.read_csv(metadata_csv)
    print(f"  Found columns in metadata.csv: {list(metadata_df.columns)}")
    
    if 'file_identifier' not in metadata_df.columns:
        raise ValueError("metadata.csv must have 'file_identifier' column for TRELLIS-500K format")
    if 'sha256' not in metadata_df.columns:
        raise ValueError("metadata.csv must have 'sha256' column for TRELLIS-500K format")
    
    # Detect dataset type and process accordingly
    if os.path.exists(object_paths_file):
        # ObjaverseXL structure with object-paths.json
        print(f"  Detected ObjaverseXL structure (object-paths.json found)")
        metadata_df = _load_objaversexl_paths(object_paths_file, input_dir, metadata_df)
    else:
        # HSSD structure: file_identifier contains direct path
        print(f"  Detected HSSD structure (no object-paths.json, using file_identifier as path)")
        metadata_df = _load_hssd_paths(input_dir, metadata_df)
    
    # Print statistics
    print(f"  Total objects in metadata: {len(metadata_df)}")
    if 'captions' in metadata_df.columns:
        caption_count = metadata_df['captions'].notna().sum()
        print(f"  Objects with captions: {caption_count}")
    if 'aesthetic_score' in metadata_df.columns:
        score_count = metadata_df['aesthetic_score'].notna().sum()
        print(f"  Objects with aesthetic scores: {score_count}")
    
    return metadata_df


def _load_objaversexl_paths(object_paths_file: str, input_dir: str, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load ObjaverseXL paths from object-paths.json and merge with metadata.
    
    Args:
        object_paths_file: Path to object-paths.json
        input_dir: Input directory
        metadata_df: Metadata DataFrame with file_identifier and sha256
    
    Returns:
        Merged DataFrame with glb_path column
    """
    with open(object_paths_file, 'r') as f:
        object_paths = json.load(f)
    
    print(f"  Found {len(object_paths)} objects in object-paths.json")
    
    # The keys in object-paths.json are file_identifier (Sketchfab IDs), not sha256
    records = []
    for file_identifier, rel_path in object_paths.items():
        glb_path = os.path.join(input_dir, rel_path)
        records.append({
            'file_identifier': file_identifier,
            'glb_path': glb_path
        })
    
    paths_df = pd.DataFrame(records)
    
    # Normalize file_identifier:
    # - metadata.csv often stores full URLs like https://sketchfab.com/3d-models/<id>
    # - object-paths.json stores only the <id>
    metadata_df['file_identifier'] = (
        metadata_df['file_identifier']
        .astype(str)
        .str.strip()
        .str.replace(r'.*/', '', regex=True)
    )
    paths_df['file_identifier'] = paths_df['file_identifier'].astype(str).str.strip()
    
    # Merge paths with metadata
    merged_df = paths_df.merge(
        metadata_df,
        on='file_identifier',
        how='inner'
    )
    
    # Print merge statistics
    matched = len(merged_df)
    total_paths = len(paths_df)
    print(f"  Matched {matched}/{total_paths} objects via file_identifier")
    
    return merged_df


def _load_hssd_paths(input_dir: str, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load HSSD paths directly from file_identifier in metadata.
    
    For HSSD, file_identifier contains the relative path to the GLB file,
    e.g., "objects/3/3e4790548c158671fec162757053201da04b6259.glb"
    
    Args:
        input_dir: Input directory (should be the 'objects' directory)
        metadata_df: Metadata DataFrame with file_identifier and sha256
    
    Returns:
        DataFrame with glb_path column added
    """
    # For HSSD, file_identifier already contains the relative path
    # We need to construct the full path based on input_dir
    
    # The input_dir might be pointing to ./TRELLIS-500K/HSSD/raw/objects
    # And file_identifier is like "objects/3/xyz.glb"
    # So we need to go up one level to the 'raw' directory
    
    records = []
    valid_count = 0
    missing_count = 0
    
    for _, row in metadata_df.iterrows():
        file_identifier = row['file_identifier']
        sha256 = row['sha256']
        
        # Handle different path formats
        if file_identifier.startswith('objects/'):
            # Remove 'objects/' prefix since input_dir already points to objects/
            rel_path = file_identifier[8:]  # Remove "objects/"
        else:
            rel_path = file_identifier
        
        # Construct full path
        glb_path = os.path.join(input_dir, rel_path)
        
        # Check if file exists
        if os.path.exists(glb_path):
            valid_count += 1
        else:
            missing_count += 1
            if missing_count <= 5:  # Show first 5 missing files
                print(f"  Warning: File not found: {glb_path}")
        
        record = row.to_dict()
        record['glb_path'] = glb_path
        records.append(record)
    
    print(f"  Found {valid_count} existing GLB files")
    if missing_count > 0:
        print(f"  Warning: {missing_count} GLB files not found on disk")
    
    return pd.DataFrame.from_records(records)


def load_custom_metadata(input_dir: str) -> pd.DataFrame:
    """
    Load metadata for custom labeled data format.
    
    Expected structure:
        input_dir/
        ├── metadata.csv
        ├── {sha256}.glb
        └── ...
    
    metadata.csv should have columns:
        sha256, glb_file, overall_label, materials_captions, image_dir_path, view_keys
    
    Args:
        input_dir: Path to custom dataset directory
    
    Returns:
        DataFrame with columns including sha256, glb_path, overall_label, materials_captions
    """
    print(f"Loading custom metadata from: {input_dir}")
    
    # Load metadata.csv
    metadata_csv = os.path.join(input_dir, 'metadata.csv')
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"metadata.csv not found at: {metadata_csv}")
    
    metadata_df = pd.read_csv(metadata_csv)
    print(f"  Loaded {len(metadata_df)} records from metadata.csv")
    
    # Verify required columns
    required_cols = ['sha256', 'glb_file']
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata.csv: {missing_cols}")
    
    # Build full paths to GLB files
    metadata_df['glb_path'] = metadata_df['glb_file'].apply(
        lambda x: os.path.join(input_dir, x)
    )
    
    print(f"  Total objects in metadata: {len(metadata_df)}")
    return metadata_df


def get_cuda_device_specifiers_for_children() -> List[str]:
    """
    Return one string per visible GPU, each suitable as CUDA_VISIBLE_DEVICES for a child process.
    When CUDA_VISIBLE_DEVICES is unset, uses 0,1,...,n-1. When set to a comma list, uses that list
    (numeric IDs or other tokens supported by the driver).
    """
    if not torch.cuda.is_available():
        return []
    cv = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    n = torch.cuda.device_count()
    if not cv:
        return [str(i) for i in range(n)]
    parts = [p.strip() for p in cv.split(",") if p.strip()]
    if len(parts) >= n and n > 0:
        return parts[:n]
    return [str(i) for i in range(n)]


def resolve_num_gpu_workers(gpu_nums: int, num_visible: int) -> int:
    """
    gpu_nums: -1 = use all visible GPUs; 1 = single process; >1 = cap at available count.
    """
    if num_visible <= 0:
        return 1
    if gpu_nums == -1:
        return num_visible
    if gpu_nums <= 0:
        return 1
    return min(gpu_nums, num_visible)


def resolve_workers_per_gpu(max_workers: Optional[int]) -> int:
    """Number of independent mesh-processing worker slots to run on each GPU."""
    if max_workers is None:
        return 1
    return max(1, int(max_workers))


def slice_metadata_shard(metadata: pd.DataFrame, rank: int, world_size: int) -> pd.DataFrame:
    """Non-overlapping contiguous shards for multi-GPU workers."""
    if world_size <= 1:
        return metadata
    n = len(metadata)
    chunk = (n + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n)
    return metadata.iloc[start:end].reset_index(drop=True)


def _partial_results_path(output_dir: str, rank: int) -> str:
    return os.path.join(output_dir, f"results_partial_{rank:03d}.csv")


def cleanup_partial_result_files(output_dir: str) -> None:
    for p in glob.glob(os.path.join(output_dir, "results_partial_*.csv")):
        try:
            os.remove(p)
        except OSError:
            pass


def _ensure_results_schema_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.columns) == 0:
        return pd.DataFrame(columns=["sha256", "sdf_computed"])
    return df


def build_child_argv(args: argparse.Namespace, rank: int, world_size: int) -> List[str]:
    """Argv tail for a worker subprocess (same flags as user, plus shard + single-GPU)."""
    argv: List[str] = [
        "--format",
        args.format,
        "--input_dir",
        args.input_dir,
        "--output_dir",
        args.output_dir,
        "--resolutions",
        args.resolutions,
        "--threshold_factor",
        str(args.threshold_factor),
        "--gpu_nums",
        "1",
        "--worker_rank",
        str(rank),
        "--worker_world_size",
        str(world_size),
    ]
    if args.max_workers is not None:
        argv.extend(["--max_workers", str(args.max_workers)])
    if args.no_skip:
        argv.append("--no_skip")
    if args.filter_aesthetic_score is not None:
        argv.extend(["--filter_aesthetic_score", str(args.filter_aesthetic_score)])
    if args.max_samples is not None:
        argv.extend(["--max_samples", str(args.max_samples)])
    if getattr(args, "watertight", False):
        argv.append("--watertight")
    if getattr(args, "no_edge_mask", False):
        argv.append("--no_edge_mask")
    if getattr(args, "sharp_grad_dev_thresh", 0.3) != 0.3:
        argv.extend(["--sharp_grad_dev_thresh", str(args.sharp_grad_dev_thresh)])
    return argv


def merge_partial_result_csvs(output_dir: str, num_parts: int) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for rank in range(num_parts):
        path = _partial_results_path(output_dir, rank)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing partial results file (worker {rank}): {path}"
            )
        part = pd.read_csv(path)
        dfs.append(part)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def filter_existing_files(metadata: pd.DataFrame, output_dir: str, resolutions: List[int]) -> pd.DataFrame:
    """
    Filter out already processed files and add their records.
    
    Args:
        metadata: Input metadata DataFrame
        output_dir: Output directory to check for existing files
        resolutions: List of resolutions to check
    
    Returns:
        Tuple of (filtered_metadata, existing_records)
    """
    existing_records = []
    remaining_sha256s = []
    
    for sha256 in metadata['sha256'].values:
        # Check if all resolution files exist
        all_exist = True
        record = {'sha256': sha256, 'sdf_computed': True}
        
        for resolution in resolutions:
            output_path = os.path.join(output_dir, f'{sha256}_r{resolution}.npz')
            if not os.path.exists(output_path):
                all_exist = False
                break
            else:
                # Load and record number of points
                try:
                    data = np.load(output_path)
                    record[f'r{resolution}_num_points'] = len(data['sparse_index'])
                except:
                    all_exist = False
                    break
        
        if all_exist:
            existing_records.append(record)
        else:
            remaining_sha256s.append(sha256)
    
    # Filter metadata to only unprocessed items
    filtered_metadata = metadata[metadata['sha256'].isin(remaining_sha256s)]
    
    print(f"  Found {len(existing_records)} already processed files")
    print(f"  Remaining to process: {len(filtered_metadata)}")
    
    return filtered_metadata, existing_records


def process_dataset(
    metadata: pd.DataFrame,
    output_dir: str,
    resolutions: List[int] = [512],
    threshold_factor: float = 4.0,
    max_workers: Optional[int] = None,
    skip_existing: bool = True,
    watertight: bool = False,
    compute_edge_mask: bool = True,
    sharp_grad_dev_thresh: float = 0.3,
) -> pd.DataFrame:
    """
    Process all meshes in the metadata to SDF format.
    
    Args:
        metadata: DataFrame with sha256 and glb_path columns
        output_dir: Output directory for SDF files
        resolutions: List of resolutions to compute
        threshold_factor: Sparse shell half-width in voxels (>= 4 recommended)
        max_workers: Number of parallel workers (None = single process with GPU)
        skip_existing: Whether to skip already processed files
        watertight: If True, repair mesh with pymeshfix before UDF (see mesh_utils.make_watertight)
        compute_edge_mask: Compute and store GPU sharp-edge mask in npz
        sharp_grad_dev_thresh: Threshold for sharp-edge detection (default 0.3)
    
    Returns:
        DataFrame with processing results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter existing files if requested
    existing_records = []
    if skip_existing:
        metadata, existing_records = filter_existing_files(metadata, output_dir, resolutions)
    
    if len(metadata) == 0:
        print("No files to process (all already completed)")
        return pd.DataFrame.from_records(existing_records)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n⚠️  Warning: CUDA not available. SDF computation requires GPU support.")
        print("    Please ensure you have a CUDA-capable GPU and PyTorch with CUDA installed.")
        return pd.DataFrame()
    
    print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  Processing {len(metadata)} meshes at resolutions: {resolutions}")
    print(f"  Threshold factor (extra_band_factor): {threshold_factor} voxels")
    print(f"  Compute edge_mask: {compute_edge_mask}, sharp_grad_dev_thresh={sharp_grad_dev_thresh}")
    print(f"  Output directory: {output_dir}")
    
    # Determine processing mode
    max_workers = max_workers or 1  # Default to single process for GPU safety
    
    # For CUDA operations, we recommend single-process mode to avoid GPU conflicts
    if max_workers > 1:
        print(f"\n⚠️  WARNING: Using {max_workers} workers with GPU operations may cause issues.")
        print("    If processing gets stuck, try max_workers=1 for stability.")
    
    print(f"  Parallel workers: {max_workers}")
    print("\nProcessing meshes...")
    
    # Process with appropriate mode
    records = []
    
    if max_workers == 1:
        # Single-process mode (recommended for GPU)
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing"):
            result = _process_mesh_to_sdf(
                row['glb_path'], 
                row['sha256'], 
                output_dir, 
                resolutions, 
                threshold_factor,
                watertight=watertight,
                compute_edge_mask=compute_edge_mask,
                sharp_grad_dev_thresh=sharp_grad_dev_thresh,
            )
            if result is not None:
                records.append(result)
                
            # Periodic GPU cache cleanup
            if len(records) % 100 == 0:
                torch.cuda.empty_cache()
    else:
        # Multi-process mode (use with caution)
        import torch.multiprocessing as mp
        # Set spawn method for better CUDA compatibility
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        func = partial(
            _process_mesh_to_sdf_wrapper,
            output_dir=output_dir,
            resolutions=resolutions,
            threshold_factor=threshold_factor,
            watertight=watertight,
            compute_edge_mask=compute_edge_mask,
            sharp_grad_dev_thresh=sharp_grad_dev_thresh,
        )
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for _, row in metadata.iterrows():
                future = executor.submit(func, (row['glb_path'], row['sha256']))
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per mesh
                    if result is not None:
                        records.append(result)
                except Exception as e:
                    print(f"\n  Error in worker: {e}")
    
    # Combine with existing records
    if existing_records:
        records.extend(existing_records)
    
    return pd.DataFrame.from_records(records)


def _process_mesh_to_sdf_wrapper(
    args,
    output_dir,
    resolutions,
    threshold_factor,
    watertight=False,
    compute_edge_mask=True,
    sharp_grad_dev_thresh=0.3,
):
    """Wrapper function for multiprocessing."""
    mesh_path, sha256 = args
    return _process_mesh_to_sdf(
        mesh_path,
        sha256,
        output_dir,
        resolutions,
        threshold_factor,
        watertight=watertight,
        compute_edge_mask=compute_edge_mask,
        sharp_grad_dev_thresh=sharp_grad_dev_thresh,
    )


def save_results(
    results: pd.DataFrame,
    original_metadata: pd.DataFrame,
    output_dir: str,
    format_type: str
):
    """
    Save processing results merged with original metadata.
    
    Args:
        results: DataFrame with processing results
        original_metadata: Original metadata DataFrame
        output_dir: Output directory
        format_type: Format type ('trellis500k' or 'custom')
    """
    # Merge results with original metadata
    merged = original_metadata.merge(results, on='sha256', how='left')
    
    # Fill NaN values for unprocessed items
    merged['sdf_computed'] = merged['sdf_computed'].fillna(False)
    
    # Remove internal path columns
    if 'glb_path' in merged.columns:
        merged = merged.drop(columns=['glb_path'])
    
    # Save metadata
    output_csv = os.path.join(output_dir, 'metadata.csv')
    merged.to_csv(output_csv, index=False)
    print(f"\n✓ Metadata saved to: {output_csv}")
    
    # Print statistics
    success_count = merged['sdf_computed'].sum()
    total_count = len(merged)
    
    print(f"\nProcessing Statistics:")
    print(f"  Total items: {total_count}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed/Skipped: {total_count - success_count}")
    
    # Print resolution statistics
    for col in merged.columns:
        if col.startswith('r') and col.endswith('_num_points'):
            resolution = col.replace('_num_points', '')
            valid_count = merged[col].notna().sum()
            if valid_count > 0:
                avg_points = merged[col].mean()
                print(f"  {resolution}: {valid_count} files, avg {avg_points:.0f} points")


def main():
    parser = argparse.ArgumentParser(
        description='Convert mesh files to sparse SDF format for VQVAE training'
    )
    
    # Required arguments
    parser.add_argument(
        '--format',
        type=str,
        required=True,
        choices=['trellis500k', 'custom'],
        help='Dataset format type'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing mesh files and metadata'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for SDF files'
    )
    
    # Optional arguments
    parser.add_argument(
        '--resolutions',
        type=str,
        default='512',
        help='Comma-separated list of resolutions (e.g., "64,512,1024"). Default: 512'
    )
    parser.add_argument(
        '--threshold_factor',
        '--extra_band_factor',
        type=float,
        default=4.0,
        dest='threshold_factor',
        help=(
            'Sparse shell half-width in voxels (default: 4.0).  '
            'Must be >= 4 so the GT SDF band covers decoder-predicted extra voxels.  '
            'Also accepted as --extra_band_factor for clarity.'
        ),
    )
    parser.add_argument(
        '--sharp_grad_dev_thresh',
        type=float,
        default=0.3,
        help=(
            'Gradient-deviation threshold for sharp/edge detection: '
            'voxels with |1 - |∇SDF|| > this value are flagged as sharp (default: 0.3).  '
            'Smaller values flag more voxels as sharp.'
        ),
    )
    parser.add_argument(
        '--no_edge_mask',
        action='store_true',
        help=(
            'Disable computation of the sharp-edge boolean mask.  '
            'By default edge_mask is computed on the GPU and stored in each npz.'
        ),
    )
    parser.add_argument(
        '--watertight',
        action='store_true',
        help=(
            'After normalize, run pymeshfix.MeshFix.repair() before UDF '
            '(requires: pip install pymeshfix pyvista). Default: off.'
        ),
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=None,
        help=(
            'Worker processes per GPU (default: 1). Increase to 2-4 when GPU utilization is low '
            'and VRAM is available; total workers = selected GPUs * max_workers.'
        )
    )
    parser.add_argument(
        '--gpu_nums',
        type=int,
        default=-1,
        help='Number of GPUs to use in parallel (-1 = all visible GPUs, 1 = single process). '
        'Shards data across GPUs; each GPU writes disjoint .npz files and a partial CSV, then merges metadata.',
    )
    parser.add_argument(
        '--worker_rank',
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--worker_world_size',
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--no_skip',
        action='store_true',
        help='Reprocess all files even if they already exist'
    )
    parser.add_argument(
        '--filter_aesthetic_score',
        type=float,
        default=None,
        help='Filter objects with aesthetic score below this value (TRELLIS-500K only)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Only process this many samples (default: process all). Uses first N rows after filters.'
    )
    
    args = parser.parse_args()

    try:
        _preflight_udf_ext()
    except Exception as exc:
        print(f"\n❌ [sdf_voxelize] udf_ext 预检失败:\n{exc}", flush=True)
        return 1

    # Soft check for compute_sharp_mask (only warn; edge_mask will degrade to all-False)
    try:
        import udf_ext as _udf_ext_check
        if not hasattr(_udf_ext_check, 'compute_sharp_mask'):
            print(
                "\n⚠️  [sdf_voxelize] udf_ext.compute_sharp_mask 未找到。\n"
                "   edge_mask 将退化为全 False（所有样本视为无 sharp 区域）。\n"
                "   要启用锐利边缘检测，请重新编译扩展:\n"
                "     cd third_party/voxelize && "
                "pip install -v -e . --no-build-isolation",
                flush=True,
            )
    except Exception:
        pass
    
    # Parse resolutions
    resolutions = [int(r.strip()) for r in args.resolutions.split(',')]
    print(f"\n{'='*70}")
    print("SDF Voxelization Preprocessing")
    print(f"{'='*70}")
    print(f"Format: {args.format}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resolutions: {resolutions}")
    print(f"Threshold factor (extra_band_factor): {args.threshold_factor} voxels")
    print(f"Compute edge_mask: {not args.no_edge_mask}")
    print(f"Sharp grad dev thresh: {args.sharp_grad_dev_thresh}")
    print(f"Watertight (pymeshfix): {args.watertight}")
    print(f"Max samples: {args.max_samples if args.max_samples is not None else 'all'}")
    print(f"Skip existing: {not args.no_skip}")
    print(f"gpu_nums: {args.gpu_nums}")
    
    # Load metadata based on format
    try:
        if args.format == 'trellis500k':
            metadata = load_trellis500k_metadata(args.input_dir)
            
            # Apply aesthetic score filter if specified
            if args.filter_aesthetic_score is not None:
                if 'aesthetic_score' in metadata.columns:
                    before_count = len(metadata)
                    metadata = metadata[metadata['aesthetic_score'] >= args.filter_aesthetic_score]
                    print(f"  Filtered by aesthetic score >={args.filter_aesthetic_score}: {len(metadata)}/{before_count} remaining")
                else:
                    print(f"  Warning: No aesthetic_score column found, skipping filter")
        else:  # custom
            metadata = load_custom_metadata(args.input_dir)
    except Exception as e:
        print(f"\n❌ Error loading metadata: {e}")
        return 1
    
    if len(metadata) == 0:
        print("\n❌ No data found in metadata")
        return 1
    
    # Limit to max_samples if specified
    if args.max_samples is not None:
        metadata = metadata.head(args.max_samples)
        print(f"  Limited to first {args.max_samples} samples")

    # Subprocess worker: one GPU, one shard, writes partial CSV only (no final metadata merge)
    if args.worker_world_size is not None and args.worker_world_size > 1:
        if args.worker_rank is None:
            print("\n❌ Internal error: worker_world_size > 1 requires worker_rank")
            return 1
        print(f"\n  [Worker {args.worker_rank}/{args.worker_world_size}] "
              f"rows in shard: {len(slice_metadata_shard(metadata, args.worker_rank, args.worker_world_size))}")
        shard = slice_metadata_shard(metadata, args.worker_rank, args.worker_world_size)
        try:
            results = process_dataset(
                metadata=shard,
                output_dir=args.output_dir,
                resolutions=resolutions,
                threshold_factor=args.threshold_factor,
                max_workers=1,
                skip_existing=not args.no_skip,
                watertight=args.watertight,
                compute_edge_mask=not args.no_edge_mask,
                sharp_grad_dev_thresh=args.sharp_grad_dev_thresh,
            )
            results = _ensure_results_schema_for_csv(results)
            partial_path = _partial_results_path(args.output_dir, args.worker_rank)
            results.to_csv(partial_path, index=False)
            print(f"\n  [Worker {args.worker_rank}] wrote {partial_path} ({len(results)} rows)")
        except Exception as e:
            import traceback
            print(f"\n❌ Error in worker {args.worker_rank}: {e}")
            traceback.print_exc()
            return 1
        return 0

    # Orchestrator: split data across independent worker slots. Multiple slots can share
    # the same GPU to overlap mesh loading, CUDA kernels, and compressed npz writes.
    device_specs = get_cuda_device_specifiers_for_children()
    n_gpu_devices = resolve_num_gpu_workers(args.gpu_nums, len(device_specs))
    workers_per_gpu = resolve_workers_per_gpu(args.max_workers)

    if not torch.cuda.is_available():
        print("\n⚠️  Warning: CUDA not available. Running single-process CPU path is not supported for SDF.")
        n_gpu_devices = 1
        workers_per_gpu = 1

    total_worker_slots = n_gpu_devices * workers_per_gpu

    if total_worker_slots > 1 and torch.cuda.is_available():
        print(
            f"\n  Parallel GPU preprocessing: {n_gpu_devices} GPU(s) * "
            f"{workers_per_gpu} worker(s)/GPU = {total_worker_slots} worker slots"
        )
        if workers_per_gpu > 1:
            print(
                "  Tip: if CUDA OOM occurs, lower --max_workers; if GPU utilization is still low, "
                "raise it gradually while watching VRAM."
            )
        cleanup_partial_result_files(args.output_dir)
        script_path = os.path.abspath(__file__)
        procs: List[Tuple[int, int, subprocess.Popen]] = []
        try:
            for rank in range(total_worker_slots):
                gpu_idx = rank % n_gpu_devices
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = device_specs[gpu_idx]
                cmd = [sys.executable, "-u", script_path] + build_child_argv(
                    args, rank, total_worker_slots
                )
                print(
                    f"\n  Launching worker slot {rank}/{total_worker_slots} "
                    f"on GPU index {gpu_idx} (CUDA_VISIBLE_DEVICES={device_specs[gpu_idx]})"
                )
                procs.append((rank, gpu_idx, subprocess.Popen(cmd, env=env)))
            for rank, gpu_idx, p in procs:
                rc = p.wait()
                if rc != 0:
                    print(
                        f"\n❌ Subprocess failed for worker slot {rank} "
                        f"on GPU index {gpu_idx} (exit {rc})"
                    )
                    for _, _, q in procs:
                        if q.poll() is None:
                            q.terminate()
                    return 1
        except Exception:
            for _, _, q in procs:
                if q.poll() is None:
                    q.terminate()
            raise
        try:
            combined = merge_partial_result_csvs(args.output_dir, total_worker_slots)
            if len(combined) == 0:
                print("\n⚠️  No results generated")
                return 1
            save_results(combined, metadata, args.output_dir, args.format)
            for rank in range(total_worker_slots):
                try:
                    os.remove(_partial_results_path(args.output_dir, rank))
                except OSError:
                    pass
        except Exception as e:
            import traceback
            print(f"\n❌ Error merging partial results: {e}")
            traceback.print_exc()
            return 1
        print(f"\n{'='*70}")
        print("✓ Processing complete!")
        print(f"{'='*70}")
        return 0

    # Single-GPU / single-process path
    try:
        results = process_dataset(
            metadata=metadata,
            output_dir=args.output_dir,
            resolutions=resolutions,
            threshold_factor=args.threshold_factor,
            max_workers=args.max_workers,
            skip_existing=not args.no_skip,
            watertight=args.watertight,
            compute_edge_mask=not args.no_edge_mask,
            sharp_grad_dev_thresh=args.sharp_grad_dev_thresh,
        )
        
        if len(results) == 0:
            print("\n⚠️  No results generated")
            return 1
        
        # Save results
        save_results(results, metadata, args.output_dir, args.format)
        
        print(f"\n{'='*70}")
        print("✓ Processing complete!")
        print(f"{'='*70}")
        
        return 0
        
    except Exception as e:
        import traceback
        print(f"\n❌ Error during processing: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

