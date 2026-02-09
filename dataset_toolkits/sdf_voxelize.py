"""
SDF Voxelization Preprocessing Script

Convert mesh files (GLB format) to sparse SDF format for VQVAE training.
Supports three data formats:
1. TRELLIS-500K format with ObjaverseXL (with object-paths.json)
2. TRELLIS-500K format with HSSD (file_identifier as direct path)
3. Custom labeled format (with metadata.csv)

Usage:
    # TRELLIS-500K format - ObjaverseXL
    python dataset_toolkits/sdf_voxelize.py \
        --format trellis500k \
        --input_dir ./TRELLIS-500K/raw/hf-objaverse-v1 \
        --output_dir ./train_sdf_dataset \
        --resolutions 64,512 \
        --max_workers 4
    
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
import argparse
from pathlib import Path
from functools import partial
from typing import Dict, List, Any, Optional
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


def _process_mesh_to_sdf(
    mesh_path: str,
    sha256: str,
    output_dir: str,
    resolutions: List[int] = [512],
    threshold_factor: float = 4.0
) -> Dict[str, Any]:
    """
    Process a single mesh file and convert to sparse SDF at multiple resolutions.
    
    Args:
        mesh_path: Path to the mesh file (GLB format)
        sha256: SHA256 hash identifier for the mesh
        output_dir: Output directory for SDF files
        resolutions: List of resolutions to compute
        threshold_factor: UDF threshold factor for sparse extraction
    
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
                # Convert mesh to sparse SDF
                sdf_data = mesh2sparse_sdf(
                    mesh,
                    resolution=resolution,
                    threshold_factor=threshold_factor,
                    normalize=True,
                    scale=0.95
                )
                
                # Save to npz file
                output_path = os.path.join(
                    output_dir,
                    f'{sha256}_r{resolution}.npz'
                )
                np.savez_compressed(
                    output_path,
                    sparse_sdf=sdf_data['sparse_sdf'],
                    sparse_index=sdf_data['sparse_index'],
                    resolution=sdf_data['resolution']
                )
                
                # Record number of points
                result[f'r{resolution}_num_points'] = len(sdf_data['sparse_index'])
                
                # Clean up GPU memory after each resolution
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                result[f'r{resolution}_error'] = str(e)
                result['sdf_computed'] = False
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        result['sdf_computed'] = False
        return result


def load_trellis500k_metadata(input_dir: str) -> pd.DataFrame:
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
    
    Returns:
        DataFrame with columns: sha256, glb_path, captions (optional), aesthetic_score (optional)
    """
    print(f"Loading TRELLIS-500K metadata from: {input_dir}")
    
    # Check for object-paths.json to determine dataset type
    object_paths_file = os.path.join(input_dir, 'object-paths.json')
    
    # Try to find metadata.csv in parent directories
    parent_dir = os.path.dirname(input_dir)  # Go up one level first
    metadata_csv = os.path.join(parent_dir, 'metadata.csv')
    
    if not os.path.exists(metadata_csv):
        # Try going up two levels (for ObjaverseXL structure)
        parent_dir = os.path.dirname(parent_dir)
        metadata_csv = os.path.join(parent_dir, 'metadata.csv')
    
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(
            f"metadata.csv not found. Tried:\n"
            f"  - {os.path.join(os.path.dirname(input_dir), 'metadata.csv')}\n"
            f"  - {os.path.join(os.path.dirname(os.path.dirname(input_dir)), 'metadata.csv')}"
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
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    Process all meshes in the metadata to SDF format.
    
    Args:
        metadata: DataFrame with sha256 and glb_path columns
        output_dir: Output directory for SDF files
        resolutions: List of resolutions to compute
        threshold_factor: UDF threshold factor
        max_workers: Number of parallel workers (None = single process with GPU)
        skip_existing: Whether to skip already processed files
    
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
    print(f"  Threshold factor: {threshold_factor}")
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
                threshold_factor
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
            threshold_factor=threshold_factor
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


def _process_mesh_to_sdf_wrapper(args, output_dir, resolutions, threshold_factor):
    """Wrapper function for multiprocessing."""
    mesh_path, sha256 = args
    return _process_mesh_to_sdf(mesh_path, sha256, output_dir, resolutions, threshold_factor)


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
        type=float,
        default=4.0,
        help='UDF threshold factor for sparse extraction. Default: 4.0'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=None,
        help='Maximum number of parallel workers (default: CPU count, max 4)'
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
    
    args = parser.parse_args()
    
    # Parse resolutions
    resolutions = [int(r.strip()) for r in args.resolutions.split(',')]
    print(f"\n{'='*70}")
    print("SDF Voxelization Preprocessing")
    print(f"{'='*70}")
    print(f"Format: {args.format}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resolutions: {resolutions}")
    print(f"Skip existing: {not args.no_skip}")
    
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
    
    # Process dataset
    try:
        results = process_dataset(
            metadata=metadata,
            output_dir=args.output_dir,
            resolutions=resolutions,
            threshold_factor=args.threshold_factor,
            max_workers=args.max_workers,
            skip_existing=not args.no_skip
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

