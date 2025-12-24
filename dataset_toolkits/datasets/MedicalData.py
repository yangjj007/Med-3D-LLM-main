"""
Medical data processing utilities.
Supports custom medical 3D datasets for sparse SDF training.
"""

import os
import glob
import hashlib
from typing import List, Dict, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm


def get_file_list(
    data_root: str,
    extensions: List[str] = ['.obj', '.ply', '.stl', '.glb', '.gltf', '.npy', '.npz']
) -> List[Dict[str, str]]:
    """
    Get list of medical data files.
    
    Args:
        data_root: Root directory containing medical data
        extensions: List of supported file extensions
    
    Returns:
        List of dictionaries with file information
    """
    files = []
    
    # Search for all supported files recursively
    for ext in extensions:
        pattern = os.path.join(data_root, '**', f'*{ext}')
        found_files = glob.glob(pattern, recursive=True)
        
        for file_path in found_files:
            # Generate SHA256 hash from relative path
            rel_path = os.path.relpath(file_path, data_root)
            sha256 = hashlib.sha256(rel_path.encode()).hexdigest()
            
            files.append({
                'sha256': sha256,
                'file_path': file_path,
                'relative_path': rel_path,
                'extension': ext,
            })
    
    return files


def add_args(parser):
    """
    Add medical dataset specific arguments.
    
    Args:
        parser: argparse.ArgumentParser
    """
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing medical data files'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        default='.obj,.ply,.stl,.glb,.gltf,.npy,.npz',
        help='Comma-separated list of file extensions to include (default: .obj,.ply,.stl,.glb,.gltf,.npy,.npz)'
    )


def foreach_instance(
    metadata: pd.DataFrame,
    output_dir: str,
    func: Callable,
    max_workers: int = None,
    desc: str = 'Processing'
) -> pd.DataFrame:
    """
    Apply function to each instance in parallel.
    
    Args:
        metadata: DataFrame with instance information
        output_dir: Output directory
        func: Function to apply to each instance (file_path, sha256) -> result_dict
        max_workers: Maximum number of worker processes
        desc: Description for progress bar
    
    Returns:
        DataFrame with processing results
    """
    results = []
    
    if max_workers is None or max_workers == 1:
        # Sequential processing
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=desc):
            file_path = row.get('file_path', '')
            sha256 = row['sha256']
            result = func(file_path, sha256)
            results.append(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for _, row in metadata.iterrows():
                file_path = row.get('file_path', '')
                sha256 = row['sha256']
                futures.append(executor.submit(func, file_path, sha256))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing instance: {e}")
                    results.append({'error': str(e)})
    
    return pd.DataFrame.from_records(results)


def build_metadata(opt) -> pd.DataFrame:
    """
    Build metadata for medical dataset.
    
    Args:
        opt: Options from argparse
    
    Returns:
        DataFrame with metadata
    """
    # Parse extensions
    extensions = [ext.strip() for ext in opt.extensions.split(',')]
    if not all(ext.startswith('.') for ext in extensions):
        extensions = ['.' + ext if not ext.startswith('.') else ext for ext in extensions]
    
    print(f"Searching for files with extensions: {extensions}")
    print(f"Data root: {opt.data_root}")
    
    # Get file list
    files = get_file_list(opt.data_root, extensions)
    
    print(f"Found {len(files)} files")
    
    # Create metadata DataFrame
    metadata = pd.DataFrame(files)
    
    # Add default columns
    metadata['rendered'] = False
    metadata['voxelized'] = False
    metadata['sparse_sdf_computed'] = False
    metadata['aesthetic_score'] = 5.0  # Default score
    
    return metadata


if __name__ == '__main__':
    """
    Example usage:
    python dataset_toolkits/datasets/MedicalData.py \
        --data_root /path/to/medical/data \
        --output_dir ./data/medical
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Build medical dataset metadata')
    add_args(parser)
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for metadata'
    )
    
    opt = parser.parse_args()
    
    # Build metadata
    metadata = build_metadata(opt)
    
    # Save metadata
    os.makedirs(opt.output_dir, exist_ok=True)
    output_path = os.path.join(opt.output_dir, 'metadata.csv')
    metadata.to_csv(output_path, index=False)
    
    print(f"\nMetadata saved to: {output_path}")
    print(f"Total instances: {len(metadata)}")
    print("\nFile type distribution:")
    print(metadata['extension'].value_counts())

