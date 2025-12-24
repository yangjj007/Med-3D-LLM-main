"""
Compute sparse SDF from mesh files for multiple resolutions.
Supports both mesh files and dense voxel arrays.
"""

import os
import sys
import copy
import argparse
import importlib
from functools import partial
from typing import Dict, Any
import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm
from easydict import EasyDict as edict

# Add parent directory to path to import trellis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trellis.utils.mesh_utils import mesh2sparse_sdf, dense_voxel_to_sparse_sdf


def _compute_sparse_sdf_from_mesh(
    file: str,
    sha256: str,
    output_dir: str,
    resolutions: list = [64, 512, 1024]
) -> Dict[str, Any]:
    """
    Compute sparse SDF from a mesh file at multiple resolutions.
    
    Args:
        file: Path to the mesh file (not used, mesh loaded from output_dir)
        sha256: SHA256 hash of the instance
        output_dir: Output directory
        resolutions: List of resolutions to compute
    
    Returns:
        Dictionary with processing results
    """
    try:
        # Load mesh
        mesh_path = os.path.join(output_dir, 'renders', sha256, 'mesh.ply')
        if not os.path.exists(mesh_path):
            return {'sha256': sha256, 'sparse_sdf_computed': False, 'error': 'mesh not found'}
        
        mesh = trimesh.load(mesh_path)
        
        # Compute sparse SDF for each resolution
        results = {'sha256': sha256, 'sparse_sdf_computed': True}
        for resolution in resolutions:
            try:
                sdf_data = mesh2sparse_sdf(
                    mesh,
                    resolution=resolution,
                    threshold_factor=4.0,
                    normalize=True
                )
                
                # Save to npz file
                os.makedirs(os.path.join(output_dir, 'sparse_sdf'), exist_ok=True)
                output_path = os.path.join(
                    output_dir,
                    'sparse_sdf',
                    f'{sha256}_r{resolution}.npz'
                )
                np.savez_compressed(
                    output_path,
                    sparse_sdf=sdf_data['sparse_sdf'],
                    sparse_index=sdf_data['sparse_index'],
                    resolution=sdf_data['resolution']
                )
                
                results[f'r{resolution}_num_points'] = len(sdf_data['sparse_index'])
                
            except Exception as e:
                results[f'r{resolution}_error'] = str(e)
        
        return results
        
    except Exception as e:
        return {'sha256': sha256, 'sparse_sdf_computed': False, 'error': str(e)}


def _compute_sparse_sdf_from_voxel(
    file: str,
    sha256: str,
    output_dir: str,
    resolutions: list = [512, 1024]
) -> Dict[str, Any]:
    """
    Compute sparse SDF from a dense voxel file at multiple resolutions.
    
    Args:
        file: Path to the voxel file
        sha256: SHA256 hash of the instance
        output_dir: Output directory
        resolutions: List of resolutions to compute
    
    Returns:
        Dictionary with processing results
    """
    try:
        # Load voxel grid (assume .npy or .npz format)
        if file.endswith('.npy'):
            voxel_grid = np.load(file)
        elif file.endswith('.npz'):
            data = np.load(file)
            # Try common keys
            if 'voxels' in data:
                voxel_grid = data['voxels']
            elif 'data' in data:
                voxel_grid = data['data']
            else:
                voxel_grid = data[list(data.keys())[0]]
        else:
            return {'sha256': sha256, 'sparse_sdf_computed': False, 'error': 'unsupported file format'}
        
        # Compute sparse SDF for each resolution
        results = {'sha256': sha256, 'sparse_sdf_computed': True}
        for resolution in resolutions:
            try:
                sdf_data = dense_voxel_to_sparse_sdf(voxel_grid, resolution=resolution)
                
                # Save to npz file
                os.makedirs(os.path.join(output_dir, 'sparse_sdf'), exist_ok=True)
                output_path = os.path.join(
                    output_dir,
                    'sparse_sdf',
                    f'{sha256}_r{resolution}.npz'
                )
                np.savez_compressed(
                    output_path,
                    sparse_sdf=sdf_data['sparse_sdf'],
                    sparse_index=sdf_data['sparse_index'],
                    resolution=sdf_data['resolution']
                )
                
                results[f'r{resolution}_num_points'] = len(sdf_data['sparse_index'])
                
            except Exception as e:
                results[f'r{resolution}_error'] = str(e)
        
        return results
        
    except Exception as e:
        return {'sha256': sha256, 'sparse_sdf_computed': False, 'error': str(e)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute sparse SDF from mesh or voxel files')
    parser.add_argument('dataset', nargs='?', default=None,
                        help='Dataset name (e.g., ObjaverseXL, MedicalData). Optional if using --input_type voxel')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the sparse SDF data')
    parser.add_argument('--resolutions', type=str, default='64,512,1024',
                        help='Comma-separated list of resolutions (default: 64,512,1024)')
    parser.add_argument('--input_type', type=str, default='mesh', choices=['mesh', 'voxel'],
                        help='Input data type: mesh or voxel (default: mesh)')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process (comma-separated or file path)')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank for distributed processing')
    parser.add_argument('--world_size', type=int, default=1,
                        help='World size for distributed processing')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of worker processes')
    
    opt = parser.parse_args()
    opt = edict(vars(opt))
    
    # Parse resolutions
    resolutions = [int(r) for r in opt.resolutions.split(',')]
    print(f"Computing sparse SDF at resolutions: {resolutions}")
    
    # Create output directory
    os.makedirs(os.path.join(opt.output_dir, 'sparse_sdf'), exist_ok=True)
    
    # Load metadata
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found. Please run build_metadata.py first.')
    
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    # Filter instances
    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        # Only process rendered instances for mesh input
        if opt.input_type == 'mesh':
            if 'rendered' not in metadata.columns:
                raise ValueError('metadata.csv does not have "rendered" column. Please run render.py first.')
            metadata = metadata[metadata['rendered'] == True]
    
    # Filter already processed instances
    if 'sparse_sdf_computed' in metadata.columns:
        metadata = metadata[metadata['sparse_sdf_computed'] != True]
    
    # Distributed processing
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    
    # Check for already processed files
    for sha256 in copy.copy(metadata['sha256'].values):
        all_exist = True
        for resolution in resolutions:
            output_path = os.path.join(opt.output_dir, 'sparse_sdf', f'{sha256}_r{resolution}.npz')
            if not os.path.exists(output_path):
                all_exist = False
                break
        
        if all_exist:
            result = {'sha256': sha256, 'sparse_sdf_computed': True}
            for resolution in resolutions:
                output_path = os.path.join(opt.output_dir, 'sparse_sdf', f'{sha256}_r{resolution}.npz')
                data = np.load(output_path)
                result[f'r{resolution}_num_points'] = len(data['sparse_index'])
            records.append(result)
            metadata = metadata[metadata['sha256'] != sha256]
    
    print(f'Processing {len(metadata)} instances...')
    
    # Process instances
    if opt.input_type == 'mesh':
        if opt.dataset is None:
            raise ValueError('Dataset name is required for mesh input type')
        dataset_utils = importlib.import_module(f'datasets.{opt.dataset}')
        func = partial(_compute_sparse_sdf_from_mesh, output_dir=opt.output_dir, resolutions=resolutions)
        processed = dataset_utils.foreach_instance(
            metadata,
            opt.output_dir,
            func,
            max_workers=opt.max_workers,
            desc='Computing sparse SDF'
        )
    else:
        # For voxel input, iterate through metadata directly
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        func = partial(_compute_sparse_sdf_from_voxel, output_dir=opt.output_dir, resolutions=resolutions)
        
        if opt.max_workers is None or opt.max_workers > 1:
            with ProcessPoolExecutor(max_workers=opt.max_workers) as executor:
                futures = []
                for _, row in metadata.iterrows():
                    file_path = row.get('file_path', '')
                    sha256 = row['sha256']
                    futures.append(executor.submit(func, file_path, sha256))
                
                results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc='Computing sparse SDF'):
                    results.append(future.result())
                processed = pd.DataFrame.from_records(results)
        else:
            results = []
            for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc='Computing sparse SDF'):
                file_path = row.get('file_path', '')
                sha256 = row['sha256']
                results.append(func(file_path, sha256))
            processed = pd.DataFrame.from_records(results)
    
    # Combine with already processed records
    processed = pd.concat([processed, pd.DataFrame.from_records(records)])
    
    # Save results
    output_csv = os.path.join(opt.output_dir, f'sparse_sdf_computed_{opt.rank}.csv')
    processed.to_csv(output_csv, index=False)
    print(f'Results saved to {output_csv}')
    
    # Print statistics
    success_count = processed['sparse_sdf_computed'].sum()
    print(f'\nProcessing complete:')
    print(f'  Successful: {success_count}/{len(processed)}')
    for resolution in resolutions:
        col = f'r{resolution}_num_points'
        if col in processed.columns:
            avg_points = processed[col].mean()
            print(f'  Resolution {resolution}: avg {avg_points:.0f} points')

