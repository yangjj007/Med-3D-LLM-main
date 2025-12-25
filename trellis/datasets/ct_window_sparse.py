"""
CT Window Sparse Dataset for VQVAE Training

Loads CT window binarized data from preprocessing output and converts to sparse format.
Supports recursive dataset discovery and multiple window types (lung, bone, soft_tissue, brain).
"""

import os
import glob
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

# Import window configurations
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dataset_toolkits'))
from ct_preprocessing.config import WINDOW_CONFIGS
from ct_preprocessing.window_processor import get_window_filename


class CTWindowSparseSDF(Dataset):
    """
    CT Window Sparse Dataset for VQVAE training.
    
    Loads CT window binarized data and converts to sparse format on-the-fly.
    Supports recursive discovery of processed datasets.
    
    Dataset Structure:
        processed_dataset/
        ├── dataset_name/
        │   ├── metadata.csv
        │   ├── dataset_config.json
        │   └── processed/
        │       ├── case_001/
        │       │   ├── windows/
        │       │   │   ├── lung_w1500_l-600.npy
        │       │   │   ├── bone_w1500_l300.npy
        │       │   │   └── ...
        │       └── case_002/
        │           └── ...
    
    Args:
        roots (str): Root directory or comma-separated directories to search
        resolution (int): Grid resolution to use (default: 512)
        window_type (str): Window type to load ('lung', 'bone', 'soft_tissue', 'brain')
        min_points (int): Minimum number of sparse points required (default: 100)
        max_points (int): Maximum number of sparse points, will subsample if exceeded (default: 500000)
    """
    
    def __init__(
        self,
        roots: str,
        resolution: int = 512,
        window_type: str = 'lung',
        min_points: int = 100,
        max_points: int = 500000,
    ):
        super().__init__()
        
        self.resolution = resolution
        self.window_type = window_type
        self.min_points = min_points
        self.max_points = max_points
        self.value_range = (0, 1)  # For visualization - binary window values
        
        # Validate window type
        if window_type not in WINDOW_CONFIGS:
            available = ', '.join(WINDOW_CONFIGS.keys())
            raise ValueError(f"Invalid window_type: {window_type}. Available: {available}")
        
        # Get window filename
        self.window_filename = get_window_filename(window_type)
        
        # Parse roots
        self.roots = [r.strip() for r in roots.split(',')]
        
        # Discover all datasets
        self.instances = []
        self.metadata = {}
        self._discover_datasets()
        
        print(f"\nCTWindowSparseSDF Dataset:")
        print(f"  Window type: {window_type}")
        print(f"  Resolution: {resolution}")
        print(f"  Total instances: {len(self.instances)}")
        print(f"  Min points: {min_points}, Max points: {max_points}")
    
    def _discover_datasets(self):
        """
        Recursively discover all processed datasets.
        
        Searches for directories with structure:
        {root}/{dataset_name}/processed/{case_id}/
        """
        for root in self.roots:
            root = os.path.expanduser(root)
            if not os.path.exists(root):
                print(f"Warning: Root directory not found: {root}")
                continue
            
            # Find all metadata.csv files (indicates a dataset)
            metadata_files = glob.glob(os.path.join(root, '**/metadata.csv'), recursive=True)
            
            for metadata_file in metadata_files:
                dataset_root = os.path.dirname(metadata_file)
                processed_dir = os.path.join(dataset_root, 'processed')
                
                if not os.path.exists(processed_dir):
                    continue
                
                # Load metadata
                try:
                    metadata_df = pd.read_csv(metadata_file)
                except Exception as e:
                    print(f"Warning: Failed to load metadata from {metadata_file}: {e}")
                    continue
                
                # Find all case directories
                case_dirs = glob.glob(os.path.join(processed_dir, '*'))
                case_dirs = [d for d in case_dirs if os.path.isdir(d)]
                
                for case_dir in case_dirs:
                    case_id = os.path.basename(case_dir)
                    
                    # Check if window file exists
                    window_path = os.path.join(
                        case_dir, 
                        'windows', 
                        self.window_filename
                    )
                    
                    if not os.path.exists(window_path):
                        continue
                    
                    # Add to instances
                    self.instances.append({
                        'dataset_root': dataset_root,
                        'case_id': case_id,
                        'case_dir': case_dir,
                        'window_path': window_path
                    })
                    
                    # Store metadata if available
                    if 'case_id' in metadata_df.columns:
                        case_metadata = metadata_df[metadata_df['case_id'] == case_id]
                        if not case_metadata.empty:
                            self.metadata[case_id] = case_metadata.iloc[0].to_dict()
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Load and convert a single instance to sparse format.
        
        Args:
            index: Instance index
        
        Returns:
            Dictionary containing:
                - sparse_sdf: Tensor[N, 1] - Binary values (0 or 1)
                - sparse_index: Tensor[N, 3] - 3D coordinates
        """
        try:
            instance = self.instances[index]
            
            # Load window data
            window_data = np.load(instance['window_path'])
            
            # Convert to sparse format
            # Find non-zero voxels
            indices = np.argwhere(window_data > 0)
            
            # Check minimum points requirement
            if len(indices) < self.min_points:
                # Return a random valid instance instead
                return self.__getitem__(np.random.randint(0, len(self)))
            
            # Extract values at non-zero locations
            values = window_data[indices[:, 0], indices[:, 1], indices[:, 2]]
            
            # Subsample if too many points
            if self.max_points is not None and len(indices) > self.max_points:
                subsample_indices = np.random.choice(
                    len(indices), 
                    self.max_points, 
                    replace=False
                )
                indices = indices[subsample_indices]
                values = values[subsample_indices]
            
            # Convert to tensors
            sparse_sdf = torch.from_numpy(values).float().unsqueeze(-1)  # [N, 1]
            sparse_index = torch.from_numpy(indices).long()  # [N, 3]
            
            return {
                'sparse_sdf': sparse_sdf,
                'sparse_index': sparse_index,
            }
        
        except Exception as e:
            print(f"Error loading instance {index}: {e}")
            # Return a random valid instance
            return self.__getitem__(np.random.randint(0, len(self)))
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of instances into a single batch with batch indices.
        
        Args:
            batch: List of instances from __getitem__
        
        Returns:
            Dictionary containing:
                - sparse_sdf: Tensor[total_N, 1] - Concatenated SDF values
                - sparse_index: Tensor[total_N, 3] - Concatenated coordinates
                - batch_idx: Tensor[total_N] - Batch indices for each point
        """
        sparse_sdf_list = []
        sparse_index_list = []
        batch_idx_list = []
        
        for batch_idx, item in enumerate(batch):
            N = item['sparse_sdf'].shape[0]
            
            sparse_sdf_list.append(item['sparse_sdf'])
            sparse_index_list.append(item['sparse_index'])
            batch_idx_list.append(torch.full((N,), batch_idx, dtype=torch.long))
        
        return {
            'sparse_sdf': torch.cat(sparse_sdf_list, dim=0),
            'sparse_index': torch.cat(sparse_index_list, dim=0),
            'batch_idx': torch.cat(batch_idx_list, dim=0),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'num_instances': len(self.instances),
            'window_type': self.window_type,
            'resolution': self.resolution,
            'min_points': self.min_points,
            'max_points': self.max_points,
        }
    
    def __str__(self):
        stats = self.get_stats()
        lines = [
            "CTWindowSparseSDF Dataset",
            "=" * 50,
            f"Window Type: {stats['window_type']}",
            f"Resolution: {stats['resolution']}",
            f"Instances: {stats['num_instances']}",
            f"Point Range: [{stats['min_points']}, {stats['max_points']}]",
        ]
        return "\n".join(lines)

