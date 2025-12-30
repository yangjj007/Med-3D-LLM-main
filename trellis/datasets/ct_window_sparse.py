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
        cache_data: bool = True,  # 新增：是否缓存数据到内存
        precompute_sparse: bool = True,  # 新增：是否预计算稀疏索引
    ):
        super().__init__()
        
        self.resolution = resolution
        self.window_type = window_type
        self.min_points = min_points
        self.max_points = max_points
        self.value_range = (0, 1)  # For visualization - binary window values
        self.cache_data = cache_data
        self.precompute_sparse = precompute_sparse
        
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
        
        # 缓存机制
        self._cache = {} if cache_data else None
        self._sparse_cache = {} if precompute_sparse else None
        
        print(f"\nCTWindowSparseSDF Dataset:")
        print(f"  Window type: {window_type}")
        print(f"  Resolution: {resolution}")
        print(f"  Total instances: {len(self.instances)}")
        print(f"  Min points: {min_points}, Max points: {max_points}")
        print(f"  Cache enabled: {cache_data}")
        print(f"  Precompute sparse: {precompute_sparse}")
        
        # 预加载所有数据到内存（如果数据集不大）
        if cache_data and len(self.instances) <= 100:  # 只在数据集较小时全部缓存
            print(f"  Preloading all {len(self.instances)} instances to memory...")
            import time
            start_time = time.time()
            for idx in range(len(self.instances)):
                self._load_window_data(idx)
            elapsed = time.time() - start_time
            print(f"  Preloading completed in {elapsed:.2f}s")
    
    def _discover_datasets(self):
        """
        Recursively discover all processed datasets.
        
        Searches for directories with structure:
        {root}/{dataset_name}/processed/{case_id}/
        
        Also supports direct paths to 'processed' directories.
        """
        for root in self.roots:
            root = os.path.expanduser(root)
            if not os.path.exists(root):
                print(f"Warning: Root directory not found: {root}")
                continue
            
            # 检查是否直接指向 processed 目录
            if os.path.basename(root) == 'processed' and os.path.isdir(root):
                print(f"  Detected direct 'processed' directory: {root}")
                self._discover_in_processed_dir(root, parent_dir=os.path.dirname(root))
                continue
            
            # Find all metadata.csv files (indicates a dataset)
            metadata_files = glob.glob(os.path.join(root, '**/metadata.csv'), recursive=True)
            
            if len(metadata_files) == 0:
                print(f"  No metadata.csv found in {root}, trying direct case discovery...")
                # 尝试直接在该目录下查找case目录
                self._discover_in_processed_dir(root, parent_dir=root)
                continue
            
            for metadata_file in metadata_files:
                dataset_root = os.path.dirname(metadata_file)
                processed_dir = os.path.join(dataset_root, 'processed')
                
                if not os.path.exists(processed_dir):
                    continue
                
                # Load metadata
                try:
                    metadata_df = pd.read_csv(metadata_file)
                except Exception as e:
                    print(f"  Warning: Failed to load metadata from {metadata_file}: {e}")
                    metadata_df = None
                
                self._discover_in_processed_dir(processed_dir, dataset_root, metadata_df)
    
    def _discover_in_processed_dir(self, processed_dir, parent_dir, metadata_df=None):
        """
        在processed目录中发现case目录
        
        Args:
            processed_dir: processed目录路径
            parent_dir: 父目录（用于存储dataset_root）
            metadata_df: 可选的metadata DataFrame
        """
        # Find all case directories
        case_dirs = glob.glob(os.path.join(processed_dir, '*'))
        case_dirs = [d for d in case_dirs if os.path.isdir(d)]
        
        print(f"  Found {len(case_dirs)} case directories in {processed_dir}")
        
        for case_dir in case_dirs:
            case_id = os.path.basename(case_dir)
            
            # Check if window file exists
            window_path = os.path.join(
                case_dir, 
                'windows', 
                self.window_filename
            )
            
            if not os.path.exists(window_path):
                print(f"  Skipping {case_id}: window file not found at {window_path}")
                continue
            
            # Add to instances
            self.instances.append({
                'dataset_root': parent_dir,
                'case_id': case_id,
                'case_dir': case_dir,
                'window_path': window_path
            })
            
            # Store metadata if available
            if metadata_df is not None and 'case_id' in metadata_df.columns:
                case_metadata = metadata_df[metadata_df['case_id'] == case_id]
                if not case_metadata.empty:
                    self.metadata[case_id] = case_metadata.iloc[0].to_dict()
    
    def __len__(self):
        return len(self.instances)
    
    def _load_window_data(self, index: int):
        """
        加载窗口数据，使用内存映射或缓存。
        """
        # 检查缓存
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        
        instance = self.instances[index]
        
        # 使用mmap_mode加速加载（不立即加载到内存）
        if self._cache is None:
            window_data = np.load(instance['window_path'], mmap_mode='r')
        else:
            window_data = np.load(instance['window_path'])
            self._cache[index] = window_data
        
        return window_data
    
    def _get_sparse_indices(self, index: int, window_data: np.ndarray):
        """
        获取稀疏索引，使用缓存。
        """
        # 检查稀疏索引缓存
        if self._sparse_cache is not None:
            if index in self._sparse_cache:
                return self._sparse_cache[index]
        
        # 计算稀疏索引 - 优化的方法
        # 使用nonzero比argwhere快
        indices = np.stack(np.nonzero(window_data), axis=1)
        
        # 提取值
        if len(indices) > 0:
            values = window_data[indices[:, 0], indices[:, 1], indices[:, 2]]
        else:
            values = np.array([], dtype=window_data.dtype)
        
        # 缓存结果
        if self._sparse_cache is not None:
            self._sparse_cache[index] = (indices, values)
        
        return indices, values
    
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
            # 加载窗口数据（可能从缓存）
            window_data = self._load_window_data(index)
            
            # 获取稀疏表示（可能从缓存）
            indices, values = self._get_sparse_indices(index, window_data)
            
            # 检查最小点数要求
            if len(indices) < self.min_points:
                # 返回随机有效实例
                return self.__getitem__(np.random.randint(0, len(self)))
            
            # 如果点数过多，进行子采样
            if self.max_points is not None and len(indices) > self.max_points:
                subsample_indices = np.random.choice(
                    len(indices), 
                    self.max_points, 
                    replace=False
                )
                indices = indices[subsample_indices]
                values = values[subsample_indices]
            
            # 转换为张量（这个操作很快）
            sparse_sdf = torch.from_numpy(values.copy()).float().unsqueeze(-1)  # [N, 1]
            sparse_index = torch.from_numpy(indices.copy()).long()  # [N, 3]
            
            return {
                'sparse_sdf': sparse_sdf,
                'sparse_index': sparse_index,
            }
        
        except Exception as e:
            print(f"Error loading instance {index}: {e}")
            import traceback
            traceback.print_exc()
            # 返回随机有效实例
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
    
    @torch.no_grad()
    def visualize_sample(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Visualize sparse CT window data by converting to 2D slice grids.
        
        Creates a grid of axial, sagittal, and coronal slices for visualization.
        
        Args:
            sample: Dictionary containing sparse_sdf, sparse_index, batch_idx
        
        Returns:
            Tensor of shape [B, 1, H, W] for visualization (on the same device as input)
        """
        try:
            print("\n[DEBUG] visualize_sample called")
            print(f"[DEBUG] sample keys: {sample.keys()}")
            
            sparse_sdf = sample['sparse_sdf']
            sparse_index = sample['sparse_index']
            batch_idx = sample['batch_idx']
            
            print(f"[DEBUG] sparse_sdf shape: {sparse_sdf.shape}, dtype: {sparse_sdf.dtype}")
            print(f"[DEBUG] sparse_index shape: {sparse_index.shape}, dtype: {sparse_index.dtype}")
            print(f"[DEBUG] batch_idx shape: {batch_idx.shape}, dtype: {batch_idx.dtype}")
            
            # Get the device from input tensors (for multi-GPU compatibility)
            device = sparse_sdf.device if isinstance(sparse_sdf, torch.Tensor) else torch.device('cuda')
            print(f"[DEBUG] Using device: {device}")
            
            # Determine batch size
            if len(batch_idx) == 0:
                print("[DEBUG] batch_idx is empty! Creating dummy image")
                return torch.zeros(1, 1, 384, 512, device=device)
            
            batch_size = int(batch_idx.max().item() + 1)
            print(f"[DEBUG] batch_size determined: {batch_size}")
            
            # Parameters for visualization
            num_slices_per_axis = 4  # Show 4 slices per axis
            slice_size = 128  # Downsample to 128x128 for each slice
            
            print(f"Downsample to {slice_size}^3 for visualization") 

            images = []
            
            for b in range(min(batch_size, 16)):  # Limit to 16 samples for visualization
                print(f"[DEBUG] Processing batch {b}/{min(batch_size, 16)}")
                
                # Get data for this batch item
                mask = (batch_idx == b).squeeze()  # Ensure mask is 1D
                print(f"[DEBUG]   mask shape: {mask.shape}, sum: {mask.sum().item()}")
                
                # Handle edge case where there's no data for this batch
                if mask.sum() == 0:
                    print(f"[DEBUG]   Batch {b} has no data, creating empty grid")
                    # Create empty grid on the correct device
                    grid = np.zeros((3 * slice_size, num_slices_per_axis * slice_size), dtype=np.float32)
                    images.append(torch.from_numpy(grid).unsqueeze(0).to(device))
                    continue
                
                indices = sparse_index[mask].cpu().numpy()
                values = sparse_sdf[mask].cpu().numpy().squeeze()
                
                print(f"[DEBUG]   indices shape: {indices.shape}, values shape: {values.shape}")
                print(f"[DEBUG]   values range: [{values.min()}, {values.max()}]")
                
                # Convert sparse to dense (downsampled)
                scale = self.resolution / slice_size
                print(f"[DEBUG]   scale factor: {scale}, resolution: {self.resolution}")
                
                if scale == 0:
                    print(f"[ERROR] Scale is zero! resolution={self.resolution}, slice_size={slice_size}")
                    raise ValueError(f"Scale cannot be zero: resolution={self.resolution}, slice_size={slice_size}")
                
                dense = np.zeros((slice_size, slice_size, slice_size), dtype=np.float32)
                
                # Downsample indices
                scaled_indices = (indices / scale).astype(np.int32)
                scaled_indices = np.clip(scaled_indices, 0, slice_size - 1)
                
                print(f"[DEBUG]   scaled_indices shape: {scaled_indices.shape}")
                print(f"[DEBUG]   scaled_indices range: [{scaled_indices.min()}, {scaled_indices.max()}]")
                
                # # Fill dense array with max values at each location
                # for idx_i, (idx, val) in enumerate(zip(scaled_indices, values)):
                #     if idx_i < 3:  # Print first few for debugging
                #         print(f"[DEBUG]   Setting dense[{idx[0]}, {idx[1]}, {idx[2]}] = max(current, {val})")
                #     dense[idx[0], idx[1], idx[2]] = max(dense[idx[0], idx[1], idx[2]], val)

                # Fill dense array with max values at each location (vectorized)
                # Convert 3D indices to linear indices for efficient processing
                linear_indices = np.ravel_multi_index(
                    (scaled_indices[:, 0], scaled_indices[:, 1], scaled_indices[:, 2]),
                    (slice_size, slice_size, slice_size),
                    mode='clip'
                )

                # Use np.maximum.at for vectorized max operation
                dense_flat = dense.flatten()
                np.maximum.at(dense_flat, linear_indices, values)
                dense = dense_flat.reshape(slice_size, slice_size, slice_size)
                
                print(f"[DEBUG]   dense array filled, non-zero count: {np.count_nonzero(dense)}")
                
                # Sample slices from three axes
                # Axial (XY plane), Sagittal (YZ plane), Coronal (XZ plane)
                slice_positions = np.linspace(slice_size // 4, 3 * slice_size // 4, num_slices_per_axis, dtype=int)
                print(f"[DEBUG]   slice_positions: {slice_positions}")
                
                slices = []
                
                # Axial slices (top row)
                for z in slice_positions:
                    slices.append(dense[:, :, z])
                
                # Sagittal slices (middle row)
                for x in slice_positions:
                    slices.append(dense[x, :, :])
                
                # Coronal slices (bottom row)
                for y in slice_positions:
                    slices.append(dense[:, y, :])
                
                print(f"[DEBUG]   Created {len(slices)} slices")
                
                # Arrange slices in a grid (3 rows x num_slices_per_axis cols)
                grid = np.zeros((3 * slice_size, num_slices_per_axis * slice_size), dtype=np.float32)
                
                for i, slice_data in enumerate(slices):
                    row = i // num_slices_per_axis
                    col = i % num_slices_per_axis
                    grid[row * slice_size:(row + 1) * slice_size, 
                         col * slice_size:(col + 1) * slice_size] = slice_data
                
                print(f"[DEBUG]   Grid created with shape: {grid.shape}")
                
                # Convert to tensor [1, H, W] and move to correct device
                image = torch.from_numpy(grid).unsqueeze(0).to(device)
                images.append(image)
                print(f"[DEBUG]   Batch {b} completed, image shape: {image.shape}")
            
            # Stack batch [B, 1, H, W]
            print(f"[DEBUG] Stacking {len(images)} images")
            result = torch.stack(images)
            print(f"[DEBUG] Final result shape: {result.shape}")
            print(f"[DEBUG] Final result device: {result.device}")
            return result
            
        except Exception as e:
            print(f"\n[ERROR] Exception in visualize_sample:")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            print(f"[ERROR] Exception message: {str(e)}")
            import traceback
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            raise
    
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

