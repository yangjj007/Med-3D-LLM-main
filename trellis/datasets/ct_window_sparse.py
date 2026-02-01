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
        self.value_range = (0, 1)  # For visualization - SDF distance values
        
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
        print(f"  使用预计算的SDF文件 (.npz格式)")
    
    def _find_mask_mode_sdf(self, case_dir: str) -> Dict[str, str]:
        """
        在masks/文件夹中查找SDF文件（支持--use_mask模式）
        
        Args:
            case_dir: case目录路径
            
        Returns:
            字典，键为label_id，值为SDF文件路径；如果未找到返回空字典
        """
        import json
        
        masks_dir = os.path.join(case_dir, 'masks')
        if not os.path.exists(masks_dir):
            return {}
        
        # 检查info.json确认是use_mask模式
        info_path = os.path.join(case_dir, 'info.json')
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    if not info.get('use_mask', False):
                        return {}
            except:
                pass
        
        # 查找organ_labels.json
        organ_labels_path = os.path.join(masks_dir, 'organ_labels.json')
        if not os.path.exists(organ_labels_path):
            # 如果没有organ_labels.json，尝试直接查找所有*_sdf.npz文件
            sdf_files = glob.glob(os.path.join(masks_dir, '*_sdf.npz'))
            if sdf_files:
                result = {}
                for sdf_path in sdf_files:
                    filename = os.path.basename(sdf_path)
                    # 提取标签ID（支持两种格式）
                    # 格式1: 1_sdf.npz -> label_id = "1"
                    # 格式2: 11_leftsurrenalgland_sdf.npz -> label_id = "11"
                    filename_without_ext = filename.replace('_sdf.npz', '')
                    # 提取第一个下划线之前的部分作为label_id（如果没有额外下划线，就是整个字符串）
                    parts = filename_without_ext.split('_')
                    label_id = parts[0]  # 只取第一个数字部分
                    result[label_id] = sdf_path
                return result
            return {}
        
        # 读取标签映射
        try:
            with open(organ_labels_path, 'r') as f:
                label_info = json.load(f)
                label_to_name = label_info.get('label_to_name', {})
        except:
            return {}
        
        # 查找所有标签的SDF文件
        result = {}
        for label_id in label_to_name.keys():
            # 首先尝试标准格式: {label_id}_sdf.npz
            sdf_path = os.path.join(masks_dir, f"{label_id}_sdf.npz")
            if os.path.exists(sdf_path):
                result[label_id] = sdf_path
            else:
                # 如果标准格式不存在，尝试查找带器官名的格式: {label_id}_*_sdf.npz
                pattern = os.path.join(masks_dir, f"{label_id}_*_sdf.npz")
                matching_files = glob.glob(pattern)
                if matching_files:
                    # 使用第一个匹配的文件
                    result[label_id] = matching_files[0]
        
        return result
    
    def _find_window_mode_sdf(self, case_dir: str) -> str:
        """
        在windows/文件夹中查找SDF文件（标准窗口模式）
        
        Args:
            case_dir: case目录路径
            
        Returns:
            SDF文件路径，如果未找到返回None
        """
        # Check if window file exists (.npy or .npz)
        window_path_npy = os.path.join(
            case_dir, 
            'windows', 
            self.window_filename
        )
        window_path_npz = window_path_npy.replace('.npy', '.npz')
        
        # 优先使用.npz文件（预计算的SDF），如果不存在则尝试.npy
        if os.path.exists(window_path_npz):
            return window_path_npz
        elif os.path.exists(window_path_npy):
            return window_path_npy
        else:
            return None
    
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
            
            # 首先检查是否使用mask模式（优先在masks/文件夹中查找）
            mask_sdf_paths = self._find_mask_mode_sdf(case_dir)
            
            # 如果找到mask模式的数据，将每个标签作为独立实例
            if mask_sdf_paths:
                for label_id, sdf_path in mask_sdf_paths.items():
                    self.instances.append({
                        'dataset_root': parent_dir,
                        'case_id': f"{case_id}_label{label_id}",
                        'case_dir': case_dir,
                        'window_path': sdf_path,
                        'mode': 'mask',
                        'label_id': label_id
                    })
                
                # Store metadata if available
                if metadata_df is not None and 'case_id' in metadata_df.columns:
                    case_metadata = metadata_df[metadata_df['case_id'] == case_id]
                    if not case_metadata.empty:
                        base_metadata = case_metadata.iloc[0].to_dict()
                        for label_id in mask_sdf_paths.keys():
                            self.metadata[f"{case_id}_label{label_id}"] = base_metadata.copy()
                continue
            
            # 如果没有找到mask模式的数据，尝试在windows/文件夹查找
            window_path = self._find_window_mode_sdf(case_dir)
            if window_path is not None:
                # Add to instances
                self.instances.append({
                    'dataset_root': parent_dir,
                    'case_id': case_id,
                    'case_dir': case_dir,
                    'window_path': window_path,
                    'mode': 'window'
                })
                
                # Store metadata if available
                if metadata_df is not None and 'case_id' in metadata_df.columns:
                    case_metadata = metadata_df[metadata_df['case_id'] == case_id]
                    if not case_metadata.empty:
                        self.metadata[case_id] = case_metadata.iloc[0].to_dict()
                continue
            
            # 如果两种模式都没找到，跳过
            print(f"  Skipping {case_id}: SDF file not found in masks/ or windows/")
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Load and convert a single instance to sparse format.
        
        Args:
            index: Instance index
        
        Returns:
            Dictionary containing:
                - sparse_sdf: Tensor[N, 1] - SDF distance values (not binary!)
                - sparse_index: Tensor[N, 3] - 3D coordinates
        """
        try:
            instance = self.instances[index]
            window_path = instance['window_path']
            
            # 确定SDF文件路径
            if window_path.endswith('.npz'):
                sdf_path = window_path  # 直接使用npz文件
            else:
                sdf_path = window_path.replace('.npy', '.npz')  # 从npy路径推导npz路径
            
            if not os.path.exists(sdf_path):
                raise FileNotFoundError(
                    f"预计算的SDF文件不存在: {sdf_path}\n"
                    f"请先运行: python scripts/precompute_ct_window_sdf.py --data_root <your_data_root>\n"
                    f"或者使用 --compute_sdf 参数在预处理时生成SDF文件"
                )
            
            # 加载SDF数据
            sdf_data = np.load(sdf_path)
            sparse_sdf = sdf_data['sparse_sdf']  # [N, 1] - 距离值
            sparse_index = sdf_data['sparse_index']  # [N, 3] - 3D坐标
            
            # 检查最小点数要求
            if len(sparse_index) < self.min_points:
                # 返回随机有效实例
                return self.__getitem__(np.random.randint(0, len(self)))
            
            # 如果点数过多，进行子采样
            if self.max_points is not None and len(sparse_index) > self.max_points:
                subsample_indices = np.random.choice(
                    len(sparse_index), 
                    self.max_points, 
                    replace=False
                )
                sparse_sdf = sparse_sdf[subsample_indices]
                sparse_index = sparse_index[subsample_indices]
            
            # 转换为张量
            sparse_sdf = torch.from_numpy(sparse_sdf.copy()).float()  # [N, 1]
            sparse_index = torch.from_numpy(sparse_index.copy()).long()  # [N, 3]
            
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

