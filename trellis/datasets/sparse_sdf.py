"""
Sparse SDF dataset for training VQVAE models.
Loads pre-computed sparse SDF data from .npz files.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List
from .components import StandardDatasetBase


class SparseSDF(StandardDatasetBase):
    """
    Sparse SDF dataset for VQVAE training.
    
    Each instance is stored as {sha256}_r{resolution}.npz containing:
        - sparse_sdf: SDF values [N, 1]
        - sparse_index: 3D coordinates [N, 3]
        - resolution: Grid resolution
    
    Args:
        roots (str): Comma-separated paths to dataset directories
        resolution (int): Grid resolution to load (default: 64)
        min_points (int): Minimum number of sparse points (default: 10)
        max_points (int): Maximum number of sparse points, will subsample if exceeded (default: None)
    """
    
    def __init__(
        self,
        roots: str,
        resolution: int = 64,
        min_points: int = 10,
        max_points: int = None,
    ):
        self.resolution = resolution
        self.min_points = min_points
        self.max_points = max_points
        self.value_range = (0, 1)  # For visualization
        
        super().__init__(roots)
    
    def filter_metadata(self, metadata):
        """Filter metadata to only include instances with sparse SDF computed."""
        stats = {}
        
        # Check if sparse_sdf_computed column exists
        if 'sparse_sdf_computed' in metadata.columns:
            metadata = metadata[metadata['sparse_sdf_computed'] == True]
            stats['Sparse SDF computed'] = len(metadata)
        
        # Check if resolution-specific point count exists
        points_col = f'r{self.resolution}_num_points'
        if points_col in metadata.columns:
            metadata = metadata[metadata[points_col] >= self.min_points]
            stats[f'Min {self.min_points} points at r{self.resolution}'] = len(metadata)
        
        return metadata, stats
    
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        """
        Load a single instance.
        
        Args:
            root: Root directory
            instance: Instance SHA256 hash
        
        Returns:
            Dictionary containing:
                - sparse_sdf: Tensor[N, 1]
                - sparse_index: Tensor[N, 3]
        """
        # Load sparse SDF data
        sdf_path = os.path.join(root, 'sparse_sdf', f'{instance}_r{self.resolution}.npz')
        data = np.load(sdf_path)
        
        sparse_sdf = torch.from_numpy(data['sparse_sdf']).float()  # [N, 1]
        sparse_index = torch.from_numpy(data['sparse_index']).long()  # [N, 3]
        
        # Subsample if too many points
        if self.max_points is not None and len(sparse_sdf) > self.max_points:
            indices = torch.randperm(len(sparse_sdf))[:self.max_points]
            sparse_sdf = sparse_sdf[indices]
            sparse_index = sparse_index[indices]
        
        return {
            'sparse_sdf': sparse_sdf,
            'sparse_index': sparse_index,
        }
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of instances.
        
        Args:
            batch: List of instances from get_instance
        
        Returns:
            Dictionary containing:
                - sparse_sdf: Tensor[total_N, 1]
                - sparse_index: Tensor[total_N, 3]
                - batch_idx: Tensor[total_N] - batch indices for each point
        """
        # Concatenate all sparse data
        sparse_sdfs = []
        sparse_indices = []
        batch_indices = []
        
        for batch_idx, item in enumerate(batch):
            sparse_sdfs.append(item['sparse_sdf'])
            sparse_indices.append(item['sparse_index'])
            batch_indices.append(
                torch.full((len(item['sparse_sdf']),), batch_idx, dtype=torch.long)
            )
        
        return {
            'sparse_sdf': torch.cat(sparse_sdfs, dim=0),
            'sparse_index': torch.cat(sparse_indices, dim=0),
            'batch_idx': torch.cat(batch_indices, dim=0),
        }
    
    @torch.no_grad()
    def visualize_sample(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Visualize sparse SDF samples by converting to dense voxel grids.
        
        Args:
            sample: Dictionary containing sparse_sdf, sparse_index, batch_idx
        
        Returns:
            Tensor of shape [B, 3, H, W] for visualization
        """
        from ..representations.octree import DfsOctree as Octree
        from ..renderers import OctreeRenderer
        import utils3d
        
        sparse_sdf = sample['sparse_sdf']
        sparse_index = sample['sparse_index']
        batch_idx = sample['batch_idx']
        
        # Determine batch size
        batch_size = int(batch_idx.max().item() + 1)
        
        # Setup renderer
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 2
        renderer.pipe.primitive = 'voxel'
        
        # Setup camera views
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        pitch = [0, 0, 0, 0]
        
        images = []
        
        for b in range(min(batch_size, 16)):  # Limit to 16 samples for memory
            # Extract points for this batch
            mask = batch_idx == b
            coords = sparse_index[mask]  # [N, 3]
            
            if len(coords) == 0:
                continue
            
            # Create octree representation
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda' if coords.is_cuda else 'cpu',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            
            # Set positions (normalized to [0, 1])
            representation.position = coords.float() / self.resolution
            representation.depth = torch.full(
                (representation.position.shape[0], 1),
                int(np.log2(self.resolution)),
                dtype=torch.uint8,
                device=representation.position.device
            )
            
            # Render 4 views
            batch_images = []
            for yaw, p in zip(yaws, pitch):
                orig = torch.tensor([
                    np.sin(yaw) * np.cos(p),
                    np.cos(yaw) * np.cos(p),
                    np.sin(p),
                ]).float() * 2
                
                if representation.position.is_cuda:
                    orig = orig.cuda()
                
                fov = torch.deg2rad(torch.tensor(30.0))
                if representation.position.is_cuda:
                    fov = fov.cuda()
                
                target = torch.zeros(3).float()
                up = torch.tensor([0, 0, 1]).float()
                if representation.position.is_cuda:
                    target = target.cuda()
                    up = up.cuda()
                
                extrinsics = utils3d.torch.extrinsics_look_at(orig, target, up)
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                
                # Render
                res = renderer.render(
                    representation,
                    extrinsics,
                    intrinsics,
                    colors_overwrite=representation.position
                )
                batch_images.append(res['color'])
            
            # Arrange in 2x2 grid
            top = torch.cat([batch_images[0], batch_images[1]], dim=2)
            bottom = torch.cat([batch_images[2], batch_images[3]], dim=2)
            grid = torch.cat([top, bottom], dim=1)
            images.append(grid)
        
        if len(images) == 0:
            # Return empty image if no valid samples
            return torch.zeros(1, 3, 512, 512)
        
        return torch.stack(images)

