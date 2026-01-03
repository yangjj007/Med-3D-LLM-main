# -*- coding: utf-8 -*-
"""
Mesh utilities for sparse SDF generation from Direct3D.
"""

import torch
import numpy as np
import trimesh
from typing import Dict, Union, Tuple
from skimage import measure


def normalize_mesh(mesh: trimesh.Trimesh, scale: float = 0.95) -> trimesh.Trimesh:
    """
    Normalize mesh to fit within [-scale, scale] cube.
    
    Args:
        mesh: Input trimesh object
        scale: Scale factor (default: 0.95)
    
    Returns:
        Normalized trimesh object
    """
    vertices = mesh.vertices
    min_coords, max_coords = vertices.min(axis=0), vertices.max(axis=0)
    dxyz = max_coords - min_coords
    dist = max(dxyz)
    mesh_scale = 2.0 * scale / dist
    mesh_offset = -(min_coords + max_coords) / 2
    vertices = (vertices + mesh_offset) * mesh_scale
    mesh.vertices = vertices
    return mesh


def compute_valid_udf(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    dim: int = 512,
    threshold: float = 8.0
) -> torch.Tensor:
    """
    Compute unsigned distance field (UDF) using CUDA extension.
    
    Args:
        vertices: Vertex coordinates [N, 3] on CUDA
        faces: Face indices [M, 3] on CUDA
        dim: Resolution of the UDF grid
        threshold: Distance threshold
    
    Returns:
        UDF values as 1D tensor [dim^3]
    
    Note:
        Requires udf_ext CUDA extension from Direct3D.
        Install with: cd third_party/voxelize && python setup.py install
    """
    if not faces.is_cuda or not vertices.is_cuda:
        raise ValueError("Both vertices and faces tensors must be CUDA tensors")
    
    try:
        import udf_ext
    except ImportError:
        raise ImportError(
            "udf_ext CUDA extension not found. "
            "Please compile in the third_party/voxelize directory by runing: cd third_party/voxelize && pip install -e . --no-build-isolation"
        )
    
    udf = torch.zeros(dim**3, device=vertices.device).int() + 10000000
    n_faces = faces.shape[0]
    udf_ext.compute_valid_udf(vertices, faces, udf, n_faces, dim, threshold)
    return udf.float() / 10000000.


def mesh2sparse_sdf(
    mesh: trimesh.Trimesh,
    resolution: int = 512,
    threshold_factor: float = 4.0,
    normalize: bool = True,
    scale: float = 0.95
) -> Dict[str, np.ndarray]:
    """
    Convert mesh to sparse SDF representation.
    
    Args:
        mesh: Input trimesh object
        resolution: Grid resolution
        threshold_factor: UDF threshold factor (threshold = threshold_factor / resolution)
        normalize: Whether to normalize mesh first
        scale: Normalization scale factor
    
    Returns:
        Dictionary containing:
            - sparse_sdf: SDF values [N, 1]
            - sparse_index: 3D coordinates [N, 3]
            - resolution: Grid resolution
    """
    # Normalize mesh
    if normalize:
        mesh = normalize_mesh(mesh, scale=scale)
    
    # Convert to torch tensors on GPU
    vertices = torch.Tensor(mesh.vertices).float().cuda() * 0.5
    faces = torch.Tensor(mesh.faces).int().cuda()
    
    # Compute UDF
    threshold = threshold_factor
    udf = compute_valid_udf(vertices, faces, dim=resolution, threshold=threshold)
    udf = udf.reshape(resolution, resolution, resolution)
    
    # Extract sparse coordinates where UDF < threshold/resolution
    sparse_mask = (udf < threshold_factor / resolution)
    sparse_index = sparse_mask.nonzero(as_tuple=False)  # [N, 3]
    sparse_sdf = udf[sparse_mask].unsqueeze(-1)  # [N, 1]
    
    # Convert to numpy
    result = {
        'sparse_sdf': sparse_sdf.cpu().numpy().astype(np.float32),
        'sparse_index': sparse_index.cpu().numpy().astype(np.int32),
        'resolution': resolution,
    }
    
    return result


def dense_voxel_to_sparse_sdf(
    voxel_grid: np.ndarray,
    resolution: int = 512,
    threshold_factor: float = 4.0,
    marching_cubes_level: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Convert dense 3D voxel array to sparse SDF representation.
    
    This function uses Marching Cubes to extract mesh from voxel grid,
    then computes UDF to generate sparse SDF representation with only
    points near the surface.
    
    Args:
        voxel_grid: Dense 3D binary voxel array [H, W, D]
        resolution: Target resolution (will resize if different)
        threshold_factor: UDF threshold factor for sparse extraction (default: 4.0)
        marching_cubes_level: Threshold for marching cubes (default: 0.5 for binary voxels)
    
    Returns:
        Dictionary containing:
            - sparse_sdf: SDF values [N, 1] with actual distance information
            - sparse_index: 3D coordinates [N, 3]
            - resolution: Grid resolution
    """
    import scipy.ndimage as ndimage
    
    # Ensure correct shape
    if voxel_grid.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {voxel_grid.shape}")
    
    # Resize if needed
    current_res = voxel_grid.shape[0]
    if current_res != resolution:
        zoom_factor = resolution / current_res
        voxel_grid = ndimage.zoom(
            voxel_grid.astype(np.float32),
            zoom_factor,
            order=0  # Nearest neighbor for binary data
        )
        voxel_grid = (voxel_grid > 0.5).astype(np.float32)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dense_voxel_to_sparse_sdf. Please use a GPU.")
    
    # Step 1: Voxel → Mesh using Marching Cubes
    try:
        vertices, faces, _, _ = measure.marching_cubes(
            voxel_grid, 
            level=marching_cubes_level, 
            method="lewiner"
        )
    except Exception as e:
        raise RuntimeError(f"Marching Cubes failed: {str(e)}. The voxel grid may be empty or invalid.")
    
    # Check if mesh is valid
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Marching Cubes produced empty mesh. The voxel grid may contain no occupied voxels.")
    
    # Step 2: Normalize vertices to [-0.5, 0.5] range
    vertices = vertices / resolution - 0.5
    
    # Step 3: Convert to torch tensors on CUDA
    vertices_cuda = torch.Tensor(vertices).float().cuda() * 0.5
    faces_cuda = torch.Tensor(faces).int().cuda()
    
    # Step 4: Mesh → SDF using compute_valid_udf
    udf = torch.zeros(resolution**3, device='cuda').int() + 10000000
    
    try:
        import udf_ext
        udf_ext.compute_valid_udf(
            vertices_cuda, 
            faces_cuda, 
            udf, 
            faces_cuda.shape[0], 
            resolution, 
            threshold_factor
        )
    except ImportError:
        raise ImportError(
            "udf_ext CUDA extension not found. "
            "Please compile in the third_party/voxelize directory by running: "
            "cd third_party/voxelize && pip install -e . --no-build-isolation"
        )
    
    # Convert UDF to float and reshape
    sdf = udf.float() / 10000000.0
    sdf = sdf.reshape(resolution, resolution, resolution)
    
    # Step 5: Extract sparse representation (only points near surface)
    threshold = threshold_factor / resolution
    sparse_mask = sdf < threshold
    sparse_indices = sparse_mask.nonzero(as_tuple=False)  # [N, 3]
    sparse_sdf_values = sdf[sparse_mask].unsqueeze(-1)  # [N, 1]
    
    # Step 6: Convert to numpy and return
    result = {
        'sparse_sdf': sparse_sdf_values.cpu().numpy().astype(np.float32),
        'sparse_index': sparse_indices.cpu().numpy().astype(np.int32),
        'resolution': resolution,
    }
    
    return result


def mesh2index(
    mesh: trimesh.Trimesh,
    size: int = 1024,
    factor: int = 8
) -> torch.Tensor:
    """
    Convert mesh to latent index representation (downsampled sparse coordinates).
    Used for hierarchical/progressive training.
    
    Args:
        mesh: Input trimesh object
        size: Full resolution
        factor: Downsampling factor
    
    Returns:
        Unique latent indices [N, 4] where first column is batch index (0)
    """
    vertices = torch.Tensor(mesh.vertices).float().cuda() * 0.5
    faces = torch.Tensor(mesh.faces).int().cuda()
    
    sdf = compute_valid_udf(vertices, faces, dim=size, threshold=4.0)
    sdf = sdf.reshape(size, size, size).unsqueeze(0)
    
    sparse_index = (sdf < 4/size).nonzero()
    sparse_index[..., 1:] = sparse_index[..., 1:] // factor
    latent_index = torch.unique(sparse_index, dim=0)
    
    return latent_index


# Alias for backward compatibility
mesh2latent_index = mesh2index