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
    # Use torch.from_numpy with copy to ensure contiguous arrays
    vertices = torch.from_numpy(np.array(mesh.vertices, copy=True)).float().cuda() * 0.5
    faces = torch.from_numpy(np.array(mesh.faces, copy=True)).int().cuda()
    
    # Compute UDF
    threshold = threshold_factor
    udf = compute_valid_udf(vertices, faces, dim=resolution, threshold=threshold)
    udf = udf.reshape(resolution, resolution, resolution)
    
    # Extract sparse coordinates where UDF < threshold/resolution
    sparse_mask = (udf < threshold_factor / resolution)
    sparse_index = sparse_mask.nonzero(as_tuple=False)  # [N, 3]
    sparse_sdf = udf[sparse_mask].unsqueeze(-1)  # [N, 1]
    
    # Step 6: 去重 (Deduplication)
    # Although theoretically nonzero() should return unique indices,
    # we apply deduplication to ensure no duplicates from CUDA atomic operations
    sparse_index_unique, inverse_indices = torch.unique(sparse_index, dim=0, return_inverse=True)
    
    # For duplicate indices, keep the minimum SDF value
    sparse_sdf_unique = torch.zeros(len(sparse_index_unique), 1, device=sparse_sdf.device, dtype=sparse_sdf.dtype)
    for i in range(len(sparse_index_unique)):
        mask = (inverse_indices == i)
        sparse_sdf_unique[i] = sparse_sdf[mask].min()
    
    # Convert to numpy
    result = {
        'sparse_sdf': sparse_sdf_unique.cpu().numpy().astype(np.float32),
        'sparse_index': sparse_index_unique.cpu().numpy().astype(np.int32),
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
    import sys
    
    # 调试函数
    def _debug(msg):
        print(f"[MESH_UTILS_DEBUG] {msg}", file=sys.stderr, flush=True)
    
    _debug(f"开始dense_voxel_to_sparse_sdf: voxel_grid.shape={voxel_grid.shape}, resolution={resolution}")
    _debug(f"  voxel_grid统计: dtype={voxel_grid.dtype}, min={voxel_grid.min()}, max={voxel_grid.max()}, sum={voxel_grid.sum()}")
    
    # Ensure correct shape
    if voxel_grid.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {voxel_grid.shape}")
    
    # Resize if needed
    current_res = voxel_grid.shape[0]
    if current_res != resolution:
        _debug(f"  调整分辨率: {current_res} -> {resolution}")
        zoom_factor = resolution / current_res
        voxel_grid = ndimage.zoom(
            voxel_grid.astype(np.float32),
            zoom_factor,
            order=0  # Nearest neighbor for binary data
        )
        voxel_grid = (voxel_grid > 0.5).astype(np.float32)
        _debug(f"  调整后: shape={voxel_grid.shape}, sum={voxel_grid.sum()}")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dense_voxel_to_sparse_sdf. Please use a GPU.")
    
    # CUDA内存检查
    device = torch.cuda.current_device()
    mem_before = torch.cuda.memory_allocated(device) / 1024**3
    _debug(f"  CUDA内存 (开始): {mem_before:.2f}GB")
    
    # Step 1: Voxel → Mesh using Marching Cubes
    _debug(f"  步骤1: Marching Cubes (level={marching_cubes_level})")
    try:
        vertices, faces, _, _ = measure.marching_cubes(
            voxel_grid, 
            level=marching_cubes_level, 
            method="lewiner"
        )
        _debug(f"  Marching Cubes成功: vertices={len(vertices)}, faces={len(faces)}")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        _debug(f"  Marching Cubes失败: {type(e).__name__}: {e}")
        _debug(f"  错误堆栈:\n{error_trace}")
        raise RuntimeError(f"Marching Cubes failed: {str(e)}. The voxel grid may be empty or invalid.") from e
    
    # Check if mesh is valid
    if len(vertices) == 0 or len(faces) == 0:
        _debug(f"  Marching Cubes生成空网格: vertices={len(vertices)}, faces={len(faces)}")
        raise ValueError("Marching Cubes produced empty mesh. The voxel grid may contain no occupied voxels.")
    
    # Step 2: Normalize vertices to [-0.5, 0.5] range
    _debug(f"  步骤2: 归一化顶点")
    vertices = vertices / resolution - 0.5
    _debug(f"  顶点范围: min={vertices.min()}, max={vertices.max()}")
    
    # Step 3: Convert to torch tensors on CUDA
    _debug(f"  步骤3: 转换到CUDA")
    try:
        # Make copies to avoid negative stride issues with numpy arrays from Marching Cubes
        vertices_cuda = torch.from_numpy(vertices.copy()).float().cuda() * 0.5
        faces_cuda = torch.from_numpy(faces.copy()).int().cuda()
        mem_after_mesh = torch.cuda.memory_allocated(device) / 1024**3
        _debug(f"  CUDA内存 (网格数据): {mem_after_mesh:.2f}GB (增加 {mem_after_mesh-mem_before:.2f}GB)")
    except Exception as e:
        import traceback
        _debug(f"  CUDA转换失败: {type(e).__name__}: {e}")
        _debug(f"  错误堆栈:\n{traceback.format_exc()}")
        raise RuntimeError(f"Failed to move mesh to CUDA: {e}") from e
    
    # Step 4: Mesh → SDF using compute_valid_udf
    _debug(f"  步骤4: 计算UDF (resolution={resolution}, threshold_factor={threshold_factor})")
    try:
        udf = torch.zeros(resolution**3, device='cuda').int() + 10000000
        mem_after_udf_alloc = torch.cuda.memory_allocated(device) / 1024**3
        _debug(f"  CUDA内存 (UDF分配): {mem_after_udf_alloc:.2f}GB (增加 {mem_after_udf_alloc-mem_after_mesh:.2f}GB)")
    except Exception as e:
        _debug(f"  UDF内存分配失败: {type(e).__name__}: {e}")
        raise RuntimeError(f"Failed to allocate UDF tensor: {e}") from e
    
    try:
        import udf_ext
        _debug(f"  调用udf_ext.compute_valid_udf...")
        udf_ext.compute_valid_udf(
            vertices_cuda, 
            faces_cuda, 
            udf, 
            faces_cuda.shape[0], 
            resolution, 
            threshold_factor
        )
        _debug(f"  UDF计算完成")
    except ImportError as e:
        _debug(f"  udf_ext未安装: {e}")
        raise ImportError(
            "udf_ext CUDA extension not found. "
            "Please compile in the third_party/voxelize directory by running: "
            "cd third_party/voxelize && pip install -e . --no-build-isolation"
        ) from e
    except Exception as e:
        import traceback
        _debug(f"  UDF计算失败: {type(e).__name__}: {e}")
        _debug(f"  错误堆栈:\n{traceback.format_exc()}")
        raise RuntimeError(f"UDF computation failed: {e}") from e
    
    # Convert UDF to float and reshape
    _debug(f"  步骤5: 转换UDF为SDF")
    try:
        sdf = udf.float() / 10000000.0
        sdf = sdf.reshape(resolution, resolution, resolution)
        _debug(f"  SDF统计: min={sdf.min().item():.6f}, max={sdf.max().item():.6f}, mean={sdf.mean().item():.6f}")
    except Exception as e:
        _debug(f"  SDF转换失败: {type(e).__name__}: {e}")
        raise RuntimeError(f"SDF conversion failed: {e}") from e
    
    # Step 5: Extract sparse representation (only points near surface)
    _debug(f"  步骤6: 提取稀疏表示")
    threshold = threshold_factor / resolution
    _debug(f"  阈值: {threshold:.6f}")
    sparse_mask = sdf < threshold
    sparse_count = sparse_mask.sum().item()
    _debug(f"  稀疏点数: {sparse_count}")
    
    if sparse_count == 0:
        _debug(f"  警告: 没有点低于阈值")
        raise ValueError(f"No points found below threshold {threshold:.6f}")
    
    sparse_indices = sparse_mask.nonzero(as_tuple=False)  # [N, 3]
    sparse_sdf_values = sdf[sparse_mask].unsqueeze(-1)  # [N, 1]
    
    # Step 6: 去重 (Deduplication)
    _debug(f"  步骤6.5: 去重")
    sparse_indices_unique, inverse_indices = torch.unique(sparse_indices, dim=0, return_inverse=True)
    
    # For duplicate indices, keep the minimum SDF value
    sparse_sdf_unique = torch.zeros(len(sparse_indices_unique), 1, device=sparse_sdf_values.device, dtype=sparse_sdf_values.dtype)
    for i in range(len(sparse_indices_unique)):
        mask = (inverse_indices == i)
        sparse_sdf_unique[i] = sparse_sdf_values[mask].min()
    
    _debug(f"  去重前: {len(sparse_indices)} 个点, 去重后: {len(sparse_indices_unique)} 个点")
    
    # Step 7: Convert to numpy and return
    _debug(f"  步骤7: 转换回numpy")
    try:
        result = {
            'sparse_sdf': sparse_sdf_unique.cpu().numpy().astype(np.float32),
            'sparse_index': sparse_indices_unique.cpu().numpy().astype(np.int32),
            'resolution': resolution,
        }
        _debug(f"  完成: sparse_sdf.shape={result['sparse_sdf'].shape}, sparse_index.shape={result['sparse_index'].shape}")
    except Exception as e:
        _debug(f"  numpy转换失败: {type(e).__name__}: {e}")
        raise RuntimeError(f"Failed to convert result to numpy: {e}") from e
    finally:
        # 清理CUDA内存
        try:
            del udf, sdf, sparse_mask, sparse_indices, sparse_sdf_values, vertices_cuda, faces_cuda
            torch.cuda.empty_cache()
            mem_final = torch.cuda.memory_allocated(device) / 1024**3
            _debug(f"  CUDA内存 (清理后): {mem_final:.2f}GB")
        except:
            pass
    
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
    # Use torch.from_numpy with copy to ensure contiguous arrays
    vertices = torch.from_numpy(np.array(mesh.vertices, copy=True)).float().cuda() * 0.5
    faces = torch.from_numpy(np.array(mesh.faces, copy=True)).int().cuda()
    
    sdf = compute_valid_udf(vertices, faces, dim=size, threshold=4.0)
    sdf = sdf.reshape(size, size, size).unsqueeze(0)
    
    sparse_index = (sdf < 4/size).nonzero()
    sparse_index[..., 1:] = sparse_index[..., 1:] // factor
    latent_index = torch.unique(sparse_index, dim=0)
    
    return latent_index


# Alias for backward compatibility
mesh2latent_index = mesh2index