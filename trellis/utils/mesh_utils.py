# -*- coding: utf-8 -*-
"""
Mesh utilities for sparse SDF generation from Direct3D.
"""

import sys
import torch
import numpy as np
import trimesh
from typing import Dict, Union, Tuple, Any
from skimage import measure


def _import_udf_ext() -> Any:
    """Load the ``udf_ext`` CUDA extension; surface the real loader error."""
    try:
        import udf_ext as mod  # type: ignore[import-not-found]

        return mod
    except Exception as exc:
        raise ImportError(
            "无法加载 CUDA 扩展 udf_ext。\n"
            f"  当前 Python: {sys.executable}\n"
            f"  原始错误: {type(exc).__name__}: {exc}\n"
            "  常见原因: 用错了 pip（与运行脚本的 python 不是同一环境）。请用:\n"
            "    cd third_party/voxelize && python -m pip install -v -e . --no-build-isolation\n"
            "  安装后自检:\n"
            '    python -c "import udf_ext; print(udf_ext.__file__); print(hasattr(udf_ext, \'compute_valid_sdf\'))"'
        ) from exc


def make_watertight(mesh: trimesh.Trimesh, verbose: bool = False) -> trimesh.Trimesh:
    """
    Convert a triangle mesh toward watertight / manifold using pymeshfix.MeshFix.

    Uses pyvista PolyData + pymeshfix.repair() (hole filling, self-intersection cleanup).
    On failure, returns the input mesh unchanged.

    Requires: pip install pymeshfix pyvista

    Args:
        mesh: Input trimesh (typically already normalized).
        verbose: Pass through to pymeshfix repair if supported.

    Returns:
        Repaired trimesh.Trimesh, or original mesh on error.
    """
    try:
        import pymeshfix
        import pyvista as pv
    except ImportError as e:
        print(
            f"[make_watertight] pymeshfix/pyvista not installed ({e}); "
            "install with: pip install pymeshfix pyvista. Skipping watertight step."
        )
        return mesh

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh

    try:
        verts = np.asarray(mesh.vertices, dtype=np.float64, order="C")
        faces = np.asarray(mesh.faces, dtype=np.int64, order="C")
        # PyVista face array: [n_pts, v0, v1, v2, n_pts, ...]
        face_array = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
        pv_mesh = pv.PolyData(verts, face_array)

        mf = pymeshfix.MeshFix(pv_mesh)
        try:
            mf.repair(verbose=verbose)
        except TypeError:
            mf.repair()

        out = mf.mesh
        pts = np.asarray(out.points, dtype=np.float32)
        # PyVista faces: [3, a, b, c, 3, ...]
        fc = out.faces
        if fc.size == 0:
            if verbose:
                print("[make_watertight] repair produced empty mesh; keeping original.")
            return mesh
        n_faces = fc.size // 4
        tri = fc.reshape(n_faces, 4)[:, 1:4].astype(np.int64)

        repaired = trimesh.Trimesh(vertices=pts, faces=tri, process=False)
        repaired.remove_unreferenced_vertices()
        if len(repaired.vertices) == 0 or len(repaired.faces) == 0:
            if verbose:
                print("[make_watertight] repair yielded empty trimesh; keeping original.")
            return mesh
        return repaired
    except Exception as ex:
        print(f"[make_watertight] repair failed ({type(ex).__name__}: {ex}); using original mesh.")
        return mesh


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
    
    udf_ext = _import_udf_ext()
    udf = torch.zeros(dim**3, device=vertices.device).int() + 10000000
    n_faces = faces.shape[0]
    udf_ext.compute_valid_udf(vertices, faces, udf, n_faces, dim, threshold)
    return udf.float() / 10000000.


def compute_valid_sdf_packed(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    dim: int = 512,
    threshold: float = 8.0,
) -> torch.Tensor:
    """
    Compute packed signed distance field using the CUDA extension.

    The CUDA kernel updates only voxels in each triangle's local threshold
    bounding box. It packs nearest distance and sign into int64:

        packed = (int_distance << 1) | inside_bit

    where ``inside_bit == 1`` means negative SDF.
    """
    if not faces.is_cuda or not vertices.is_cuda:
        raise ValueError("Both vertices and faces tensors must be CUDA tensors")

    udf_ext = _import_udf_ext()

    if not hasattr(udf_ext, "compute_valid_sdf"):
        raise RuntimeError(
            "udf_ext.compute_valid_sdf not found. Rebuild the CUDA extension: "
            "cd third_party/voxelize && pip install -e . --no-build-isolation"
        )

    far = 10000000 << 1
    packed = torch.full((dim**3,), far, device=vertices.device, dtype=torch.long)
    n_faces = faces.shape[0]
    udf_ext.compute_valid_sdf(vertices, faces, packed, n_faces, dim, threshold)
    return packed


def _point_tri_closest_and_dist2(
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorised closest-point-on-triangle (Ericson, "Real-Time Collision Detection").

    Args:
        p: query points,  shape [..., 3]   (broadcastable with a/b/c)
        a, b, c: triangle vertices, shape [..., 3]

    Returns:
        closest:  closest point on triangle to ``p``,  shape [..., 3]
        dist2:    squared distance,                    shape [...]
    """
    eps = 1e-30
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)

    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)

    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    denom = 1.0 / (va + vb + vc + eps)
    v = vb * denom
    w = vc * denom

    region_A  = (d1 <= 0) & (d2 <= 0)
    region_B  = (d3 >= 0) & (d4 <= d3)
    region_C  = (d6 >= 0) & (d5 <= d6)
    region_AB = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~region_A & ~region_B
    region_AC = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~region_A & ~region_C
    region_BC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~region_B & ~region_C

    v_AB = d1 / (d1 - d3 + eps)
    w_AC = d2 / (d2 - d6 + eps)
    w_BC = (d4 - d3) / ((d4 - d3) + (d5 - d6) + eps)

    zero = torch.zeros_like(v)
    one = torch.ones_like(v)
    v = torch.where(region_A,  zero, v)
    w = torch.where(region_A,  zero, w)
    v = torch.where(region_B,  one,  v)
    w = torch.where(region_B,  zero, w)
    v = torch.where(region_C,  zero, v)
    w = torch.where(region_C,  one,  w)
    v = torch.where(region_AB, v_AB, v)
    w = torch.where(region_AB, zero, w)
    v = torch.where(region_AC, zero, v)
    w = torch.where(region_AC, w_AC, w)
    v = torch.where(region_BC, 1.0 - w_BC, v)
    w = torch.where(region_BC, w_BC, w)

    closest = a + v.unsqueeze(-1) * ab + w.unsqueeze(-1) * ac
    diff = p - closest
    dist2 = (diff * diff).sum(-1)
    return closest, dist2


@torch.no_grad()
def _compute_sparse_sdf_signs_gpu(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor,
    query_points: torch.Tensor,
    point_chunk: int = 8192,
    face_chunk: int = 4096,
) -> torch.Tensor:
    """
    GPU implementation of nearest-triangle sign test.

    Brute-force search of the closest mesh triangle for every query point,
    chunked so that intermediate ``[P, F]`` buffers fit in GPU memory.  For
    millions of query points and ~10^4-10^5 triangles this is orders of
    magnitude faster than ``trimesh.proximity.closest_point`` (CPU/Python).

    Args:
        vertices:     [V, 3] float, on CUDA
        faces:        [F, 3] long,  on CUDA
        face_normals: [F, 3] float, on CUDA (already normalised, outward)
        query_points: [N, 3] float, on CUDA
        point_chunk:  number of points per outer-loop chunk
        face_chunk:   number of triangles per inner-loop chunk

    Returns:
        ``[N]`` bool tensor: True where the query point is inside the surface.
    """
    device = query_points.device
    n_points = query_points.shape[0]
    if n_points == 0:
        return torch.zeros((0,), dtype=torch.bool, device=device)

    a_all = vertices[faces[:, 0]].contiguous()  # [F, 3]
    b_all = vertices[faces[:, 1]].contiguous()
    c_all = vertices[faces[:, 2]].contiguous()
    n_faces = a_all.shape[0]

    inside = torch.zeros(n_points, dtype=torch.bool, device=device)
    inf_const = torch.full((1,), float("inf"), device=device, dtype=query_points.dtype).item()

    for ps in range(0, n_points, point_chunk):
        pe = min(ps + point_chunk, n_points)
        pts = query_points[ps:pe]                          # [P, 3]
        n_pts = pts.shape[0]

        best_d2 = torch.full((n_pts,), inf_const, device=device, dtype=pts.dtype)
        best_face = torch.zeros(n_pts, dtype=torch.long, device=device)

        for fs in range(0, n_faces, face_chunk):
            fe = min(fs + face_chunk, n_faces)
            a = a_all[fs:fe].unsqueeze(0)                  # [1, Fc, 3]
            b = b_all[fs:fe].unsqueeze(0)
            c = c_all[fs:fe].unsqueeze(0)
            p_b = pts.unsqueeze(1)                         # [P, 1, 3]
            _, d2 = _point_tri_closest_and_dist2(p_b, a, b, c)  # [P, Fc]
            d2_min, d2_idx = d2.min(dim=1)
            update = d2_min < best_d2
            best_d2 = torch.where(update, d2_min, best_d2)
            best_face = torch.where(update, fs + d2_idx, best_face)

        a = a_all[best_face]
        b = b_all[best_face]
        c = c_all[best_face]
        closest, _ = _point_tri_closest_and_dist2(pts, a, b, c)
        normals = face_normals[best_face]
        signed_offset = ((pts - closest) * normals).sum(-1)
        inside[ps:pe] = signed_offset < 0.0

    return inside


def _compute_sparse_sdf_signs_by_normals(
    mesh: trimesh.Trimesh,
    query_points: torch.Tensor,
    point_chunk: int = 8192,
    face_chunk: int = 4096,
) -> torch.Tensor:
    """
    GPU sign estimator: for each query point find its closest mesh triangle
    and use ``sign = dot(query_point - closest_surface_point, face_normal)``.

    With outward face normals this is positive outside and negative inside.
    Inputs and outputs are CUDA tensors so the caller can stay on GPU.

    Args:
        mesh:         input trimesh.Trimesh
        query_points: [N, 3] float CUDA tensor in the same coordinate frame
                      as ``mesh.vertices``

    Returns:
        ``[N]`` bool CUDA tensor: True where the query point is inside.
    """
    if not query_points.is_cuda:
        raise RuntimeError(
            "_compute_sparse_sdf_signs_by_normals expects a CUDA query_points tensor"
        )
    if query_points.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=query_points.device)

    try:
        mesh.fix_normals()
    except Exception:
        pass

    device = query_points.device
    dtype = query_points.dtype
    vertices_t = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(
        device=device, dtype=dtype
    )
    faces_t = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)).to(device)
    face_normals_t = torch.from_numpy(np.asarray(mesh.face_normals, dtype=np.float32)).to(
        device=device, dtype=dtype
    )

    return _compute_sparse_sdf_signs_gpu(
        vertices_t,
        faces_t,
        face_normals_t,
        query_points,
        point_chunk=point_chunk,
        face_chunk=face_chunk,
    )


def mesh2sparse_sdf(
    mesh: trimesh.Trimesh,
    resolution: int = 512,
    threshold_factor: float = 4.0,
    normalize: bool = True,
    scale: float = 0.95,
    watertight: bool = True,
    watertight_verbose: bool = True,
    compute_edge_mask: bool = True,
    sharp_grad_dev_thresh: float = 0.3,
) -> Dict[str, np.ndarray]:
    """
    Convert mesh to sparse **signed** SDF representation.

    Each sparse voxel stores a signed distance value normalized to [-1, 1]:
      - Negative values  → voxel center is inside the mesh
      - Zero             → voxel is on the geometric surface
      - Positive values  → voxel center is outside the mesh
      - Background (not stored) = 1.0 (far exterior)

    The sign is estimated by a nearest-triangle normal test for speed.  Values
    are normalized by the shell half-width
    ``udf_max = threshold_factor / resolution`` so that the deepest shell
    voxels map to ±1 and surface-crossing voxels map to ~0.  This matches
    the decoder's SparseTanh output range [-1, 1] and allows Marching Cubes
    at iso-level 0.0 to extract a smooth surface.

    ``threshold_factor`` controls the sparse band half-width in voxels.  A
    larger value is required so that decoder-predicted "extra" voxels (voxels
    beyond the input sparse set) still have a valid GT SDF to supervise against.
    Recommended value ≥ 4 (4-voxel shell = good coverage for typical decoders).

    Args:
        mesh: Input trimesh object
        resolution: Grid resolution
        threshold_factor: Shell half-width in voxels.  Set ≥ 4.0 so the GT SDF
            covers decoder-predicted extra voxels.
        normalize: Whether to normalize mesh into [-scale, scale] first
        scale: Normalization scale factor (default 0.95)
        watertight: Deprecated compatibility argument; no repair is run here.
        watertight_verbose: Deprecated compatibility argument.
        compute_edge_mask: If True (default) and udf_ext.compute_sharp_mask is
            available, run the GPU sharp-edge kernel and return ``edge_mask``.
            Falls back gracefully (all-False mask + warning) when unavailable.
        sharp_grad_dev_thresh: Threshold for |1 - |∇SDF||; voxels exceeding
            this value are marked as sharp/edge (default 0.3).

    Returns:
        Dictionary containing:
            - sparse_sdf:        Signed SDF values in [-1, 1],  shape [N, 1]  (float32)
            - sparse_index:      3-D grid coordinates,          shape [N, 3]  (int32)
            - edge_mask:         Sharp-edge boolean mask,        shape [N]     (bool)
            - extra_band_factor: threshold_factor value stored as metadata     (float32 scalar)
            - resolution:        Grid resolution (int)
    """
    # Normalize mesh
    if normalize:
        mesh = normalize_mesh(mesh, scale=scale)

    try:
        mesh.fix_normals()
    except Exception as exc:
        raise RuntimeError(f"[mesh2sparse_sdf] failed to fix mesh normals: {exc}") from exc

    # Convert to torch tensors on GPU
    vertices = torch.from_numpy(np.array(mesh.vertices, copy=True)).float().cuda() * 0.5
    faces = torch.from_numpy(np.array(mesh.faces, copy=True)).int().cuda()

    # Compute signed SDF in the CUDA voxelization pass.
    # packed = (int_distance << 1) | inside_bit
    packed = compute_valid_sdf_packed(vertices, faces, dim=resolution, threshold=threshold_factor)
    packed_3d = packed.reshape(resolution, resolution, resolution)

    # Extract sparse shell: voxels within threshold_factor / resolution of the surface.
    # Work in the packed integer domain to avoid a dense float conversion.
    udf_max = threshold_factor / resolution
    threshold_int = int(udf_max * 10000000)
    dist_int = packed_3d >> 1
    sparse_mask = dist_int < threshold_int
    sparse_index = sparse_mask.nonzero(as_tuple=False)   # [N, 3]
    packed_sparse = packed_3d[sparse_mask].unsqueeze(-1)    # [N, 1]
    sparse_sdf = (packed_sparse >> 1).float() / 10000000.0
    inside_t = (packed_sparse & 1).bool()
    sparse_sdf[inside_t] = -sparse_sdf[inside_t]
    sparse_sdf = (sparse_sdf / udf_max).clamp(-1.0, 1.0)

    # ---- de-duplication ----
    num_before = sparse_index.shape[0]
    unique_idx = torch.unique(sparse_index, dim=0)
    num_after  = unique_idx.shape[0]
    if num_before != num_after:
        print(f"\n⚠️  [mesh2sparse_sdf] 发现重复点: {num_before} → {num_after}")
        sparse_index = unique_idx
        packed_sparse = packed_3d[sparse_index[:, 0], sparse_index[:, 1], sparse_index[:, 2]].unsqueeze(-1)
        sparse_sdf = (packed_sparse >> 1).float() / 10000000.0
        inside_t = (packed_sparse & 1).bool()
        sparse_sdf[inside_t] = -sparse_sdf[inside_t]
        sparse_sdf = (sparse_sdf / udf_max).clamp(-1.0, 1.0)
    else:
        print(f"✓ [mesh2sparse_sdf] 无重复点 (共 {num_before} 个点, "
              f"resolution={resolution}, udf_max={udf_max:.6f})")

    inside_count = int((sparse_sdf < 0).sum().item())
    outside_count = int(sparse_sdf.numel() - inside_count)
    print(f"✓ [mesh2sparse_sdf] 有符号SDF: interior={inside_count}, exterior={outside_count}, "
          f"range=[{sparse_sdf.min().item():.3f}, {sparse_sdf.max().item():.3f}]")

    # ---- GPU sharp-edge mask ----
    # Run compute_sharp_kernel on the same packed buffer; no extra mesh/CPU ops.
    edge_mask_np = np.zeros(sparse_index.shape[0], dtype=np.bool_)
    if compute_edge_mask:
        try:
            udf_ext = _import_udf_ext()
            if hasattr(udf_ext, 'compute_sharp_mask'):
                # band: normalised SDF units (same as sparse_sdf values), full band width
                band_norm = 1.0  # cover the entire [-1, 1] sparse band
                sharp_mask_dense = torch.zeros(
                    resolution ** 3, device=packed.device, dtype=torch.uint8
                )
                udf_ext.compute_sharp_mask(
                    packed,          # flat [DIM^3] int64
                    sharp_mask_dense,
                    resolution,
                    band_norm,
                    sharp_grad_dev_thresh,
                )
                sharp_mask_3d = sharp_mask_dense.reshape(resolution, resolution, resolution)
                # Extract sparse subset aligned to sparse_index
                edge_mask_sparse = sharp_mask_3d[
                    sparse_index[:, 0], sparse_index[:, 1], sparse_index[:, 2]
                ].bool()
                sharp_count = int(edge_mask_sparse.sum().item())
                total_n = sparse_index.shape[0]
                print(
                    f"✓ [mesh2sparse_sdf] edge_mask 计算完成: "
                    f"sharp={sharp_count}/{total_n} ({100.0*sharp_count/max(total_n,1):.2f}%), "
                    f"grad_dev_thresh={sharp_grad_dev_thresh}"
                )
                edge_mask_np = edge_mask_sparse.cpu().numpy().astype(np.bool_)
            else:
                print(
                    "[mesh2sparse_sdf] ⚠️  udf_ext 缺少 compute_sharp_mask "
                    "(需重新编译 third_party/voxelize); edge_mask 退化为全 False"
                )
        except Exception as exc:
            print(f"[mesh2sparse_sdf] ⚠️  edge_mask 计算失败 ({exc}); 退化为全 False")

    # Convert to numpy
    return {
        'sparse_sdf':        sparse_sdf.cpu().numpy().astype(np.float32),
        'sparse_index':      sparse_index.cpu().numpy().astype(np.int32),
        'edge_mask':         edge_mask_np,
        'extra_band_factor': np.float32(threshold_factor),
        'resolution':        resolution,
    }


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
        udf_ext = _import_udf_ext()
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
    except ImportError:
        raise
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
    
    # ============ 去重检查与调试 ============
    num_points_before = sparse_indices.shape[0]
    unique_indices = torch.unique(sparse_indices, dim=0)
    num_points_after = unique_indices.shape[0]
    
    if num_points_before != num_points_after:
        _debug(f"  ⚠️  发现重复点!")
        _debug(f"    去重前: {num_points_before} 个点")
        _debug(f"    去重后: {num_points_after} 个点")
        _debug(f"    重复数: {num_points_before - num_points_after} 个 ({(num_points_before-num_points_after)/num_points_before*100:.2f}%)")
        # 使用去重后的索引
        sparse_indices = unique_indices
        # 重新从SDF中提取对应位置的值
        sparse_sdf_values = sdf[sparse_indices[:, 0], sparse_indices[:, 1], sparse_indices[:, 2]].unsqueeze(-1)
    else:
        _debug(f"  ✓ 无重复点 (共 {num_points_before} 个点)")
    # ========================================
    
    # Step 7: Convert to numpy and return
    _debug(f"  步骤8: 转换回numpy")
    try:
        result = {
            'sparse_sdf': sparse_sdf_values.cpu().numpy().astype(np.float32),
            'sparse_index': sparse_indices.cpu().numpy().astype(np.int32),
            'resolution': resolution,
        }
        _debug(f"  步骤9完成: sparse_sdf.shape={result['sparse_sdf'].shape}, sparse_index.shape={result['sparse_index'].shape}")
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


# def mesh2index(
#     mesh: trimesh.Trimesh,
#     size: int = 1024,
#     factor: int = 8
# ) -> torch.Tensor:
#     """
#     Convert mesh to latent index representation (downsampled sparse coordinates).
#     Used for hierarchical/progressive training.
    
#     Args:
#         mesh: Input trimesh object
#         size: Full resolution
#         factor: Downsampling factor
    
#     Returns:
#         Unique latent indices [N, 4] where first column is batch index (0)
#     """
#     # Use torch.from_numpy with copy to ensure contiguous arrays
#     vertices = torch.from_numpy(np.array(mesh.vertices, copy=True)).float().cuda() * 0.5
#     faces = torch.from_numpy(np.array(mesh.faces, copy=True)).int().cuda()
    
#     sdf = compute_valid_udf(vertices, faces, dim=size, threshold=4.0)
#     sdf = sdf.reshape(size, size, size).unsqueeze(0)
    
#     sparse_index = (sdf < 4/size).nonzero()
#     sparse_index[..., 1:] = sparse_index[..., 1:] // factor
#     latent_index = torch.unique(sparse_index, dim=0)
    
#     return latent_index


# # Alias for backward compatibility
# mesh2latent_index = mesh2index