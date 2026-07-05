"""
Sparse signed SDF → trimesh via dense fill + Marching Cubes (aligned with eval sparse_mesh_export).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import trimesh
from skimage import measure


def _sparse_indices_to_mesh(
    sparse_sdf_i: torch.Tensor,
    sparse_index_i: torch.Tensor,
    voxel_resolution: int,
    mc_threshold: float,
) -> Optional[trimesh.Trimesh]:
    """One batch element: (N,), (N, 3) int indices → trimesh or None."""
    if sparse_sdf_i.numel() == 0:
        return None

    sparse_index_i = sparse_index_i.long()
    valid = ((sparse_index_i >= 0) & (sparse_index_i < voxel_resolution)).all(dim=1)
    if not bool(valid.all()):
        sparse_sdf_i = sparse_sdf_i[valid]
        sparse_index_i = sparse_index_i[valid]
    if sparse_sdf_i.numel() == 0:
        return None

    sdf = torch.ones(
        (voxel_resolution, voxel_resolution, voxel_resolution),
        dtype=torch.float32,
    )
    sdf[
        sparse_index_i[:, 0],
        sparse_index_i[:, 1],
        sparse_index_i[:, 2],
    ] = sparse_sdf_i
    try:
        vertices, faces, _, _ = measure.marching_cubes(
            sdf.numpy(),
            level=float(mc_threshold),
            method="lewiner",
        )
    except ValueError:
        return None
    vertices = vertices / float(voxel_resolution) * 2.0 - 1.0
    return trimesh.Trimesh(vertices, faces, process=False)


def sparse_sample_dict_to_trimeshes(
    sample: Dict[str, torch.Tensor],
    voxel_resolution: int,
    mc_threshold: float = 0.0,
) -> List[Optional[trimesh.Trimesh]]:
    """
    Collated sparse SDF batch dict → list of meshes (eval-compatible).

    Keys: ``sparse_sdf`` [N, 1], ``sparse_index`` [N, 3], ``batch_idx`` [N].
    """
    sparse_sdf = sample["sparse_sdf"]
    sparse_index = sample["sparse_index"]
    batch_idx = sample["batch_idx"]

    if batch_idx.numel() == 0:
        return []

    batch_size = int(batch_idx.max().item() + 1)
    meshes: List[Optional[trimesh.Trimesh]] = []
    for b in range(batch_size):
        mask = batch_idx == b
        s = sparse_sdf[mask].float().squeeze(-1).detach().cpu()
        idx = sparse_index[mask].detach().cpu().long()
        meshes.append(_sparse_indices_to_mesh(s, idx, voxel_resolution, mc_threshold))
    return meshes


def sparse_feats_coords_to_trimeshes(
    feats: torch.Tensor,
    coords: torch.Tensor,
    voxel_resolution: int,
    mc_threshold: float = 0.0,
) -> List[Optional[trimesh.Trimesh]]:
    """
    SparseTensor-style ``feats`` [N, C] and ``coords`` [N, 4] (batch, z, y, x or batch + 3 spatial).

    Matches ``SparseSDFVQVAE.sparse2mesh`` layout: spatial columns are ``coords[:, 1:4]``.
    """
    if coords.numel() == 0:
        return []

    batch_size = int(coords[:, 0].max().detach().cpu().item() + 1)
    feats = feats.float()
    meshes: List[Optional[trimesh.Trimesh]] = []
    for i in range(batch_size):
        idx = coords[:, 0] == i
        sparse_sdf_i = feats[idx].squeeze(-1).detach().cpu()
        sparse_index_i = coords[idx, 1:4].detach().cpu().long()
        meshes.append(_sparse_indices_to_mesh(sparse_sdf_i, sparse_index_i, voxel_resolution, mc_threshold))
    return meshes
