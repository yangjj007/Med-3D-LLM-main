"""
Variable-length 3D tokenization: use all VAE output points (no grid pooling).
- Morton code (Z-order) sorting so adjacent tokens are spatially adjacent.
- Optional FPS downsample only when exceeding max_safe_length (soft cap).
- No fixed 512; sequence length = N (typically 8k~12k) with dynamic padding in batch.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch


# Default soft cap: only trigger FPS when point count exceeds this (e.g. noisy outliers)
DEFAULT_MAX_SAFE_LENGTH = 15000


def _morton_code_3d(x: int, y: int, z: int, bits: int = 10) -> int:
    """Compute Morton (Z-order) code for (x, y, z), each in [0, 2^bits - 1]."""
    code = 0
    for i in range(bits):
        code |= ((x >> i) & 1) << (3 * i)
        code |= ((y >> i) & 1) << (3 * i + 1)
        code |= ((z >> i) & 1) << (3 * i + 2)
    return code


def morton_sort_indices(coords_xyz: np.ndarray, coord_max: int = 512) -> np.ndarray:
    """
    Return permutation indices that sort points by Morton (Z-order) curve.
    coords_xyz: [N, 3] int, (x, y, z). coord_max: max coordinate value (e.g. 512 or 8).
    """
    n = coords_xyz.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    bits = max(1, int(np.ceil(np.log2(coord_max + 1))))
    codes = np.zeros(n, dtype=np.int64)
    for i in range(n):
        x, y, z = int(coords_xyz[i, 0]), int(coords_xyz[i, 1]), int(coords_xyz[i, 2])
        codes[i] = _morton_code_3d(x, y, z, bits)
    return np.argsort(codes)


def fps_downsample_indices(
    coords_xyz: torch.Tensor,
    num_sample: int,
    start_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Farthest Point Sampling: return indices of num_sample points.
    coords_xyz: [N, 3]. Returns long tensor of shape [num_sample].
    """
    N = coords_xyz.shape[0]
    if N <= num_sample:
        return torch.arange(N, device=coords_xyz.device, dtype=torch.long)
    if start_idx is None:
        start_idx = 0
    selected = [start_idx]
    pts = coords_xyz.float()
    for _ in range(num_sample - 1):
        # [N, 3] vs [len(selected), 3] -> min dist [N]
        chosen = pts[selected]
        dists = torch.cdist(pts, chosen).min(dim=1).values
        # Farthest from current set
        next_idx = dists.argmax().item()
        selected.append(next_idx)
    return torch.tensor(selected, device=coords_xyz.device, dtype=torch.long)


def encoding_indices_to_variable_length_sequence(
    encoding_indices: "SparseTensor",
    batch_idx: int,
    max_safe_length: int = DEFAULT_MAX_SAFE_LENGTH,
    coord_max: int = 64,
) -> np.ndarray:
    """
    From encoding_indices for one sample, produce variable-length token index sequence:
    1. Extract (indices, coords) for this batch_idx.
    2. If N > max_safe_length, FPS downsample to max_safe_length (soft cap).
    3. Morton sort by (x,y,z) so adjacent tokens are spatially adjacent.
    4. Return 1D int64 array of codebook indices, length L (variable).

    Args:
        encoding_indices: SparseTensor .feats [N], .coords [N, 4] (batch_idx, x, y, z).
        batch_idx: which sample.
        max_safe_length: only FPS-downsample when N > this (e.g. 15000).
        coord_max: max coordinate value for Morton (current trellis VAE: 64^3 latent -> 64;
                  if you use a VAE that outputs 512^3 coords, set 512).

    Returns:
        indices: [L] int64, L = min(N, max_safe_length), values 0..8191.
    """
    indices = encoding_indices.feats.squeeze(-1).long()
    coords = encoding_indices.coords
    if coords.shape[0] == 0:
        return np.array([], dtype=np.int64)

    mask = coords[:, 0] == batch_idx
    if not mask.any():
        return np.array([], dtype=np.int64)

    idx_b = indices[mask]
    xyz_b = coords[mask][:, 1:4]

    N = idx_b.shape[0]
    if N > max_safe_length:
        xyz_cpu = xyz_b.cpu().float()
        fps_idx = fps_downsample_indices(xyz_cpu, max_safe_length)
        idx_b = idx_b[fps_idx]
        xyz_b = xyz_b[fps_idx]
        N = idx_b.shape[0]

    xyz_np = xyz_b.cpu().numpy().astype(np.int64)
    order = morton_sort_indices(xyz_np, coord_max=coord_max)
    idx_np = idx_b.cpu().numpy()
    return idx_np[order]


def variable_length_sequence_to_mesh_token_string(indices: np.ndarray) -> str:
    """Convert variable-length index array to mesh token string (no padding here; batch will pad)."""
    parts = ["<mesh_start>"]
    for v in indices:
        parts.append(f"<mesh_{int(v)}>")
    parts.append("<mesh_end>")
    return "".join(parts)


def batch_encoding_indices_to_variable_length_sequences(
    encoding_indices: "SparseTensor",
    batch_size: int,
    max_safe_length: int = DEFAULT_MAX_SAFE_LENGTH,
    coord_max: int = 64,
) -> List[np.ndarray]:
    """For batched encoding_indices, return list of variable-length index arrays."""
    return [
        encoding_indices_to_variable_length_sequence(
            encoding_indices, b, max_safe_length=max_safe_length, coord_max=coord_max
        )
        for b in range(batch_size)
    ]
