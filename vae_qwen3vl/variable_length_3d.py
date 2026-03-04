"""
Variable-length 3D tokenization: use all VAE output points (no grid pooling).
- Morton code (Z-order) sorting so adjacent tokens are spatially adjacent.
- When N > max_safe_length: skip sample (return None), do NOT use for training.
- No fixed 512; sequence length = N (typically 8k~12k) with dynamic padding in batch.
"""

from typing import List, Optional
import time
import numpy as np
import torch


# Hard cap: samples with N > this are skipped (not used for training)
DEFAULT_MAX_SAFE_LENGTH = 15000


def morton_sort_indices(coords_xyz: np.ndarray, coord_max: int = 512) -> np.ndarray:
    """
    Return permutation indices that sort points by Morton (Z-order) curve.
    Vectorized: no per-point Python loop.
    coords_xyz: [N, 3] int, (x, y, z). coord_max: max coordinate value (e.g. 512 or 64).
    """
    n = coords_xyz.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    bits = max(1, int(np.ceil(np.log2(coord_max + 1))))
    x = coords_xyz[:, 0].astype(np.int64)
    y = coords_xyz[:, 1].astype(np.int64)
    z = coords_xyz[:, 2].astype(np.int64)
    codes = np.zeros(n, dtype=np.int64)
    for i in range(bits):
        codes |= ((x >> i) & 1) << (3 * i)
        codes |= ((y >> i) & 1) << (3 * i + 1)
        codes |= ((z >> i) & 1) << (3 * i + 2)
    return np.argsort(codes)


def encoding_indices_to_variable_length_sequence(
    encoding_indices: "SparseTensor",
    batch_idx: int,
    max_safe_length: int = DEFAULT_MAX_SAFE_LENGTH,
    coord_max: int = 64,
) -> Optional[np.ndarray]:
    """
    From encoding_indices for one sample, produce variable-length token index sequence:
    1. Extract (indices, coords) for this batch_idx.
    2. If N > max_safe_length: return None (sample skipped, not used for training).
    3. Morton sort by (x,y,z) so adjacent tokens are spatially adjacent.
    4. Return 1D int64 array of codebook indices, length L (variable).

    Args:
        encoding_indices: SparseTensor .feats [N], .coords [N, 4] (batch_idx, x, y, z).
        batch_idx: which sample.
        max_safe_length: samples with N > this are skipped (return None).
        coord_max: max coordinate value for Morton (current trellis VAE: 64^3 latent -> 64;
                  if you use a VAE that outputs 512^3 coords, set 512).

    Returns:
        indices: [L] int64, values 0..8191; or None if N > max_safe_length (skip).
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
    print(f"[DEBUG varlen] batch_idx={batch_idx} N={N} (max_safe={max_safe_length})", flush=True)
    if N > max_safe_length:
        print(
            f"[SKIP] batch_idx={batch_idx} 3D token count N={N} exceeds max_safe_3d_length={max_safe_length}, "
            "skipping this sample (not used for training).",
            flush=True,
        )
        return None

    t0 = time.time()
    xyz_np = xyz_b.cpu().numpy().astype(np.int64)
    order = morton_sort_indices(xyz_np, coord_max=coord_max)
    idx_np = idx_b.cpu().numpy()
    print(f"[DEBUG varlen]   Morton sort {N} pts took {time.time()-t0:.3f}s", flush=True)
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
) -> List[Optional[np.ndarray]]:
    """For batched encoding_indices, return list of variable-length index arrays (None = skip)."""
    return [
        encoding_indices_to_variable_length_sequence(
            encoding_indices, b, max_safe_length=max_safe_length, coord_max=coord_max
        )
        for b in range(batch_size)
    ]
