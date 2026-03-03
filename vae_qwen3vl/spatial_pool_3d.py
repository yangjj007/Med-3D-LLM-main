"""
8x8x8 spatial pooling for 3D latent: aggregate VAE encoding_indices into a fixed
512-token grid (one token per cell) via majority vote. Used for discrete-token
3D-VL alignment (no Projector); output is flattened in (z, y, x) order to match
existing sequence_3d lexsort convention.
"""

from typing import Tuple, Union, List
import torch
import numpy as np


# Sentinel for empty grid cells (no point in that cell). Map to <mesh_empty> when textifying.
MESH_EMPTY = -1

GRID_SIZE = 8
POOLED_LEN = 8 * 8 * 8  # 512


def _majority_vote(indices: torch.Tensor) -> int:
    """Return the most frequent value; if empty return MESH_EMPTY."""
    if indices.numel() == 0:
        return MESH_EMPTY
    indices_np = indices.cpu().numpy().ravel()
    counts = np.bincount(indices_np.astype(np.int64), minlength=8192)
    winner = int(np.argmax(counts))
    return winner


def spatial_pool_8x8x8(
    indices: torch.Tensor,
    coords_xyz: torch.Tensor,
) -> np.ndarray:
    """
    Aggregate points into an 8x8x8 grid by majority vote per cell.
    coords_xyz: [N, 3] with values in 0..7 (x, y, z).
    indices: [N] with values in 0..8191 (codebook indices).

    Returns:
        flat: length-512 int64 array, values in 0..8191 or MESH_EMPTY (-1).
        Order: (z, y, x) i.e. flat[k] = grid[z, y, x] with k = z*64 + y*8 + x.
    """
    grid = np.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), MESH_EMPTY, dtype=np.int64)
    if indices.numel() == 0:
        return grid.ravel(order="C")  # C order: last index (x) varies fastest -> z,y,x

    idx = indices.cpu().numpy().ravel()
    xyz = coords_xyz.cpu().numpy()  # [N, 3]
    if xyz.shape[0] != idx.shape[0]:
        xyz = xyz.reshape(-1, 3)[: idx.shape[0]]

    x, y, z = xyz[:, 0].astype(np.int64), xyz[:, 1].astype(np.int64), xyz[:, 2].astype(np.int64)
    np.clip(x, 0, GRID_SIZE - 1, out=x)
    np.clip(y, 0, GRID_SIZE - 1, out=y)
    np.clip(z, 0, GRID_SIZE - 1, out=z)

    # Group by (z, y, x) and majority vote
    for gz in range(GRID_SIZE):
        for gy in range(GRID_SIZE):
            for gx in range(GRID_SIZE):
                mask = (z == gz) & (y == gy) & (x == gx)
                if not np.any(mask):
                    continue
                grid[gz, gy, gx] = _majority_vote(torch.from_numpy(idx[mask]))

    # Flatten in (z, y, x) order: index k = z*64 + y*8 + x
    flat = grid.ravel(order="C")
    return flat


def encoding_indices_to_pooled_sequence(
    encoding_indices: "SparseTensor",
    batch_idx: int,
) -> np.ndarray:
    """
    From a single-sample encoding_indices (or one sample from batched), produce
    the length-512 pooled index sequence.

    Args:
        encoding_indices: SparseTensor with .feats [N, 1] or [N] (indices),
                         .coords [N, 4] (batch_idx, x, y, z).
        batch_idx: which sample index to take (if batched).

    Returns:
        length-512 int64 array (0..8191 or MESH_EMPTY).
    """
    indices = encoding_indices.feats.squeeze(-1).long()
    coords = encoding_indices.coords  # [N, 4]: batch_idx, x, y, z
    if coords.shape[0] == 0:
        return np.full(POOLED_LEN, MESH_EMPTY, dtype=np.int64)

    mask = coords[:, 0] == batch_idx
    if not mask.any():
        return np.full(POOLED_LEN, MESH_EMPTY, dtype=np.int64)
    idx_b = indices[mask]
    xyz_b = coords[mask][:, 1:4]  # [M, 3] x,y,z
    return spatial_pool_8x8x8(idx_b, xyz_b)


def pooled_sequence_to_mesh_token_string(flat: np.ndarray) -> str:
    """
    Convert length-512 pooled sequence (ints 0..8191 or MESH_EMPTY) to the
    token string for the LLM: <mesh_start> <mesh_0> ... <mesh_end>.

    Uses no space between tokens so tokenizer sees one token per <mesh_*>.
    """
    parts = ["<mesh_start>"]
    for v in flat:
        if v == MESH_EMPTY:
            parts.append("<mesh_empty>")
        else:
            parts.append(f"<mesh_{int(v)}>")
    parts.append("<mesh_end>")
    return "".join(parts)


def batch_encoding_indices_to_pooled_sequences(
    encoding_indices: "SparseTensor",
    batch_size: int,
) -> List[np.ndarray]:
    """
    For batched encoding_indices, return a list of length batch_size, each
    element a length-512 int64 array.
    """
    return [
        encoding_indices_to_pooled_sequence(encoding_indices, b)
        for b in range(batch_size)
    ]
