"""
Prepare 3D latent sequence for LLM: sort by (z, y, x), truncate or pad to
max_3d_tokens, and build attention_mask. Supports batched (batch_idx in coords).
"""

from typing import Tuple, Optional
import torch
import numpy as np


def _sort_coords_feats(
    coords: torch.Tensor,
    feats: torch.Tensor,
    batch_idx_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sort points by (batch_idx, z, y, x) so order is deterministic and
    spatially coherent. coords: [N, 4] (batch_idx, x, y, z).
    """
    # coords: [N, 4] -> (batch_idx, x, y, z); lexsort sorts by last key first
    # so (x, y, z, batch_idx) -> order by batch_idx, then z, y, x
    coords_np = coords.cpu().numpy()
    order = np.lexsort((coords_np[:, 1], coords_np[:, 2], coords_np[:, 3], coords_np[:, 0]))
    order = torch.from_numpy(order).to(coords.device)
    return coords[order], feats[order]


def prepare_3d_sequence(
    feats: torch.Tensor,
    coords: torch.Tensor,
    max_3d_tokens: int = 2048,
    pad_value: float = 0.0,
    sort: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Turn (feats, coords) into fixed-length token sequence and mask for LLM.

    Args:
        feats: [N, 16] continuous latent.
        coords: [N, 4] (batch_idx, x, y, z).
        max_3d_tokens: Target sequence length; longer sequences are truncated
                       (or downsampled), shorter are padded.
        pad_value: Value to use for padding feats.
        sort: Whether to sort by (batch_idx, z, y, x) before truncate/pad.

    Returns:
        feats_out: [max_3d_tokens, 16] padded/truncated.
        attention_mask: [max_3d_tokens] bool, True where valid (not padding).
        coords_out: [max_3d_tokens, 4] padded (padding coords can be 0); optional for pos encoding.
    """
    N, C = feats.shape
    device = feats.device

    if sort:
        coords, feats = _sort_coords_feats(coords, feats)

    if N >= max_3d_tokens:
        feats = feats[:max_3d_tokens]
        coords = coords[:max_3d_tokens]
        attention_mask = torch.ones(max_3d_tokens, dtype=torch.bool, device=device)
    else:
        pad_len = max_3d_tokens - N
        feats = torch.cat([
            feats,
            torch.full((pad_len, C), pad_value, dtype=feats.dtype, device=device),
        ], dim=0)
        coords = torch.cat([
            coords,
            torch.zeros(pad_len, 4, dtype=coords.dtype, device=device),
        ], dim=0)
        attention_mask = torch.cat([
            torch.ones(N, dtype=torch.bool, device=device),
            torch.zeros(pad_len, dtype=torch.bool, device=device),
        ], dim=0)

    return feats, attention_mask, coords


def prepare_3d_sequence_batched(
    feats: torch.Tensor,
    coords: torch.Tensor,
    batch_size: int,
    max_3d_tokens: int = 2048,
    pad_value: float = 0.0,
    sort: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched version: coords have coords[:, 0] = batch_idx. Produces
    feats [batch_size, max_3d_tokens, 16], attention_mask [batch_size, max_3d_tokens].
    """
    device = feats.device
    out_feats = []
    out_mask = []
    out_coords = []

    for b in range(batch_size):
        mask_b = coords[:, 0] == b
        if mask_b.any():
            fb = feats[mask_b]
            cb = coords[mask_b]
        else:
            fb = feats.new_zeros(0, feats.shape[-1])
            cb = coords.new_zeros(0, 4)
        fb, mb, cob = prepare_3d_sequence(fb, cb, max_3d_tokens, pad_value, sort)
        out_feats.append(fb)
        out_mask.append(mb)
        out_coords.append(cob)

    feats_out = torch.stack(out_feats, dim=0)
    attention_mask = torch.stack(out_mask, dim=0)
    coords_out = torch.stack(out_coords, dim=0)
    return feats_out, attention_mask, coords_out
