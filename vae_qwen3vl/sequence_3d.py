"""
Prepare 3D latent sequence for LLM: sort by (z, y, x), truncate or pad to
max_3d_tokens, and build attention_mask. Supports batched (batch_idx in coords).

Truncate modes when sequence is longer than max_3d_tokens:
  - "head": take the first L tokens after sort (original behavior).
  - "random_sample": randomly sample L tokens without replacement, preserving
    spatial spread for better understanding while avoiding OOM.
"""

from typing import Tuple, Optional, Literal
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
    truncate_mode: Literal["head", "random_sample"] = "head",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Turn (feats, coords) into fixed-length token sequence and mask for LLM.

    Args:
        feats: [N, 16] continuous latent.
        coords: [N, 4] (batch_idx, x, y, z).
        max_3d_tokens: Target sequence length. If <= 0: no truncation, output length = N (use all).
                       If > 0: longer sequences truncated, shorter padded to max_3d_tokens.
        pad_value: Value to use for padding feats.
        sort: Whether to sort by (batch_idx, z, y, x) before truncate/pad.
        truncate_mode: When N > max_3d_tokens, "head" = take first L tokens;
                       "random_sample" = randomly sample L tokens (better spatial coverage, avoids OOM).

    Returns:
        feats_out: [L, 16] with L = max_3d_tokens (or N when max_3d_tokens <= 0).
        attention_mask: [L] bool, True where valid (not padding).
        coords_out: [L, 4] padded (padding coords can be 0); optional for pos encoding.
    """
    N, C = feats.shape
    device = feats.device

    if sort:
        coords, feats = _sort_coords_feats(coords, feats)

    # max_3d_tokens <= 0: 不截断，有多少用多少，仅当需要对齐长度时才 pad（由 batched 调用时传入 target_len）
    target_len = max_3d_tokens if max_3d_tokens > 0 else N

    if N >= target_len:
        if truncate_mode == "random_sample":
            perm = torch.randperm(N, device=device)
            idx = perm[:target_len]
            feats = feats[idx]
            coords = coords[idx]
        else:
            feats = feats[:target_len]
            coords = coords[:target_len]
        attention_mask = torch.ones(target_len, dtype=torch.bool, device=device)
    else:
        pad_len = target_len - N
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
    truncate_mode: Literal["head", "random_sample"] = "head",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched version: coords have coords[:, 0] = batch_idx. Produces
    feats [batch_size, L, 16], attention_mask [batch_size, L].
    When max_3d_tokens <= 0: L = batch max length (no truncation, pad only to batch max).
    When max_3d_tokens > 0: L = min(max_3d_tokens, batch_max) — 比上限少则有多少进多少不填充，多则截断。
    truncate_mode: passed to prepare_3d_sequence when truncating each sample.
    """
    device = feats.device
    # 先算每个样本点数与 batch 内最大长度
    per_sample_len = []
    for b in range(batch_size):
        mask_b = (coords[:, 0] == b)
        per_sample_len.append(mask_b.sum().item())
    batch_actual_max = max(max(per_sample_len), 1)

    # 确定本 batch 使用的序列长度 L
    if max_3d_tokens is not None and max_3d_tokens <= 0:
        target_len = batch_actual_max  # 不截断，按 batch 内实际最大
    else:
        # 打开截断时：L = min(上限, batch 实际最大)，不把短 batch 填充到上限
        cap = max_3d_tokens if max_3d_tokens and max_3d_tokens > 0 else batch_actual_max
        target_len = min(cap, batch_actual_max)
    target_len = max(target_len, 1)

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
        fb, mb, cob = prepare_3d_sequence(fb, cb, target_len, pad_value, sort, truncate_mode=truncate_mode)
        out_feats.append(fb)
        out_mask.append(mb)
        out_coords.append(cob)

    feats_out = torch.stack(out_feats, dim=0)
    attention_mask = torch.stack(out_mask, dim=0)
    coords_out = torch.stack(out_coords, dim=0)
    return feats_out, attention_mask, coords_out
