"""
Sequence Parallelism utilities for 3D mesh (DP, SP, TP).

Dimension order MUST be (dp, sp, tp) — tp innermost (last) — because
``parallelize_module`` requires TP to be the innermost dimension on its parent mesh.

- create_3d_mesh: Create DeviceMesh with (DP, SP, TP) dimensions
- get_sp_group_from_mesh: Get ProcessGroup for SP ring communication
- split_for_sp: Split batch along sequence dimension for SP
- sp_cross_entropy_loss: Distributed cross-entropy over SP ranks
- apply_sp_attention_patch: Patch attention layers to use Ring Flash Attention
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, Optional, Tuple, List
import os
import time


def _sp_dbg_enabled() -> bool:
    return os.getenv("PARALLEL_DEBUG", "0") == "1"


def _sp_dbg_verbose() -> bool:
    return os.getenv("PARALLEL_DEBUG_VERBOSE", "0") == "1"


def _sp_dbg(msg: str, *, force: bool = False) -> None:
    if not (force or _sp_dbg_enabled()):
        return
    rank = "?"
    try:
        if dist.is_initialized():
            rank = str(dist.get_rank())
    except Exception:
        pass
    ts = time.strftime("%H:%M:%S")
    print(f"[SP-DBG {ts}][rank{rank}] {msg}", flush=True)


def create_3d_mesh(
    dp_size: int,
    sp_size: int,
    tp_size: int,
    world_size: Optional[int] = None,
):
    """
    Create 3D DeviceMesh with dimensions (dp, sp, tp).

    tp is the innermost (last) dimension, which is required by
    ``parallelize_module``: "TP needs to be the innermost dimension on its
    parent mesh."

    Rank layout (C-order, tp fastest):
        rank = dp * (sp_size * tp_size) + sp * tp_size + tp

    Returns:
        mesh_3d: Full 3D DeviceMesh
        dp_mesh, sp_mesh, tp_mesh: 1D sub-meshes
    """
    from torch.distributed.device_mesh import init_device_mesh

    if world_size is None:
        world_size = dist.get_world_size()
    if dp_size * sp_size * tp_size != world_size:
        raise ValueError(
            f"dp_size * sp_size * tp_size ({dp_size * sp_size * tp_size}) "
            f"must equal world_size ({world_size})"
        )
    mesh_3d = init_device_mesh(
        "cuda",
        (dp_size, sp_size, tp_size),
        mesh_dim_names=("dp", "sp", "tp"),
    )
    return mesh_3d, mesh_3d["dp"], mesh_3d["sp"], mesh_3d["tp"]


_sp_group_cache: Dict[Tuple[int, ...], Any] = {}


def get_sp_group_from_mesh(mesh_3d) -> Optional[Any]:
    """
    Get ProcessGroup for SP ring communication from 3D mesh (dp, sp, tp).

    Each (dp, tp) slice has an SP group — ranks that share the same dp and tp
    index but differ in sp index form a ring.

    Rank layout for ("dp", "sp", "tp") mesh with C-order (tp fastest):
        rank = dp * (sp_size * tp_size) + sp * tp_size + tp

    SP group for a given (my_dp, my_tp):
        ranks = [my_dp * (sp_size * tp_size) + s * tp_size + my_tp
                 for s in range(sp_size)]
    """
    sp_mesh = mesh_3d["sp"]
    tp_mesh = mesh_3d["tp"]
    dp_mesh = mesh_3d["dp"]
    sp_size = sp_mesh.size()
    tp_size = tp_mesh.size()
    dp_size = dp_mesh.size()
    if sp_size <= 1:
        _sp_dbg("sp_size<=1, skip sp_group creation")
        return None
    try:
        pg = sp_mesh.get_group()
        if pg is not None:
            return pg
    except Exception:
        pass
    my_rank = dist.get_rank()
    # rank = dp * (sp_size * tp_size) + sp * tp_size + tp  [tp innermost]
    my_tp = my_rank % tp_size
    rest = my_rank // tp_size
    my_sp = rest % sp_size
    my_dp = rest // sp_size
    cache_key = (dp_size, sp_size, tp_size, my_dp, my_tp)
    if cache_key not in _sp_group_cache:
        ranks = [
            my_dp * (sp_size * tp_size) + s * tp_size + my_tp
            for s in range(sp_size)
        ]
        _sp_dbg(f"create sp_group ranks={ranks}", force=True)
        _sp_group_cache[cache_key] = dist.new_group(ranks)
    else:
        _sp_dbg(f"reuse cached sp_group key={cache_key}")
    return _sp_group_cache[cache_key]


def split_for_sp(
    batch: Dict[str, torch.Tensor],
    sp_group: Optional[Any],
) -> Dict[str, torch.Tensor]:
    """
    Split batch along sequence dimension for SP.
    Each SP rank gets [B, S/sp_size, ...] of input_ids, attention_mask, labels.

    Args:
        batch: {"input_ids", "attention_mask", "labels"} (and optionally "position_ids")
        sp_group: ProcessGroup for SP dimension, or None for no split

    Returns:
        Split batch with truncated tensors.
    """
    if sp_group is None:
        return batch
    sp_rank = dist.get_rank(group=sp_group)
    sp_size = dist.get_world_size(group=sp_group)
    if sp_size <= 1:
        return batch

    out = {}
    seq_dim = 1
    for key in ("input_ids", "attention_mask", "labels"):
        if key not in batch or batch[key] is None:
            out[key] = batch.get(key)
            continue
        x = batch[key]
        S = x.size(seq_dim)
        chunk_size = (S + sp_size - 1) // sp_size
        start = sp_rank * chunk_size
        end = min(start + chunk_size, S)
        if end <= start:
            out[key] = x[:, :0].clone()
        else:
            out[key] = x[:, start:end].contiguous()
        if _sp_dbg_verbose():
            _sp_dbg(
                f"split key={key} sp_rank={sp_rank}/{sp_size} "
                f"orig_S={S} chunk=[{start}:{end}) out_shape={tuple(out[key].shape)}"
            )
    if "position_ids" in batch and batch["position_ids"] is not None:
        pid = batch["position_ids"]
        S = pid.size(seq_dim)
        chunk_size = (S + sp_size - 1) // sp_size
        start = sp_rank * chunk_size
        end = min(start + chunk_size, S)
        if end > start:
            out["position_ids"] = pid[:, start:end].contiguous()
        else:
            out["position_ids"] = pid[:, :0].clone()
    return out


def sp_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sp_group: Optional[Any],
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute cross-entropy loss over SP-sharded logits and labels.
    Each SP rank has logits [B, S_local, V] and labels [B, S_local].
    Returns mean loss (averaged over SP ranks).
    """
    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[..., 1:].contiguous().view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = torch.nn.functional.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=-100,
        reduction="mean",
    )
    if sp_group is not None and dist.get_world_size(group=sp_group) > 1:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=sp_group)
    return loss


def _get_llm_layers(vl_model: nn.Module) -> nn.ModuleList:
    """Extract transformer layers from Qwen2-VL / Qwen3-VL (same as tensor_parallel_utils)."""
    m = vl_model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    else:
        m = getattr(m, "model", None)
    if m is None:
        return None
    layers = getattr(getattr(m, "language_model", m), "layers", None)
    return layers


def apply_sp_attention_patch(
    model: nn.Module,
    sp_group: Optional[Any],
) -> None:
    """
    Patch attention layers in the VL model to use Ring Flash Attention when sp_group is set.
    Replaces the core attention computation with ring_flash_attn while keeping Q,K,V projections.
    """
    if sp_group is None:
        return

    from .ring_attention import ring_flash_attn

    layers = _get_llm_layers(model.vl_model if hasattr(model, "vl_model") else model)
    if layers is None:
        return

    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        _orig_forward = attn.forward

        def make_patched(attn_module, _sp_grp):
            def _patched(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
                bsz, q_len, _ = hidden_states.size()
                _self = attn_module
                if _sp_dbg_verbose():
                    _sp_dbg(f"attn patched forward enter hidden_states={tuple(hidden_states.shape)}")
                key_states = _self.k_proj(hidden_states)
                value_states = _self.v_proj(hidden_states)
                query_states = _self.q_proj(hidden_states)
                # NOTE: do not use getattr(..., fallback_attr) directly here:
                # fallback expression is evaluated eagerly and may raise AttributeError.
                head_dim = _self.head_dim
                # 使用实际输出维度推断本地 head 数，兼容 TP 分片情况
                # (TP 下 q_proj 输出已沿 head 维度分片，全局 num_heads 不可直接使用)
                local_num_heads = query_states.shape[-1] // head_dim
                local_num_kv_heads = key_states.shape[-1] // head_dim
                num_kv_groups = local_num_heads // local_num_kv_heads
                query_states = query_states.view(bsz, q_len, local_num_heads, head_dim)
                key_states = key_states.view(bsz, -1, local_num_kv_heads, head_dim)
                value_states = value_states.view(bsz, -1, local_num_kv_heads, head_dim)
                if num_kv_groups > 1:
                    key_states = key_states.repeat_interleave(num_kv_groups, dim=2)
                    value_states = value_states.repeat_interleave(num_kv_groups, dim=2)
                if _sp_dbg_verbose():
                    _sp_dbg(
                        f"attn qkv reshaped q={tuple(query_states.shape)} "
                        f"k={tuple(key_states.shape)} v={tuple(value_states.shape)} kv_groups={num_kv_groups}"
                    )
                attn_output = ring_flash_attn(
                    query_states, key_states, value_states,
                    sp_group=_sp_grp, causal=True,
                )
                attn_output = attn_output.reshape(bsz, q_len, -1)
                # Qwen3-VL / Qwen2-VL decoder expects (attn_output, attn_weights) — 2 values
                return _self.o_proj(attn_output), None
            return _patched

        attn.forward = make_patched(attn, sp_group)
        if _sp_dbg_verbose() or idx < 2:
            _sp_dbg(f"patched self_attn forward at layer={idx}")

    print(f"[SP] Patched {len(layers)} attention layers with Ring Flash Attention", flush=True)
