"""
Ring Attention for Sequence Parallelism.

Implements causal Ring Attention by passing K,V in a ring across SP ranks.
Each rank computes attention of its local Q against all past K,V chunks,
merging results with online softmax. Compatible with GQA (no head splitting).
"""

from __future__ import annotations

import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, Any


def _get_sp_rank_size(sp_group: Any) -> Tuple[int, int]:
    """Get rank and world size within SP group."""
    if sp_group is None:
        return 0, 1
    return dist.get_rank(group=sp_group), dist.get_world_size(group=sp_group)


def _get_group_global_ranks(sp_group: Any) -> list[int]:
    """Return global rank list for the provided process group in group-rank order."""
    if sp_group is None:
        return [dist.get_rank()]
    if hasattr(dist, "get_process_group_ranks"):
        try:
            ranks = dist.get_process_group_ranks(sp_group)
            if ranks:
                return list(ranks)
        except Exception:
            pass
    # Fallback for older torch versions.
    sp_size = dist.get_world_size(group=sp_group)
    cur = torch.tensor([dist.get_rank()], device="cuda", dtype=torch.int64)
    gathered = [torch.zeros_like(cur) for _ in range(sp_size)]
    dist.all_gather(gathered, cur, group=sp_group)
    return [int(x.item()) for x in gathered]


def _ring_send_recv(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    sp_group: Any,
) -> None:
    """Send send_tensor to next rank, receive into recv_tensor from prev rank."""
    sp_rank, sp_size = _get_sp_rank_size(sp_group)
    if sp_size <= 1:
        return
    global_ranks = _get_group_global_ranks(sp_group)
    my_group_rank = sp_rank
    next_rank = global_ranks[(my_group_rank + 1) % sp_size]
    prev_rank = global_ranks[(my_group_rank - 1 + sp_size) % sp_size]
    if _ring_attn_debug_enabled():
        _ring_attn_debug(
            f"ring send/recv send_shape={tuple(send_tensor.shape)} recv_shape={tuple(recv_tensor.shape)} "
            f"prev={prev_rank} next={next_rank}",
            sp_group=sp_group,
        )

    # NOTE:
    # Subgroup P2P on NCCL can hang during lazy communicator initialization on
    # some environments. Use default-world P2P with global ranks for robustness.
    # Ring topology is unchanged (peer ranks are still derived from sp_group).
    t0 = time.time() if _ring_attn_debug_enabled() else 0.0
    req_recv = dist.irecv(recv_tensor, src=prev_rank)
    req_send = dist.isend(send_tensor, dst=next_rank)
    req_recv.wait()
    req_send.wait()
    if _ring_attn_debug_enabled():
        _ring_attn_debug(
            f"ring send/recv done elapsed={time.time() - t0:.4f}s "
            f"send_numel={send_tensor.numel()} recv_numel={recv_tensor.numel()}",
            sp_group=sp_group,
        )


def _gather_sp_seq_lens(local_len: int, sp_group: Any) -> list[int]:
    """Gather per-rank local sequence lengths within SP group."""
    sp_rank, sp_size = _get_sp_rank_size(sp_group)
    if sp_size <= 1:
        return [local_len]
    send = torch.tensor([local_len], device="cuda", dtype=torch.int64)
    recvs = [torch.zeros_like(send) for _ in range(sp_size)]
    dist.all_gather(recvs, send, group=sp_group)
    return [int(x.item()) for x in recvs]


def _ring_attn_debug_enabled() -> bool:
    """Enable debug logs with env var RING_ATTN_DEBUG=1."""
    return os.getenv("RING_ATTN_DEBUG", "0") == "1" or os.getenv("PARALLEL_DEBUG", "0") == "1"


def _ring_attn_debug(msg: str, sp_group: Any = None) -> None:
    if not _ring_attn_debug_enabled():
        return
    try:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank(group=sp_group) if sp_group is not None else dist.get_rank()
            print(f"[RingAttn][rank={rank}] {msg}", flush=True)
            return
    except Exception:
        pass
    print(f"[RingAttn] {msg}", flush=True)


def _ring_attn_sync_enabled() -> bool:
    """Force cuda synchronize after key steps for precise fault localization."""
    return os.getenv("RING_ATTN_SYNC_DEBUG", "0") == "1"


def _ring_attn_sync(tag: str, sp_group: Any = None) -> None:
    """Optional sync point to turn async CUDA/NCCL failures into sync errors."""
    if not _ring_attn_sync_enabled():
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        _ring_attn_debug(f"sync ok @ {tag}", sp_group=sp_group)
    except Exception as e:
        _ring_attn_debug(f"sync failed @ {tag}: {repr(e)}", sp_group=sp_group)
        raise


def _normalize_lse_to_bsh1(
    lse: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Normalize LSE tensor to [B, S, H, 1] for safe broadcasting."""
    b, s, h, _ = q.shape

    if lse.dim() == 3:
        # [B, H, S] -> [B, S, H, 1]
        if lse.shape == (b, h, s):
            return lse.permute(0, 2, 1).unsqueeze(-1).contiguous()
        # [B, S, H] -> [B, S, H, 1]
        if lse.shape == (b, s, h):
            return lse.unsqueeze(-1).contiguous()
    elif lse.dim() == 4:
        # [B, S, H, 1] (expected)
        if lse.shape[:3] == (b, s, h) and lse.shape[-1] == 1:
            return lse.contiguous()
        # [B, H, S, 1] -> [B, S, H, 1]
        if lse.shape[:3] == (b, h, s) and lse.shape[-1] == 1:
            return lse.permute(0, 2, 1, 3).contiguous()
        # Some flash-attn variants may expose a per-key dimension: [B,S,H,S_k] / [B,H,S,S_k].
        if lse.shape[:3] == (b, s, h):
            reduced = torch.logsumexp(lse.float(), dim=-1, keepdim=True)
            return reduced.to(q.dtype).contiguous()
        if lse.shape[:3] == (b, h, s):
            reduced = torch.logsumexp(lse.float(), dim=-1, keepdim=True).permute(0, 2, 1, 3)
            return reduced.to(q.dtype).contiguous()

    raise RuntimeError(
        f"Unexpected LSE shape {tuple(lse.shape)} for q shape {tuple(q.shape)}; "
        "cannot normalize to [B,S,H,1]."
    )


def _flash_attn_single_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute attention for one Q,K,V chunk. Returns (out, lse).

    q, k, v are in [B, S, H, D] layout.
    Returned lse is reshaped to [B, S, H, 1] so it broadcasts with out [B, S, H, D].
    """
    head_dim = q.size(-1)
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    try:
        from flash_attn import flash_attn_func
        # flash_attn natively accepts [B, S, H, D] and returns:
        #   out:  [B, S, H, D]
        #   lse:  version-dependent layout (usually [B, H, S])
        result = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale, return_attn_probs=True)
        out, lse = result[0], result[1]
        raw_lse_shape = tuple(lse.shape)
        # Normalize lse to [B, S, H, 1] to avoid shape instability across flash-attn versions.
        lse = _normalize_lse_to_bsh1(lse, q)
        _ring_attn_debug(
            f"flash_attn chunk shapes q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
            f"out={tuple(out.shape)} lse_raw={raw_lse_shape} lse_norm={tuple(lse.shape)} causal={causal}"
        )
        return out, lse
    except ImportError:
        # Fallback: transpose to [B, H, S, D] for standard matmul / SDPA, then transpose back.
        q_t = q.permute(0, 2, 1, 3)   # [B, H, S_q, D]
        k_t = k.permute(0, 2, 1, 3)   # [B, H, S_k, D]
        v_t = v.permute(0, 2, 1, 3)   # [B, H, S_k, D]
        out_t = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=scale,
        )
        out = out_t.permute(0, 2, 1, 3).contiguous()  # [B, S, H, D]
        # Recompute scores to derive LSE (SDPA doesn't expose it).
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # [B, H, S_q, S_k]
        if causal:
            S = scores.size(-1)
            mask = torch.triu(
                torch.ones(S, S, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
        # lse: [B, H, S_q, 1] -> permute -> [B, S_q, H, 1]
        lse = torch.logsumexp(scores.float(), dim=-1, keepdim=True).to(q.dtype)
        lse = lse.permute(0, 2, 1, 3).contiguous()  # [B, S, H, 1]
        return out, lse


def _online_softmax_merge(
    out_accum: torch.Tensor,
    lse_accum: torch.Tensor,
    out_new: torch.Tensor,
    lse_new: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge new attention output into accumulated using online softmax."""
    # Defensive normalization for unexpected LSE layouts from different kernels.
    if lse_accum.shape[-1] != 1:
        lse_accum = torch.logsumexp(lse_accum.float(), dim=-1, keepdim=True).to(out_accum.dtype)
    if lse_new.shape[-1] != 1:
        lse_new = torch.logsumexp(lse_new.float(), dim=-1, keepdim=True).to(out_new.dtype)

    # out_accum, lse_accum: accumulated so far
    # out_new, lse_new: from new chunk
    if _ring_attn_debug_enabled():
        _ring_attn_debug(
            f"merge shapes out_accum={tuple(out_accum.shape)} lse_accum={tuple(lse_accum.shape)} "
            f"out_new={tuple(out_new.shape)} lse_new={tuple(lse_new.shape)}"
        )
    lse_max = torch.maximum(lse_accum, lse_new)
    exp_old = torch.exp(lse_accum - lse_max)
    exp_new = torch.exp(lse_new - lse_max)
    if _ring_attn_debug_enabled():
        _ring_attn_debug(
            f"merge exp shapes exp_old={tuple(exp_old.shape)} exp_new={tuple(exp_new.shape)}"
        )
    out_accum = out_accum * exp_old + out_new * exp_new
    lse_accum = lse_max + torch.log(exp_old + exp_new)
    return out_accum, lse_accum


class RingFlashAttnFunc(torch.autograd.Function):
    """
    Ring Flash Attention: causal attention with K,V passed in a ring.
    Forward: each rank attends its Q to all past K,V chunks via ring.
    Backward: gradient of O w.r.t. Q,K,V propagated through ring.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: Any,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        sp_rank, sp_size = _get_sp_rank_size(sp_group)
        if _ring_attn_debug_enabled():
            _ring_attn_debug(
                f"forward entry q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
                f"sp_rank={sp_rank} sp_size={sp_size}",
                sp_group=sp_group,
            )
        _ring_attn_sync("forward_entry", sp_group=sp_group)
        if sp_size <= 1:
            out, lse = _flash_attn_single_chunk(q, k, v, causal=causal, scale=softmax_scale)
            _ring_attn_sync("single_chunk_done", sp_group=sp_group)
            # saved layout: [q, lse_total, k0, v0]  -- same as multi-rank path
            ctx.save_for_backward(q, lse, k, v)
            ctx.chunk_outputs = [out]
            ctx.chunk_lses = [lse]
            ctx.sp_group = sp_group
            ctx.causal = causal
            ctx.softmax_scale = softmax_scale
            return out

        head_dim = q.size(-1)
        scale = softmax_scale if softmax_scale is not None else 1.0 / (head_dim ** 0.5)

        # Variable-length SP chunks are common with ceil-based split; ring P2P needs
        # uniform tensor shapes, so we communicate padded K/V and slice by real length.
        local_k_len = int(k.size(1))
        seq_lens = _gather_sp_seq_lens(local_k_len, sp_group)
        max_k_len = max(seq_lens)
        if _ring_attn_debug_enabled():
            _ring_attn_debug(
                f"seq_lens={seq_lens} local_k_len={local_k_len} max_k_len={max_k_len}",
                sp_group=sp_group,
            )
        _ring_attn_sync("seq_lens_gather_done", sp_group=sp_group)

        out_acc = torch.zeros_like(q)
        lse_acc = torch.full(
            (q.size(0), q.size(1), q.size(2), 1),
            float("-inf"),
            device=q.device,
            dtype=q.dtype,
        )
        if q.dtype == torch.bfloat16:
            lse_acc = lse_acc.float()

        if local_k_len < max_k_len:
            pad_len = max_k_len - local_k_len
            k_pad = torch.zeros(
                (k.size(0), pad_len, k.size(2), k.size(3)),
                device=k.device,
                dtype=k.dtype,
            )
            v_pad = torch.zeros(
                (v.size(0), pad_len, v.size(2), v.size(3)),
                device=v.device,
                dtype=v.dtype,
            )
            k_comm = torch.cat([k, k_pad], dim=1).contiguous()
            v_comm = torch.cat([v, v_pad], dim=1).contiguous()
        else:
            k_comm = k
            v_comm = v

        cur_k = k_comm.clone()
        cur_v = v_comm.clone()
        recv_k = torch.empty_like(k_comm)
        recv_v = torch.empty_like(v_comm)

        chunk_outputs = []
        chunk_lses = []
        chunk_ks = []
        chunk_vs = []

        for step in range(sp_size):
            src_rank = (sp_rank - step + sp_size) % sp_size
            use_chunk = src_rank <= sp_rank
            is_self = src_rank == sp_rank
            if _ring_attn_debug_enabled():
                _ring_attn_debug(
                    f"step={step} src_rank={src_rank} use_chunk={use_chunk} is_self={is_self} "
                    f"cur_k={tuple(cur_k.shape)} cur_v={tuple(cur_v.shape)} src_len={seq_lens[src_rank]}",
                    sp_group=sp_group,
                )
            _ring_attn_sync(f"step{step}_pre_chunk", sp_group=sp_group)

            if use_chunk:
                src_len = seq_lens[src_rank]
                k_eff = cur_k[:, :src_len].contiguous()
                v_eff = cur_v[:, :src_len].contiguous()
                _ring_attn_sync(f"step{step}_pre_flash", sp_group=sp_group)
                attn_out, chunk_lse = _flash_attn_single_chunk(
                    q, k_eff, v_eff, causal=is_self, scale=scale
                )
                _ring_attn_sync(f"step{step}_post_flash", sp_group=sp_group)
                out_acc, lse_acc = _online_softmax_merge(out_acc, lse_acc, attn_out, chunk_lse)
                _ring_attn_sync(f"step{step}_post_merge", sp_group=sp_group)
                chunk_outputs.append(attn_out)
                chunk_lses.append(chunk_lse)
                chunk_ks.append(k_eff.clone())
                chunk_vs.append(v_eff.clone())

            if step < sp_size - 1:
                _ring_attn_sync(f"step{step}_pre_ring_k", sp_group=sp_group)
                _ring_send_recv(cur_k, recv_k, sp_group)
                _ring_attn_sync(f"step{step}_post_ring_k", sp_group=sp_group)
                _ring_attn_sync(f"step{step}_pre_ring_v", sp_group=sp_group)
                _ring_send_recv(cur_v, recv_v, sp_group)
                _ring_attn_sync(f"step{step}_post_ring_v", sp_group=sp_group)
                cur_k, recv_k = recv_k, cur_k
                cur_v, recv_v = recv_v, cur_v

        _ring_attn_sync("forward_before_save_ctx", sp_group=sp_group)
        ctx.save_for_backward(q, lse_acc, *chunk_ks, *chunk_vs)
        ctx.chunk_outputs = chunk_outputs
        ctx.chunk_lses = chunk_lses
        ctx.sp_group = sp_group
        ctx.causal = causal
        ctx.softmax_scale = scale
        _ring_attn_sync("forward_before_return", sp_group=sp_group)
        return out_acc

    @staticmethod
    def backward(
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        saved = ctx.saved_tensors
        q = saved[0]
        n = len(ctx.chunk_outputs)
        # saved: [q, lse_acc, k0, k1, ..., v0, v1, ...]
        lse_total = saved[1]
        chunk_ks = list(saved[2 : 2 + n])
        chunk_vs = list(saved[2 + n : 2 + 2 * n])
        chunk_outputs = ctx.chunk_outputs
        chunk_lses = ctx.chunk_lses
        sp_group = ctx.sp_group
        causal = ctx.causal
        scale = ctx.softmax_scale

        sp_rank, sp_size = _get_sp_rank_size(sp_group)
        if sp_size <= 1:
            k, v = chunk_ks[0], chunk_vs[0]
            grad_q, grad_k, grad_v = _flash_attn_bwd(
                grad_out, chunk_outputs[0], q, k, v, chunk_lses[0], causal, scale
            )
            return grad_q, grad_k, grad_v, None, None, None

        # Use the accumulated total LSE (not just the last chunk's LSE) for correct weighting
        lse_final = lse_total.float() if lse_total.dtype == torch.bfloat16 else lse_total
        weights = [
            torch.exp(ls.float() - lse_final).to(grad_out.dtype)
            for ls in chunk_lses
        ]

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(chunk_ks[0])
        grad_v = torch.zeros_like(chunk_vs[0])
        to_send_k = torch.zeros_like(chunk_ks[0])
        to_send_v = torch.zeros_like(chunk_vs[0])
        prev_rank = (sp_rank - 1 + sp_size) % sp_size

        for i, (out_i, k_i, v_i, lse_i, w_i) in enumerate(zip(chunk_outputs, chunk_ks, chunk_vs, chunk_lses, weights)):
            src_rank = (sp_rank - i + sp_size) % sp_size
            is_self = src_rank == sp_rank
            grad_chunk = grad_out * w_i
            gq, gk, gv = _flash_attn_bwd(grad_chunk, out_i, q, k_i, v_i, lse_i, causal=is_self, scale=scale)
            grad_q = grad_q + gq
            if src_rank == sp_rank:
                grad_k = grad_k + gk
                grad_v = grad_v + gv
            elif src_rank == prev_rank:
                to_send_k = to_send_k + gk
                to_send_v = to_send_v + gv

        if sp_size > 1:
            recv_k = torch.empty_like(chunk_ks[0])
            recv_v = torch.empty_like(chunk_vs[0])
            _ring_send_recv(to_send_k, recv_k, sp_group)
            _ring_send_recv(to_send_v, recv_v, sp_group)
            grad_k = grad_k + recv_k
            grad_v = grad_v + recv_v
        return grad_q, grad_k, grad_v, None, None, None


def _flash_attn_bwd(
    dout: torch.Tensor,
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    causal: bool,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of attention: dL/dQ, dL/dK, dL/dV. Recompute scores for SDPA-style backward.

    q, k, v, dout are in [B, S, H, D] layout; returned gradients use the same layout.
    """
    head_dim = q.size(-1)
    sc = scale if scale else 1.0 / (head_dim ** 0.5)
    # Transpose to [B, H, S, D] for correct batched matmul
    q_t    = q.permute(0, 2, 1, 3)    # [B, H, S_q, D]
    k_t    = k.permute(0, 2, 1, 3)    # [B, H, S_k, D]
    v_t    = v.permute(0, 2, 1, 3)    # [B, H, S_k, D]
    dout_t = dout.permute(0, 2, 1, 3) # [B, H, S_q, D]
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * sc  # [B, H, S_q, S_k]
    if causal:
        S = scores.size(-1)
        mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # [B, H, S_q, S_k]
    dv_t = torch.matmul(probs.transpose(-2, -1), dout_t)        # [B, H, S_k, D]
    dP   = torch.matmul(dout_t, v_t.transpose(-2, -1))          # [B, H, S_q, S_k]
    d_scores = probs * (dP - (probs * dP).sum(dim=-1, keepdim=True))
    dq_t = torch.matmul(d_scores, k_t) * sc                     # [B, H, S_q, D]
    dk_t = torch.matmul(d_scores.transpose(-2, -1), q_t) * sc   # [B, H, S_k, D]
    # Transpose back to [B, S, H, D]
    dq = dq_t.permute(0, 2, 1, 3).contiguous()
    dk = dk_t.permute(0, 2, 1, 3).contiguous()
    dv = dv_t.permute(0, 2, 1, 3).contiguous()
    return dq, dk, dv


def ring_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sp_group: Any = None,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Ring Flash Attention. Call this from patched attention forward.

    Args:
        q, k, v: [B, S_local, H, D] - local chunk (S_local = S / SP_size)
        sp_group: ProcessGroup for SP dimension, or None for no parallelism
        causal: use causal mask for self-chunk
        softmax_scale: attention scale (default 1/sqrt(head_dim))

    Returns:
        attn_output: [B, S_local, H, D]
    """
    return RingFlashAttnFunc.apply(q, k, v, sp_group, causal, softmax_scale)
