"""
Wrapper that adds a 3D branch to Qwen2-VL / Qwen3-VL: VAE (frozen) + projector
produce 3D token embeddings that are prepended to text and fed to the LLM.
"""

from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.distributed as dist
import os


def _is_main_for_debug() -> bool:
    """仅主进程打印显存调试，避免重复输出。"""
    try:
        import torch.distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


def _debug_mem_log(phase: str, extra: str = "") -> None:
    """细粒度显存调试：当模型 _debug_memory_fine=True 时由 forward 内调用。"""
    if not torch.cuda.is_available():
        return
    rank_s = ""
    try:
        if dist.is_initialized():
            rank_s = f"[rank{dist.get_rank()}] "
    except Exception:
        pass
    parts = []
    for i in range(torch.cuda.device_count()):
        try:
            a = torch.cuda.memory_allocated(i) / (1024**3)
            r = torch.cuda.memory_reserved(i) / (1024**3)
            p = torch.cuda.max_memory_allocated(i) / (1024**3)
            parts.append(f"GPU{i}:{a:.1f}/{p:.1f}GiB")
        except Exception:
            parts.append(f"GPU{i}:err")
    msg = f"[MEM] {rank_s}@{phase}"
    if extra:
        msg += f" | {extra}"
    msg += f" | {' '.join(parts)}"
    print(msg, flush=True)


def _parallel_debug_enabled() -> bool:
    return os.getenv("PARALLEL_DEBUG", "0") == "1"


def _get_vl_layers_for_debug(vl_model: nn.Module):
    """获取 vl_model 的 decoder layers，用于注册 debug hooks。失败返回 None。"""
    try:
        m = vl_model
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        else:
            m = getattr(m, "model", m)
        layers = getattr(getattr(m, "language_model", m), "layers", None)
        return layers
    except Exception:
        return None


def _register_discrete_debug_hooks(vl_model: nn.Module, dbg_flag: bool) -> None:
    """为 discrete 模式在 vl_model 的 Transformer 层上注册 debug hooks。"""
    if not dbg_flag or getattr(vl_model, "_discrete_debug_hooks_registered", False):
        return
    layers = _get_vl_layers_for_debug(vl_model)
    if layers is None or len(layers) == 0:
        return
    n = len(layers)
    # 前 6 层每层打点（layer 0→1 间易出问题），之后每 4 层，再加最后一层
    indices = list(range(min(6, n))) + list(range(6, n, 4)) + ([n - 1] if n > 6 else [])
    indices = sorted(set(indices))
    handles = []

    for idx in indices:
        layer = layers[idx]
        layer_idx = idx

        all_ranks = __import__("os").environ.get("DEBUG_MEMORY_ALL_RANKS", "0") == "1"

        def make_hook(li):
            def _hook(module, inp, out):
                if all_ranks or _is_main_for_debug():
                    _debug_mem_log(f"vl_L{li}/{n}", f"layer {li}/{n} done")
            return _hook

        h = layer.register_forward_hook(make_hook(layer_idx))
        handles.append(h)
    setattr(vl_model, "_discrete_debug_hooks_registered", True)
    setattr(vl_model, "_discrete_debug_handles", handles)


from .projector import Projector3D
from .sequence_3d import prepare_3d_sequence, prepare_3d_sequence_batched
from .vae_latent_extractor import extract_3d_latent, extract_3d_latent_and_indices


def _get_vl_model_and_config(model_name_or_path: str, **kwargs):
    try:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
        model_cls = Qwen3VLForConditionalGeneration
        config_cls = Qwen3VLConfig
    except Exception:
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig
        model_cls = Qwen2VLForConditionalGeneration
        config_cls = Qwen2VLConfig
    model = model_cls.from_pretrained(model_name_or_path, **kwargs)
    # 应用 self_attn 返回值兼容补丁，修复 "too many values to unpack (expected 2)"
    try:
        from .qwen3vl_debug_patch import apply_qwen3vl_self_attn_patch
        apply_qwen3vl_self_attn_patch(verbose=True, debug=False)
    except Exception as e:
        import warnings
        warnings.warn(f"[Qwen3VL] 应用 self_attn 补丁失败: {e}", stacklevel=2)
    return model, model.config


class Qwen3VLWith3DBranch(nn.Module):
    """
    Qwen2-VL or Qwen3-VL with an added 3D branch. For 3D input: VAE encode ->
    prepare_3d_sequence -> projector -> 3D token embeddings, then prepended to
    text embeddings and fed to the LLM.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        vae_model: Optional[nn.Module] = None,
        latent_dim: int = 16,
        max_3d_tokens: int = 2048,
        use_3d_pos: bool = False,
        projector_num_layers: int = 1,
        torch_dtype: Optional[torch.dtype] = None,
        use_vl_model: Optional[nn.Module] = None,
        truncate_mode: str = "head",
        use_discrete_3d_tokens: bool = False,
        **vl_kwargs,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.max_3d_tokens = max_3d_tokens
        self.truncate_mode = truncate_mode
        self.use_discrete_3d_tokens = use_discrete_3d_tokens
        if torch_dtype is None:
            torch_dtype = torch.float32
        if use_vl_model is not None:
            self.vl_model = use_vl_model
            cfg = getattr(use_vl_model, "config", None)
            if cfg is None:
                cfg = type("_C", (), {"hidden_size": 1024, "text_config": type("_T", (), {"hidden_size": 1024})()})()
            self.vl_config = cfg
        else:
            self.vl_model, self.vl_config = _get_vl_model_and_config(
                model_name_or_path, torch_dtype=torch_dtype, **vl_kwargs
            )
        hidden_size = getattr(
            self.vl_config, "hidden_size", None
        ) or getattr(
            getattr(self.vl_config, "text_config", None), "hidden_size", 1024
        )
        if use_discrete_3d_tokens:
            self.projector = None
        else:
            if vae_model is not None and hasattr(vae_model, "vq") and hasattr(vae_model.vq, "embeddings"):
                latent_dim = vae_model.vq.embeddings.weight.shape[1]
            self.projector = Projector3D(
                latent_dim=latent_dim,
                hidden_size=hidden_size,
                num_layers=projector_num_layers,
                use_3d_pos=use_3d_pos,
                pos_mode="sinusoidal",
                max_coord=64,
            )
        if vae_model is not None:
            for p in vae_model.parameters():
                p.requires_grad = False
        self.sp_group = None

    def get_3d_embeds(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        attention_mask_3d: Optional[torch.Tensor] = None,
        coords_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D latent to LLM hidden size.
        feats: [N, 16] or [B, N, 16]; coords: [N, 4] or [B, N, 4].
        Returns: embeddings [seq_len, hidden_size] or [B, seq_len, hidden_size],
                 attention_mask [seq_len] or [B, seq_len].
        """
        if self.use_discrete_3d_tokens or self.projector is None:
            raise RuntimeError("get_3d_embeds is for projector path only; use discrete token input_ids for discrete mode.")
        if feats.dim() == 2:
            feats, attention_mask_3d, coords_out = prepare_3d_sequence(
                feats, coords, max_3d_tokens=self.max_3d_tokens, truncate_mode=self.truncate_mode
            )
            feats = feats.unsqueeze(0)
            attention_mask_3d = attention_mask_3d.unsqueeze(0)
            coords_out = coords_out.unsqueeze(0)
        embeds = self.projector(feats, coords_out)
        return embeds, attention_mask_3d

    def get_3d_embeds_and_encoding_indices(
        self,
        inputs_3d: Dict[str, torch.Tensor],
        device: Union[str, torch.device] = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Get 3D token embeddings and encoding_indices for mesh reconstruction.

        Use encoding_indices with self.vae_model.Decode(encoding_indices) to reconstruct
        sparse SDF, or decode_mesh / sparse2mesh for mesh output.

        Args:
            inputs_3d: Batch dict with 'sparse_sdf', 'sparse_index', 'batch_idx'.
            device: Device for VAE and tensors.

        Returns:
            embeds_3d: [1, seq_len, hidden_size] 3D token embeddings.
            attention_mask_3d: [1, seq_len] mask for 3D tokens.
            encoding_indices: SparseTensor to pass to vae_model.Decode(...).
        """
        if self.use_discrete_3d_tokens or self.projector is None:
            raise RuntimeError("get_3d_embeds_and_encoding_indices is for projector path only.")
        if self.vae_model is None:
            raise ValueError("vae_model is required for get_3d_embeds_and_encoding_indices.")
        feats_3d, coords_3d, encoding_indices = extract_3d_latent_and_indices(
            inputs_3d, self.vae_model, device=device
        )
        feats_seq, mask_3d, coords_seq = prepare_3d_sequence(
            feats_3d, coords_3d, max_3d_tokens=self.max_3d_tokens, truncate_mode=self.truncate_mode
        )
        feats_seq = feats_seq.unsqueeze(0).to(device)
        mask_3d = mask_3d.unsqueeze(0).to(device)
        coords_seq = coords_seq.unsqueeze(0).to(device)
        embeds_3d = self.projector(feats_seq, coords_seq)
        return embeds_3d, mask_3d, encoding_indices

    def forward_with_3d(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_3d: Optional[Dict[str, torch.Tensor]] = None,
        feats_3d: Optional[torch.Tensor] = None,
        coords_3d: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward when input includes 3D. Either pass inputs_3d (batch dict for VAE)
        or precomputed feats_3d [B, N, latent_dim] and coords_3d [B, N, 4].
        When inputs_3d + vae_model are used, outputs include encoding_indices_3d
        for mesh reconstruction via vae_model.Decode(encoding_indices_3d).
        Not used in discrete-token mode (use forward with input_ids only).
        """
        if self.use_discrete_3d_tokens or self.projector is None:
            raise RuntimeError("forward_with_3d is for projector path only; use forward(input_ids=..., labels=...) for discrete mode.")
        _dbg = getattr(self, "_debug_memory_fine", False)
        encoding_indices = None
        if inputs_3d is not None and self.vae_model is not None:
            feats_3d, coords_3d, encoding_indices = extract_3d_latent_and_indices(
                inputs_3d, self.vae_model, device=input_ids.device
            )
            if _dbg and _is_main_for_debug():
                _debug_mem_log("after_extract_3d", f"feats={feats_3d.shape if feats_3d is not None else None} coords={coords_3d.shape if coords_3d is not None else None}")
        elif inputs_3d is None and (feats_3d is None or coords_3d is None):
            raise ValueError("Provide either inputs_3d (with vae_model) or feats_3d and coords_3d.")
        if feats_3d is None or coords_3d is None:
            raise ValueError("Provide either inputs_3d (with vae_model) or feats_3d and coords_3d.")

        batch_size = input_ids.shape[0]
        if feats_3d.dim() == 2:
            feats_seq, mask_3d, coords_seq = prepare_3d_sequence_batched(
                feats_3d, coords_3d, batch_size=batch_size, max_3d_tokens=self.max_3d_tokens,
                truncate_mode=self.truncate_mode
            )
        else:
            if feats_3d.dim() == 3:
                feats_flat = feats_3d.reshape(-1, feats_3d.shape[-1])
                b_idx = torch.arange(batch_size, device=feats_3d.device).repeat_interleave(feats_3d.shape[1])
                coords_flat = coords_3d.reshape(-1, 4)
                coords_flat = torch.cat([b_idx.unsqueeze(1).to(coords_flat.dtype), coords_flat[:, 1:4]], dim=1)
                feats_3d, coords_3d = feats_flat, coords_flat
            feats_seq, mask_3d, coords_seq = prepare_3d_sequence_batched(
                feats_3d, coords_3d, batch_size=batch_size, max_3d_tokens=self.max_3d_tokens,
                truncate_mode=self.truncate_mode
            )
        if feats_seq.device != input_ids.device:
            feats_seq = feats_seq.to(input_ids.device)
            mask_3d = mask_3d.to(input_ids.device)
            coords_seq = coords_seq.to(input_ids.device)
        if _dbg and _is_main_for_debug():
            _debug_mem_log("after_prepare_3d_seq", f"feats_seq={feats_seq.shape} mask_3d={mask_3d.shape}")
        embeds_3d = self.projector(feats_seq, coords_seq)
        if _dbg and _is_main_for_debug():
            _debug_mem_log("after_projector", f"embeds_3d={embeds_3d.shape}")

        embed_tokens = self.vl_model.get_input_embeddings()
        text_embeds = embed_tokens(input_ids)
        embeds_3d = embeds_3d.to(text_embeds.dtype)
        seq_3d = embeds_3d.shape[1]
        if _dbg and _is_main_for_debug():
            _debug_mem_log("after_text_embed", f"text_embeds={text_embeds.shape} seq_3d={seq_3d}")
        combined_embeds = torch.cat([embeds_3d, text_embeds], dim=1)
        if attention_mask is not None:
            combined_attention_mask = torch.cat(
                [mask_3d.to(attention_mask.dtype), attention_mask], dim=1
            )
        else:
            combined_attention_mask = torch.cat(
                [mask_3d.to(text_embeds.dtype), torch.ones_like(input_ids, dtype=text_embeds.dtype, device=input_ids.device)], dim=1
            )

        if labels is not None:
            pad_labels = torch.full(
                (labels.shape[0], seq_3d),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            combined_labels = torch.cat([pad_labels, labels], dim=1)
        else:
            combined_labels = None
        # Debug: 打印输入形状，便于排查 forward 异常
        _debug = getattr(self, "_debug_vl_forward", False)
        if _debug:
            print(
                f"[Qwen3VL-Debug] forward_with_3d -> vl_model: "
                f"combined_embeds={combined_embeds.shape}, "
                f"combined_attention_mask={combined_attention_mask.shape if combined_attention_mask is not None else None}, "
                f"use_cache={use_cache}",
                flush=True,
            )
        if _dbg and _is_main_for_debug():
            total_seq = combined_embeds.shape[1]
            _debug_mem_log("before_vl_model", f"combined_seq={total_seq} (3d={seq_3d}+text={total_seq-seq_3d}) [此处易OOM]")
        outputs = self.vl_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            use_cache=use_cache,
        )
        if _dbg and _is_main_for_debug():
            _debug_mem_log("after_vl_model")
        if encoding_indices is not None:
            outputs["encoding_indices_3d"] = encoding_indices
        return outputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_3d: Optional[Dict[str, torch.Tensor]] = None,
        feats_3d: Optional[torch.Tensor] = None,
        coords_3d: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Any]:
        kwargs.setdefault("use_cache", False)
        if inputs_3d is not None or feats_3d is not None:
            return self.forward_with_3d(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                inputs_3d=inputs_3d,
                feats_3d=feats_3d,
                coords_3d=coords_3d,
                use_cache=kwargs["use_cache"],
            )
        sp_group = getattr(self, "sp_group", None)
        if sp_group is not None and input_ids is not None:
            from .sequence_parallel_utils import split_for_sp
            if _parallel_debug_enabled():
                try:
                    sp_rank = dist.get_rank(group=sp_group)
                    sp_size = dist.get_world_size(group=sp_group)
                except Exception:
                    sp_rank, sp_size = -1, -1
                print(
                    f"[SP-FWD] before split input_ids={tuple(input_ids.shape)} "
                    f"attention_mask={tuple(attention_mask.shape) if attention_mask is not None else None} "
                    f"labels={tuple(labels.shape) if labels is not None else None} "
                    f"sp_rank={sp_rank}/{sp_size}",
                    flush=True,
                )
            batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            batch = split_for_sp(batch, sp_group)
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels")
            if _parallel_debug_enabled():
                print(
                    f"[SP-FWD] after split input_ids={tuple(input_ids.shape)} "
                    f"attention_mask={tuple(attention_mask.shape) if attention_mask is not None else None} "
                    f"labels={tuple(labels.shape) if labels is not None else None}",
                    flush=True,
                )
        _dbg = getattr(self, "_debug_memory_fine", False)
        _register_discrete_debug_hooks(self.vl_model, _dbg)
        _all_r = __import__("os").environ.get("DEBUG_MEMORY_ALL_RANKS", "0") == "1"
        if _dbg and (_all_r or _is_main_for_debug()) and input_ids is not None:
            seq_len = input_ids.shape[1]
            _debug_mem_log("before_vl(discrete)", f"seq={seq_len}")
        outputs = self.vl_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs,
        )
        if _dbg and (_all_r or _is_main_for_debug()):
            _debug_mem_log("after_vl(discrete)")
        if sp_group is not None and "loss" in outputs and outputs["loss"] is not None:
            dist.all_reduce(outputs["loss"], op=dist.ReduceOp.AVG, group=sp_group)
        return outputs

    def generate(self, **kwargs):
        return self.vl_model.generate(**kwargs)
