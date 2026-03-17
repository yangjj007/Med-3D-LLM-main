"""
Tensor Parallelism utilities for Qwen3-VL / Qwen2-VL training.

GPU topology for 4 GPUs (TP=2, DP=2):
  [GPU0, GPU1] — TP group 0  (dp_rank=0, same data)
  [GPU2, GPU3] — TP group 1  (dp_rank=1, different data)

Each TP group splits every transformer layer across 2 GPUs:
  - Attention: q/k/v_proj → ColwiseParallel (output sharded by head)
               o_proj     → RowwiseParallel (input sharded, output all-reduced)
  - MLP:       gate/up_proj → ColwiseParallel
               down_proj    → RowwiseParallel

Constraint for Qwen3-VL 2B (GQA: 16 Q-heads, 2 KV-heads):
  TP degree must divide num_key_value_heads (2) → max TP=2.
  With 4 GPUs: TP=2, DP=2.

Requires PyTorch >= 2.3 for stable DTensor TP + FSDP2 composability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
import os
import time


def _pdebug_enabled() -> bool:
    return os.getenv("PARALLEL_DEBUG", "0") == "1"


def _pdebug_verbose() -> bool:
    return os.getenv("PARALLEL_DEBUG_VERBOSE", "0") == "1"


def _pdbg(msg: str, *, force: bool = False) -> None:
    if not (force or _pdebug_enabled()):
        return
    rank = "?"
    try:
        if dist.is_initialized():
            rank = str(dist.get_rank())
    except Exception:
        pass
    ts = time.strftime("%H:%M:%S")
    print(f"[TP-DBG {ts}][rank{rank}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Device mesh helpers
# ---------------------------------------------------------------------------

def get_tp_dp_mesh(tp_size: int, world_size: Optional[int] = None):
    """
    Create a 2D device mesh with dimensions (dp, tp).

    Example: world_size=4, tp_size=2 → mesh shape (2, 2).

    Returns:
        mesh_2d: The full 2D DeviceMesh.
        dp_mesh: Sub-mesh for the DP dimension.
        tp_mesh: Sub-mesh for the TP dimension.
    """
    from torch.distributed.device_mesh import init_device_mesh

    if world_size is None:
        world_size = dist.get_world_size()
    if world_size % tp_size != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by tp_size={tp_size}. "
            f"For Qwen3-VL 2B (2 KV heads), the only valid TP sizes are 1 and 2."
        )
    dp_size = world_size // tp_size
    _pdbg(f"create 2D mesh request world_size={world_size} tp_size={tp_size} dp_size={dp_size}")
    mesh_2d = init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    _pdbg("create 2D mesh done", force=True)
    return mesh_2d, mesh_2d["dp"], mesh_2d["tp"]


def get_3d_mesh(
    tp_size: int,
    sp_size: int = 1,
    world_size: Optional[int] = None,
):
    """
    Create 3D device mesh (dp, sp, tp) for TP+SP hybrid.

    Dimension order MUST be (dp, sp, tp) — tp innermost (last) — because
    ``parallelize_module`` requires: "TP needs to be the innermost dimension
    on its parent mesh."

    Note on FSDP2+TP composition:
        FSDP2 also requires ("dp","tp") to be a *contiguous* subsequence of
        the parent mesh dims when composing with TP.  With 3 dims this
        conflicts: tp innermost → ("dp","tp") non-contiguous (sp is between).
        Therefore, when SP is enabled, skip apply_fsdp2_dp() and use
        register_dp_grad_hooks() for DP gradient synchronisation instead.

    Rank layout (C-order, tp fastest):
        rank = dp * (sp_size * tp_size) + sp * tp_size + tp

    Example: world_size=8, tp_size=2, sp_size=2 → dp=2, mesh (2, 2, 2).

    Returns:
        mesh_3d: Full 3D DeviceMesh (dp, sp, tp).
        dp_mesh, sp_mesh, tp_mesh: 1D sub-meshes.
    """
    from torch.distributed.device_mesh import init_device_mesh

    if world_size is None:
        world_size = dist.get_world_size()
    if world_size % (tp_size * sp_size) != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by tp_size*sp_size ({tp_size * sp_size})"
        )
    dp_size = world_size // (tp_size * sp_size)
    _pdbg(
        f"create 3D mesh request world_size={world_size} tp_size={tp_size} "
        f"sp_size={sp_size} dp_size={dp_size}"
    )
    # tp innermost (last dim) — required by parallelize_module
    mesh_3d = init_device_mesh(
        "cuda",
        (dp_size, sp_size, tp_size),
        mesh_dim_names=("dp", "sp", "tp"),
    )
    _pdbg("create 3D mesh done", force=True)
    return mesh_3d, mesh_3d["dp"], mesh_3d["sp"], mesh_3d["tp"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_llm_layers(vl_model: nn.Module) -> nn.ModuleList:
    """
    Extract the transformer decoder layer list from Qwen2-VL or Qwen3-VL.

    Structure for Qwen2-VL / Qwen3-VL:
      - model.model.language_model.layers  (standard VL structure)
      - model.model.layers                (fallback for older/different layouts)
    When wrapped by PeftModel:
      - model.base_model.model.language_model.layers
    """
    # Handle PeftModel wrapper: get underlying base model first
    m = vl_model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    else:
        m = getattr(m, "model", None)
        if m is None:
            raise AttributeError(
                "Cannot find 'model' attribute in vl_model. "
                "Expected Qwen2VLForConditionalGeneration or Qwen3VLForConditionalGeneration."
            )
    # Qwen2-VL / Qwen3-VL: layers live under language_model
    layers = getattr(getattr(m, "language_model", m), "layers", None)
    if layers is None:
        raise AttributeError(
            "Cannot find 'model.language_model.layers' or 'model.layers' in vl_model. "
            "Ensure you are using Qwen2-VL or Qwen3-VL from transformers."
        )
    return layers


# ---------------------------------------------------------------------------
# Tensor Parallelism
# ---------------------------------------------------------------------------

def apply_tp_to_qwen3vl(
    vl_model: nn.Module,
    tp_mesh,
    apply_to_attention: bool = True,
    apply_to_mlp: bool = True,
) -> int:
    """
    Apply tensor parallelism to each transformer layer of Qwen3-VL/Qwen2-VL.

    Calling convention by training stage
    -------------------------------------
    warmup stage (no LoRA):
        apply_to_attention=True, apply_to_mlp=True
        → full TP on all projections; ~2× activation memory reduction per layer.

    SFT stage with LoRA (peft applied BEFORE this call):
        - If LoRA targets attention only:
            apply_to_attention=False, apply_to_mlp=True
        - If LoRA also targets MLP (e.g. discrete mode with gate/up/down_proj LoRA):
            apply_to_attention=False, apply_to_mlp=False
        → SKIP attention TP to avoid conflict with peft LoRA wrappers.
        Reason: peft replaces nn.Linear with a custom module whose lora_A/B
        matrices are regular (non-DTensor) tensors.  Applying ColwiseParallel /
        RowwiseParallel on top of a peft layer would shard the base weight but
        leave lora_A/B unsharded, producing incorrect all-reduce boundaries for
        RowwiseParallel (o_proj) that expects sharded input.
        → MLP layers are safe to parallelize only when they are not LoRA-targeted.
        Memory savings: MLP contributes ~65% of per-layer activations for
        Qwen3-VL 2B (intermediate=8960 vs. hidden=1536), so MLP-only TP still
        yields meaningful memory reduction.

    Args:
        vl_model:           Qwen3VLForConditionalGeneration or Qwen2VL...
        tp_mesh:            1-D DeviceMesh for the TP dimension.
        apply_to_attention: Whether to shard attention projections.
                            Set False when peft LoRA has already been applied.
        apply_to_mlp:       Whether to shard MLP projections.
                            Always True unless debugging.

    Returns:
        Number of sub-modules that were parallelized.
    """
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )

    layers = _get_llm_layers(vl_model)
    n_parallelized = 0

    # Sanity check: warn if LoRA layers are detected but apply_to_attention=True
    if apply_to_attention and len(layers) > 0:
        first_attn = getattr(layers[0], "self_attn", None)
        if first_attn is not None:
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(first_attn, proj_name, None)
                if proj is not None and not isinstance(proj, nn.Linear):
                    print(
                        f"[TP WARNING] {proj_name} in layer 0 is {type(proj).__name__}, "
                        f"not nn.Linear — likely a peft LoRA wrapper. "
                        f"Attention TP will be skipped to avoid DTensor+LoRA incompatibility. "
                        f"Pass apply_to_attention=False explicitly to suppress this warning.",
                        flush=True,
                    )
                    apply_to_attention = False
                    break

    # Sanity check for MLP LoRA wrappers
    if apply_to_mlp and len(layers) > 0:
        first_mlp = getattr(layers[0], "mlp", None)
        if first_mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(first_mlp, proj_name, None)
                if proj is not None and not isinstance(proj, nn.Linear):
                    print(
                        f"[TP WARNING] {proj_name} in layer 0 is {type(proj).__name__}, "
                        f"not nn.Linear — likely a peft LoRA wrapper. "
                        f"MLP TP will be skipped to avoid DTensor+LoRA incompatibility. "
                        f"Pass apply_to_mlp=False explicitly to suppress this warning.",
                        flush=True,
                    )
                    apply_to_mlp = False
                    break

    for idx, layer in enumerate(layers):
        if apply_to_attention:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn_plan = {}
                for name in ("q_proj", "k_proj", "v_proj"):
                    if hasattr(attn, name) and isinstance(getattr(attn, name), nn.Linear):
                        attn_plan[name] = ColwiseParallel()
                if hasattr(attn, "o_proj") and isinstance(attn.o_proj, nn.Linear):
                    attn_plan["o_proj"] = RowwiseParallel()
                if attn_plan:
                    parallelize_module(attn, tp_mesh, attn_plan)
                    n_parallelized += 1
                    if _pdebug_verbose() or idx < 2:
                        _pdbg(f"layer={idx} attn TP plan keys={list(attn_plan.keys())}")

        if apply_to_mlp:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                mlp_plan = {}
                for name in ("gate_proj", "up_proj"):
                    if hasattr(mlp, name) and isinstance(getattr(mlp, name), nn.Linear):
                        mlp_plan[name] = ColwiseParallel()
                if hasattr(mlp, "down_proj") and isinstance(mlp.down_proj, nn.Linear):
                    mlp_plan["down_proj"] = RowwiseParallel()
                if mlp_plan:
                    parallelize_module(mlp, tp_mesh, mlp_plan)
                    n_parallelized += 1
                    if _pdebug_verbose() or idx < 2:
                        _pdbg(f"layer={idx} mlp TP plan keys={list(mlp_plan.keys())}")

    print(
        f"[TP] Tensor parallelism applied: {n_parallelized} sub-module(s) across "
        f"{len(layers)} transformer layers "
        f"(attn={apply_to_attention}, mlp={apply_to_mlp})",
        flush=True,
    )
    return n_parallelized


# ---------------------------------------------------------------------------
# FSDP2 data parallelism
# ---------------------------------------------------------------------------

def apply_fsdp2_dp(
    model: nn.Module,
    dp_mesh,
    use_bf16: bool = True,
) -> None:
    """
    Apply FSDP2 (fully_shard) on the DP dimension for data-parallel sharding.

    Shards each transformer layer first (finer granularity → better overlap),
    then wraps the outer model to shard embedding, lm_head, and projector.

    Must be called AFTER apply_tp_to_qwen3vl() so that TP and FSDP2 compose
    correctly on top of the already-parallelized layers.

    Args:
        model:      The Qwen3VLWith3DBranch wrapper (contains vl_model + projector).
        dp_mesh:    1-D DeviceMesh for the DP dimension.
        use_bf16:   Enable bfloat16 mixed precision policy.
    """
    from torch.distributed._composable.fsdp import fully_shard

    kwargs = {"mesh": dp_mesh}
    if use_bf16:
        try:
            from torch.distributed._composable.fsdp import MixedPrecisionPolicy
            kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        except ImportError:
            pass  # older PyTorch without MixedPrecisionPolicy

    layers = _get_llm_layers(model.vl_model)
    t0 = time.time()
    for idx, layer in enumerate(layers):
        fully_shard(layer, **kwargs)
        if _pdebug_verbose() and idx < 4:
            _pdbg(f"fully_shard layer={idx} done")

    # Wrap the outer model to cover embeddings, lm_head, projector, VAE (frozen)
    fully_shard(model, **kwargs)
    _pdbg(f"fully_shard total elapsed={time.time() - t0:.3f}s", force=True)

    print(
        f"[FSDP2] Data-parallel sharding applied: "
        f"{len(layers)} transformer layers + outer model wrapper.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# DP gradient sync (used instead of FSDP2 when SP+TP are both active)
# ---------------------------------------------------------------------------

def register_dp_grad_hooks(model: nn.Module, dp_mesh) -> None:
    """
    Register per-parameter gradient hooks that all-reduce gradients across
    the DP dimension.

    Used when TP and SP are *both* active (3D mesh).  In that case FSDP2+TP
    composition is not possible because the 3D mesh must be ordered
    ("dp", "sp", "tp") for TP (innermost constraint), which makes ("dp","tp")
    non-contiguous and breaks FSDP2's internal sub-mesh lookup.

    For DTensor parameters (TP-sharded): all-reduces the local shard in-place
    across dp ranks — each TP rank independently averages its own shard.
    For regular parameters: all-reduces the full gradient across dp ranks.

    Hooks fire on every ``backward()`` call, which is correct for gradient
    accumulation: each micro-step's contribution is individually averaged,
    and the results accumulate correctly.
    """
    try:
        dp_group = dp_mesh.get_group()
    except Exception:
        dp_group = dp_mesh.get_group(mesh_dim=0)

    if dp_group is None or dist.get_world_size(group=dp_group) <= 1:
        print("[DP] dp_size=1, skipping gradient hook registration.", flush=True)
        return

    try:
        from torch.distributed.tensor import DTensor as _DTensor
    except ImportError:
        _DTensor = None

    n_hooked = 0
    verbose = _pdebug_verbose()
    for param in model.parameters():
        if not param.requires_grad:
            continue

        def _make_hook(grp, dtensor_cls):
            _fired = {"n": 0}
            def _hook(grad):
                if grad is None:
                    return grad
                _fired["n"] += 1
                t0 = time.time()
                if dtensor_cls is not None and isinstance(grad, dtensor_cls):
                    dist.all_reduce(grad._local_tensor, op=dist.ReduceOp.AVG, group=grp)
                else:
                    dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=grp)
                if verbose and _fired["n"] <= 2:
                    shape = tuple(grad.shape) if hasattr(grad, "shape") else "dtensor"
                    _pdbg(
                        f"dp_grad_all_reduce fired={_fired['n']} shape={shape} "
                        f"elapsed={time.time() - t0:.4f}s"
                    )
                return grad
            return _hook

        param.register_hook(_make_hook(dp_group, _DTensor))
        n_hooked += 1

    print(
        f"[DP] Registered gradient hooks on {n_hooked} parameters "
        f"for DP sync (dp_size={dp_mesh.size()}, no FSDP2).",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers for FSDP2
# ---------------------------------------------------------------------------

def gather_full_state_dict(model: nn.Module) -> dict:
    """
    Gather the full (unsharded) state dict from an FSDP2-wrapped model.
    ALL ranks must call this function simultaneously (requires collective ops).

    Returns a CPU state dict (only meaningful on rank 0 after the call).
    """
    try:
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            StateDictOptions,
        )
        full_sd = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    except Exception as exc:
        # Fallback: collect DTensor params manually
        print(f"[FSDP2] get_model_state_dict failed ({exc}), using DTensor fallback", flush=True)
        full_sd = {}
        for name, param in model.named_parameters():
            if hasattr(param, "full_tensor"):
                full_sd[name] = param.full_tensor().detach().cpu()
            else:
                full_sd[name] = param.detach().cpu()
    return full_sd


def save_projector_tp(model: nn.Module, save_path: str, is_main: bool) -> None:
    """Save projector weights from an FSDP2-wrapped model. All ranks participate."""
    full_sd = gather_full_state_dict(model)
    if is_main:
        proj_sd = {
            k.removeprefix("projector."): v
            for k, v in full_sd.items()
            if k.startswith("projector.")
        }
        torch.save(proj_sd, save_path)
        print(f"[TP] Saved projector → {save_path}", flush=True)


def save_lora_tp(model: nn.Module, save_dir: str, is_main: bool) -> None:
    """Save LoRA adapter weights from an FSDP2-wrapped model. All ranks participate."""
    import os
    import json
    full_sd = gather_full_state_dict(model)
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        # Extract vl_model sub-keys (peft LoRA weights live under vl_model.*)
        vl_sd = {
            k.removeprefix("vl_model."): v
            for k, v in full_sd.items()
            if k.startswith("vl_model.")
        }
        # Filter to LoRA-specific tensors only (lora_A, lora_B, lora_embedding*)
        lora_sd = {k: v for k, v in vl_sd.items() if "lora_" in k}
        if lora_sd:
            lora_path = os.path.join(save_dir, "adapter_model.bin")
            torch.save(lora_sd, lora_path)
            print(f"[TP] Saved LoRA adapter → {lora_path}", flush=True)
        else:
            # Fallback: save entire vl_model state (no peft, or non-standard naming)
            fallback_path = os.path.join(save_dir, "vl_model_state.pt")
            torch.save(vl_sd, fallback_path)
            print(f"[TP] (No lora_ keys found) Saved vl_model state → {fallback_path}", flush=True)

        # Write adapter_config.json so that `PeftModel.from_pretrained(..., save_dir)`
        # works in eval code. We must NOT fabricate defaults silently: extract the
        # real config from the live peft model if available.
        peft_model = getattr(model, "vl_model", None)
        config_obj = None
        if peft_model is not None and hasattr(peft_model, "peft_config"):
            try:
                config_obj = peft_model.peft_config.get("default", None)
            except Exception:
                config_obj = None
        if config_obj is not None:
            try:
                config_dict = config_obj.to_dict()
                # Ensure JSON-serializable (target_modules may be set)
                def _json_compatible(obj):
                    if obj is None or isinstance(obj, (bool, int, float, str)):
                        return obj
                    if isinstance(obj, dict):
                        return {str(k): _json_compatible(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_json_compatible(v) for v in obj]
                    if isinstance(obj, set):
                        try:
                            return sorted([_json_compatible(v) for v in obj])
                        except Exception:
                            return [_json_compatible(v) for v in obj]
                    try:
                        import enum
                        if isinstance(obj, enum.Enum):
                            return obj.value
                    except Exception:
                        pass
                    try:
                        from pathlib import Path
                        if isinstance(obj, Path):
                            return str(obj)
                    except Exception:
                        pass
                    return str(obj)

                config_dict = _json_compatible(config_dict)
                config_dict["peft_type"] = config_dict.get("peft_type") or "LORA"
                config_path = os.path.join(save_dir, "adapter_config.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                print(f"[TP] Saved adapter_config.json → {config_path}", flush=True)
            except Exception as e:
                print(f"[TP WARN] Failed to write adapter_config.json: {e}", flush=True)
        else:
            print(
                "[TP WARN] peft_config not found on model.vl_model; "
                "adapter_config.json not written (eval may not load LoRA by directory).",
                flush=True,
            )


def save_warmup_embed_lmhead_tp(model: nn.Module, save_path: str, is_main: bool) -> None:
    """
    Save only embedding layer and lm_head from vl_model (for warmup stage, discrete mode).
    Used so SFT can load these weights and continue from warmup. All ranks must call.
    """
    full_sd = gather_full_state_dict(model)
    if is_main:
        # Keep keys under vl_model that are text embed_tokens or lm_head.
        # Avoid broad "embedding" substring which may match non-text embeddings.
        embed_lmhead_sd = {
            k: v for k, v in full_sd.items()
            if k.startswith("vl_model.")
            and (
                "embed_tokens" in k
                or "lm_head" in k
            )
        }
        if embed_lmhead_sd:
            torch.save(embed_lmhead_sd, save_path)
            print(f"[TP] Saved warmup embed+lm_head ({len(embed_lmhead_sd)} keys) → {save_path}", flush=True)
        else:
            print(f"[TP] WARN: no embed_tokens/embedding/lm_head keys found in state_dict, skip save", flush=True)


# ---------------------------------------------------------------------------
# Checkpoint Loading helpers for FSDP2 + TP
# ---------------------------------------------------------------------------

def load_projector_tp(model: nn.Module, load_path: str, is_main: bool) -> None:
    """
    将 Projector 权重加载到 FSDP2 封装的模型中。
    所有 rank 都必须参与调用。
    """
    from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
    
    # 1. 只有主进程读取磁盘文件
    if is_main:
        if not os.path.isfile(load_path):
            print(f"[TP WARN] Projector file not found: {load_path}", flush=True)
            checkpoint = {}
        else:
            checkpoint = torch.load(load_path, map_location="cpu")
            # 补回保存时去掉的 "projector." 前缀
            checkpoint = {f"projector.{k}": v for k, v in checkpoint.items()}
    else:
        checkpoint = {}

    # 2. 使用 set_model_state_dict 自动处理广播和分片加载
    try:
        set_model_state_dict(
            model,
            model_state_dict=checkpoint,
            options=StateDictOptions(full_state_dict=True)
        )
        if is_main:
            print(f"[TP] Projector loaded from {load_path}", flush=True)
    except Exception as e:
        if is_main:
            print(f"[TP ERROR] Failed to load projector: {e}", flush=True)


def load_lora_tp(model: nn.Module, load_dir: str, is_main: bool) -> None:
    """
    将 LoRA 权重加载到 FSDP2 封装的模型中。
    所有 rank 都必须参与调用。
    """
    from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
    import os

    # 1. 确定权重文件路径
    lora_path = os.path.join(load_dir, "adapter_model.bin")
    fallback_path = os.path.join(load_dir, "vl_model_state.pt")
    
    if is_main:
        target_path = lora_path if os.path.exists(lora_path) else fallback_path
        if not os.path.exists(target_path):
            print(f"[TP WARN] LoRA file not found in {load_dir}", flush=True)
            checkpoint = {}
        else:
            checkpoint = torch.load(target_path, map_location="cpu")
            # 补回保存时去掉的 "vl_model." 前缀
            checkpoint = {f"vl_model.{k}": v for k, v in checkpoint.items()}
    else:
        checkpoint = {}

    # 2. 分发并加载权重
    try:
        set_model_state_dict(
            model,
            model_state_dict=checkpoint,
            options=StateDictOptions(full_state_dict=True)
        )
        if is_main:
            print(f"[TP] LoRA weights loaded from {load_dir}", flush=True)
    except Exception as e:
        if is_main:
            print(f"[TP ERROR] Failed to load LoRA: {e}", flush=True)


def load_warmup_embed_lmhead_tp(model: nn.Module, load_path: str, is_main: bool) -> None:
    """
    加载由 save_warmup_embed_lmhead_tp 保存的权重（用于离散模式 SFT 承接 Warmup）。
    """
    from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
    
    if is_main:
        if not os.path.isfile(load_path):
            print(f"[TP WARN] Warmup weights not found: {load_path}", flush=True)
            checkpoint = {}
        else:
            checkpoint = torch.load(load_path, map_location="cpu")
            # 这里的 key 在保存时已经是带 vl_model. 前缀的了，不需要处理
    else:
        checkpoint = {}

    try:
        set_model_state_dict(
            model,
            model_state_dict=checkpoint,
            options=StateDictOptions(full_state_dict=True)
        )
        if is_main:
            print(f"[TP] Warmup embed/lm_head loaded from {load_path}", flush=True)
    except Exception as e:
        if is_main:
            print(f"[TP ERROR] Failed to load warmup weights: {e}", flush=True)

__all__ = [
    "get_tp_dp_mesh",
    "get_3d_mesh", 
    "apply_tp_to_qwen3vl",
    "apply_fsdp2_dp",
    "register_dp_grad_hooks",
    "gather_full_state_dict",
    "save_projector_tp",
    "save_lora_tp",
    "save_warmup_embed_lmhead_tp",
    "load_projector_tp",
    "load_lora_tp",  # ✅ 确保这一行存在
    "load_warmup_embed_lmhead_tp",
]