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
    mesh_2d = init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    return mesh_2d, mesh_2d["dp"], mesh_2d["tp"]


def get_3d_mesh(
    tp_size: int,
    sp_size: int = 1,
    world_size: Optional[int] = None,
):
    """
    Create 3D device mesh (dp, tp, sp) for TP+SP hybrid.

    Dimension order (dp, tp, sp) is required for FSDP2+TP composition:
    FSDP2 internally needs mesh[("dp", "tp")] for TP-parallelized layers.
    ("dp", "tp") must be a contiguous subsequence of mesh_dim_names;
    (dp, sp, tp) would make ("dp", "tp") invalid (sp in between).

    Example: world_size=4, tp_size=2, sp_size=2 -> dp=1, mesh (1, 2, 2).

    Returns:
        mesh_3d: Full 3D DeviceMesh (dp, tp, sp).
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
    # (dp, tp, sp) so that ("dp", "tp") is contiguous for FSDP2
    mesh_3d = init_device_mesh(
        "cuda",
        (dp_size, tp_size, sp_size),
        mesh_dim_names=("dp", "tp", "sp"),
    )
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
        apply_to_attention=False, apply_to_mlp=True
        → SKIP attention TP to avoid conflict with peft LoRA wrappers.
        Reason: peft replaces nn.Linear with a custom module whose lora_A/B
        matrices are regular (non-DTensor) tensors.  Applying ColwiseParallel /
        RowwiseParallel on top of a peft layer would shard the base weight but
        leave lora_A/B unsharded, producing incorrect all-reduce boundaries for
        RowwiseParallel (o_proj) that expects sharded input.
        → MLP layers are not LoRA-targeted → always safe to parallelize.
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

    for layer in layers:
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
    for layer in layers:
        fully_shard(layer, **kwargs)

    # Wrap the outer model to cover embeddings, lm_head, projector, VAE (frozen)
    fully_shard(model, **kwargs)

    print(
        f"[FSDP2] Data-parallel sharding applied: "
        f"{len(layers)} transformer layers + outer model wrapper.",
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
