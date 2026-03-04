"""
Qwen3-VL 调试与补丁模块

解决问题: ValueError: too many values to unpack (expected 2)
- 原因: 不同版本的 transformers 中 Qwen3VLTextAttention.forward 可能返回 2 个值
  (attn_output, attn_weights) 或 3 个值 (attn_output, attn_weights, past_key_value)。
  Qwen3VLTextDecoderLayer 使用 hidden_states, _ = self.self_attn(...) 只期望 2 个值，
  当返回 3 个时会报错。

- 补丁: 将 decoder layer 的 unpack 改为 attn_outputs = self.self_attn(...); hidden_states = attn_outputs[0]，
  兼容 2 个或更多返回值的版本。

用法:
  from vae_qwen3vl.qwen3vl_debug_patch import apply_qwen3vl_self_attn_patch, enable_debug_logging
  apply_qwen3vl_self_attn_patch()   # 在 model 构建后、第一次 forward 前调用
  enable_debug_logging(True)        # 可选：开启详细 debug 日志
"""

from __future__ import annotations

import logging
from typing import Any, Optional

# 控制是否打印详细 debug 信息
_DEBUG_ENABLED = False

logger = logging.getLogger(__name__)


def enable_debug_logging(enabled: bool = True):
    """开启或关闭详细 debug 日志。"""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = enabled


def _debug_print(msg: str, *args, **kwargs):
    """仅在 _DEBUG_ENABLED 时打印。"""
    if _DEBUG_ENABLED:
        full = msg
        if args:
            full += " " + " ".join(str(a) for a in args)
        if kwargs:
            full += " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"[Qwen3VL-Debug] {full}", flush=True)


def print_transformers_info():
    """打印 transformers 版本等信息，便于排查。"""
    try:
        import transformers as tf
        ver = getattr(tf, "__version__", "unknown")
        print(f"[Qwen3VL-Debug] transformers 版本: {ver}", flush=True)
    except Exception as e:
        print(f"[Qwen3VL-Debug] 无法获取 transformers 版本: {e}", flush=True)


def apply_qwen3vl_self_attn_patch(
    vl_model: Optional[Any] = None,
    verbose: bool = True,
    debug: bool = False,
) -> bool:
    """
    对 Qwen3-VL / Qwen2-VL 的 DecoderLayer 类应用补丁，解决
    "ValueError: too many values to unpack (expected 2)" 错误。

    通过猴子补丁修改 Qwen3VLTextDecoderLayer / Qwen2VLTextDecoderLayer 的 forward，
    使 self_attn 的返回值用 attn_outputs[0] 取第一个元素，兼容 2 值或 3+ 值返回。

    Args:
        vl_model: 可选，若传入则额外做一次实例级诊断；补丁主要作用于类
        verbose: 是否打印补丁应用提示
        debug: 是否同时开启 enable_debug_logging

    Returns:
        True 若成功应用补丁，否则 False
    """
    if debug:
        enable_debug_logging(True)
    if _DEBUG_ENABLED or debug:
        print_transformers_info()

    layer_cls = None
    layer_cls_name = None
    try:
        from transformers.models import qwen3_vl
        if hasattr(qwen3_vl, "Qwen3VLTextDecoderLayer"):
            layer_cls = qwen3_vl.Qwen3VLTextDecoderLayer
            layer_cls_name = "Qwen3VLTextDecoderLayer"
    except ImportError:
        pass
    if layer_cls is None:
        try:
            from transformers.models import qwen2_vl
            if hasattr(qwen2_vl, "Qwen2VLTextDecoderLayer"):
                layer_cls = qwen2_vl.Qwen2VLTextDecoderLayer
                layer_cls_name = "Qwen2VLTextDecoderLayer"
        except ImportError:
            pass

    if layer_cls is None:
        if verbose:
            print(
                "[Qwen3VL-Patch] 未找到 Qwen3VLTextDecoderLayer / Qwen2VLTextDecoderLayer，跳过。",
                flush=True,
            )
        return False

    if getattr(layer_cls, "_qwen3vl_unpack_patched", False):
        if verbose:
            print(f"[Qwen3VL-Patch] {layer_cls_name} 已应用过补丁，跳过。", flush=True)
        return True

    _orig_forward = layer_cls.forward

    def _patched_forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # 关键修复：用 attn_outputs[0] 代替 hidden_states, _ = self.self_attn(...)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_output = attn_outputs[0] if isinstance(attn_outputs, (tuple, list)) else attn_outputs
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    layer_cls.forward = _patched_forward
    layer_cls._qwen3vl_unpack_patched = True
    if verbose:
        print(
            f"[Qwen3VL-Patch] 已对 {layer_cls_name} 应用 self_attn 多返回值兼容补丁 "
            "(修复 'too many values to unpack')。",
            flush=True,
        )
    return True
