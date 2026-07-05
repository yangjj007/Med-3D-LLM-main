"""
Evaluate 3D-VL alignment: run 3D -> text generation and optionally validate mesh reconstruction.

Usage:
  # 使用配置文件（与训练一致，含 LoRA + 真实数据）
  python vae_qwen3vl/eval_3d_vl.py --config configs/3d_align_train_variable_length.yaml

  # 指定路径
  python vae_qwen3vl/eval_3d_vl.py --config configs/3d_align_train_variable_length.yaml  --lora_dir outputs_3d_align/ep4_lr5e-5_bs1_nall_lora64_20260321_051741/lora_epoch1 --data_dir train_sdf_dataset/res512_thre0.1_small 2>&1 | tee eval_debug.log
  
  # With mesh reconstruction check (save decoded mesh)
  python vae_qwen3vl/eval_3d_vl.py ... --save_mesh
"""

import argparse
import os
import re

# 确保 3D-VL 评估使用 torchsparse（新终端直接 python 运行时不会走 shell 脚本的 export）
os.environ.setdefault("SPARSE_BACKEND", "torchsparse")

import sys
import json

import torch
from transformers import AutoTokenizer


def _resolve_latest_projector_ckpt(run_dir: str) -> str | None:
    """若 run_dir 下无 projector_final.pt，则返回 epoch 最大的 projector_epoch{N}.pt，否则返回 None。"""
    import glob
    pattern = os.path.join(run_dir, "projector_epoch*.pt")
    files = glob.glob(pattern)
    best_epoch, best_path = -1, None
    for p in files:
        name = os.path.basename(p)
        m = re.match(r"projector_epoch(\d+)\.pt", name)
        if m and os.path.isfile(p):
            e = int(m.group(1))
            if e > best_epoch:
                best_epoch, best_path = e, p
    return best_path


def _resolve_latest_lora_dir(run_dir: str) -> str | None:
    """若 run_dir 下无 lora_final 目录，则返回 epoch 最大的 lora_epoch{N} 目录，否则返回 None。"""
    if not os.path.isdir(run_dir):
        return None
    best_epoch, best_path = -1, None
    for name in os.listdir(run_dir):
        m = re.match(r"lora_epoch(\d+)$", name)
        if not m:
            continue
        p = os.path.join(run_dir, name)
        if os.path.isdir(p):
            e = int(m.group(1))
            if e > best_epoch:
                best_epoch, best_path = e, p
    return best_path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_config(config_path: str, project_root: str) -> dict:
    """Load YAML config. Returns flattened dict."""
    try:
        import yaml
    except ImportError:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        return {}
    out = {}
    path_keys = ("vl_model", "vae_config", "vae_ckpt", "data_dir", "output_dir")
    for k, v in cfg.items():
        if k == "accelerate":
            continue
        if isinstance(v, dict):
            for sk, sv in v.items():
                if sk != "accelerate":
                    out[sk] = sv
        else:
            out[k] = v
    for k in path_keys:
        if k in out and isinstance(out[k], str) and out[k] and not os.path.isabs(out[k]):
            out[k] = os.path.normpath(os.path.join(project_root, out[k]))
    return out


def load_vae_from_config(vae_config_path: str, vae_ckpt_path: str, device: str = "cuda"):
    """Load SparseSDFVQVAE from project config and checkpoint."""
    from trellis.models import SparseSDFVQVAE
    with open(vae_config_path, "r") as f:
        config = json.load(f)
    vae_args = config["models"]["vqvae"]["args"]
    model = SparseSDFVQVAE(**vae_args)
    ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    return model


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def _find_adapter_bin(lora_dir: str):
    """Locate the adapter weight file (safetensors or bin)."""
    import glob as _glob
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = os.path.join(lora_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _check_lora_weights_are_nonzero(model) -> bool:
    """
    检查 LoRA-B 权重是否为非零（非默认初始化）。
    PEFT 默认 lora_B 初始化为全零，若仍为全零则说明权重未被真正注入。
    """
    lora_b_all_zero = True
    lora_b_count = 0
    for name, param in model.vl_model.named_parameters():
        if "lora_B" in name and "weight" in name:
            lora_b_count += 1
            if param.abs().max().item() > 1e-8:
                lora_b_all_zero = False
                break
    if lora_b_count == 0:
        return False
    return not lora_b_all_zero


def _force_inject_lora_weights(model, ckpt_sd: dict, ckpt_keys: set, model_lora_keys: set) -> int:
    """
    将 checkpoint 中的 LoRA 权重强制注入模型，支持 key 完全匹配和 remap 两种路径。
    返回成功注入的权重数量。
    """
    matched = ckpt_keys & model_lora_keys
    if matched == ckpt_keys:
        final_sd = ckpt_sd
    else:
        remap = _build_lora_key_remap(ckpt_keys, model_lora_keys)
        if not remap:
            print(f"[LoRA INJECT] 无法建立 key 映射 (ckpt 示例: {sorted(ckpt_keys)[:2]}, model 示例: {sorted(model_lora_keys)[:2]})")
            return 0
        print(f"[LoRA INJECT] 发现 key 映射规则，remap {len(remap)} 个 key")
        final_sd = {remap.get(k, k): v for k, v in ckpt_sd.items()}

    param_dict = dict(model.vl_model.named_parameters())
    loaded_count = 0
    shape_mismatch = 0
    for k, v in final_sd.items():
        if k in param_dict:
            if param_dict[k].shape == v.shape:
                with torch.no_grad():
                    param_dict[k].copy_(v.to(param_dict[k].device))
                loaded_count += 1
            else:
                shape_mismatch += 1
                if shape_mismatch <= 3:
                    print(f"[LoRA INJECT] Shape mismatch: {k} ckpt={v.shape} model={param_dict[k].shape}")
    if shape_mismatch > 3:
        print(f"[LoRA INJECT] ... 共 {shape_mismatch} 个 shape 不匹配")
    return loaded_count


def _verify_and_fix_lora_loading(model, lora_dir: str, device: str) -> bool:
    """
    PeftModel.from_pretrained 之后，检查 LoRA 权重是否真正加载到模型中。
    核心验证：检查 lora_B 是否仍为全零（PEFT 默认初始化），若是则说明加载失败，
    尝试手动 remap 并注入。返回 True 仅当权重值验证通过。
    """
    from peft import PeftModel
    if not isinstance(model.vl_model, PeftModel):
        print("[LoRA DIAG] model.vl_model 不是 PeftModel，跳过验证")
        return False

    adapter_path = _find_adapter_bin(lora_dir)
    if adapter_path is None:
        print(f"[LoRA DIAG] 未找到 adapter_model.bin/.safetensors in {lora_dir}")
        return False

    if adapter_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        ckpt_sd = load_file(adapter_path, device="cpu")
    else:
        ckpt_sd = torch.load(adapter_path, map_location="cpu")

    model_lora_keys = {
        n for n, _ in model.vl_model.named_parameters() if "lora_" in n
    }
    ckpt_keys = set(ckpt_sd.keys())

    matched = ckpt_keys & model_lora_keys
    missing_in_model = ckpt_keys - model_lora_keys
    missing_in_ckpt = model_lora_keys - ckpt_keys

    print(f"[LoRA DIAG] Checkpoint keys: {len(ckpt_keys)}, Model LoRA keys: {len(model_lora_keys)}")
    print(f"[LoRA DIAG] Key 匹配: {len(matched)}, 仅在 ckpt: {len(missing_in_model)}, 仅在 model: {len(missing_in_ckpt)}")

    # === 打印 checkpoint 权重详细信息 ===
    model_param_dict = dict(model.vl_model.named_parameters())
    _printed = 0
    _shape_mismatch_count = 0
    _shape_ratios = set()
    for ck in sorted(ckpt_keys):
        ckpt_tensor = ckpt_sd[ck]
        model_shape = tuple(model_param_dict[ck].shape) if ck in model_param_dict else None
        ckpt_shape = tuple(ckpt_tensor.shape)
        is_mismatch = model_shape is not None and ckpt_shape != model_shape
        if is_mismatch:
            _shape_mismatch_count += 1
            for d in range(len(ckpt_shape)):
                if ckpt_shape[d] != model_shape[d]:
                    _shape_ratios.add(model_shape[d] / ckpt_shape[d])
        if _printed < 4:
            abs_max = ckpt_tensor.abs().max().item()
            abs_mean = ckpt_tensor.abs().mean().item()
            tag = " *** SHAPE MISMATCH ***" if is_mismatch else ""
            print(f"[LoRA CKPT] {ck}: ckpt_shape={ckpt_shape} model_shape={model_shape} "
                  f"|max|={abs_max:.6f} |mean|={abs_mean:.6f}{tag}")
            _printed += 1

    if _shape_mismatch_count > 0:
        print(f"[LoRA DIAG] ⚠️  共 {_shape_mismatch_count}/{len(ckpt_keys)} 个权重 shape 不匹配！")
        print(f"[LoRA DIAG]   model/ckpt 维度比率: {sorted(_shape_ratios)}")
        if 2.0 in _shape_ratios:
            print(f"[LoRA DIAG]   检测到 2x 比率 → 极可能是 FSDP2 分片问题：")
            print(f"[LoRA DIAG]   训练时 save_lora_tp 使用了 to_local() 保存了 FSDP 的 1/2 分片，")
            print(f"[LoRA DIAG]   而非完整权重。请使用修复后的 save_lora_tp 重新保存 LoRA 权重。")
        return False

    # 第 1 步：检查 PEFT from_pretrained 是否已经成功注入了权重
    weights_live = _check_lora_weights_are_nonzero(model)
    if weights_live:
        print(f"[LoRA DIAG] ✓ lora_B 权重非零，PEFT 加载已成功生效")
        return True

    print(f"[LoRA DIAG] ⚠️  lora_B 权重全零！PEFT from_pretrained 未真正注入权重，尝试手动注入 ...")

    # 第 2 步：手动注入（支持 key 完全匹配或 remap）
    loaded_count = _force_inject_lora_weights(model, ckpt_sd, ckpt_keys, model_lora_keys)

    if loaded_count > 0:
        # 第 3 步：注入后再次验证
        weights_live_after = _check_lora_weights_are_nonzero(model)
        if weights_live_after:
            print(f"[LoRA DIAG] ✓ 手动注入成功：{loaded_count}/{len(ckpt_sd)} 个权重，lora_B 验证通过")
            return True
        else:
            print(f"[LoRA DIAG] ✗ 手动注入了 {loaded_count} 个权重，但 lora_B 仍为全零（可能注入到了错误位置）")
            return False
    else:
        if len(missing_in_model) > 0:
            print(f"[LoRA DIAG] ✗ Key 不匹配且无法建立映射")
            print(f"[LoRA DIAG]   Checkpoint 示例 key: {sorted(missing_in_model)[:2]}")
            print(f"[LoRA DIAG]   Model 示例 key:      {sorted(model_lora_keys)[:2]}")
        else:
            print(f"[LoRA DIAG] ✗ Key 匹配但注入失败（shape 不一致？）")
        return False


def _build_lora_key_remap(ckpt_keys, model_keys):
    """
    尝试找到 checkpoint key → model key 的系统性映射。
    处理常见的结构差异：
    - Qwen3-VL (model.language_model.layers) vs Qwen2-VL (model.layers)
    - PEFT 版本差异（有/无 base_model.model. 前缀）
    """
    remap = {}

    remap_rules = [
        ("model.language_model.layers.", "model.layers."),
        ("model.layers.", "model.language_model.layers."),
        ("model.language_model.model.layers.", "model.language_model.layers."),
        ("model.language_model.layers.", "model.language_model.model.layers."),
    ]

    prefix_rules = [
        ("base_model.model.", ""),
        ("", "base_model.model."),
    ]

    for rule_old, rule_new in remap_rules:
        candidate = {}
        for ck in ckpt_keys:
            if rule_old in ck:
                new_key = ck.replace(rule_old, rule_new)
                if new_key in model_keys:
                    candidate[ck] = new_key
        if len(candidate) == len(ckpt_keys):
            return candidate

    for prefix_old, prefix_new in prefix_rules:
        for rule_old, rule_new in remap_rules + [("", "")]:
            candidate = {}
            for ck in ckpt_keys:
                nk = ck
                if prefix_old and nk.startswith(prefix_old):
                    nk = nk[len(prefix_old):]
                if prefix_new:
                    nk = prefix_new + nk
                if rule_old and rule_old in nk:
                    nk = nk.replace(rule_old, rule_new)
                if nk in model_keys:
                    candidate[ck] = nk
            if len(candidate) == len(ckpt_keys):
                return candidate

    return {}


def _manual_load_lora(model, lora_dir: str, device: str) -> bool:
    """
    当 PeftModel.from_pretrained 完全失败时，手动创建 PeftModel 并加载权重。
    加载后通过 lora_B 非零检查验证是否真正生效。
    """
    adapter_path = _find_adapter_bin(lora_dir)
    config_path = os.path.join(lora_dir, "adapter_config.json")
    if not adapter_path or not os.path.isfile(config_path):
        print(f"[LoRA Manual] 缺少 adapter 文件: bin={adapter_path}, config={os.path.isfile(config_path)}")
        return False

    try:
        from peft import PeftModel, LoraConfig, get_peft_model

        with open(config_path, "r") as f:
            cfg = json.load(f)

        target_modules = cfg.get("target_modules", ["q_proj", "v_proj"])
        if isinstance(target_modules, dict):
            target_modules = list(target_modules.keys())

        lora_config = LoraConfig(
            r=cfg.get("r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            target_modules=target_modules,
            lora_dropout=cfg.get("lora_dropout", 0.0),
            bias=cfg.get("bias", "none"),
        )

        model.vl_model = get_peft_model(model.vl_model, lora_config)

        if adapter_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt_sd = load_file(adapter_path, device="cpu")
        else:
            ckpt_sd = torch.load(adapter_path, map_location="cpu")

        model_lora_keys = {n for n, _ in model.vl_model.named_parameters() if "lora_" in n}
        loaded = _force_inject_lora_weights(model, ckpt_sd, set(ckpt_sd.keys()), model_lora_keys)

        if loaded > 0:
            weights_ok = _check_lora_weights_are_nonzero(model)
            if weights_ok:
                print(f"[LoRA Manual] ✓ 手动注入 {loaded}/{len(ckpt_sd)} 个权重，lora_B 验证通过")
                return True
            else:
                print(f"[LoRA Manual] ✗ 注入了 {loaded} 个权重，但 lora_B 仍全零")
                return False
        else:
            print(f"[LoRA Manual] ✗ 未能注入任何权重 (ckpt keys: {len(ckpt_sd)}, model keys: {len(model_lora_keys)})")
            return False
    except Exception as e:
        print(f"[LoRA Manual] ✗ 手动加载异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def _run_generation(model, inputs_3d, tokenizer, prompt, max_new_tokens, device, vae_model=None,
                    max_safe_3d_length=15000, coord_max_3d=64):
    """Run 3D -> text generation for one sample."""
    if getattr(model, "use_discrete_3d_tokens", False) and vae_model is not None:
        return _run_generation_discrete(model, inputs_3d, tokenizer, prompt, max_new_tokens, device, vae_model,
                                        max_safe_3d_length=max_safe_3d_length, coord_max_3d=coord_max_3d)

    with torch.no_grad():
        embeds_3d, mask_3d, encoding_indices = model.get_3d_embeds_and_encoding_indices(inputs_3d, device=device)
    seq_3d = embeds_3d.shape[1]

    try:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt_text = prompt
    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)
    embed_tokens = model.vl_model.get_input_embeddings()
    text_embeds = embed_tokens(prompt_ids)
    model_dtype = next(model.vl_model.parameters()).dtype
    combined_embeds = torch.cat([embeds_3d, text_embeds], dim=1).to(model_dtype)
    prompt_mask = torch.ones(1, prompt_ids.shape[1], dtype=model_dtype, device=device)
    combined_mask = torch.cat([mask_3d.to(model_dtype), prompt_mask], dim=1)

    prompt_len = prompt_ids.shape[1]
    with torch.no_grad():
        out = model.vl_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    start = seq_3d + prompt_len
    generated_ids = out[:, start:] if out.shape[1] > start else out
    
    # 【修复】处理 encoding_indices 转 list
    if encoding_indices is not None:
        if hasattr(encoding_indices, 'coords') and hasattr(encoding_indices, 'feats'):
            encoding_indices_list = {
                "coords": encoding_indices.coords.cpu().tolist(),
                "feats": encoding_indices.feats.cpu().tolist()
            }
        elif isinstance(encoding_indices, torch.Tensor):
            encoding_indices_list = encoding_indices.cpu().tolist()
        else:
            encoding_indices_list = str(encoding_indices)
    else:
        encoding_indices_list = []
    
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    
    print(f"[DEBUG Continuous] Prompt Len: {prompt_len}, Gen Len: {generated_ids.shape[1]}")
    print(f"[DEBUG Continuous] Generated IDs (first 10): {generated_ids[0].cpu().tolist()[:10]}")
    if not text:
        print(f"[WARN] Continuous path generated empty text!")

    return {
        "text": text, 
        "encoding_indices": encoding_indices_list, 
        "prompt_ids": prompt_ids[0].cpu().tolist(), 
        "generated_ids": generated_ids[0].cpu().tolist(),
        "embeds_3d_shape": list(embeds_3d.shape)
    }

def _run_generation_discrete(model, inputs_3d, tokenizer, prompt, max_new_tokens, device, vae_model,
                             max_safe_3d_length=15000, coord_max_3d=64):
    """Discrete 3D token path: VAE Encode -> Text Serialization -> Chat Template -> Generate.
    序列结构与训练完全一致：
    先将 VAE codes 转换为 <mesh_start><mesh_X>...<mesh_end> 字符串，
    再与 prompt 拼接为 user_content，整体使用 tokenizer.apply_chat_template 编码。
    当点数超过 max_safe_3d_length 时，使用 FPS 下采样（与训练一致）。
    """
    with torch.no_grad():
        inputs_3d_dev = {}
        for k, v in inputs_3d.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                # 与 collate_sdf_caption_discrete 保持一致，VAE 编码强制要求 fp32
                if v.is_floating_point():
                    v = v.to(dtype=torch.float32)
                inputs_3d_dev[k] = v
            else:
                inputs_3d_dev[k] = v

        vae_model = vae_model.to(device)
        if next(vae_model.parameters()).dtype != torch.float32:
            vae_model = vae_model.float()

        # VAE 编码
        encoding_indices = vae_model.Encode(inputs_3d_dev)
    
    # 获取 3D Code 序列
    if hasattr(encoding_indices, 'feats'):
        codes_3d = encoding_indices.feats.long().squeeze(-1)
    elif isinstance(encoding_indices, torch.Tensor):
        codes_3d = encoding_indices.long().squeeze(-1)
    else:
        raise TypeError(f"Unsupported encoding_indices type: {type(encoding_indices)}")
    
    num_3d_tokens = codes_3d.shape[0]
    print(f"[DEBUG Discrete] VAE encoded {num_3d_tokens} 3D tokens")
    
    # 构建 mesh_str：FPS 下采样 + Morton Sort，与训练时 collate 完全一致
    mesh_str = ""
    try:
        from vae_qwen3vl.variable_length_3d import (
            batch_encoding_indices_to_variable_length_sequences,
            variable_length_sequence_to_mesh_token_string
        )
        seq_list = batch_encoding_indices_to_variable_length_sequences(
            encoding_indices, batch_size=1, max_safe_length=max_safe_3d_length, coord_max=coord_max_3d
        )
        mesh_str = variable_length_sequence_to_mesh_token_string(seq_list[0])
    except Exception:
        mesh_tokens = [f"<mesh_{c.item()}>" for c in codes_3d]
        mesh_str = "<mesh_start>" + "".join(mesh_tokens) + "<mesh_end>"
    
    # 【核心修复】与训练数据加载保持 100% 一致的构建逻辑
    user_content = mesh_str + "\n" + prompt
    #user_content =  prompt
    #user_content =  "Who are you? What can you do?"
    
    try:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt_text = user_content
    
    # 统一通过 tokenizer 完成 ID 转换和 Mask 构建
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    print(f"[DEBUG Discrete] Input IDs shape: {input_ids.shape}")
    
    # 生成
    with torch.no_grad():
        out = model.vl_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 提取生成结果 (去除输入部分的 prompt)
    input_len = input_ids.shape[1]
    generated_ids = out[:, input_len:] if out.shape[1] > input_len else out
    
    print(f"[DEBUG Discrete] Output Len: {out.shape[1]}, Generated Len: {generated_ids.shape[1]}")
    
    # 解码文本
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    if not text:
        raw_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        print(f"[WARN] Generated text is empty after stripping. Raw: {raw_text}")
    
    # 准备 debug 数据
    if encoding_indices is not None:
        if hasattr(encoding_indices, 'coords') and hasattr(encoding_indices, 'feats'):
            encoding_indices_list = {
                "coords": encoding_indices.coords.cpu().tolist(),
                "feats": encoding_indices.feats.cpu().tolist()
            }
        elif isinstance(encoding_indices, torch.Tensor):
            encoding_indices_list = encoding_indices.cpu().tolist()
        else:
            encoding_indices_list = str(encoding_indices)
    else:
        encoding_indices_list = []
    
    return {
        "text": text, 
        "encoding_indices": encoding_indices_list, 
        "prompt_ids": input_ids[0].cpu().tolist(),
        "generated_ids": generated_ids[0].cpu().tolist(),
        "input_ids_full": input_ids[0].cpu().tolist(),
        "num_3d_tokens": num_3d_tokens,
        "mesh_str": mesh_str[:200] + ("..." if len(mesh_str) > 200 else ""),
        "attention_mask_sample": attention_mask[0, :50].cpu().tolist(),
        "has_pad_tokens": (input_ids == pad_token_id).any().item(),
        "input_structure": {
            "prompt_len": input_ids.shape[1],
            "vae_tokens": num_3d_tokens,
            "total_input_len": input_ids.shape[1]
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D-VL: generation and optional mesh reconstruction")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (e.g. configs/3d_align_train.yaml)")
    parser.add_argument("--vl_model", type=str, default=None)
    parser.add_argument("--vae_config", type=str, default=None)
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--projector_ckpt", type=str, default=None)
    parser.add_argument("--lora_dir", type=str, default=None, help="Path to LoRA adapter (e.g. outputs_3d_align/lora_final)")
    parser.add_argument("--data_dir", type=str, default=None, help="SDF dataset dir (same as training, e.g. train_sdf_dataset)")
    parser.add_argument("--data_path", type=str, default=None, help="Path to .pt batch file (legacy); if set, overrides data_dir")
    parser.add_argument("--prompt", type=str, default="Describe this 3D shape in one sentence:")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_eval_samples", type=int, default=10, help="Max samples when using data_dir")
    parser.add_argument("--output_dir", type=str, default="./eval_3d_vl_out")
    parser.add_argument("--output_run", type=str, default=None, help="训练子目录名，用于指定要评估的 run")
    parser.add_argument("--save_mesh", action="store_true")
    parser.add_argument("--voxel_resolution", type=int, default=512)
    parser.add_argument("--mc_threshold", type=float, default=0.1)
    parser.add_argument("--max_3d_tokens", type=int, default=8000)
    parser.add_argument("--truncate_mode", type=str, default=None, choices=["head", "random_sample", "fps"],
                        help="When 3D tokens exceed max_3d_tokens: fps=Farthest Point Sampling (matches training, default); head=first L; random_sample=randomly sample L")
    parser.add_argument("--max_safe_3d_length", type=int, default=None,
                        help="Soft cap for variable-length discrete path: FPS downsample only when N > this (default from config or 15000, matches training)")
    parser.add_argument("--use_3d_pos", action="store_true", default=None, help="Use 3D positional encoding")
    parser.add_argument("--projector_layers", type=int, default=None, help="Projector MLP layers (须与训练一致，默认 1)")
    args, remaining = parser.parse_known_args()

    # ========================================================================
    # 【新增】权重加载确认信息 - 记录命令行原始输入
    # ========================================================================
    cmd_projector_ckpt = args.projector_ckpt
    cmd_lora_dir = args.lora_dir
    cmd_vae_ckpt = args.vae_ckpt
    cmd_vae_config = args.vae_config
    cmd_vl_model = args.vl_model
    print("=" * 80)
    print("[WEIGHT LOADING INFO] 命令行原始输入:")
    print(f"  --projector_ckpt: {cmd_projector_ckpt}")
    print(f"  --lora_dir: {cmd_lora_dir}")
    print(f"  --vae_ckpt: {cmd_vae_ckpt}")
    print(f"  --vae_config: {cmd_vae_config}")
    print(f"  --vl_model: {cmd_vl_model}")
    print("=" * 80)

    run_d = None
    # Load config
    if args.config:
        cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    else:
        cfg_path = None
        
    if cfg_path and os.path.isfile(cfg_path):
        cfg = _load_config(cfg_path, PROJECT_ROOT)
        for k in (
            "vl_model", "vae_config", "vae_ckpt", "data_dir", "prompt",
            "max_3d_tokens", "projector_layers", "truncate_mode", "max_safe_3d_length", "use_discrete_3d_tokens",
        ):
            if k in cfg and getattr(args, k, None) is None:
                setattr(args, k, cfg[k])

        def _run_dir():
            out_d = cfg.get("output_dir", "outputs_3d_align")
            if not os.path.isabs(out_d):
                out_d = os.path.join(PROJECT_ROOT, out_d)

            run = args.output_run or cfg.get("output_run")
            if run:
                # 【修复核心1】智能拼接路径，防止重复
                if os.path.isabs(run) and os.path.isdir(run):
                    return run
                abs_run = os.path.join(PROJECT_ROOT, run)
                if os.path.isdir(abs_run):
                    return abs_run
                return os.path.join(out_d, run)

            if not os.path.isdir(out_d):
                return out_d

            candidate_dirs = []
            for name in os.listdir(out_d):
                p = os.path.join(out_d, name)
                if not os.path.isdir(p):
                    continue
                import glob
                if (
                    os.path.isdir(os.path.join(p, "lora_final")) or len(glob.glob(os.path.join(p, "lora_epoch*"))) > 0
                    or os.path.isdir(os.path.join(p, "tokenizer_final")) or len(glob.glob(os.path.join(p, "tokenizer_epoch*"))) > 0
                    or os.path.isfile(os.path.join(p, "train_args.json"))
                ):
                    candidate_dirs.append(p)

            if candidate_dirs:
                candidate_dirs.sort(key=lambda x: os.path.getmtime(x))
                best = candidate_dirs[-1]
                print(f"[Eval] 未指定 --output_run，自动使用最近的 run 目录：{os.path.basename(best)}", flush=True)
                return best

            return out_d

        run_d = _run_dir()
        
        # 【修复核心2】自动判断使用 discrete 模式，哪怕只有 epoch
        import glob
        if os.path.isdir(run_d):
            if os.path.isdir(os.path.join(run_d, "tokenizer_final")) or len(glob.glob(os.path.join(run_d, "tokenizer_epoch*"))) > 0:
                args.use_discrete_3d_tokens = True
                
        if args.output_dir is None or args.output_dir == "./eval_3d_vl_out":
            args.output_dir = os.path.join(run_d, "eval_out")
        args.use_3d_pos = args.use_3d_pos if args.use_3d_pos is not None else cfg.get("use_3d_pos", False)
        
        # ========================================================================
        # Projector 权重加载逻辑
        # ========================================================================
        if args.projector_ckpt is None:
            args.projector_ckpt = os.path.join(run_d, "projector_final.pt")
            print(f"[WEIGHT] projector_ckpt 未指定，尝试默认路径：{args.projector_ckpt}")
        
        if not os.path.isfile(args.projector_ckpt):
            latest = _resolve_latest_projector_ckpt(run_d)
            if latest:
                old_path = args.projector_ckpt
                args.projector_ckpt = latest
                print(f"[WEIGHT] ⚠️  {old_path} 不存在，自动使用最新 epoch: {os.path.basename(args.projector_ckpt)}")
            else:
                print(f"[WEIGHT] ⚠️  未找到任何 projector 权重文件！")
                args.projector_ckpt = None # 【修复核心3】找不到时置空，避免骗人
        else:
            print(f"[WEIGHT] ✓ 使用 projector 权重：{args.projector_ckpt}")
        
        # ========================================================================
        # LoRA 权重加载逻辑
        # ========================================================================
        # 你的输出里只有 lora 和 tokenizer，说明很可能是使用 LoRA 的
        if args.lora_dir is None:
            args.lora_dir = os.path.join(run_d, "lora_final")
            print(f"[WEIGHT] lora_dir 未指定，尝试默认路径：{args.lora_dir}")
        
        if args.lora_dir and not os.path.isdir(args.lora_dir):
            latest_lora = _resolve_latest_lora_dir(run_d)
            if latest_lora:
                old_path = args.lora_dir
                args.lora_dir = latest_lora
                print(f"[WEIGHT] ⚠️  {old_path} 不存在，自动使用最新 epoch: {os.path.basename(args.lora_dir)}")
            else:
                print(f"[WEIGHT] ⚠️  未找到任何 LoRA 权重目录！")
                args.lora_dir = None # 【修复核心3】找不到时置空
        elif args.lora_dir and os.path.isdir(args.lora_dir):
            print(f"[WEIGHT] ✓ 使用 LoRA 权重：{args.lora_dir}")
        
        if args.data_dir is None:
            args.data_dir = cfg.get("data_dir")
    
    if args.use_3d_pos is None:
        args.use_3d_pos = False

    use_discrete = getattr(args, "use_discrete_3d_tokens", False)

    if not args.vae_config or not args.vae_ckpt:
        parser.error("Require --vae_config, --vae_ckpt (or --config)")
    if not use_discrete and not args.projector_ckpt:
        parser.error("Require --projector_ckpt (or --config) unless evaluating discrete-token run")

    args.vl_model = args.vl_model or "Qwen/Qwen2-VL-2B-Instruct"
    if args.data_dir and not os.path.isabs(args.data_dir):
        args.data_dir = os.path.normpath(os.path.join(PROJECT_ROOT, args.data_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========================================================================
    # 【修复核心4】动态查找最新的 tokenizer 目录
    # ========================================================================
    discrete_tok_path = None
    if use_discrete and run_d:
        tok_final = os.path.join(run_d, "tokenizer_final")
        if os.path.isdir(tok_final):
            discrete_tok_path = tok_final
        else:
            tok_dirs = glob.glob(os.path.join(run_d, "tokenizer_epoch*"))
            best_epoch, best_tok = -1, None
            for d in tok_dirs:
                m = re.match(r"tokenizer_epoch(\d+)$", os.path.basename(d))
                if m and os.path.isdir(d):
                    e = int(m.group(1))
                    if e > best_epoch:
                        best_epoch, best_tok = e, d
            discrete_tok_path = best_tok

    print("=" * 80)
    print("[WEIGHT LOADING INFO] 最终使用的权重路径:")
    print(f"  VAE Config: {args.vae_config}")
    print(f"  VAE Checkpoint: {args.vae_ckpt}")
    print(f"  VL Model: {args.vl_model}")
    if not use_discrete:
        print(f"  Projector: {args.projector_ckpt}")
    if args.lora_dir:
        print(f"  LoRA: {args.lora_dir}")
    if use_discrete and discrete_tok_path:
        print(f"  Tokenizer (discrete): {discrete_tok_path}")
    print("=" * 80)

    vae_model = load_vae_from_config(args.vae_config, args.vae_ckpt, device)
    from vae_qwen3vl import Qwen3VLWith3DBranch, resize_token_embeddings_and_init_mesh

    projector_layers = args.projector_layers if args.projector_layers is not None else 1
    model = Qwen3VLWith3DBranch(
        model_name_or_path=args.vl_model,
        vae_model=vae_model,
        max_3d_tokens=args.max_3d_tokens,
        use_3d_pos=args.use_3d_pos,
        projector_num_layers=projector_layers,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        truncate_mode=args.truncate_mode or "fps",
        use_discrete_3d_tokens=use_discrete,
    )
    
    if not use_discrete and model.projector is not None and args.projector_ckpt:
        print(f"[WEIGHT] 正在加载 Projector 权重：{args.projector_ckpt}")
        ckpt = torch.load(args.projector_ckpt, map_location="cpu")
        model.projector.load_state_dict(ckpt, strict=True)
        print(f"[WEIGHT] ✓ Projector 权重加载完成")

    model = model.to(device)

    tokenizer = None
    if use_discrete and discrete_tok_path and os.path.isdir(discrete_tok_path):
        tokenizer = AutoTokenizer.from_pretrained(discrete_tok_path, trust_remote_code=True)
        resize_token_embeddings_and_init_mesh(model, tokenizer)
        print(f"[WEIGHT] ✓ 加载扩展 Tokenizer: {discrete_tok_path}")
        print(f"[WEIGHT]   词表大小：{len(tokenizer)}")
    else:
        try:
            tokenizer = model.vl_model.get_tokenizer() if hasattr(model.vl_model, "get_tokenizer") else None
        except Exception:
            tokenizer = None
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.vl_model, trust_remote_code=True)
                print(f"[WEIGHT] ✓ 使用默认 Tokenizer: {args.vl_model}")
            except Exception:
                from transformers import Qwen2VLProcessor
                processor = Qwen2VLProcessor.from_pretrained(args.vl_model)
                tokenizer = processor.tokenizer
                print(f"[WEIGHT] ✓ 使用 Processor Tokenizer: {args.vl_model}")

    _lora_loaded = False
    if args.lora_dir and os.path.isdir(args.lora_dir):
        try:
            from peft import PeftModel
            model.vl_model = PeftModel.from_pretrained(model.vl_model, args.lora_dir)
            _lora_loaded = _verify_and_fix_lora_loading(model, args.lora_dir, device)
        except Exception as e:
            print(f"[WEIGHT] ⚠️  PeftModel.from_pretrained 失败 ({e})，尝试手动加载...")
            _lora_loaded = _manual_load_lora(model, args.lora_dir, device)

        if _lora_loaded:
            print(f"[WEIGHT] ✓ LoRA 权重已验证生效：{args.lora_dir}")
        else:
            print(f"[WEIGHT] ✗ LoRA 权重未能正确加载！模型将以 base model 运行（输出大概率无意义）")
            print(f"[WEIGHT]   请检查: 1) adapter 文件是否完整  2) 模型结构是否匹配  3) PEFT 版本是否兼容")
    
    model = model.to(device)
    model.eval()

    print("=" * 80)
    print("[WEIGHT LOADING SUMMARY] 权重加载完成总结:")
    print(f"  模型模式：{'Discrete (无 Projector)' if use_discrete else 'Continuous (有 Projector)'}")
    print(f"  VAE: {args.vae_ckpt}")
    print(f"  VL Base: {args.vl_model}")
    if not use_discrete:
        print(f"  Projector: {args.projector_ckpt}")
    if args.lora_dir:
        lora_status = "✓ 已验证生效" if _lora_loaded else "✗ 加载失败（lora_B 全零）"
        print(f"  LoRA: {args.lora_dir}  [{lora_status}]")
    if use_discrete:
        print(f"  3D Token Embeddings: 已扩展 (discrete mode, tokenizer: {os.path.basename(discrete_tok_path) if discrete_tok_path else 'None'})")
    print(f"  设备：{device}")
    print("=" * 80)
    
    if args.lora_dir and not _lora_loaded:
        print("[WARNING] LoRA 未生效，生成结果将等同于未微调的 base model。建议排查后重试。")
        print("=" * 80)

    # Build eval samples
    samples = []
    if args.data_path and os.path.isfile(args.data_path):
        batch = torch.load(args.data_path, map_location=device)
        if isinstance(batch, dict) and "inputs_3d" in batch:
            inputs_3d = _to_device(batch["inputs_3d"], device)
        elif isinstance(batch, dict):
            inputs_3d = _to_device(batch, device)
        else:
            inputs_3d = batch.to(device)
        samples = [(inputs_3d, None)]
    elif args.data_dir and os.path.isdir(args.data_dir):
        from vae_qwen3vl.dataset_sdf_caption import SDF3DCaptionDataset
        import random
        dataset = SDF3DCaptionDataset(
            sdf_dir=args.data_dir,
            resolution=512,
            min_points=100,
            max_points=500000,
            max_samples=None,
        )
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:args.max_eval_samples]
        for idx in indices:
            item = dataset[idx]
            inputs_3d = _to_device(item["inputs_3d"], device)
            gt_caption = item.get("caption", "")
            samples.append((inputs_3d, gt_caption))
        print(f"Loaded {len(samples)} samples from {args.data_dir}")
    else:
        print("No data_path or data_dir; using dummy 3D.")
        N = 500
        inputs_3d = {
            "sparse_sdf": torch.randn(N, 1, device=device),
            "sparse_index": torch.randint(0, 64, (N, 3), device=device),
            "batch_idx": torch.zeros(N, dtype=torch.long, device=device),
        }
        samples = [(inputs_3d, None)]

    results = []
    for idx, (inputs_3d, gt_caption) in enumerate(samples):
        gen_info = _run_generation(
            model, inputs_3d, tokenizer, args.prompt, args.max_new_tokens, device, vae_model=vae_model,
            max_safe_3d_length=getattr(args, "max_safe_3d_length", None) or 15000,
            coord_max_3d=getattr(args, "coord_max_3d", 64),
        )
        
        generated = gen_info["text"]
        print(f"[{idx}] Generated: {generated[:80]}{'...' if len(generated) > 80 else ''}")
        if gt_caption:
            print(f"     GT: {gt_caption[:80]}{'...' if len(gt_caption) > 80 else ''}")
        
        results.append({
            "idx": idx, 
            "generated": generated, 
            "gt_caption": gt_caption or "",
            "debug_prompt_ids": gen_info.get("prompt_ids", []),
            "debug_generated_ids": gen_info.get("generated_ids", []),
            "debug_encoding_indices": gen_info.get("encoding_indices", []),
            "debug_embeds_shape": gen_info.get("embeds_3d_shape", None),
            "debug_mesh_str": gen_info.get("mesh_str", None)
        })

        if args.save_mesh and idx == 0 and hasattr(vae_model, "Decode") and hasattr(vae_model, "sparse2mesh"):
            with torch.no_grad():
                recon = vae_model.Decode(encoding_indices)
            meshes = vae_model.sparse2mesh(recon, voxel_resolution=args.voxel_resolution, mc_threshold=args.mc_threshold)
            for i, mesh in enumerate(meshes):
                out_path = os.path.join(args.output_dir, f"recon_mesh_{i}.obj")
                if hasattr(mesh, "export"):
                    mesh.export(out_path)
                else:
                    try:
                        import trimesh
                        if isinstance(mesh, dict) and "vertices" in mesh and "faces" in mesh:
                            m = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"])
                            m.export(out_path)
                    except Exception:
                        pass
                if os.path.isfile(out_path):
                    print("Saved", out_path)

    with open(os.path.join(args.output_dir, "eval_results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if len(results) == 1:
        with open(os.path.join(args.output_dir, "generated.txt"), "w") as f:
            f.write(results[0]["generated"] + "\n")

    print("Done. Results in", args.output_dir)
    
if __name__ == "__main__":
    main()
