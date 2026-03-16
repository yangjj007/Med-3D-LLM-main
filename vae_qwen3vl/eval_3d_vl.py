"""
Evaluate 3D-VL alignment: run 3D -> text generation and optionally validate mesh reconstruction.

Usage:
  # 使用配置文件（与训练一致，含 LoRA + 真实数据）
  python vae_qwen3vl/eval_3d_vl.py --config configs/3d_align_train.yaml

  # 指定路径
  python vae_qwen3vl/eval_3d_vl.py --vae_config ... --vae_ckpt ... --projector_ckpt outputs_3d_align/projector_final.pt --lora_dir outputs_3d_align/lora_final --vl_model ... --data_dir train_sdf_dataset --output_dir ./eval_out

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


def _run_generation(model, inputs_3d, tokenizer, prompt, max_new_tokens, device, vae_model=None):
    """Run 3D -> text generation for one sample."""
    if getattr(model, "use_discrete_3d_tokens", False) and vae_model is not None:
        return _run_generation_discrete(model, inputs_3d, tokenizer, prompt, max_new_tokens, device, vae_model)

    with torch.no_grad():
        embeds_3d, mask_3d, encoding_indices = model.get_3d_embeds_and_encoding_indices(inputs_3d, device=device)
    seq_3d = embeds_3d.shape[1]

    try:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}],
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

def _run_generation_discrete(model, inputs_3d, tokenizer, prompt, max_new_tokens, device, vae_model):
    """Discrete 3D token path: VAE Encode -> Text Serialization -> Chat Template -> Generate.
    序列结构与训练完全一致：
    先将 VAE codes 转换为 <mesh_start><mesh_X>...<mesh_end> 字符串，
    再与 prompt 拼接为 user_content，整体使用 tokenizer.apply_chat_template 编码。
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
    
    # 构建 mesh_str (优先尝试使用训练时带可能 Morton Sort 的方法，失败则回退至直接拼接)
    mesh_str = ""
    try:
        from vae_qwen3vl.dataset.variable_length_3d import (
            batch_encoding_indices_to_variable_length_sequences,
            variable_length_sequence_to_mesh_token_string
        )
        seq_list = batch_encoding_indices_to_variable_length_sequences(
            encoding_indices, batch_size=1, max_safe_length=16384, coord_max=64
        )
        mesh_str = variable_length_sequence_to_mesh_token_string(seq_list[0])
    except Exception:
        # 兜底逻辑：直接遍历 code 生成特殊 token 字符串
        mesh_tokens = [f"<mesh_{c.item()}>" for c in codes_3d]
        mesh_str = "<mesh_start>" + "".join(mesh_tokens) + "<mesh_end>"
    
    # 【核心修复】与训练数据加载保持 100% 一致的构建逻辑
    user_content = mesh_str + "\n" + prompt
    #user_content =  prompt
    #user_content =  "Who are you? What can you do?"
    
    try:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}, {"role": "assistant", "content": ""}],
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
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_eval_samples", type=int, default=50, help="Max samples when using data_dir")
    parser.add_argument("--output_dir", type=str, default="./eval_3d_vl_out")
    parser.add_argument("--output_run", type=str, default=None, help="训练子目录名，用于指定要评估的 run")
    parser.add_argument("--save_mesh", action="store_true")
    parser.add_argument("--voxel_resolution", type=int, default=512)
    parser.add_argument("--mc_threshold", type=float, default=0.5)
    parser.add_argument("--max_3d_tokens", type=int, default=16384)
    parser.add_argument("--truncate_mode", type=str, default=None, choices=["head", "random_sample"],
                        help="When 3D tokens exceed max_3d_tokens: head=first L; random_sample=randomly sample L (default from config or head)")
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
            "max_3d_tokens", "projector_layers", "truncate_mode", "use_discrete_3d_tokens",
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
        truncate_mode=args.truncate_mode or "head",
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

    if args.lora_dir and os.path.isdir(args.lora_dir):
        try:
            from peft import PeftModel
            model.vl_model = PeftModel.from_pretrained(model.vl_model, args.lora_dir)
            print(f"[WEIGHT] ✓ LoRA 权重加载完成：{args.lora_dir}")
        except Exception as e:
            print(f"[WEIGHT] ⚠️  LoRA 加载失败 ({e})，继续不使用 LoRA")
    
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
        print(f"  LoRA: {args.lora_dir}")
    if use_discrete:
        print(f"  3D Token Embeddings: 已扩展 (discrete mode, tokenizer: {os.path.basename(discrete_tok_path) if discrete_tok_path else 'None'})")
    print(f"  设备：{device}")
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
        dataset = SDF3DCaptionDataset(
            sdf_dir=args.data_dir,
            resolution=512,
            min_points=100,
            max_points=500000,
            max_samples=args.max_eval_samples,
        )
        import random
        indices = list(range(len(dataset)))
        random.shuffle(indices)  # 打乱索引
        for idx in indices[:args.max_eval_samples]:  # 取前 N 个随机索引
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
            model, inputs_3d, tokenizer, args.prompt, args.max_new_tokens, device, vae_model=vae_model
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
