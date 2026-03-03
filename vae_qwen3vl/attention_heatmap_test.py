"""
测试：大模型第一层和最后一层对 3D 前缀序列的 Cross-Attention / Self-Attention 权重热力图。

若「文本 token 对 8192 个 3D token 的注意力权重几乎全是 0」，则说明模型可能已把 3D 特征静音，
主要依赖 user prompt 在生成。

Usage:
  # 使用与训练一致的 config（会从 output_run 解析 projector / LoRA 路径）
  python vae_qwen3vl/attention_heatmap_test.py --config configs/3d_align_train.yaml --output_run ep30_lr5e-5_bs2_nall_lora16_20260226_132718

  # 或显式指定
  python vae_qwen3vl/attention_heatmap_test.py --vae_config configs/vae/sdf_vqvae_stage2.json --vae_ckpt outputs/.../vqvae_step0000100.pt \\
    --projector_ckpt outputs_3d_align/.../projector_epoch9.pt --lora_dir outputs_3d_align/.../lora_epoch9 --vl_model /yangjunjie/model_qwen3vl_2B \\
    --data_dir train_sdf_dataset --max_3d_tokens 8192 --projector_layers 3
"""

import argparse
import os
import re
import sys

os.environ.setdefault("SPARSE_BACKEND", "torchsparse")

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _resolve_latest_projector_ckpt(run_dir: str) -> str | None:
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


def _load_config(config_path: str, project_root: str) -> dict:
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
    import json
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


def run_forward_with_attentions(model, inputs_3d, tokenizer, prompt, device):
    """
    做一次前向（不生成），返回 combined_embeds 对应的 attentions，以及 seq_3d。
    """
    with torch.no_grad():
        embeds_3d, mask_3d, _ = model.get_3d_embeds_and_encoding_indices(inputs_3d, device=device)
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

    with torch.no_grad():
        outputs = model.vl_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_attentions=True,
        )
    # attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("vl_model 未返回 attentions，请确认模型支持 output_attentions=True")
    return attentions, seq_3d, combined_embeds.shape[1], prompt_ids.shape[1]


def extract_text_to_3d_attention(attn_tensor, seq_3d):
    """
    attn_tensor: (batch, num_heads, seq_len, seq_len)
    取「文本位置（query）对 3D 位置（key）」的注意力：attn[:, :, seq_3d:, :seq_3d]
    返回 (num_heads, prompt_len, seq_3d)，再对 head 取平均 -> (prompt_len, seq_3d)
    """
    b, nh, L, _ = attn_tensor.shape
    text_to_3d = attn_tensor[0, :, seq_3d:, :seq_3d].cpu().float().numpy()
    return text_to_3d


def print_attention_stats(name, attn_2d, seq_3d, threshold=1e-5):
    """attn_2d: (num_text_pos, num_3d_tokens)，已对 head 平均或取某一 head."""
    flat = attn_2d.ravel()
    mean_ = float(np.mean(flat))
    max_ = float(np.max(flat))
    min_ = float(np.min(flat))
    frac_near_zero = float(np.mean(flat < threshold))
    print(f"  [{name}] 对 3D token 注意力: mean={mean_:.2e}, max={max_:.2e}, min={min_:.2e}, "
          f"比例 < {threshold:.0e}: {frac_near_zero*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="3D-VL 注意力热力图：首层/末层文本→3D 注意力")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_run", type=str, default=None)
    parser.add_argument("--vl_model", type=str, default=None)
    parser.add_argument("--vae_config", type=str, default=None)
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--projector_ckpt", type=str, default=None)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Describe this 3D shape in one sentence:")
    parser.add_argument("--max_3d_tokens", type=int, default=8192)
    parser.add_argument("--truncate_mode", type=str, default=None, choices=["head", "random_sample"])
    parser.add_argument("--use_3d_pos", action="store_true", default=None)
    parser.add_argument("--projector_layers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="热力图保存目录，默认 run_dir/attention_heatmaps")
    parser.add_argument("--sample_index", type=int, default=0, help="使用 data_dir 时第几个样本")
    args = parser.parse_args()

    if args.config:
        cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    else:
        cfg_path = None
    if cfg_path and os.path.isfile(cfg_path):
        cfg = _load_config(cfg_path, PROJECT_ROOT)
        for k in ("vl_model", "vae_config", "vae_ckpt", "data_dir", "prompt", "max_3d_tokens", "projector_layers", "truncate_mode"):
            if k in cfg and getattr(args, k, None) is None:
                setattr(args, k, cfg[k])
        out_d = cfg.get("output_dir", "outputs_3d_align")
        if not os.path.isabs(out_d):
            out_d = os.path.join(PROJECT_ROOT, out_d)
        run_d = os.path.join(out_d, args.output_run or cfg.get("output_run", ""))
        if args.output_dir is None:
            args.output_dir = os.path.join(run_d, "attention_heatmaps")
        args.use_3d_pos = args.use_3d_pos if args.use_3d_pos is not None else cfg.get("use_3d_pos", False)
        if args.projector_ckpt is None:
            args.projector_ckpt = os.path.join(run_d, "projector_final.pt")
        if not os.path.isfile(args.projector_ckpt):
            latest = _resolve_latest_projector_ckpt(run_d)
            if latest:
                args.projector_ckpt = latest
                print(f"[Attention Test] 使用最新 projector: {os.path.basename(args.projector_ckpt)}")
        if args.lora_dir is None and cfg.get("use_lora"):
            args.lora_dir = os.path.join(run_d, "lora_final")
        if args.lora_dir and not os.path.isdir(args.lora_dir):
            latest_lora = _resolve_latest_lora_dir(run_d)
            if latest_lora:
                args.lora_dir = latest_lora
                print(f"[Attention Test] 使用最新 LoRA: {os.path.basename(args.lora_dir)}")
        if args.data_dir is None:
            args.data_dir = cfg.get("data_dir")
    if args.use_3d_pos is None:
        args.use_3d_pos = False

    if not args.vae_config or not args.vae_ckpt or not args.projector_ckpt:
        parser.error("需要 --vae_config, --vae_ckpt, --projector_ckpt（或 --config + 含 projector 的 run）")
    if not os.path.isfile(args.projector_ckpt):
        parser.error(f"projector 权重文件不存在: {args.projector_ckpt}")

    args.vl_model = args.vl_model or "Qwen/Qwen2-VL-2B-Instruct"
    if args.data_dir and not os.path.isabs(args.data_dir):
        args.data_dir = os.path.normpath(os.path.join(PROJECT_ROOT, args.data_dir))
    os.makedirs(args.output_dir or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae_model = load_vae_from_config(args.vae_config, args.vae_ckpt, device)
    from vae_qwen3vl import Qwen3VLWith3DBranch

    # 必须用 eager 注意力才能返回 output_attentions（sdpa 不支持）
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        from transformers import Qwen3VLForConditionalGeneration
        vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.vl_model,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
    except Exception:
        from transformers import Qwen2VLForConditionalGeneration
        vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.vl_model,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
    print("已使用 attn_implementation='eager' 加载 VL 模型以支持 output_attentions")

    projector_layers = args.projector_layers if args.projector_layers is not None else 1
    model = Qwen3VLWith3DBranch(
        use_vl_model=vl_model,
        vae_model=vae_model,
        max_3d_tokens=args.max_3d_tokens,
        use_3d_pos=args.use_3d_pos,
        projector_num_layers=projector_layers,
        truncate_mode=args.truncate_mode or "head",
    )
    ckpt = torch.load(args.projector_ckpt, map_location="cpu")
    model.projector.load_state_dict(ckpt, strict=True)

    if args.lora_dir and os.path.isdir(args.lora_dir):
        try:
            from peft import PeftModel
            model.vl_model = PeftModel.from_pretrained(model.vl_model, args.lora_dir)
            print(f"已加载 LoRA: {args.lora_dir}")
        except Exception as e:
            print(f"LoRA 加载失败 ({e})，继续无 LoRA")

    model = model.to(device)
    model.eval()

    try:
        tokenizer = model.vl_model.get_tokenizer() if hasattr(model.vl_model, "get_tokenizer") else None
    except Exception:
        tokenizer = None
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.vl_model, trust_remote_code=True)
        except Exception:
            from transformers import Qwen2VLProcessor
            processor = Qwen2VLProcessor.from_pretrained(args.vl_model)
            tokenizer = processor.tokenizer

    # 一个样本
    if args.data_path and os.path.isfile(args.data_path):
        batch = torch.load(args.data_path, map_location=device)
        if isinstance(batch, dict) and "inputs_3d" in batch:
            inputs_3d = _to_device(batch["inputs_3d"], device)
        elif isinstance(batch, dict):
            inputs_3d = _to_device(batch, device)
        else:
            inputs_3d = batch.to(device)
    elif args.data_dir and os.path.isdir(args.data_dir):
        from vae_qwen3vl.dataset_sdf_caption import SDF3DCaptionDataset
        dataset = SDF3DCaptionDataset(
            sdf_dir=args.data_dir,
            resolution=512,
            min_points=100,
            max_points=500000,
            max_samples=max(10, args.sample_index + 1),
        )
        item = dataset[args.sample_index]
        inputs_3d = _to_device(item["inputs_3d"], device)
    else:
        N = 500
        inputs_3d = {
            "sparse_sdf": torch.randn(N, 1, device=device),
            "sparse_index": torch.randint(0, 64, (N, 3), device=device),
            "batch_idx": torch.zeros(N, dtype=torch.long, device=device),
        }

    print("运行一次 forward（output_attentions=True）...")
    attentions, seq_3d, full_len, prompt_len = run_forward_with_attentions(
        model, inputs_3d, tokenizer, args.prompt, device
    )
    num_layers = len(attentions)
    print(f"序列长度: 3D={seq_3d}, prompt={prompt_len}, 总={full_len}; 层数={num_layers}")

    # 首层、末层
    first_layer_attn = attentions[0]
    last_layer_attn = attentions[-1]
    text_to_3d_first = extract_text_to_3d_attention(first_layer_attn, seq_3d)
    text_to_3d_last = extract_text_to_3d_attention(last_layer_attn, seq_3d)
    # text_to_3d_*: (num_heads, prompt_len, seq_3d)
    mean_over_heads_first = np.mean(text_to_3d_first, axis=0)
    mean_over_heads_last = np.mean(text_to_3d_last, axis=0)

    print("\n--- 文本 token 对 3D token 的注意力统计（对 head 取平均）---")
    print_attention_stats("第一层", mean_over_heads_first, seq_3d)
    print_attention_stats("最后一层", mean_over_heads_last, seq_3d)

    # 热力图（使用英文标签避免 CJK 字体缺失警告）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过保存热力图")
        return

    out_dir = args.output_dir or "."
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 3D token 可能很多，下采样显示
    step_3d = max(1, seq_3d // 512)
    x_slice = np.arange(0, seq_3d, step_3d)
    plot_first = mean_over_heads_first[:, x_slice]
    plot_last = mean_over_heads_last[:, x_slice]

    im0 = axes[0].imshow(plot_first, aspect="auto", cmap="hot", interpolation="nearest")
    axes[0].set_title("Layer 0: text -> 3D attention")
    axes[0].set_xlabel("3D token index (downsampled)")
    axes[0].set_ylabel("Text token index")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(plot_last, aspect="auto", cmap="hot", interpolation="nearest")
    axes[1].set_title("Last Layer: text -> 3D attention")
    axes[1].set_xlabel("3D token index (downsampled)")
    axes[1].set_ylabel("Text token index")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    path = os.path.join(out_dir, "attention_text_to_3d_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"热力图已保存: {path}")

    # 额外：仅看「最后一个文本位置」对 3D 的注意力分布（最能反映生成时是否看 3D）
    last_text_row_first = mean_over_heads_first[-1]
    last_text_row_last = mean_over_heads_last[-1]
    print("\n--- 最后一个文本位置（assistant 首 token 前）对 3D 的注意力 ---")
    print_attention_stats("第一层-最后位置", last_text_row_first[np.newaxis, :], seq_3d)
    print_attention_stats("最后一层-最后位置", last_text_row_last[np.newaxis, :], seq_3d)

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 2))
    ax2.plot(last_text_row_first, label="Layer 0", alpha=0.8)
    ax2.plot(last_text_row_last, label="Last layer", alpha=0.8)
    ax2.set_xlabel("3D token index")
    ax2.set_ylabel("Attention weight")
    ax2.legend()
    ax2.set_title("Last text position -> 3D tokens attention")
    path2 = os.path.join(out_dir, "attention_last_text_to_3d.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"最后一位置→3D 曲线已保存: {path2}")

    # 结论提示
    th = 1e-5
    frac_first = np.mean(mean_over_heads_first.ravel() < th)
    frac_last = np.mean(mean_over_heads_last.ravel() < th)
    if frac_first > 0.95 and frac_last > 0.95:
        print("\n>>> 结论：文本 token 对 3D 的注意力绝大部分接近 0，与「模型把 3D 静音、主要靠 prompt 硬猜」一致。")
    else:
        print("\n>>> 结论：存在一定比例的文本→3D 注意力，模型仍有利用 3D 特征。")


if __name__ == "__main__":
    main()
