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
import sys
import json

import torch

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


def _run_generation(model, inputs_3d, tokenizer, prompt, max_new_tokens, device):
    """Run 3D -> text generation for one sample. Returns (generated_text, encoding_indices)."""
    with torch.no_grad():
        embeds_3d, mask_3d, encoding_indices = model.get_3d_embeds_and_encoding_indices(inputs_3d, device=device)
    seq_3d = embeds_3d.shape[1]

    # 与训练一致：使用 chat 模板 + add_generation_prompt，模型在 assistant 后生成
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
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    return text, encoding_indices


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
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_eval_samples", type=int, default=20, help="Max samples when using data_dir")
    parser.add_argument("--output_dir", type=str, default="./eval_3d_vl_out")
    parser.add_argument("--output_run", type=str, default=None, help="训练子目录名，如 ep2_lr1e-4_bs2_n500_lora8，用于指定要评估的 run")
    parser.add_argument("--save_mesh", action="store_true")
    parser.add_argument("--voxel_resolution", type=int, default=256)
    parser.add_argument("--mc_threshold", type=float, default=0.2)
    parser.add_argument("--max_3d_tokens", type=int, default=2048)
    parser.add_argument("--use_3d_pos", action="store_true", default=None, help="Use 3D positional encoding")
    args, remaining = parser.parse_known_args()

    # Load config
    if args.config:
        cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    else:
        cfg_path = None
    if cfg_path and os.path.isfile(cfg_path):
        cfg = _load_config(cfg_path, PROJECT_ROOT)
        for k in ("vl_model", "vae_config", "vae_ckpt", "data_dir", "prompt", "max_3d_tokens"):
            if k in cfg and getattr(args, k, None) is None:
                setattr(args, k, cfg[k])
        def _run_dir():
            out_d = cfg.get("output_dir", "outputs_3d_align")
            if not os.path.isabs(out_d):
                out_d = os.path.join(PROJECT_ROOT, out_d)
            run = args.output_run or cfg.get("output_run")
            if run:
                out_d = os.path.join(out_d, run)
            return out_d

        run_d = _run_dir()
        if args.output_dir is None or args.output_dir == "./eval_3d_vl_out":
            args.output_dir = os.path.join(run_d, "eval_out")
        args.use_3d_pos = args.use_3d_pos if args.use_3d_pos is not None else cfg.get("use_3d_pos", False)
        if args.projector_ckpt is None:
            args.projector_ckpt = os.path.join(run_d, "projector_final.pt")
        if args.lora_dir is None and cfg.get("use_lora"):
            args.lora_dir = os.path.join(run_d, "lora_final")
        if args.data_dir is None:
            args.data_dir = cfg.get("data_dir")
    if args.use_3d_pos is None:
        args.use_3d_pos = False

    if not args.vae_config or not args.vae_ckpt or not args.projector_ckpt:
        parser.error("Require --vae_config, --vae_ckpt, --projector_ckpt (or --config)")

    args.vl_model = args.vl_model or "Qwen/Qwen2-VL-2B-Instruct"
    if args.data_dir and not os.path.isabs(args.data_dir):
        args.data_dir = os.path.normpath(os.path.join(PROJECT_ROOT, args.data_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae_model = load_vae_from_config(args.vae_config, args.vae_ckpt, device)
    from vae_qwen3vl import Qwen3VLWith3DBranch

    model = Qwen3VLWith3DBranch(
        model_name_or_path=args.vl_model,
        vae_model=vae_model,
        max_3d_tokens=args.max_3d_tokens,
        use_3d_pos=args.use_3d_pos,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    ckpt = torch.load(args.projector_ckpt, map_location="cpu")
    model.projector.load_state_dict(ckpt, strict=True)

    if args.lora_dir and os.path.isdir(args.lora_dir):
        try:
            from peft import PeftModel
            model.vl_model = PeftModel.from_pretrained(model.vl_model, args.lora_dir)
            print(f"Loaded LoRA from {args.lora_dir}")
        except Exception as e:
            print(f"LoRA load failed ({e}), continuing without LoRA")

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
        for idx in range(min(len(dataset), args.max_eval_samples)):
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
        generated, encoding_indices = _run_generation(
            model, inputs_3d, tokenizer, args.prompt, args.max_new_tokens, device
        )
        results.append({"idx": idx, "generated": generated, "gt_caption": gt_caption or ""})
        print(f"[{idx}] Generated: {generated[:80]}{'...' if len(generated) > 80 else ''}")
        if gt_caption:
            print(f"     GT: {gt_caption[:80]}{'...' if len(gt_caption) > 80 else ''}")

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
