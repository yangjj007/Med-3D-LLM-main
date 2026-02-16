"""
Evaluate 3D-VL alignment: run 3D -> text generation and optionally validate mesh reconstruction.

Usage:
  # 3D -> text generation only
  python vae_qwen3vl/eval_3d_vl.py --vae_config configs/vae/sdf_vqvae_stage1.json --vae_ckpt path/to/vae.pt --projector_ckpt path/to/projector_final.pt --vl_model Qwen/Qwen2-VL-2B-Instruct --data_path path/to/one_batch.pt --output_dir ./eval_out

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D-VL: generation and optional mesh reconstruction")
    parser.add_argument("--vl_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--vae_config", type=str, required=True, help="Path to VAE config JSON")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--projector_ckpt", type=str, required=True, help="Path to projector checkpoint (e.g. projector_final.pt)")
    parser.add_argument("--data_path", type=str, default=None, help="Path to a .pt file containing one batch dict (sparse_sdf, sparse_index, batch_idx). If None, use dummy 3D for quick test.")
    parser.add_argument("--prompt", type=str, default="Describe this 3D shape in one sentence:", help="Text prompt after 3D tokens")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="./eval_3d_vl_out")
    parser.add_argument("--save_mesh", action="store_true", help="Decode 3D to mesh and save to output_dir")
    parser.add_argument("--voxel_resolution", type=int, default=256, help="For mesh extraction")
    parser.add_argument("--mc_threshold", type=float, default=0.2, help="Marching cubes threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae_model = load_vae_from_config(args.vae_config, args.vae_ckpt, device)
    from vae_qwen3vl import Qwen3VLWith3DBranch

    model = Qwen3VLWith3DBranch(
        model_name_or_path=args.vl_model,
        vae_model=vae_model,
        max_3d_tokens=2048,
        use_3d_pos=False,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    ckpt = torch.load(args.projector_ckpt, map_location="cpu")
    model.projector.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()

    if args.data_path and os.path.isfile(args.data_path):
        batch = torch.load(args.data_path, map_location=device)
        if isinstance(batch, dict):
            inputs_3d = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        else:
            inputs_3d = batch.to(device)
    else:
        print("No data_path or file not found; using dummy 3D batch for quick test.")
        N = 500
        inputs_3d = {
            "sparse_sdf": torch.randn(N, 1, device=device),
            "sparse_index": torch.randint(0, 64, (N, 3), device=device),
            "batch_idx": torch.zeros(N, dtype=torch.long, device=device),
        }

    with torch.no_grad():
        embeds_3d, mask_3d, encoding_indices = model.get_3d_embeds_and_encoding_indices(inputs_3d, device=device)
    seq_3d = embeds_3d.shape[1]

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

    prompt_ids = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)
    embed_tokens = model.vl_model.get_input_embeddings()
    text_embeds = embed_tokens(prompt_ids)
    combined_embeds = torch.cat([embeds_3d, text_embeds], dim=1)
    prompt_mask = torch.ones(1, prompt_ids.shape[1], dtype=combined_embeds.dtype, device=device)
    combined_mask = torch.cat([mask_3d.to(combined_embeds.dtype), prompt_mask], dim=1)

    prompt_len = prompt_ids.shape[1]
    with torch.no_grad():
        out = model.vl_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Output is full sequence; generated part is after 3D + prompt
    start = seq_3d + prompt_len
    generated_ids = out[:, start:] if out.shape[1] > start else out
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    print("Generated:", text)
    with open(os.path.join(args.output_dir, "generated.txt"), "w") as f:
        f.write(text + "\n")

    if args.save_mesh and hasattr(vae_model, "Decode") and hasattr(vae_model, "sparse2mesh"):
        with torch.no_grad():
            recon = vae_model.Decode(encoding_indices)
        meshes = vae_model.sparse2mesh(recon, voxel_resolution=args.voxel_resolution, mc_threshold=args.mc_threshold)
        for i, mesh in enumerate(meshes):
            out_path = os.path.join(args.output_dir, f"recon_mesh_{i}.obj")
            if hasattr(mesh, "export"):
                mesh.export(out_path)
            else:
                import trimesh
                if isinstance(mesh, dict) and "vertices" in mesh and "faces" in mesh:
                    m = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"])
                    m.export(out_path)
                else:
                    print("Mesh format not recognized, skipping save.")
            print("Saved", out_path)
    elif args.save_mesh:
        print("VAE has no Decode/sparse2mesh; skipping mesh save.")

    print("Done. Results in", args.output_dir)


if __name__ == "__main__":
    main()
