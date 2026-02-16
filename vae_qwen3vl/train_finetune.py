"""
Fine-tune the 3D projector (and optional LoRA) on Qwen2-VL / Qwen3-VL checkpoint
using 3D-text paired data. VAE is frozen.
"""

import argparse
import os
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root for trellis
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


class Dummy3DTextDataset(Dataset):
    """
    Placeholder dataset: yields (feats_3d, coords_3d, input_ids, attention_mask, labels).
    latent_dim must match the VAE codebook dimension (e.g. 16). Replace with your
    3D-text pairs for real alignment training.
    """

    def __init__(
        self,
        num_samples: int = 100,
        max_3d_tokens: int = 512,
        seq_len: int = 64,
        vocab_size: int = 152064,
        latent_dim: int = 16,
    ):
        self.num_samples = num_samples
        self.max_3d_tokens = max_3d_tokens
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        N = torch.randint(100, self.max_3d_tokens + 1, (1,)).item()
        feats_3d = torch.randn(N, self.latent_dim)
        coords_3d = torch.cat([
            torch.zeros(N, 1, dtype=torch.long),
            torch.randint(0, 64, (N, 3)),
        ], dim=1)
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        attention_mask = torch.ones(self.seq_len)
        labels = input_ids.clone()
        labels[:-1] = -100
        return {
            "feats_3d": feats_3d,
            "coords_3d": coords_3d,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_3d_text(batch, latent_dim: int = 16):
    """Collate batch: pad 3D to max N in batch, stack text. latent_dim must match projector."""
    feats = [b["feats_3d"] for b in batch]
    coords = [b["coords_3d"] for b in batch]
    max_n = max(f.size(0) for f in feats)
    pad_feats = []
    pad_coords = []
    for i, (f, c) in enumerate(zip(feats, coords)):
        n = f.size(0)
        if n < max_n:
            f = torch.cat([f, torch.zeros(max_n - n, latent_dim)], dim=0)
            c = torch.cat([c, torch.zeros(max_n - n, 4, dtype=c.dtype)], dim=0)
        pad_feats.append(f)
        pad_coords.append(c)
    return {
        "feats_3d": torch.stack(pad_feats),
        "coords_3d": torch.stack(pad_coords),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune 3D projector on Qwen2-VL / Qwen3-VL")
    parser.add_argument("--vl_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace model name or path")
    parser.add_argument("--vae_config", type=str, default=None, help="Path to VAE config JSON (optional)")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="Path to VAE checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="./outputs_3d_vl", help="Output directory")
    parser.add_argument("--max_3d_tokens", type=int, default=2048, help="Max 3D token length")
    parser.add_argument("--use_3d_pos", action="store_true", help="Use 3D positional encoding")
    parser.add_argument("--projector_layers", type=int, default=1, help="Projector MLP layers (1 = Linear)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for projector")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA on LLM (requires peft)")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--dummy_data", action="store_true", help="Use dummy dataset for testing")
    parser.add_argument("--dummy_samples", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae_model = None
    if args.vae_config and args.vae_ckpt:
        vae_model = load_vae_from_config(args.vae_config, args.vae_ckpt, device)

    from vae_qwen3vl import Qwen3VLWith3DBranch

    model = Qwen3VLWith3DBranch(
        model_name_or_path=args.vl_model,
        vae_model=vae_model,
        latent_dim=16,
        max_3d_tokens=args.max_3d_tokens,
        use_3d_pos=args.use_3d_pos,
        projector_num_layers=args.projector_layers,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    latent_dim = model.projector.latent_dim

    for name, p in model.named_parameters():
        if "projector" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    if args.use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model.vl_model = get_peft_model(model.vl_model, lora_config)
            for name, p in model.vl_model.named_parameters():
                if "lora" in name:
                    p.requires_grad = True
        except ImportError:
            print("peft not installed; skipping LoRA")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    if args.dummy_data:
        dataset = Dummy3DTextDataset(
            num_samples=args.dummy_samples,
            max_3d_tokens=min(256, args.max_3d_tokens),
            latent_dim=latent_dim,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_3d_text(b, latent_dim=latent_dim),
        )
    else:
        raise ValueError("Provide your 3D-text dataset. Use --dummy_data for a quick test.")

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            use_inputs_3d = "inputs_3d" in batch and batch["inputs_3d"] is not None
            if use_inputs_3d:
                outputs = model.forward_with_3d(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    inputs_3d=batch["inputs_3d"],
                )
            else:
                outputs = model.forward_with_3d(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    feats_3d=batch["feats_3d"],
                    coords_3d=batch["coords_3d"],
                )
            loss = getattr(outputs, "loss", None) or outputs.get("loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            if step % 10 == 0:
                print(f"Epoch {epoch} step {step} loss {loss.item():.4f}")
        ckpt_path = os.path.join(args.output_dir, f"projector_epoch{epoch}.pt")
        torch.save(model.projector.state_dict(), ckpt_path)
        print(f"Saved {ckpt_path}")

    torch.save(model.projector.state_dict(), os.path.join(args.output_dir, "projector_final.pt"))
    with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
