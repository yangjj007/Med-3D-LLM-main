"""
Fine-tune the 3D projector (and optional LoRA) on Qwen2-VL / Qwen3-VL checkpoint
using 3D-text paired data. VAE is frozen.
Supports multi-GPU training via HuggingFace Accelerate.
"""

import argparse
import os

# 确保 3D-VL 训练使用 spconv（新终端直接 python 运行时不会走 shell 脚本的 export）
os.environ.setdefault("SPARSE_BACKEND", "spconv")

import sys
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root for trellis
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


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


def _training_log_stream(log_path: str, orig_stdout, echo_to_console: bool = True):
    """
    Returns a file-like object that:
    - Writes all output to log_path
    - If echo_to_console: echoes to orig_stdout only training-related lines (filters out [DEBUG])
    - 多卡时仅 rank0 echo_to_console=True，避免多进程输出交错导致控制台格式混乱
    """

    class TrainingLogTee:
        def __init__(self, path, original, echo):
            self._log = open(path, "w", encoding="utf-8")
            self._orig = original
            self._echo = echo

        def write(self, data):
            self._log.write(data)
            self._log.flush()
            if self._echo and "[DEBUG" not in data and data.strip():
                self._orig.write(data)
                self._orig.flush()

        def flush(self):
            self._log.flush()
            if self._echo:
                self._orig.flush()

        def close(self):
            self._log.close()

    return TrainingLogTee(log_path, orig_stdout, echo_to_console)


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


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


def _make_output_subdir(args) -> str:
    """
    根据关键训练参数生成输出子目录名，便于多次训练不混淆。
    格式: ep{epochs}_lr{lr}_bs{batch_size}_n{samples}_lora{lora_r}_{timestamp}
    """
    lr_str = f"{args.lr:.0e}".replace("-0", "-").replace("+0", "+")  # 1e-4, 5e-5
    n_str = "all" if args.max_samples == 0 else str(args.max_samples)
    if args.dummy_data:
        n_str = f"d{args.dummy_samples}"
    lora_str = f"lora{args.lora_r}" if args.use_lora else "nolora"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ep{args.epochs}_lr{lr_str}_bs{args.batch_size}_n{n_str}_{lora_str}_{ts}"


def _load_config(config_path: str, project_root: str) -> dict:
    """Load YAML config. Returns flattened dict (excluding nested 'accelerate')."""
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
            out["_accelerate_config"] = v
            continue
        if isinstance(v, dict):
            for sk, sv in v.items():
                out[sk] = sv
        else:
            out[k] = v
    for k in path_keys:
        if k in out and isinstance(out[k], str) and out[k] and not os.path.isabs(out[k]):
            out[k] = os.path.normpath(os.path.join(project_root, out[k]))
    return out


def main():
    parser = argparse.ArgumentParser(description="Fine-tune 3D projector on Qwen2-VL / Qwen3-VL")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (training + paths)")
    parser.add_argument("--vl_model", type=str, default=None, help="HuggingFace model name or path")
    parser.add_argument("--vae_config", type=str, default=None, help="Path to VAE config JSON")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="Path to VAE checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_3d_tokens", type=int, default=None, help="Max 3D token length")
    parser.add_argument("--truncate_mode", type=str, default="head", choices=["head", "random_sample"],
                        help="When 3D tokens exceed max_3d_tokens: head=take first L; random_sample=randomly sample L (better coverage, avoids OOM)")
    parser.add_argument("--use_3d_pos", action="store_true", help="Use 3D positional encoding")
    parser.add_argument("--projector_layers", type=int, default=1, help="Projector MLP layers (1 = Linear)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping max norm (0=disabled)")
    parser.add_argument("--use_lr_scheduler", action="store_true", help="Use cosine LR schedule with warmup")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio of total steps")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA on LLM")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha (default 2*r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--dummy_data", action="store_true", help="Use dummy dataset")
    parser.add_argument("--dummy_samples", type=int, default=50)
    parser.add_argument("--data_dir", type=str, default=None, help="Path to SDF dataset")
    parser.add_argument("--data_format", type=str, default="sdf_caption", choices=["sdf_caption"])
    parser.add_argument("--max_samples", type=int, default=2000, help="Max samples (0 = all)")
    parser.add_argument("--prompt", type=str, default="Describe this 3D shape in one sentence:", help="Caption prompt")
    parser.add_argument("--use_discrete_3d_tokens", action="store_true", help="Use discrete mesh tokens (8x8x8 pool) and no Projector")
    parser.add_argument("--use_variable_length_3d_tokens", action="store_true", help="Variable-length 3D: keep all VAE points, Morton sort, dynamic pad (use with use_discrete_3d_tokens)")
    parser.add_argument("--max_safe_3d_length", type=int, default=15000, help="Soft cap: FPS downsample only when N > this (variable-length mode)")
    parser.add_argument("--coord_max_3d", type=int, default=64, help="Max coord for Morton (current VAE: 64^3 latent=64; if VAE outputs 512^3 use 512)")
    parser.add_argument("--max_length_variable", type=int, default=32768, help="Max token length when variable-length 3D (avoid truncating 8k~12k mesh)")
    parser.add_argument("--reconstruction_ratio", type=float, default=0.0, help="Fraction of samples as 3D reconstruction task (0..1), only when use_discrete_3d_tokens")
    parser.add_argument("--training_stage", type=str, default="sft", choices=["warmup", "sft"], help="warmup = only embed+lm_head; sft = LoRA + embed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation (recommend 4~8 for variable-length with batch_size=1)")
    parser.add_argument("--use_flash_attn_2", action="store_true", help="Use Flash Attention 2 for long sequences (variable-length)")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Gradient checkpointing to save VRAM (variable-length)")
    args, remaining = parser.parse_known_args()
    known_keys = {a.dest for a in parser._actions if a.dest != "help"}
    if args.config and os.path.isfile(args.config):
        cfg = _load_config(args.config, PROJECT_ROOT)
        overrides = {k: v for k, v in cfg.items() if not k.startswith("_") and k in known_keys}
        parser.set_defaults(**overrides)
        args = parser.parse_args(remaining)
    else:
        args = parser.parse_args(remaining if remaining else [])

    accelerator = Accelerator(
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
    ) if ACCELERATE_AVAILABLE else None
    device = accelerator.device if accelerator is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if accelerator is not None:
        set_seed(42)
        accelerator.print(f"Accelerate: {accelerator.num_processes} processes, device={device}")

    output_base = args.output_dir or os.path.join(PROJECT_ROOT, "outputs_3d_align")
    args.output_dir = os.path.join(output_base, _make_output_subdir(args))
    if accelerator is None or accelerator.is_main_process:
        (accelerator.print if accelerator else print)(f"Output dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}.log")
    _orig_stdout = sys.stdout
    _log_tee = _training_log_stream(log_path, _orig_stdout, echo_to_console=(rank == 0))
    sys.stdout = _log_tee
    if accelerator is None or accelerator.is_main_process:
        (accelerator.print if accelerator else print)(f"[Train] Full log -> {log_path}")

    metrics_file = None
    use_discrete = getattr(args, "use_discrete_3d_tokens", False)
    training_stage = getattr(args, "training_stage", "sft")
    try:
        vae_model = None
        if args.vae_config and args.vae_ckpt:
            vae_model = load_vae_from_config(args.vae_config, args.vae_ckpt, str(device))

        from transformers import AutoTokenizer
        from vae_qwen3vl import Qwen3VLWith3DBranch, add_mesh_tokens_to_tokenizer, resize_token_embeddings_and_init_mesh

        tokenizer = AutoTokenizer.from_pretrained(args.vl_model, trust_remote_code=True)
        if use_discrete:
            add_mesh_tokens_to_tokenizer(tokenizer)

        vl_kwargs = {}
        if getattr(args, "use_flash_attn_2", False):
            vl_kwargs["attn_implementation"] = "flash_attention_2"
        model = Qwen3VLWith3DBranch(
            model_name_or_path=args.vl_model,
            vae_model=vae_model,
            latent_dim=16,
            max_3d_tokens=args.max_3d_tokens or 512,
            use_3d_pos=args.use_3d_pos,
            projector_num_layers=args.projector_layers,
            torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
            truncate_mode=args.truncate_mode,
            use_discrete_3d_tokens=use_discrete,
            **vl_kwargs,
        )
        model = model.to(device)
        if getattr(args, "use_gradient_checkpointing", False) and hasattr(model.vl_model, "gradient_checkpointing_enable"):
            model.vl_model.gradient_checkpointing_enable()
        if use_discrete:
            resize_token_embeddings_and_init_mesh(model, tokenizer)

        latent_dim = model.projector.latent_dim if model.projector is not None else 16

        # Stage1 warmup: only embedding + lm_head; Stage2 sft: projector (if any) or LoRA + embed
        for name, p in model.named_parameters():
            p.requires_grad = False
        if training_stage == "warmup":
            for name, p in model.named_parameters():
                if "embed_tokens" in name or "embedding" in name or "lm_head" in name:
                    p.requires_grad = True
        else:
            if model.projector is not None:
                for name, p in model.named_parameters():
                    if "projector" in name:
                        p.requires_grad = True
            for name, p in model.named_parameters():
                if "embed_tokens" in name or "embedding" in name or "lm_head" in name:
                    p.requires_grad = True
        use_lora_applied = False
        if args.use_lora and training_stage == "sft":
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                lora_alpha = args.lora_alpha if args.lora_alpha is not None else (2 * args.lora_r)
                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                model.vl_model = get_peft_model(model.vl_model, lora_config)
                for name, p in model.vl_model.named_parameters():
                    if "lora" in name:
                        p.requires_grad = True
                use_lora_applied = True
                (accelerator.print if accelerator else print)("LoRA applied successfully.")
            except ImportError:
                (accelerator.print if accelerator else print)("peft not installed; skipping LoRA")

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

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
        elif args.data_dir:
            from vae_qwen3vl.dataset_sdf_caption import (
                SDF3DCaptionDataset,
                collate_sdf_caption,
                collate_sdf_caption_discrete,
            )
            if not use_discrete:
                tokenizer = AutoTokenizer.from_pretrained(args.vl_model, trust_remote_code=True)
            dataset = SDF3DCaptionDataset(
                sdf_dir=args.data_dir,
                resolution=512,
                min_points=100,
                max_points=500000,
                max_samples=args.max_samples if args.max_samples > 0 else None,
            )
            if use_discrete:
                if vae_model is None:
                    raise ValueError("Discrete 3D token mode requires --vae_config and --vae_ckpt for collate encoding.")
                reconstruction_ratio = getattr(args, "reconstruction_ratio", 0.0)
                use_variable_length_3d = getattr(args, "use_variable_length_3d_tokens", False)
                collate_fn = lambda b: collate_sdf_caption_discrete(
                    b,
                    tokenizer,
                    vae_model,
                    device,
                    prompt=args.prompt,
                    max_length=2048,
                    reconstruction_ratio=reconstruction_ratio,
                    use_variable_length_3d=use_variable_length_3d,
                    max_safe_3d_length=getattr(args, "max_safe_3d_length", 15000),
                    coord_max_3d=getattr(args, "coord_max_3d", 64),
                    max_length_variable=getattr(args, "max_length_variable", 32768) if use_variable_length_3d else None,
                )
            else:
                collate_fn = lambda b: collate_sdf_caption(b, tokenizer, prompt=args.prompt, max_length=256)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
            )
        else:
            if use_discrete:
                raise ValueError("Discrete 3D token mode requires --data_dir (no dummy_data path).")
            raise ValueError("Provide --dummy_data or --data_dir. Use --dummy_data for a quick test.")

        if accelerator is not None:
            model, opt, dataloader = accelerator.prepare(model, opt, dataloader)

        # LR scheduler (cosine with warmup)
        scheduler = None
        if getattr(args, "use_lr_scheduler", False):
            num_training_steps = len(dataloader) * args.epochs
            num_warmup_steps = int(getattr(args, "warmup_ratio", 0.05) * num_training_steps)
            try:
                from transformers import get_cosine_schedule_with_warmup
                scheduler = get_cosine_schedule_with_warmup(
                    opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
                )
                if accelerator is not None:
                    scheduler = accelerator.prepare(scheduler)
                _pr = accelerator.print if accelerator is not None else print
                _pr(f"LR scheduler: cosine, warmup {num_warmup_steps} / {num_training_steps} steps")
            except ImportError:
                pass

        _print = accelerator.print if accelerator is not None else print
        model.train()
        # 多卡时 model 被 DDP 包装，需用 .module 调用自定义方法 forward_with_3d
        _model = getattr(model, "module", model)
        global_step = 0
        # 训练指标记录（仅 main process），用于可视化
        metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
        metrics_file = open(metrics_path, "w", encoding="utf-8") if (accelerator is None or accelerator.is_main_process) else None
        for epoch in range(args.epochs):
            for step, batch in enumerate(dataloader):
                batch = {k: _to_device(v, device) for k, v in batch.items()}
                use_inputs_3d = "inputs_3d" in batch and batch["inputs_3d"] is not None
                if use_discrete or not use_inputs_3d:
                    # Discrete path: batch has input_ids, attention_mask, labels only
                    outputs = _model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                elif use_inputs_3d:
                    outputs = _model.forward_with_3d(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        inputs_3d=batch["inputs_3d"],
                    )
                else:
                    outputs = _model.forward_with_3d(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        feats_3d=batch["feats_3d"],
                        coords_3d=batch["coords_3d"],
                    )
                loss = getattr(outputs, "loss", None) or outputs.get("loss")
                opt.zero_grad()
                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                # Gradient clipping
                grad_clip = getattr(args, "grad_clip", 0.0)
                if grad_clip > 0:
                    if accelerator is not None:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                # 记录指标（用于可视化）
                if metrics_file is not None:
                    lr = opt.param_groups[0]["lr"] if opt.param_groups else 0.0
                    metrics_file.write(json.dumps({"step": global_step, "epoch": epoch, "loss": round(loss.item(), 6), "lr": lr}) + "\n")
                    metrics_file.flush()
                if step % 10 == 0:
                    _print(f"Epoch {epoch} step {step} loss {loss.item():.4f}")
            raw_model = accelerator.unwrap_model(model) if accelerator is not None else model
            if accelerator is None or accelerator.is_main_process:
                if raw_model.projector is not None:
                    ckpt_path = os.path.join(args.output_dir, f"projector_epoch{epoch}.pt")
                    torch.save(raw_model.projector.state_dict(), ckpt_path)
                    _print(f"Saved {ckpt_path}")
                if use_discrete:
                    tok_path = os.path.join(args.output_dir, f"tokenizer_epoch{epoch}")
                    tokenizer.save_pretrained(tok_path)
                    _print(f"Saved tokenizer to {tok_path}")
                if use_lora_applied:
                    lora_dir = os.path.join(args.output_dir, f"lora_epoch{epoch}")
                    raw_model.vl_model.save_pretrained(lora_dir)
                    _print(f"Saved LoRA to {lora_dir}")
            if accelerator is not None:
                accelerator.wait_for_everyone()

        raw_model = accelerator.unwrap_model(model) if accelerator is not None else model
        if accelerator is None or accelerator.is_main_process:
            if raw_model.projector is not None:
                torch.save(raw_model.projector.state_dict(), os.path.join(args.output_dir, "projector_final.pt"))
            if use_discrete:
                tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer_final"))
            if use_lora_applied:
                lora_final_dir = os.path.join(args.output_dir, "lora_final")
                raw_model.vl_model.save_pretrained(lora_final_dir)
                _print(f"Saved LoRA final to {lora_final_dir}")
            with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            _print("Done.")
    finally:
        if metrics_file is not None:
            try:
                metrics_file.close()
            except Exception:
                pass
        sys.stdout = _orig_stdout
        _log_tee.close()


if __name__ == "__main__":
    import traceback
    import sys
    def _save_error(exc_type, exc_val, exc_tb):
        if exc_type is not None:
            tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
            err_path = os.path.join(PROJECT_ROOT, "configs", f".train_error_rank{rank}.txt")
            os.makedirs(os.path.dirname(err_path), exist_ok=True)
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(tb)
    sys.excepthook = _save_error
    try:
        main()
    except BaseException as e:
        _save_error(type(e), e, e.__traceback__)
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        err_path = os.path.join(PROJECT_ROOT, "configs", f".train_error_rank{rank}.txt")
        try:
            print(f"[rank{rank}] Error saved to {err_path}", file=sys.stderr)
        except Exception:
            pass
        raise
