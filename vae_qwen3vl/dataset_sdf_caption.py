"""
3D SDF + Caption Dataset for LLM-VAE alignment training.

Loads preprocessed sparse SDF (.npz) from train_sdf_dataset and pairs with captions
from metadata.csv. Returns inputs_3d (sparse_sdf, sparse_index, batch_idx) + text
for forward_with_3d.

Discrete-token path: use collate_sdf_caption_discrete with vae_model to produce
3D mesh token strings (8x8x8 pooled) and optionally mix reconstruction task
(assistant = same 3D token sequence) via reconstruction_ratio.
"""

import os
import json
import random
import time
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Union


class SDF3DCaptionDataset(Dataset):
    """
    Dataset of 3D SDF (from .npz) + caption for alignment training.
    
    Each sample: load npz (sparse_sdf [N,1], sparse_index [N,3]) and one caption.
    Returns inputs_3d dict + will be collated with text to form full batch.
    
    Args:
        sdf_dir: Directory containing {sha256}_r{resolution}.npz and metadata.csv
        resolution: Grid resolution (default 512, must match npz files)
        min_points: Minimum sparse points per sample
        max_points: Max points (subsample if exceeded), None = no limit
        max_samples: Limit dataset size for quick experiments
    """

    def __init__(
        self,
        sdf_dir: str,
        resolution: int = 512,
        min_points: int = 100,
        max_points: Optional[int] = 500000,
        max_samples: Optional[int] = None,
    ):
        self.sdf_dir = sdf_dir
        self.resolution = resolution
        self.min_points = min_points
        self.max_points = max_points

        metadata_path = os.path.join(sdf_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

        df = pd.read_csv(metadata_path)
        # Filter: sdf_computed + has r512 data
        if "sdf_computed" in df.columns:
            df = df[df["sdf_computed"] == True]
        points_col = f"r{resolution}_num_points"
        if points_col in df.columns:
            df = df[df[points_col].notna()]
            df = df[df[points_col] >= min_points]
        if "captions" not in df.columns or df["captions"].isna().all():
            raise ValueError("metadata must have 'captions' column with text")
        df = df[df["captions"].notna()]

        self.instances = []
        for _, row in df.iterrows():
            sha256 = row["sha256"]
            npz_path = os.path.join(sdf_dir, f"{sha256}_r{resolution}.npz")
            if os.path.exists(npz_path):
                self.instances.append(
                    {"sha256": sha256, "npz_path": npz_path, "captions": row["captions"]}
                )
            if max_samples and len(self.instances) >= max_samples:
                break

        print(f"[SDF3DCaptionDataset] Loaded {len(self.instances)} samples from {sdf_dir}")

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict:
        item = self.instances[idx]
        data = np.load(item["npz_path"])
        sparse_sdf = torch.from_numpy(data["sparse_sdf"]).float()  # [N, 1]
        sparse_index = torch.from_numpy(data["sparse_index"]).long()  # [N, 3]

        if self.max_points is not None and len(sparse_sdf) > self.max_points:
            perm = torch.randperm(len(sparse_sdf))[: self.max_points]
            sparse_sdf = sparse_sdf[perm]
            sparse_index = sparse_index[perm]

        # Parse captions (JSON array)
        captions_raw = item["captions"]
        if isinstance(captions_raw, str):
            try:
                captions = json.loads(captions_raw)
            except json.JSONDecodeError:
                captions = [str(captions_raw)]
        else:
            captions = [str(captions_raw)]
        caption = np.random.choice(captions) if captions else "A 3D object."

        return {
            "inputs_3d": {
                "sparse_sdf": sparse_sdf,
                "sparse_index": sparse_index,
                "batch_idx": torch.zeros(len(sparse_sdf), dtype=torch.long),
            },
            "caption": caption,
        }


def collate_sdf_caption(
    batch: List[Dict],
    tokenizer,
    prompt: str = "Describe this 3D shape in one sentence:",
    max_length: int = 256,
) -> Dict:
    """
    Collate batch: merge inputs_3d (concat with batch_idx) + tokenize text.
    
    Text format: User asks prompt, Assistant answers with caption.
    Labels: -100 for prompt, real ids for caption (assistant part).
    """
    # Merge inputs_3d
    sparse_sdfs = []
    sparse_indices = []
    batch_indices = []
    for i, b in enumerate(batch):
        d = b["inputs_3d"]
        n = len(d["sparse_sdf"])
        sparse_sdfs.append(d["sparse_sdf"])
        sparse_indices.append(d["sparse_index"])
        batch_indices.append(torch.full((n,), i, dtype=torch.long))

    inputs_3d = {
        "sparse_sdf": torch.cat(sparse_sdfs, dim=0),
        "sparse_index": torch.cat(sparse_indices, dim=0),
        "batch_idx": torch.cat(batch_indices, dim=0),
    }

    # Build chat for each sample
    messages_list = []
    for b in batch:
        messages_list.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": b["caption"]},
        ])

    # Tokenize with chat template
    text = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=False,
    )
    if isinstance(text, str):
        text = [text]

    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Labels: -100 for prompt (user) part, real ids only for assistant reply
    labels = input_ids.clone()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    labels[labels == pad_id] = -100
    # Get token count of "user + prompt" part to mask
    try:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}],
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_len = len(prefix)
        labels[:, :prompt_len] = -100
    except Exception:
        pass

    return {
        "inputs_3d": inputs_3d,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_sdf_caption_discrete(
    batch: List[Dict],
    tokenizer: Any,
    vae_model: torch.nn.Module,
    device: Union[str, torch.device],
    prompt: str = "Describe this 3D shape in one sentence:",
    reconstruct_prompt: str = "Reconstruct this 3D shape.",
    max_length: int = 2048,
    reconstruction_ratio: float = 0.0,
    use_variable_length_3d: bool = False,
    max_safe_3d_length: int = 15000,
    coord_max_3d: int = 64,
    max_length_variable: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Collate for discrete 3D tokens: run VAE Encode, then either
    - 8x8x8 spatial pool (fixed 512 tokens), or
    - variable-length: Morton sort + optional FPS soft cap, pad to batch max via tokenizer.

    Returns input_ids, attention_mask, labels (no inputs_3d).

    Args:
        use_variable_length_3d: if True, use all VAE points with Morton sort and FPS soft cap.
        max_safe_3d_length: only FPS-downsample when N > this (e.g. 15000).
        coord_max_3d: max coord value for Morton (current VAE 64^3 -> 64; 512^3 output -> 512).
    """
    from .spatial_pool_3d import (
        batch_encoding_indices_to_pooled_sequences,
        pooled_sequence_to_mesh_token_string,
    )
    from .variable_length_3d import (
        batch_encoding_indices_to_variable_length_sequences,
        variable_length_sequence_to_mesh_token_string,
    )

    # Merge inputs_3d
    sparse_sdfs = []
    sparse_indices = []
    batch_indices = []
    for i, b in enumerate(batch):
        d = b["inputs_3d"]
        n = len(d["sparse_sdf"])
        sparse_sdfs.append(d["sparse_sdf"])
        sparse_indices.append(d["sparse_index"])
        batch_indices.append(torch.full((n,), i, dtype=torch.long))

    inputs_3d = {
        "sparse_sdf": torch.cat(sparse_sdfs, dim=0),
        "sparse_index": torch.cat(sparse_indices, dim=0),
        "batch_idx": torch.cat(batch_indices, dim=0),
    }

    # VAE Encode (no grad)
    _collate_t0 = time.time()
    vae_model = vae_model.to(device)
    vae_model.eval()
    vae_dtype = next(vae_model.parameters()).dtype
    with torch.no_grad():
        inputs_3d_device = {}
        for k, v in inputs_3d.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                if v.is_floating_point():
                    v = v.to(dtype=vae_dtype)
                inputs_3d_device[k] = v
            else:
                inputs_3d_device[k] = v
        # Encode 内部会构造新的稀疏特征，开启 autocast 以避免 Float/BFloat16 不一致。
        use_autocast = (
            torch.cuda.is_available()
            and str(device).startswith("cuda")
            and vae_dtype in (torch.float16, torch.bfloat16)
        )
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=vae_dtype)
            if use_autocast
            else nullcontext()
        )
        with autocast_ctx:
            encoding_indices = vae_model.Encode(inputs_3d_device)
    _t_vae = time.time()
    print(f"[DEBUG collate] VAE Encode took {_t_vae - _collate_t0:.3f}s", flush=True)

    batch_size = len(batch)
    if use_variable_length_3d:
        seq_list = batch_encoding_indices_to_variable_length_sequences(
            encoding_indices,
            batch_size,
            max_safe_length=max_safe_3d_length,
            coord_max=coord_max_3d,
        )
        mesh_strings = [variable_length_sequence_to_mesh_token_string(s) for s in seq_list]
    else:
        pooled_list = batch_encoding_indices_to_pooled_sequences(
            encoding_indices, batch_size
        )
        mesh_strings = [pooled_sequence_to_mesh_token_string(p) for p in pooled_list]
    _t_seq = time.time()
    print(f"[DEBUG collate] Sequence gen took {_t_seq - _t_vae:.3f}s  mesh_str_lens={[len(s) for s in mesh_strings]}", flush=True)

    messages_list = []
    for i, b in enumerate(batch):
        mesh_str = mesh_strings[i]
        if reconstruction_ratio > 0 and random.random() < reconstruction_ratio:
            user_content = mesh_str + "\n" + reconstruct_prompt
            assistant_content = mesh_str
        else:
            user_content = mesh_str + "\n" + prompt
            assistant_content = b["caption"]
        messages_list.append([
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ])

    # Variable-length 3D: use larger max_length to avoid truncating 8k~12k mesh tokens
    effective_max_length = max_length
    if use_variable_length_3d and max_length_variable is not None:
        effective_max_length = max(max_length, max_length_variable)

    _t_tok0 = time.time()
    text = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=False,
    )
    if isinstance(text, str):
        text = [text]
    _t_template = time.time()
    print(f"[DEBUG collate] Chat template took {_t_template - _t_tok0:.3f}s  text_lens={[len(t) for t in text]}", flush=True)

    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=effective_max_length,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    _t_tokenize = time.time()
    print(f"[DEBUG collate] Tokenizer encode took {_t_tokenize - _t_template:.3f}s  input_ids.shape={input_ids.shape}", flush=True)

    labels = input_ids.clone()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    labels[labels == pad_id] = -100
    for i in range(batch_size):
        try:
            prefix_tokens = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": messages_list[i][0]["content"]},
                    {"role": "assistant", "content": ""},
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
            prefix_len = len(prefix_tokens)
            if prefix_len < labels.shape[1]:
                labels[i, :prefix_len] = -100
        except Exception:
            pass
    _t_labels = time.time()
    print(f"[DEBUG collate] Label masking took {_t_labels - _t_tokenize:.3f}s", flush=True)
    print(f"[DEBUG collate] TOTAL collate took {_t_labels - _collate_t0:.3f}s", flush=True)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
