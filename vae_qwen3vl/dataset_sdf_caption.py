"""
3D SDF + Caption Dataset for LLM-VAE alignment training.

Loads preprocessed sparse SDF (.npz) from train_sdf_dataset and pairs with captions
from metadata.csv. Returns inputs_3d (sparse_sdf, sparse_index, batch_idx) + text
for forward_with_3d.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional


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
