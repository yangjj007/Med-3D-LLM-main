"""
Build LLaMA-Factory ShareGPT JSONL for Qwen3-VL SFT with discrete 3D mesh tokens.

Pipeline (offline):
  1) Mesh -> sparse SDF .npz via dataset_toolkits/sdf_voxelize.py (+ metadata.csv with captions).
  2) This script: VQVAE Encode -> variable-length sequence (Morton + optional FPS) -> <mesh_*> string.
  3) Writes JSONL + mesh_tokens_comma.txt for LLaMA-Factory add_tokens / resize_vocab.

Data format matches LLaMA-Factory sharegpt + OpenAI-style messages (role/content), same tags as
data/mllm_demo.json. No <image> placeholder when doing text-only 3D; omit images column in dataset_info.

Usage (after SDF + VAE checkpoint exist):
  # Windows PowerShell: pass add_tokens from file
  # $t = Get-Content ..\\sft_3d_out\\mesh_tokens_comma.txt -Raw

python dataset_toolkits/build_qwen3vl_sft_3d_jsonl.py \
  --sdf_dir ./train_sdf_dataset/res512_thre0.1 \
  --vae_config ./configs/vae/sdf_vqvae_stage2.json \
  --vae_ckpt ./outputs/sdf_vqvae_stage2_512_0.5/ckpts/vqvae_step0000100.pt \
  --out_jsonl /yangjunjie/LLaMA-Factory/data/qwen3vl_3d_sft.jsonl \
  --gpu_ids 0,1,2,3 \
  --batch_size 4

  # Copy qwen3vl_3d_sft.jsonl + mesh_tokens_comma.txt into LLaMA-Factory/data/
  cd LLaMA-Factory
  # Windows: merge tokens into YAML (avoids CMD length limit); see SDF_PREPROCESSING_GUIDE.md
  python examples/train_lora/merge_qwen3vl_3d_train_yaml.py --tokens data/mesh_tokens_comma.txt
  llamafactory-cli train examples/train_lora/qwen3vl_lora_sft_3d_merged.yaml

Flags: --multiturn (one 4-message sample per object); --task caption_only|reconstruct_only.
Raise LLaMA-Factory cutoff_len for long mesh sequences (e.g. 16384). Optional images: use render.py + mllm_demo-style columns (not emitted here).

Parallel:
  - --batch_size > 1: one VQVAE Encode forward for multiple meshes (sparse points collated via batch_idx).
  - --gpu_ids 0,1,2: one process per GPU, each handles a contiguous slice of the dataset; parts merged to --out_jsonl.
    (Data-parallel sharding; not model parallel. Sparse dict format is unsuitable for nn.DataParallel.)
  - Phase-3 (BPE JSONL): --phase3_num_workers (0 = auto min(CPU,32)) uses CPU multiprocessing; set env
    BPE3D_ENCODE_VERIFY=1 to assert incremental BPE encode matches legacy on the first 16 samples/shard.

Logging:
  - By default, stdout/stderr are suppressed during each VQVAE Encode + mesh-token postprocess to reduce third-party spam.
  - Pass --vqvae_verbose to show that output again.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

# trellis / torchsparse
os.environ.setdefault("SPARSE_BACKEND", "torchsparse")

_TOOLKIT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOLKIT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vae_qwen3vl.dataset_sdf_caption import SDF3DCaptionDataset
from vae_qwen3vl.variable_length_3d import (
    fps_downsample_indices,
    morton_sort_indices,
)
from bpe_3d import BPE3DTokenizer, MORTON_MAX_CODE, serialize_morton_mesh_pairs


DEFAULT_CAPTION_PROMPT = "Describe this 3D shape in one sentence:"
DEFAULT_RECONSTRUCT_PROMPT = "Reconstruct this 3D shape in mesh token format:"


@contextlib.contextmanager
def _stdio_suppressed() -> Iterator[None]:
    """Redirect both Python-level and C/extension-level (OS fd 1/2) stdout/stderr to devnull.

    Pure Python redirection (sys.stdout/stderr) is insufficient for C extensions such as
    torchsparse or CUDA runtime which write directly to file descriptors 1 and 2.  This
    context manager saves/restores the OS-level fds so that all output is silenced.
    """
    old_py_out, old_py_err = sys.stdout, sys.stderr
    old_fd1 = os.dup(1)
    old_fd2 = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        with open(os.devnull, "w", encoding="utf-8") as _py_sink:
            sys.stdout = sys.stderr = _py_sink
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_py_out, old_py_err
    finally:
        os.dup2(old_fd1, 1)
        os.dup2(old_fd2, 2)
        os.close(devnull_fd)
        os.close(old_fd1)
        os.close(old_fd2)


def mesh_token_vocab_strings(num_embeddings: int) -> List[str]:
    """Tokens to add to Qwen tokenizer (order: specials, Morton coords, mesh ids)."""
    out = ["<mesh_start>", "<mesh_end>", "<mesh_empty>"]
    out.extend(f"<morton_{i}>" for i in range(MORTON_MAX_CODE + 1))
    out.extend(f"<mesh_{i}>" for i in range(num_embeddings))
    return out


def write_mesh_tokens_comma(path: Path, num_embeddings: int) -> None:
    tokens = mesh_token_vocab_strings(num_embeddings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(tokens), encoding="utf-8")
    print(f"[build_sft_3d] Wrote {len(tokens)} tokens to {path}")


class _SimpleSparseTensor:
    """Lightweight SparseTensor-like object for BPE encode input/output."""

    def __init__(self, feats: torch.Tensor, coords: torch.Tensor) -> None:
        self.feats = feats
        self.coords = coords

    def replace(self, new_feats: torch.Tensor) -> "_SimpleSparseTensor":
        return _SimpleSparseTensor(new_feats, self.coords)


def _mesh_token_string_from_ids(ids: np.ndarray) -> str:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return "<mesh_start><mesh_end>"
    return "<mesh_start>" + "".join(f"<mesh_{int(v)}>" for v in ids.tolist()) + "<mesh_end>"


def _mesh_token_string_from_ids_coords(ids: np.ndarray, coords_xyz: np.ndarray) -> str:
    return serialize_morton_mesh_pairs(
        np.asarray(ids, dtype=np.int64),
        np.asarray(coords_xyz, dtype=np.int64),
    )


# Phase-3 multiprocessing (spawn): module-level tokenizer per worker process.
_PHASE3_MP_TOK: Optional[BPE3DTokenizer] = None


def _phase3_pool_init(merge_table_path: str) -> None:
    global _PHASE3_MP_TOK
    _PHASE3_MP_TOK = BPE3DTokenizer.load(merge_table_path)


def _phase3_pool_encode(
    task: Tuple[int, np.ndarray, np.ndarray],
) -> Tuple[int, str]:
    global _PHASE3_MP_TOK
    if _PHASE3_MP_TOK is None:
        raise RuntimeError("Phase-3 worker tokenizer not initialized")
    idx, tokens, coords = task
    macro_ids, macro_anchors = _PHASE3_MP_TOK.encode_sparse_numpy(tokens, coords)
    return idx, _mesh_token_string_from_ids_coords(macro_ids, macro_anchors)


def morton_sort_indices_torch(coords_xyz: torch.Tensor, coord_max: int = 512) -> torch.Tensor:
    """GPU-accelerated Morton sort using PyTorch."""
    n = coords_xyz.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=coords_xyz.device)
    bits = max(1, int(np.ceil(np.log2(coord_max + 1))))
    x = coords_xyz[:, 0].long()
    y = coords_xyz[:, 1].long()
    z = coords_xyz[:, 2].long()
    bit_idx = torch.arange(bits, dtype=torch.long, device=coords_xyz.device)
    x_bits = ((x.unsqueeze(-1) >> bit_idx) & 1) << (3 * bit_idx)
    y_bits = ((y.unsqueeze(-1) >> bit_idx) & 1) << (3 * bit_idx + 1)
    z_bits = ((z.unsqueeze(-1) >> bit_idx) & 1) << (3 * bit_idx + 2)
    codes = x_bits.sum(dim=1) + y_bits.sum(dim=1) + z_bits.sum(dim=1)
    return torch.argsort(codes)


def _extract_tokens_and_coords_for_batch(
    encoding_indices: Any,
    batch_idx: int,
    max_safe_length: int,
    coord_max: int,
    over_limit_strategy: str = "drop",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return Morton-sorted (token ids, xyz coords) after optional FPS downsample."""
    indices = encoding_indices.feats.squeeze(-1).long()
    coords = encoding_indices.coords
    mask = coords[:, 0] == batch_idx
    if not mask.any():
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
            0,
        )
    idx_b = indices[mask]
    xyz_b = coords[mask][:, 1:4]
    n_raw = int(idx_b.shape[0])
    if n_raw > max_safe_length:
        if over_limit_strategy == "drop":
            # Avoid expensive FPS downsampling and sorting since the sample will be dropped anyway
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 3), dtype=np.int64),
                n_raw,
            )
        fps_idx = fps_downsample_indices(xyz_b.float(), max_safe_length)
        idx_b = idx_b[fps_idx]
        xyz_b = xyz_b[fps_idx]
    
    # Run vectorized GPU Morton sorting
    order = morton_sort_indices_torch(xyz_b, coord_max=coord_max)
    idx_b = idx_b[order]
    xyz_b = xyz_b[order]

    idx_np = idx_b.detach().cpu().numpy().astype(np.int64)
    xyz_np = xyz_b.detach().cpu().numpy().astype(np.int64)
    return idx_np, xyz_np, n_raw


def _bpe_mesh_string_from_tokens_coords(
    tokens: np.ndarray,
    coords_xyz: np.ndarray,
    bpe_tokenizer: BPE3DTokenizer,
) -> str:
    macro_ids, macro_anchors = bpe_tokenizer.encode_sparse_numpy(
        np.asarray(tokens, dtype=np.int64),
        np.asarray(coords_xyz, dtype=np.int64),
    )
    return _mesh_token_string_from_ids_coords(macro_ids, macro_anchors)


def _bpe_macro_count_from_tokens_coords(
    tokens: np.ndarray,
    coords_xyz: np.ndarray,
    bpe_tokenizer: BPE3DTokenizer,
) -> int:
    macro_ids, _ = bpe_tokenizer.encode_sparse_numpy(
        np.asarray(tokens, dtype=np.int64),
        np.asarray(coords_xyz, dtype=np.int64),
    )
    return int(macro_ids.shape[0])


def _align_vae_args_vq_group_size_to_checkpoint(
    vae_args: dict, state_dict: dict
) -> None:
    """Infer vq_group_size from the checkpoint's codebook embedding dimension.

    The VQ codebook weight has shape [num_embeddings, embed_dim * vq_group_size].
    By dividing its second dimension by the embed_dim in vae_args we can recover
    the vq_group_size that was used when the checkpoint was trained, and patch
    vae_args in-place so the model is constructed with matching dimensions.
    """
    emb_key = "vq.embeddings.weight"
    if emb_key not in state_dict:
        return  # nothing to infer from; leave vae_args unchanged

    ckpt_emb_dim: int = state_dict[emb_key].shape[1]  # embed_dim * vq_group_size
    embed_dim: int = int(
        vae_args.get("embed_dim") or vae_args.get("latent_channels") or 0
    )
    if embed_dim <= 0 or ckpt_emb_dim % embed_dim != 0:
        return  # cannot infer reliably; leave vae_args unchanged

    inferred = ckpt_emb_dim // embed_dim
    current = int(vae_args.get("vq_group_size", 8))
    if inferred != current:
        print(
            f"[load_vae] vq_group_size mismatch: config={current},"
            f" checkpoint={inferred} → using checkpoint value",
            flush=True,
        )
        vae_args["vq_group_size"] = inferred


def load_vae_from_config(
    vae_config_path: str,
    vae_ckpt_path: Optional[str],
    device: torch.device,
) -> torch.nn.Module:
    from trellis.models import SparseSDFVQVAE

    with open(vae_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    vae_args = dict(config["models"]["vqvae"]["args"])
    state_for_load = None
    if vae_ckpt_path and os.path.isfile(vae_ckpt_path):
        ckpt_raw = torch.load(vae_ckpt_path, map_location="cpu")
        state_for_load = (
            ckpt_raw["state_dict"]
            if isinstance(ckpt_raw, dict) and "state_dict" in ckpt_raw
            else ckpt_raw
        )
        if isinstance(state_for_load, dict):
            _align_vae_args_vq_group_size_to_checkpoint(vae_args, state_for_load)
    model = SparseSDFVQVAE(**vae_args)
    if state_for_load is not None:
        model.load_state_dict(state_for_load, strict=False)
        print(f"[build_sft_3d] Loaded VAE weights from {vae_ckpt_path}")
    else:
        print(
            "[build_sft_3d] WARNING: no valid --vae_ckpt; using randomly initialized VAE (debug only)."
        )
    model = model.to(device)
    model.eval()
    return model


def _inputs_3d_to_device(
    inputs_3d: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in inputs_3d.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(dtype=torch.float32)
            out[k] = v
        else:
            out[k] = v
    return out


def collate_inputs_3d(batch_samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Merge inputs_3d for VAE Encode: batch_idx = 0..B-1 per point."""
    sparse_sdfs: List[torch.Tensor] = []
    sparse_indices: List[torch.Tensor] = []
    batch_indices: List[torch.Tensor] = []
    for i, b in enumerate(batch_samples):
        d = b["inputs_3d"]
        n = len(d["sparse_sdf"])
        sparse_sdfs.append(d["sparse_sdf"])
        sparse_indices.append(d["sparse_index"])
        batch_indices.append(torch.full((n,), i, dtype=torch.long))
    return {
        "sparse_sdf": torch.cat(sparse_sdfs, dim=0),
        "sparse_index": torch.cat(sparse_indices, dim=0),
        "batch_idx": torch.cat(batch_indices, dim=0),
    }


class _IndexedSDFSlice(Dataset):
    """DataLoader wrapper that keeps per-sample read errors local to the batch."""

    def __init__(self, ds: SDF3DCaptionDataset, indices: List[int]) -> None:
        self.ds = ds
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        idx = int(self.indices[pos])
        try:
            return idx, self.ds[idx], None
        except Exception as e:
            return idx, None, str(e)


def _identity_collate(batch: List[Any]) -> List[Any]:
    return batch


def encode_batch_to_mesh_strings(
    vae: torch.nn.Module,
    collated: Dict[str, torch.Tensor],
    device: torch.device,
    max_safe_length: int,
    coord_max: int,
    fps_stats: Optional[Dict[str, Any]] = None,
    silent_vqvae: bool = True,
    over_limit_strategy: str = "drop",
    bpe_tokenizer: Optional[BPE3DTokenizer] = None,
    collect_corpus_entries: bool = False,
) -> Tuple[
    List[Optional[str]],
    Optional[List[Optional[str]]],
    Optional[List[Optional[Dict[str, np.ndarray]]]],
]:
    """Encode a collated sparse batch into mesh-token strings.

    Returns one entry per sample; entries are ``None`` when the raw sequence
    length exceeds *max_safe_length* and *over_limit_strategy* is ``"drop"``.
    When *over_limit_strategy* is ``"fps"``, FPS downsampling is applied
    instead and a non-None string is always returned.
    """
    batch_size = int(collated["batch_idx"].max().item()) + 1

    def _forward_and_tokens() -> Tuple[
        List[Optional[str]],
        Optional[List[Optional[str]]],
        Optional[List[Optional[Dict[str, np.ndarray]]]],
    ]:
        with torch.no_grad():
            x = _inputs_3d_to_device(collated, device)
            vae_f = vae.float() if next(vae.parameters()).dtype != torch.float32 else vae
            encoding_indices = vae_f.Encode(x)

        mesh_results: List[Optional[str]] = []
        bpe_results: Optional[List[Optional[str]]] = [] if bpe_tokenizer is not None else None
        corpus_entries: Optional[List[Optional[Dict[str, np.ndarray]]]] = (
            [] if collect_corpus_entries else None
        )

        for b in range(batch_size):
            tok_sorted, xyz_sorted, N = _extract_tokens_and_coords_for_batch(
                encoding_indices,
                b,
                max_safe_length=max_safe_length,
                coord_max=coord_max,
                over_limit_strategy=over_limit_strategy,
            )

            if fps_stats is not None:
                fps_stats["total_samples"] = fps_stats.get("total_samples", 0) + 1
                fps_stats["max_seq_len"] = max(fps_stats.get("max_seq_len", 0), int(N))
                fps_stats["min_seq_len"] = min(
                    fps_stats.get("min_seq_len", float("inf")), int(N)
                )
                fps_stats["seq_len_sum"] = fps_stats.get("seq_len_sum", 0.0) + float(N)
                if N > max_safe_length:
                    fps_stats["over_limit_count"] = fps_stats.get("over_limit_count", 0) + 1
                    fps_stats["excess_sum"] = fps_stats.get("excess_sum", 0.0) + float(
                        N - max_safe_length
                    )
                    if over_limit_strategy == "drop":
                        fps_stats["dropped_count"] = fps_stats.get("dropped_count", 0) + 1

            if over_limit_strategy == "drop" and N > max_safe_length:
                mesh_results.append(None)
                if bpe_results is not None:
                    bpe_results.append(None)
                if corpus_entries is not None:
                    corpus_entries.append(None)
            else:
                mesh_results.append(_mesh_token_string_from_ids_coords(tok_sorted, xyz_sorted))
                if bpe_results is not None:
                    bpe_results.append(
                        _bpe_mesh_string_from_tokens_coords(tok_sorted, xyz_sorted, bpe_tokenizer)
                    )
                if corpus_entries is not None:
                    corpus_entries.append(
                        {
                            "tokens": tok_sorted.astype(np.int64, copy=False),
                            "coords": xyz_sorted.astype(np.int64, copy=False),
                        }
                    )
        return mesh_results, bpe_results, corpus_entries

    if silent_vqvae:
        with _stdio_suppressed():
            return _forward_and_tokens()
    return _forward_and_tokens()


def encode_to_mesh_string(
    vae: torch.nn.Module,
    inputs_3d: Dict[str, torch.Tensor],
    device: torch.device,
    max_safe_length: int,
    coord_max: int,
    fps_stats: Optional[Dict[str, Any]] = None,
    silent_vqvae: bool = True,
    over_limit_strategy: str = "drop",
    bpe_tokenizer: Optional[BPE3DTokenizer] = None,
    collect_corpus_entry: bool = False,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, np.ndarray]]]:
    one = [{"inputs_3d": inputs_3d, "caption": "", "sample_id": ""}]
    mesh, bpe, corpus = encode_batch_to_mesh_strings(
        vae,
        collate_inputs_3d(one),
        device,
        max_safe_length,
        coord_max,
        fps_stats=fps_stats,
        silent_vqvae=silent_vqvae,
        over_limit_strategy=over_limit_strategy,
        bpe_tokenizer=bpe_tokenizer,
        collect_corpus_entries=collect_corpus_entry,
    )
    return (
        mesh[0],
        (bpe[0] if bpe is not None else None),
        (corpus[0] if corpus is not None else None),
    )


def build_records(
    mesh_str: str,
    caption: str,
    caption_prompt: str,
    reconstruct_prompt: str,
    multiturn: bool,
    task: str,
) -> List[Dict[str, Any]]:
    """Return one or two sharegpt rows (messages only)."""
    u_desc = f"{mesh_str}\n{caption_prompt}"
    u_recon = f"{caption}\n{reconstruct_prompt}"
    if multiturn:
        return [
            {
                "messages": [
                    {"role": "user", "content": u_desc},
                    {"role": "assistant", "content": caption},
                    {"role": "user", "content": u_recon},
                    {"role": "assistant", "content": mesh_str},
                ],
                "images": [],
            }
        ]
    rows: List[Dict[str, Any]] = []
    if task in ("both", "caption_only"):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": u_desc},
                    {"role": "assistant", "content": caption},
                ],
                "images": [],
            }
        )
    if task in ("both", "reconstruct_only"):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": u_recon},
                    {"role": "assistant", "content": mesh_str},
                ],
                "images": [],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Qwen3-VL SFT JSONL with 3D VQ tokens for LLaMA-Factory."
    )
    p.add_argument("--sdf_dir", type=str, required=True, help="Dir with metadata.csv and *_r{res}.npz")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--min_points", type=int, default=100)
    p.add_argument("--max_points", type=int, default=500000)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--vae_config", type=str, required=True, help="JSON with models.vqvae.args")
    p.add_argument("--vae_ckpt", type=str, default=None, help="VAE state dict .pt (optional for debug)")
    p.add_argument("--max_safe_3d_length", type=int, default=15000)
    p.add_argument(
        "--over_limit",
        type=str,
        choices=("drop", "fps"),
        default="drop",
        help="How to handle samples whose raw VQ sequence length exceeds --max_safe_3d_length. "
        "'drop': discard the sample entirely (default). "
        "'fps': apply Farthest-Point Sampling to downsample the sequence to the limit.",
    )
    p.add_argument("--coord_max_3d", type=int, default=512)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument(
        "--enable_bpe",
        action="store_true",
        help="Also build a BPE-replaced JSONL using BPE3DTokenizer merge table.",
    )
    p.add_argument(
        "--bpe_merge_table",
        type=str,
        default=None,
        help="Path to BPE merge_table.json (required when --enable_bpe).",
    )
    p.add_argument(
        "--out_jsonl_bpe",
        type=str,
        default=None,
        help="BPE JSONL output path (default: <out_jsonl stem>_bpe.jsonl).",
    )
    p.add_argument(
        "--bpe_extra_vocab_size",
        type=int,
        default=8192,
        help="Extra mesh token slots above base codebook size for BPE macros. Default: 8192.",
    )
    p.add_argument(
        "--bpe_num_merges",
        type=int,
        default=8192,
        help="Number of BPE merges when auto-training merge table.",
    )
    p.add_argument(
        "--bpe_min_freq",
        type=int,
        default=2,
        help="Minimum pair frequency for BPE training in auto-train mode.",
    )
    p.add_argument(
        "--bpe_num_workers",
        type=int,
        default=0,
        help=(
            "BPE training worker processes when train_mode=legacy (0 = auto: "
            "min(os.cpu_count(), 8)). Ignored for default incremental trainer."
        ),
    )
    p.add_argument(
        "--bpe_train_mode",
        type=str,
        default=None,
        choices=("incremental", "legacy"),
        help=(
            "BPE training implementation: incremental (default) or legacy "
            "(full recount + optional multiprocessing). "
            "If omitted, uses env BPE_TRAIN_MODE or incremental."
        ),
    )
    p.add_argument(
        "--phase3_num_workers",
        type=int,
        default=0,
        help=(
            "Phase-3 BPE JSONL CPU worker processes. 0 = auto min(os.cpu_count(), 32). "
            "1 = single process (no extra pool). Merge table must exist on disk for workers>1."
        ),
    )
    p.add_argument(
        "--inline_bpe_during_encode",
        action="store_true",
        help=(
            "In BPE load mode, generate BPE JSONL inside each GPU encode worker. "
            "Default is to cache VAE tokens first and apply BPE in Phase-3 CPU workers, "
            "which keeps GPU encode from waiting on CPU BPE."
        ),
    )
    p.add_argument(
        "--empty_cache_interval",
        type=int,
        default=0,
        help=(
            "Call torch.cuda.empty_cache() every N batches during encoding. "
            "0 disables per-batch cache clearing (default, faster)."
        ),
    )
    p.add_argument(
        "--force_reencode",
        action="store_true",
        help=(
            "Ignore cached *.corpus.part{r}.npz / *.meta.part{r}.jsonl shards "
            "and redo Phase-1 (VAE Encode) from scratch. "
            "By default, if a complete Phase-1 cache is found next to --out_jsonl, "
            "Phase-1 is skipped and the pipeline resumes from Phase-2 (BPE train) "
            "and Phase-3 (apply BPE)."
        ),
    )
    p.add_argument(
        "--token_list_out",
        type=str,
        default=None,
        help="Comma-separated mesh tokens for add_tokens (default: next to jsonl, mesh_tokens_comma.txt)",
    )
    p.add_argument(
        "--token_list_out_bpe",
        type=str,
        default=None,
        help="Comma-separated mesh tokens for BPE JSONL (default: next to bpe jsonl, mesh_tokens_comma_bpe.txt).",
    )
    p.add_argument("--caption_prompt", type=str, default=DEFAULT_CAPTION_PROMPT)
    p.add_argument("--reconstruct_prompt", type=str, default=DEFAULT_RECONSTRUCT_PROMPT)
    p.add_argument("--multiturn", action="store_true", help="One 4-message sample per object")
    p.add_argument(
        "--task",
        type=str,
        choices=("both", "caption_only", "reconstruct_only"),
        default="both",
        help="Ignored if --multiturn",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_embeddings", type=int, default=None, help="Override VQ codebook size (default: from config)")
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples per VQVAE Encode forward (collated sparse batch).",
    )
    p.add_argument(
        "--loader_num_workers",
        type=int,
        default=4,
        help=(
            "CPU DataLoader workers per GPU process for reading .npz samples. "
            "0 disables prefetch and uses the legacy serial ds[idx] loop."
        ),
    )
    p.add_argument(
        "--loader_prefetch_factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor when --loader_num_workers > 0.",
    )
    p.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device indices for multi-GPU sharding (e.g. 0,1,2). "
        "Each GPU writes a slice; results are concatenated in dataset order.",
    )
    p.add_argument(
        "--vqvae_verbose",
        action="store_true",
        help="Do not suppress stdout/stderr during VQVAE Encode / mesh-token postprocess "
        "(default: silent to avoid third-party debug spam).",
    )
    p.add_argument(
        "--log_interval",
        type=int,
        default=200,
        help="Print intermediate sequence-length statistics every N successfully encoded samples "
        "(0 = disable intermediate logging, only print at end).",
    )
    return p.parse_args()


def _parse_gpu_ids(s: Optional[str]) -> Optional[List[int]]:
    if s is None or not str(s).strip():
        return None
    out = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    return out or None


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu" or not torch.cuda.is_available():
        if args.device != "cpu" and not torch.cuda.is_available():
            print("[build_sft_3d] CUDA not available; using CPU.")
        return torch.device("cpu")
    if args.device == "cuda":
        return torch.device("cuda:0")
    return torch.device(args.device)


def _merge_jsonl_parts(out_path: Path, world_size: int) -> None:
    with out_path.open("wb") as fout:
        for r in range(world_size):
            part = Path(str(out_path) + f".part{r}")
            if not part.is_file():
                raise FileNotFoundError(f"Missing shard output: {part}")
            with part.open("rb") as fin:
                shutil.copyfileobj(fin, fout)
            part.unlink(missing_ok=True)


def _merge_fps_stats_parts(out_path: Path, world_size: int) -> Dict[str, Any]:
    """Merge per-rank .stats.part* written by multi-GPU workers."""
    merged: Dict[str, Any] = {
        "over_limit_count": 0,
        "dropped_count": 0,
        "excess_sum": 0.0,
        "total_samples": 0,
        "max_seq_len": 0,
        "min_seq_len": float("inf"),
        "seq_len_sum": 0.0,
    }
    for r in range(world_size):
        part = Path(str(out_path) + f".stats.part{r}")
        if not part.is_file():
            raise FileNotFoundError(f"Missing shard FPS stats: {part}")
        d = json.loads(part.read_text(encoding="utf-8"))
        merged["over_limit_count"] += int(d.get("over_limit_count", 0))
        merged["dropped_count"] += int(d.get("dropped_count", 0))
        merged["excess_sum"] += float(d.get("excess_sum", 0.0))
        merged["total_samples"] += int(d.get("total_samples", 0))
        merged["max_seq_len"] = max(merged["max_seq_len"], int(d.get("max_seq_len", 0)))
        merged["min_seq_len"] = min(
            merged["min_seq_len"], float(d.get("min_seq_len", float("inf")))
        )
        merged["seq_len_sum"] += float(d.get("seq_len_sum", 0.0))
        part.unlink(missing_ok=True)
    if merged["min_seq_len"] == float("inf"):
        merged["min_seq_len"] = 0
    return merged


def _format_seq_stats(
    stats: Dict[str, Any], max_safe_3d_length: int, prefix: str = ""
) -> str:
    """Return a human-readable summary of sequence-length statistics."""
    over = int(stats.get("over_limit_count", 0))
    dropped = int(stats.get("dropped_count", 0))
    fps_kept = over - dropped
    excess_sum = float(stats.get("excess_sum", 0.0))
    avg_excess = excess_sum / over if over else 0.0
    total = int(stats.get("total_samples", 0))
    max_len = int(stats.get("max_seq_len", 0))
    min_len_raw = stats.get("min_seq_len", 0)
    min_len = 0 if min_len_raw == float("inf") else int(min_len_raw)
    avg_len = float(stats.get("seq_len_sum", 0.0)) / total if total else 0.0
    over_pct = 100.0 * over / total if total else 0.0
    tag = f"[build_sft_3d]{prefix}"
    line1 = (
        f"{tag} seq_len: total={total}, max={max_len}, min={min_len}, avg={avg_len:.1f}"
    )
    if dropped > 0 and fps_kept == 0:
        action = f"all dropped={dropped}"
    elif dropped == 0 and fps_kept > 0:
        action = f"all FPS-downsampled={fps_kept}"
    else:
        action = f"dropped={dropped}, FPS-downsampled={fps_kept}"
    line2 = (
        f"{tag} exceeded limit (N>{max_safe_3d_length}): "
        f"{over}/{total} ({over_pct:.1f}%), {action}, "
        f"avg_excess={avg_excess:.1f}, sum_excess={excess_sum:.0f}"
    )
    return f"{line1}\n{line2}"


def _print_fps_sampling_stats(
    stats: Dict[str, Any], max_safe_3d_length: int, prefix: str = ""
) -> None:
    print(_format_seq_stats(stats, max_safe_3d_length, prefix=prefix))


def _args_dict_for_mp(ns: argparse.Namespace) -> Dict[str, Any]:
    d = vars(ns).copy()
    for k in (
        "out_jsonl",
        "out_jsonl_bpe",
        "token_list_out",
        "token_list_out_bpe",
        "bpe_merge_table",
        "sdf_dir",
        "vae_config",
        "vae_ckpt",
    ):
        v = d.get(k)
        if v is not None:
            d[k] = str(v)
    return d


def _save_corpus_part(
    path: Path,
    entries: List[Dict[str, np.ndarray]],
    base_vocab_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    tok_chunks: List[np.ndarray] = []
    coord_chunks: List[np.ndarray] = []
    for rec in entries:
        t = np.asarray(rec["tokens"], dtype=np.int64)
        c = np.asarray(rec["coords"], dtype=np.int64)
        tok_chunks.append(t)
        coord_chunks.append(c)
        offsets.append(offsets[-1] + int(t.shape[0]))
    tokens = np.concatenate(tok_chunks, axis=0) if tok_chunks else np.zeros((0,), dtype=np.int64)
    coords = (
        np.concatenate(coord_chunks, axis=0)
        if coord_chunks
        else np.zeros((0, 3), dtype=np.int64)
    )
    np.savez(
        path,
        offsets=np.asarray(offsets, dtype=np.int64),
        tokens=tokens,
        coords=coords,
        base_vocab_size=np.asarray([base_vocab_size], dtype=np.int64),
    )


def _load_corpus_part(path: Path) -> Tuple[List[Dict[str, np.ndarray]], int]:
    data = np.load(path, allow_pickle=False)
    offsets = data["offsets"]
    tokens = data["tokens"]
    coords = data["coords"]
    base_vocab_size = int(data["base_vocab_size"][0])
    out: List[Dict[str, np.ndarray]] = []
    for i in range(len(offsets) - 1):
        a, b = int(offsets[i]), int(offsets[i + 1])
        out.append(
            {
                "tokens": np.asarray(tokens[a:b], dtype=np.int64),
                "coords": np.asarray(coords[a:b], dtype=np.int64),
            }
        )
    return out, base_vocab_size


def _resolve_merge_table_output_path(args: argparse.Namespace) -> Path:
    if args.bpe_merge_table:
        return Path(args.bpe_merge_table)
    return Path(str(Path(args.out_jsonl).with_suffix("")) + ".merge_table.json")


def _resolve_phase3_merge_table_path(args: argparse.Namespace) -> Path:
    """Path to merge_table.json on disk (required for Phase-3 multiprocessing)."""
    if args.bpe_merge_table and Path(args.bpe_merge_table).is_file():
        return Path(args.bpe_merge_table)
    p = _resolve_merge_table_output_path(args)
    if p.is_file():
        return p
    raise SystemExit(
        "[build_sft_3d] Phase-3 requires merge_table JSON on disk. "
        f"Pass --bpe_merge_table or ensure {p} exists."
    )


def _phase3_effective_workers(args: argparse.Namespace) -> int:
    w = int(getattr(args, "phase3_num_workers", 0))
    if w <= 0:
        return min(os.cpu_count() or 1, 32)
    return max(1, w)


def _phase3_verify_encode_if_requested(
    tok: BPE3DTokenizer, shard: List[Dict[str, np.ndarray]], shard_label: str
) -> None:
    flag = str(os.environ.get("BPE3D_ENCODE_VERIFY", "")).strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return
    n = min(16, len(shard))
    for i in range(n):
        rec = shard[i]
        t = np.asarray(rec["tokens"], dtype=np.int64)
        c = np.asarray(rec["coords"], dtype=np.int64)
        leg = tok._encode_single_legacy(t, c)
        inc = tok._encode_single_incremental(t, c)
        if not (np.array_equal(leg[0], inc[0]) and np.array_equal(leg[1], inc[1])):
            raise SystemExit(
                f"[build_sft_3d] BPE3D_ENCODE_VERIFY failed {shard_label} index={i}: "
                "legacy vs incremental BPE encode mismatch."
            )
    print(
        f"[build_sft_3d][Phase-3] BPE3D_ENCODE_VERIFY OK ({shard_label}): first {n} sample(s)",
        flush=True,
    )


def _discover_phase1_cache(out_jsonl: str) -> int:
    """Count complete Phase-1 shards ({corpus.part{r}.npz, meta.part{r}.jsonl}) next to `out_jsonl`.

    A "complete" shard requires BOTH the corpus .npz and the meta .jsonl file to exist.
    Scans ranks 0, 1, 2, ... consecutively and stops at the first gap.
    Returns the number of complete shards (== inferred world_size for resume).
    """
    base = str(out_jsonl)
    r = 0
    while True:
        corpus = Path(f"{base}.corpus.part{r}.npz")
        meta = Path(f"{base}.meta.part{r}.jsonl")
        if not corpus.is_file() or not meta.is_file():
            return r
        r += 1


def _run_phase2_train_bpe(
    args: argparse.Namespace,
    num_emb: int,
    world_size: int,
) -> BPE3DTokenizer:
    print(f"[build_sft_3d][Phase-2] Loading corpus shards (world_size={world_size})...", flush=True)
    corpus: List[Dict[str, np.ndarray]] = []
    total_points = 0
    for r in range(world_size):
        corpus_part = Path(str(args.out_jsonl) + f".corpus.part{r}.npz")
        shard, _base_vocab = _load_corpus_part(corpus_part)
        shard_points = sum(int(rec["tokens"].shape[0]) for rec in shard)
        total_points += shard_points
        print(
            f"[build_sft_3d][Phase-2] shard={r} samples={len(shard)} points={shard_points} file={corpus_part}",
            flush=True,
        )
        corpus.extend(shard)
    if not corpus:
        raise SystemExit("[build_sft_3d] Empty corpus in auto BPE mode; no kept samples to train.")

    print(
        f"[build_sft_3d][Phase-2] Start BPE training: samples={len(corpus)} points={total_points} "
        f"num_merges={int(args.bpe_num_merges)} min_freq={int(args.bpe_min_freq)}",
        flush=True,
    )
    tok = BPE3DTokenizer(base_vocab_size=int(num_emb))
    before = sum(int(rec["tokens"].shape[0]) for rec in corpus)
    tok.train(
        corpus,
        num_merges=int(args.bpe_num_merges),
        min_freq=int(args.bpe_min_freq),
        num_workers=int(getattr(args, "bpe_num_workers", 0)),
        verbose=True,
        train_mode=getattr(args, "bpe_train_mode", None),
    )
    merge_path = _resolve_merge_table_output_path(args)
    merge_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(merge_path))

    bpe_vocab_size = int(num_emb) + int(args.bpe_extra_vocab_size)
    if tok.vocab_size > bpe_vocab_size:
        raise SystemExit(
            f"[build_sft_3d] Auto-trained BPE requires vocab_size={tok.vocab_size}, "
            f"but --num_embeddings + --bpe_extra_vocab_size = {bpe_vocab_size}. "
            "Increase --bpe_extra_vocab_size."
        )

    after = 0
    for rec in corpus:
        m = int(rec["tokens"].shape[0])
        if m == 0:
            continue
        after += _bpe_macro_count_from_tokens_coords(rec["tokens"], rec["coords"], tok)
    ratio = float(before) / max(float(after), 1.0)
    print(
        f"[build_sft_3d] Auto-trained BPE merge_table={merge_path} merges={len(tok.merge_table)} "
        f"compression={before}->{after} ({ratio:.4f}x)"
    )
    return tok


def _run_phase3_apply_bpe(
    args: argparse.Namespace,
    world_size: int,
    tok: BPE3DTokenizer,
) -> Path:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    merge_table_path = _resolve_phase3_merge_table_path(args)
    workers = _phase3_effective_workers(args)

    out_path_bpe = Path(
        args.out_jsonl_bpe
        if args.out_jsonl_bpe
        else str(Path(args.out_jsonl).with_suffix("")) + "_bpe.jsonl"
    )
    out_path_bpe.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[build_sft_3d][Phase-3] Applying BPE to cached corpus -> {out_path_bpe} "
        f"(world_size={world_size}, workers={workers}, merge_table={merge_table_path})",
        flush=True,
    )

    for r in range(world_size):
        corpus_part = Path(str(args.out_jsonl) + f".corpus.part{r}.npz")
        meta_part = Path(str(args.out_jsonl) + f".meta.part{r}.jsonl")
        shard, _ = _load_corpus_part(corpus_part)
        out_part_bpe = Path(str(out_path_bpe) + f".part{r}")
        out_part_bpe.parent.mkdir(parents=True, exist_ok=True)

        meta_lines = meta_part.read_text(encoding="utf-8").splitlines()
        if len(meta_lines) != len(shard):
            raise ValueError(
                f"meta/corpus size mismatch in shard {r}: meta_lines={len(meta_lines)}, "
                f"corpus={len(shard)}"
            )

        _phase3_verify_encode_if_requested(tok, shard, f"shard{r}")

        tasks: List[Tuple[int, np.ndarray, np.ndarray]] = [
            (
                i,
                np.asarray(shard[i]["tokens"], dtype=np.int64, order="C"),
                np.asarray(shard[i]["coords"], dtype=np.int64, order="C"),
            )
            for i in range(len(shard))
        ]

        if workers <= 1:
            bpe_mesh_strs: List[str] = []
            seq_iter: Iterable[Tuple[int, np.ndarray, np.ndarray]] = tasks
            if tqdm is not None and len(tasks) > 0:
                seq_iter = tqdm(
                    tasks,
                    total=len(tasks),
                    desc=f"[Phase-3] shard{r}/{world_size}",
                    unit="sample",
                    dynamic_ncols=True,
                    file=sys.stderr,
                )
            for _idx, tnk, cnk in seq_iter:
                macro_ids, macro_anchors = tok.encode_sparse_numpy(tnk, cnk)
                bpe_mesh_strs.append(_mesh_token_string_from_ids_coords(macro_ids, macro_anchors))
        else:
            ctx = mp.get_context("spawn")
            chunksize = max(1, len(tasks) // (workers * 4))
            with ctx.Pool(
                workers,
                initializer=_phase3_pool_init,
                initargs=(str(merge_table_path),),
            ) as pool:
                it = pool.imap(_phase3_pool_encode, tasks, chunksize=chunksize)
                if tqdm is not None and len(tasks) > 0:
                    it = tqdm(
                        it,
                        total=len(tasks),
                        desc=f"[Phase-3] shard{r}/{world_size}",
                        unit="sample",
                        dynamic_ncols=True,
                        file=sys.stderr,
                    )
                bpe_mesh_strs = [s for _i, s in it]

        with out_part_bpe.open("w", encoding="utf-8") as fout_bpe:
            for i, line in enumerate(meta_lines):
                meta = json.loads(line)
                bpe_mesh_str = bpe_mesh_strs[i]
                caps = meta.get("captions") or [""]
                for caption in caps:
                    bpe_records = build_records(
                        bpe_mesh_str,
                        str(caption),
                        args.caption_prompt,
                        args.reconstruct_prompt,
                        multiturn=args.multiturn,
                        task=args.task,
                    )
                    for row in bpe_records:
                        fout_bpe.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"[build_sft_3d][Phase-3] shard={r} applied samples={len(shard)} -> {out_part_bpe}",
            flush=True,
        )
    _merge_jsonl_parts(out_path_bpe, world_size)
    print(f"[build_sft_3d][Phase-3] merged -> {out_path_bpe}", flush=True)
    return out_path_bpe


def _mp_worker_entry(cfg: Dict[str, Any]) -> None:
    rank = cfg["rank"]
    world_size = cfg["world_size"]
    gpu_id = cfg["gpu_id"]
    args = argparse.Namespace(**cfg["args_dict"])
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    with open(args.vae_config, "r", encoding="utf-8") as f:
        vae_cfg = json.load(f)
    num_emb = args.num_embeddings
    if num_emb is None:
        num_emb = int(vae_cfg["models"]["vqvae"]["args"].get("num_embeddings", 8192))

    vae = load_vae_from_config(args.vae_config, args.vae_ckpt, device)
    ds = SDF3DCaptionDataset(
        sdf_dir=args.sdf_dir,
        resolution=args.resolution,
        min_points=args.min_points,
        max_points=args.max_points,
        max_samples=args.max_samples,
    )
    n = len(ds)
    idx_start = n * rank // world_size
    idx_end = n * (rank + 1) // world_size
    out_part = Path(str(args.out_jsonl) + f".part{rank}")
    out_part.parent.mkdir(parents=True, exist_ok=True)
    auto_train_bpe = bool(getattr(args, "auto_train_bpe", False))
    out_part_bpe: Optional[Path] = None
    bpe_tokenizer: Optional[BPE3DTokenizer] = None
    corpus_part_path: Optional[Path] = None
    meta_part_path: Optional[Path] = None
    defer_bpe_phase3 = bool(getattr(args, "defer_bpe_phase3", False))
    if bool(getattr(args, "enable_bpe", False)) and not auto_train_bpe and not defer_bpe_phase3:
        if not getattr(args, "bpe_merge_table", None):
            raise SystemExit("[build_sft_3d] --enable_bpe requires --bpe_merge_table in load mode.")
        bpe_tokenizer = BPE3DTokenizer.load(args.bpe_merge_table)
        out_jsonl_bpe = args.out_jsonl_bpe or str(Path(args.out_jsonl).with_suffix("")) + "_bpe.jsonl"
        out_part_bpe = Path(str(out_jsonl_bpe) + f".part{rank}")
        out_part_bpe.parent.mkdir(parents=True, exist_ok=True)
    if auto_train_bpe or defer_bpe_phase3:
        corpus_part_path = Path(str(args.out_jsonl) + f".corpus.part{rank}.npz")
        meta_part_path = Path(str(args.out_jsonl) + f".meta.part{rank}.jsonl")

    n_ok, n_err, fps_stats = run_build_slice(
        args=args,
        vae=vae,
        ds=ds,
        device=device,
        num_emb=num_emb,
        idx_start=idx_start,
        idx_end=idx_end,
        out_path=out_part,
        out_path_bpe=out_part_bpe,
        bpe_tokenizer=bpe_tokenizer,
        auto_train_bpe=auto_train_bpe,
        collect_bpe_cache=defer_bpe_phase3,
        corpus_part_path=corpus_part_path,
        meta_part_path=meta_part_path,
        tqdm_desc=f"rank{rank}/cuda:{gpu_id}",
    )
    stats_part = Path(str(args.out_jsonl) + f".stats.part{rank}")
    stats_part.write_text(json.dumps(fps_stats), encoding="utf-8")
    print(
        f"[build_sft_3d] shard rank={rank} cuda:{gpu_id} indices [{idx_start},{idx_end}) "
        f"wrote={n_ok} failed={n_err} -> {out_part}"
    )


def run_build_slice(
    args: argparse.Namespace,
    vae: torch.nn.Module,
    ds: SDF3DCaptionDataset,
    device: torch.device,
    num_emb: int,
    idx_start: int,
    idx_end: int,
    out_path: Path,
    out_path_bpe: Optional[Path] = None,
    bpe_tokenizer: Optional[BPE3DTokenizer] = None,
    auto_train_bpe: bool = False,
    collect_bpe_cache: bool = False,
    corpus_part_path: Optional[Path] = None,
    meta_part_path: Optional[Path] = None,
    tqdm_desc: str = "VQ encode + JSONL",
) -> Tuple[int, int, Dict[str, Any]]:
    """Process ds[idx_start:idx_end), append JSON lines to out_path.
    Returns (n_ok, n_err, fps_stats) where fps_stats counts samples with N > max_safe_3d_length (FPS applied).
    """
    n_ok, n_err = 0, 0
    fps_stats: Dict[str, Any] = {
        "over_limit_count": 0,
        "dropped_count": 0,
        "excess_sum": 0.0,
        "total_samples": 0,
        "max_seq_len": 0,
        "min_seq_len": float("inf"),
        "seq_len_sum": 0.0,
    }
    silent_vqvae = not bool(getattr(args, "vqvae_verbose", False))
    over_limit_strategy = str(getattr(args, "over_limit", "drop"))
    log_interval = max(0, int(getattr(args, "log_interval", 200)))
    _prev_log_ok = 0
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    indices = list(range(idx_start, idx_end))
    batch_size = max(1, int(args.batch_size))
    batch_starts = list(range(0, len(indices), batch_size))
    loader_num_workers = max(0, int(getattr(args, "loader_num_workers", 0)))
    loader_prefetch_factor = max(1, int(getattr(args, "loader_prefetch_factor", 2)))
    use_loader = loader_num_workers > 0
    if use_loader:
        print(
            f"[build_sft_3d] {tqdm_desc}: DataLoader workers={loader_num_workers} "
            f"prefetch_factor={loader_prefetch_factor} batch_size={batch_size}",
            flush=True,
        )
        iterator = DataLoader(
            _IndexedSDFSlice(ds, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=loader_num_workers,
            collate_fn=_identity_collate,
            persistent_workers=True,
            prefetch_factor=loader_prefetch_factor,
        )
    else:
        iterator = batch_starts
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(batch_starts), desc=tqdm_desc)

    corpus_entries: List[Dict[str, np.ndarray]] = []
    collect_corpus = bool(auto_train_bpe or collect_bpe_cache)
    if collect_corpus and (corpus_part_path is None or meta_part_path is None):
        raise ValueError("BPE corpus cache requires corpus_part_path and meta_part_path")
    empty_cache_interval = max(0, int(getattr(args, "empty_cache_interval", 0)))
    batches_done = 0

    with out_path.open("w", encoding="utf-8") as fout, (
        out_path_bpe.open("w", encoding="utf-8") if out_path_bpe is not None else contextlib.nullcontext()
    ) as fout_bpe, (
        meta_part_path.open("w", encoding="utf-8")
        if collect_corpus and meta_part_path is not None
        else contextlib.nullcontext()
    ) as fmeta:
        for batch_or_b0 in iterator:
            batch_samples: List[Dict[str, Any]] = []
            if use_loader:
                for idx, sample, err in batch_or_b0:
                    if err is not None or sample is None:
                        n_err += 1
                        print(f"[build_sft_3d] skip idx={idx} id={idx}: {err}")
                    else:
                        batch_samples.append(sample)
            else:
                b0 = int(batch_or_b0)
                chunk_idx = indices[b0 : b0 + batch_size]
                for idx in chunk_idx:
                    sample: Optional[Dict[str, Any]] = None
                    try:
                        sample = ds[idx]
                        batch_samples.append(sample)
                    except Exception as e:
                        n_err += 1
                        sid = sample["sample_id"] if sample else idx
                        print(f"[build_sft_3d] skip idx={idx} id={sid}: {e}")

            if not batch_samples:
                continue

            try:
                collated = collate_inputs_3d(batch_samples)
                mesh_strs, bpe_mesh_strs, batch_corpus = encode_batch_to_mesh_strings(
                    vae,
                    collated,
                    device,
                    max_safe_length=args.max_safe_3d_length,
                    coord_max=args.coord_max_3d,
                    fps_stats=fps_stats,
                    silent_vqvae=silent_vqvae,
                    over_limit_strategy=over_limit_strategy,
                    bpe_tokenizer=bpe_tokenizer,
                    collect_corpus_entries=collect_corpus,
                )
                for si, (sample, mesh_str) in enumerate(zip(batch_samples, mesh_strs)):
                    if mesh_str is None:
                        n_err += 1
                        print(
                            f"[build_sft_3d] drop id={sample.get('sample_id')}: "
                            f"seq_len exceeds {args.max_safe_3d_length} (--over_limit=drop)"
                        )
                        continue
                    caps = sample.get("captions_all") or [sample["caption"]]
                    if collect_corpus:
                        if batch_corpus is None:
                            raise ValueError("internal error: missing batch_corpus while collecting BPE cache")
                        rec = batch_corpus[si]
                        if rec is None:
                            raise ValueError("internal error: kept sample has empty corpus record")
                        corpus_entries.append(rec)
                        if fmeta is not None:
                            meta_row = {
                                "sample_id": sample.get("sample_id"),
                                "captions": [str(c) for c in caps],
                            }
                            fmeta.write(json.dumps(meta_row, ensure_ascii=False) + "\n")
                    for caption in caps:
                        records = build_records(
                            mesh_str,
                            caption,
                            args.caption_prompt,
                            args.reconstruct_prompt,
                            multiturn=args.multiturn,
                            task=args.task,
                        )
                        for rec in records:
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        if bpe_mesh_strs is not None and fout_bpe is not None:
                            bpe_mesh_str = bpe_mesh_strs[si]
                            if bpe_mesh_str is not None:
                                bpe_records = build_records(
                                    bpe_mesh_str,
                                    caption,
                                    args.caption_prompt,
                                    args.reconstruct_prompt,
                                    multiturn=args.multiturn,
                                    task=args.task,
                                )
                                for rec in bpe_records:
                                    fout_bpe.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok += 1
            except Exception as e_batch:
                print(f"[build_sft_3d] batch encode failed (fallback per-sample): {e_batch}")
                for sample in batch_samples:
                    try:
                        mesh_str, bpe_mesh_str, one_corpus = encode_to_mesh_string(
                            vae,
                            sample["inputs_3d"],
                            device,
                            max_safe_length=args.max_safe_3d_length,
                            coord_max=args.coord_max_3d,
                            fps_stats=fps_stats,
                            silent_vqvae=silent_vqvae,
                            over_limit_strategy=over_limit_strategy,
                            bpe_tokenizer=bpe_tokenizer,
                            collect_corpus_entry=collect_corpus,
                        )
                        if mesh_str is None:
                            n_err += 1
                            print(
                                f"[build_sft_3d] drop id={sample.get('sample_id')}: "
                                f"seq_len exceeds {args.max_safe_3d_length} (--over_limit=drop)"
                            )
                            continue
                        caps = sample.get("captions_all") or [sample["caption"]]
                        if collect_corpus:
                            if one_corpus is None:
                                raise ValueError("internal error: missing one_corpus while collecting BPE cache")
                            corpus_entries.append(one_corpus)
                            if fmeta is not None:
                                meta_row = {
                                    "sample_id": sample.get("sample_id"),
                                    "captions": [str(c) for c in caps],
                                }
                                fmeta.write(json.dumps(meta_row, ensure_ascii=False) + "\n")
                        for caption in caps:
                            records = build_records(
                                mesh_str,
                                caption,
                                args.caption_prompt,
                                args.reconstruct_prompt,
                                multiturn=args.multiturn,
                                task=args.task,
                            )
                            for rec in records:
                                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            if bpe_mesh_str is not None and fout_bpe is not None:
                                bpe_records = build_records(
                                    bpe_mesh_str,
                                    caption,
                                    args.caption_prompt,
                                    args.reconstruct_prompt,
                                    multiturn=args.multiturn,
                                    task=args.task,
                                )
                                for rec in bpe_records:
                                    fout_bpe.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_ok += 1
                    except Exception as e:
                        n_err += 1
                        print(
                            f"[build_sft_3d] skip id={sample.get('sample_id')}: {e}"
                        )
            finally:
                batches_done += 1
                if (
                    device.type == "cuda"
                    and empty_cache_interval > 0
                    and batches_done % empty_cache_interval == 0
                ):
                    torch.cuda.empty_cache()

            if log_interval > 0 and (n_ok - _prev_log_ok) >= log_interval:
                _prev_log_ok = n_ok
                _print_fps_sampling_stats(
                    fps_stats,
                    args.max_safe_3d_length,
                    prefix=f" [{tqdm_desc} | ok={n_ok} err={n_err}]",
                )

    if collect_corpus and corpus_part_path is not None:
        _save_corpus_part(corpus_part_path, corpus_entries, base_vocab_size=num_emb)
    return n_ok, n_err, fps_stats


def main() -> None:
    args = parse_args()
    if args.bpe_num_merges < 0:
        raise SystemExit("[build_sft_3d] --bpe_num_merges must be >= 0.")
    if args.bpe_min_freq < 1:
        raise SystemExit("[build_sft_3d] --bpe_min_freq must be >= 1.")
    if args.bpe_num_merges > args.bpe_extra_vocab_size:
        raise SystemExit(
            "[build_sft_3d] --bpe_num_merges cannot exceed --bpe_extra_vocab_size."
        )
    gpu_list = _parse_gpu_ids(args.gpu_ids)

    if gpu_list and len(gpu_list) > 1:
        if not torch.cuda.is_available():
            raise SystemExit("[build_sft_3d] --gpu_ids requires CUDA.")
        for g in gpu_list:
            if g < 0 or g >= torch.cuda.device_count():
                raise SystemExit(
                    f"[build_sft_3d] Invalid gpu id {g} (device_count={torch.cuda.device_count()})."
                )

    with open(args.vae_config, "r", encoding="utf-8") as f:
        vae_cfg = json.load(f)
    num_emb = args.num_embeddings
    if num_emb is None:
        num_emb = int(vae_cfg["models"]["vqvae"]["args"].get("num_embeddings", 8192))

    auto_train_bpe = bool(
        args.enable_bpe and (not args.bpe_merge_table or not Path(args.bpe_merge_table).is_file())
    )
    args.auto_train_bpe = auto_train_bpe
    defer_bpe_phase3 = bool(
        args.enable_bpe
        and not auto_train_bpe
        and not bool(getattr(args, "inline_bpe_during_encode", False))
    )
    args.defer_bpe_phase3 = defer_bpe_phase3

    token_path = Path(
        args.token_list_out
        if args.token_list_out
        else str(Path(args.out_jsonl).with_name("mesh_tokens_comma.txt"))
    )
    write_mesh_tokens_comma(token_path, num_emb)

    if not args.force_reencode and args.enable_bpe:
        cached_world_size = _discover_phase1_cache(args.out_jsonl)
        if cached_world_size > 0:
            _out_bpe_candidate = Path(
                args.out_jsonl_bpe
                if args.out_jsonl_bpe
                else str(Path(args.out_jsonl).with_suffix("")) + "_bpe.jsonl"
            )
            _merge_table_ready = (
                args.bpe_merge_table and Path(args.bpe_merge_table).is_file()
            )

            if auto_train_bpe:
                # Phase-1 cache exists but merge table not yet written → Phase-2 + Phase-3
                print(
                    f"[build_sft_3d] Resume: detected {cached_world_size} complete Phase-1 "
                    f"shard(s) next to {args.out_jsonl}; skipping Phase-1 (VAE Encode). "
                    f"Use --force_reencode to override.",
                    flush=True,
                )
                print("[build_sft_3d] BPE auto-train mode (resume).", flush=True)
                Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
                tok = _run_phase2_train_bpe(
                    args, num_emb=num_emb, world_size=cached_world_size
                )
                out_path_bpe_resume = _run_phase3_apply_bpe(
                    args, world_size=cached_world_size, tok=tok
                )
                token_path_bpe = Path(
                    args.token_list_out_bpe
                    if args.token_list_out_bpe
                    else str(out_path_bpe_resume.with_name("mesh_tokens_comma_bpe.txt"))
                )
                write_mesh_tokens_comma(
                    token_path_bpe, int(num_emb) + int(args.bpe_extra_vocab_size)
                )
                print(f"[build_sft_3d] Resume done. bpe -> {out_path_bpe_resume}")
                return

            elif _merge_table_ready and not _out_bpe_candidate.is_file():
                # merge table exists + Phase-1 cache exists, but BPE JSONL missing → Phase-3 only
                bpe_vocab_size = int(num_emb) + int(args.bpe_extra_vocab_size)
                tok_p3 = BPE3DTokenizer.load(args.bpe_merge_table)
                if tok_p3.vocab_size > bpe_vocab_size:
                    raise SystemExit(
                        f"[build_sft_3d] BPE merge table requires vocab_size={tok_p3.vocab_size}, "
                        f"but --num_embeddings + --bpe_extra_vocab_size = {bpe_vocab_size}. "
                        "Increase --bpe_extra_vocab_size."
                    )
                print(
                    f"[build_sft_3d] Resume Phase-3 only: found {cached_world_size} corpus "
                    f"shard(s) + merge_table={args.bpe_merge_table}, BPE JSONL not found "
                    f"→ generating without re-encoding. Use --force_reencode to override.",
                    flush=True,
                )
                Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
                out_path_bpe_p3 = _run_phase3_apply_bpe(
                    args, world_size=cached_world_size, tok=tok_p3
                )
                token_path_bpe_p3 = Path(
                    args.token_list_out_bpe
                    if args.token_list_out_bpe
                    else str(out_path_bpe_p3.with_name("mesh_tokens_comma_bpe.txt"))
                )
                write_mesh_tokens_comma(token_path_bpe_p3, bpe_vocab_size)
                print(f"[build_sft_3d] Resume Phase-3 done. bpe -> {out_path_bpe_p3}")
                return

    out_path_bpe: Optional[Path] = None
    bpe_tokenizer: Optional[BPE3DTokenizer] = None
    if args.enable_bpe and not auto_train_bpe:
        if not args.bpe_merge_table:
            raise SystemExit("[build_sft_3d] --enable_bpe requires --bpe_merge_table in load mode.")
        out_path_bpe = Path(
            args.out_jsonl_bpe
            if args.out_jsonl_bpe
            else str(Path(args.out_jsonl).with_suffix("")) + "_bpe.jsonl"
        )
        bpe_tokenizer_check = BPE3DTokenizer.load(args.bpe_merge_table)
        bpe_vocab_size = int(num_emb) + int(args.bpe_extra_vocab_size)
        if bpe_tokenizer_check.vocab_size > bpe_vocab_size:
            raise SystemExit(
                f"[build_sft_3d] BPE merge table requires vocab_size={bpe_tokenizer_check.vocab_size}, "
                f"but --num_embeddings + --bpe_extra_vocab_size = {bpe_vocab_size}. "
                "Increase --bpe_extra_vocab_size."
            )
        if defer_bpe_phase3:
            print(
                f"[build_sft_3d] BPE load mode (deferred Phase-3). merge_table={args.bpe_merge_table} "
                f"vocab_limit={bpe_vocab_size} out_bpe={out_path_bpe} "
                f"workers={_phase3_effective_workers(args)}",
                flush=True,
            )
        else:
            bpe_tokenizer = bpe_tokenizer_check
        token_path_bpe = Path(
            args.token_list_out_bpe
            if args.token_list_out_bpe
            else str(out_path_bpe.with_name("mesh_tokens_comma_bpe.txt"))
        )
        write_mesh_tokens_comma(token_path_bpe, bpe_vocab_size)
        if not defer_bpe_phase3:
            print(
                f"[build_sft_3d] BPE load mode. merge_table={args.bpe_merge_table} "
                f"vocab_limit={bpe_vocab_size} out_bpe={out_path_bpe}"
            )
    elif auto_train_bpe:
        print("[build_sft_3d] BPE auto-train mode enabled.", flush=True)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path_bpe is not None:
        out_path_bpe.parent.mkdir(parents=True, exist_ok=True)

    if gpu_list and len(gpu_list) > 1:
        ctx = mp.get_context("spawn")
        ad = _args_dict_for_mp(args)
        procs = []
        for rank, gpu_id in enumerate(gpu_list):
            cfg = {
                "rank": rank,
                "world_size": len(gpu_list),
                "gpu_id": gpu_id,
                "args_dict": ad,
            }
            p = ctx.Process(target=_mp_worker_entry, args=(cfg,))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        failed_codes = [p.exitcode for p in procs if p.exitcode not in (0, None)]
        if failed_codes:
            raise SystemExit(
                f"[build_sft_3d] Worker(s) failed (exit codes={failed_codes})."
            )
        _merge_jsonl_parts(out_path, len(gpu_list))
        if auto_train_bpe:
            print("[build_sft_3d][Phase-1] Raw JSONL and corpus/meta shards ready.", flush=True)
        if defer_bpe_phase3:
            print("[build_sft_3d][Phase-1] Raw JSONL and BPE corpus/meta shards ready.", flush=True)
        if args.enable_bpe and not auto_train_bpe and not defer_bpe_phase3:
            if out_path_bpe is None:
                raise SystemExit("[build_sft_3d] internal error: out_path_bpe is None while BPE enabled.")
            _merge_jsonl_parts(out_path_bpe, len(gpu_list))
        merged_fps = _merge_fps_stats_parts(out_path, len(gpu_list))
        print(f"[build_sft_3d] Merged {len(gpu_list)} shards -> {out_path}")
        if out_path_bpe is not None and not auto_train_bpe and not defer_bpe_phase3:
            print(f"[build_sft_3d] Merged {len(gpu_list)} shards -> {out_path_bpe}")
        if auto_train_bpe:
            tok = _run_phase2_train_bpe(args, num_emb=num_emb, world_size=len(gpu_list))
            out_path_bpe = _run_phase3_apply_bpe(args, world_size=len(gpu_list), tok=tok)
            token_path_bpe = Path(
                args.token_list_out_bpe
                if args.token_list_out_bpe
                else str(out_path_bpe.with_name("mesh_tokens_comma_bpe.txt"))
            )
            write_mesh_tokens_comma(token_path_bpe, int(num_emb) + int(args.bpe_extra_vocab_size))
            print(f"[build_sft_3d] Merged {len(gpu_list)} shards -> {out_path_bpe}")
        elif defer_bpe_phase3:
            tok_p3 = BPE3DTokenizer.load(args.bpe_merge_table)
            out_path_bpe = _run_phase3_apply_bpe(args, world_size=len(gpu_list), tok=tok_p3)
            print(f"[build_sft_3d] Merged {len(gpu_list)} shards -> {out_path_bpe}")
        _print_fps_sampling_stats(merged_fps, args.max_safe_3d_length)
        return

    if gpu_list and len(gpu_list) == 1:
        device = torch.device(f"cuda:{gpu_list[0]}")
    else:
        device = _resolve_device(args)

    vae = load_vae_from_config(args.vae_config, args.vae_ckpt, device)
    ds = SDF3DCaptionDataset(
        sdf_dir=args.sdf_dir,
        resolution=args.resolution,
        min_points=args.min_points,
        max_points=args.max_points,
        max_samples=args.max_samples,
    )

    corpus_part_path = (
        Path(str(args.out_jsonl) + ".corpus.part0.npz")
        if (auto_train_bpe or defer_bpe_phase3)
        else None
    )
    meta_part_path = (
        Path(str(args.out_jsonl) + ".meta.part0.jsonl")
        if (auto_train_bpe or defer_bpe_phase3)
        else None
    )
    n_ok, n_err, fps_stats = run_build_slice(
        args=args,
        vae=vae,
        ds=ds,
        device=device,
        num_emb=num_emb,
        idx_start=0,
        idx_end=len(ds),
        out_path=out_path,
        out_path_bpe=None if defer_bpe_phase3 else out_path_bpe,
        bpe_tokenizer=bpe_tokenizer,
        auto_train_bpe=auto_train_bpe,
        collect_bpe_cache=defer_bpe_phase3,
        corpus_part_path=corpus_part_path,
        meta_part_path=meta_part_path,
        tqdm_desc="VQ encode + JSONL",
    )
    if auto_train_bpe:
        print("[build_sft_3d][Phase-1] Raw JSONL and corpus/meta shards ready.", flush=True)
    if auto_train_bpe:
        tok = _run_phase2_train_bpe(args, num_emb=num_emb, world_size=1)
        out_path_bpe = _run_phase3_apply_bpe(args, world_size=1, tok=tok)
        token_path_bpe = Path(
            args.token_list_out_bpe
            if args.token_list_out_bpe
            else str(out_path_bpe.with_name("mesh_tokens_comma_bpe.txt"))
        )
        write_mesh_tokens_comma(token_path_bpe, int(num_emb) + int(args.bpe_extra_vocab_size))
        print(f"[build_sft_3d] Done. wrote={n_ok} failed={n_err} -> {out_path}")
        print(f"[build_sft_3d] Done. bpe -> {out_path_bpe}")
    elif defer_bpe_phase3:
        tok_p3 = BPE3DTokenizer.load(args.bpe_merge_table)
        out_path_bpe = _run_phase3_apply_bpe(args, world_size=1, tok=tok_p3)
        print(f"[build_sft_3d] Done. wrote={n_ok} failed={n_err} -> {out_path}")
        print(f"[build_sft_3d] Done. bpe -> {out_path_bpe}")
    else:
        print(f"[build_sft_3d] Done. wrote={n_ok} failed={n_err} -> {out_path}")
    _print_fps_sampling_stats(fps_stats, args.max_safe_3d_length)


if __name__ == "__main__":
    main()
