#!/usr/bin/env python3
"""
Apply a trained 3D BPE merge table to an existing Qwen3-VL SFT JSONL (no corpus/meta, no VAE).

The input JSONL must use Morton+mesh pairs as produced by build_qwen3vl_sft_3d_jsonl.py
(``<morton_i><mesh_j>`` inside ``<mesh_start>...<mesh_end>``). Each block is parsed back
to (token, coord), BPE-encoded, and re-serialized; other fields (captions, prompts) are unchanged.

Example::

  python dataset_toolkits/jsonl_apply_bpe.py \\
    --in_jsonl ./outputs_pad_sdf/qwen3_3d_sft_256_0.5.jsonl \\
    --out_jsonl ./outputs_pad_sdf/qwen3_3d_sft_256_0.5_bpe.jsonl \\
    --merge_table ./outputs_pad_sdf/merge_table.json \\
    --base_vocab_size 8192 \\
    --num_workers 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_TOOLKIT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOLKIT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bpe_3d import BPE3DTokenizer, parse_morton_mesh_pairs, serialize_morton_mesh_pairs  # noqa: E402

_MESH_BLOCK_RE = re.compile(r"<mesh_start>.*?<mesh_end>", re.DOTALL)

_WORKER_TOK: Optional[BPE3DTokenizer] = None
_WORKER_MAX_MESH_ID: int = 8192


def _mesh_block_replacer(block: str) -> str:
    global _WORKER_TOK, _WORKER_MAX_MESH_ID
    if _WORKER_TOK is None:
        raise RuntimeError("worker tokenizer not initialized")
    parsed = parse_morton_mesh_pairs(block, max_mesh_id=_WORKER_MAX_MESH_ID)
    if parsed.ids.size == 0:
        if re.search(r"<mesh_\d+>", block):
            raise ValueError(
                "mesh block has <mesh_*> but no valid <morton_*><mesh_*> pairs; "
                "cannot recover 3D coords for BPE. Re-build JSONL with current "
                "build_qwen3vl_sft_3d_jsonl.py (Morton format)."
            )
        return block
    macro_ids, macro_anchors = _WORKER_TOK.encode_sparse_numpy(parsed.ids, parsed.anchors)
    return serialize_morton_mesh_pairs(macro_ids, macro_anchors)


def _worker_init(merge_table: str, max_mesh_id: int) -> None:
    global _WORKER_TOK, _WORKER_MAX_MESH_ID
    _WORKER_TOK = BPE3DTokenizer.load(merge_table)
    _WORKER_MAX_MESH_ID = int(max_mesh_id)


def _encode_block_task(block: str) -> Tuple[str, str]:
    return block, _mesh_block_replacer(block)


def _replace_mesh_blocks_in_text(text: str, cache: Dict[str, str]) -> str:
    def _sub(m: re.Match[str]) -> str:
        block = m.group(0)
        if block not in cache:
            cache[block] = _mesh_block_replacer(block)
        return cache[block]

    return _MESH_BLOCK_RE.sub(_sub, text)


def _transform_row(row: dict, cache: Dict[str, str]) -> dict:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return row
    new_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            new_messages.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, str) and "<mesh_start>" in content:
            new_messages.append({**msg, "content": _replace_mesh_blocks_in_text(content, cache)})
        else:
            new_messages.append(msg)
    return {**row, "messages": new_messages}


def _collect_unique_mesh_blocks(in_path: Path) -> List[str]:
    uniq: Dict[str, None] = {}
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for msg in row.get("messages") or []:
                content = msg.get("content") or ""
                if "<mesh_start>" not in content:
                    continue
                for m in _MESH_BLOCK_RE.finditer(content):
                    uniq.setdefault(m.group(0), None)
    return list(uniq.keys())


def _build_cache(
    blocks: List[str],
    merge_table: str,
    max_mesh_id: int,
    num_workers: int,
) -> Dict[str, str]:
    if not blocks:
        return {}
    if num_workers <= 1:
        _worker_init(merge_table, max_mesh_id)
        return {b: _mesh_block_replacer(b) for b in blocks}

    ctx = mp.get_context("spawn")
    chunksize = max(1, len(blocks) // (num_workers * 4))
    cache: Dict[str, str] = {}
    with ctx.Pool(
        num_workers,
        initializer=_worker_init,
        initargs=(merge_table, max_mesh_id),
    ) as pool:
        for old, new in pool.imap(_encode_block_task, blocks, chunksize=chunksize):
            cache[old] = new
    return cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply 3D BPE merge table to an existing morton+mesh SFT JSONL."
    )
    p.add_argument("--in_jsonl", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--merge_table", type=str, required=True)
    p.add_argument(
        "--base_vocab_size",
        type=int,
        default=8192,
        help="Leaf VQ codebook size; mesh ids >= this are treated as invalid in parse.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="CPU workers for BPE encode of unique mesh blocks (0 = single process).",
    )
    p.add_argument(
        "--skip_unique_pass",
        action="store_true",
        help="Single-pass with in-memory cache (lower peak RAM for unique blocks, slower).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    merge_path = Path(args.merge_table)
    if not in_path.is_file():
        raise SystemExit(f"[jsonl_apply_bpe] input not found: {in_path}")
    if not merge_path.is_file():
        raise SystemExit(f"[jsonl_apply_bpe] merge_table not found: {merge_path}")

    tok_check = BPE3DTokenizer.load(str(merge_path))
    print(
        f"[jsonl_apply_bpe] merge_table vocab_size={tok_check.vocab_size} "
        f"merges={len(tok_check.merge_table)}",
        flush=True,
    )

    workers = int(args.num_workers)
    if workers < 0:
        raise SystemExit("--num_workers must be >= 0")

    cache: Dict[str, str] = {}
    if not args.skip_unique_pass:
        print("[jsonl_apply_bpe] Pass 1: collect unique <mesh_start> blocks …", flush=True)
        blocks = _collect_unique_mesh_blocks(in_path)
        print(f"[jsonl_apply_bpe] unique mesh blocks={len(blocks)}", flush=True)
        print("[jsonl_apply_bpe] Pass 2: BPE encode unique blocks …", flush=True)
        cache = _build_cache(blocks, str(merge_path), int(args.base_vocab_size), workers)
    else:
        _worker_init(str(merge_path), int(args.base_vocab_size))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in, n_out, n_mesh_lines = 0, 0, 0
    print(f"[jsonl_apply_bpe] Pass 3: write {out_path} …", flush=True)
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            if "<mesh_start>" in json.dumps(row, ensure_ascii=False):
                n_mesh_lines += 1
            row = _transform_row(row, cache)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
            if n_out % 50000 == 0:
                print(f"[jsonl_apply_bpe] wrote {n_out} lines …", flush=True)

    print(
        f"[jsonl_apply_bpe] done: in_lines={n_in} out_lines={n_out} "
        f"lines_with_mesh={n_mesh_lines} unique_blocks_cached={len(cache)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
