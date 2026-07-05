#!/usr/bin/env python3
"""
Verify BPE mesh JSONL against corpus + merge table using existing BPE3DTokenizer APIs.

BPE JSONL only stores macro <mesh_i> ids (no spatial anchors). decode_to_sparse() requires
(ids, anchors) produced by encode_sparse_numpy(); this script therefore loads
``<out_jsonl>.corpus.part*.npz`` (same layout as build_qwen3vl_sft_3d_jsonl.py) and:

1) encode_sparse_numpy(tokens, coords) -> macro_ids, macro_anchors
2) decode_to_sparse([{"ids": macro_ids, "anchors": macro_anchors}]) -> leaf sparse tensor
3) Assert leaf multiset matches original (tokens, coords) — same check as test_bpe_3d.py

Optionally, if --jsonl_orig / --jsonl_bpe are set together with meta shards, each sample’s
mesh token ids parsed from JSONL are compared to corpus-derived strings (macro vs leaf).

Example::

  python dataset_toolkits/verify_bpe_jsonl_decode.py \\
    --corpus_prefix ./outputs/qwen3vl_3d_sft_8cat.jsonl \\
    --merge_table ./outputs/merge_table.json \\
    --jsonl_orig ./outputs/qwen3vl_3d_sft_8cat.jsonl \\
    --jsonl_bpe ./outputs/qwen3vl_3d_sft_8cat_bpe_40W.jsonl \\
    --max_samples 10000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

_TOOLKIT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOLKIT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bpe_3d import BPE3DTokenizer  # noqa: E402


class _SimpleSparseTensor:
    """Same lightweight stand-in as test_bpe_3d.DummySparseTensor."""

    def __init__(self, feats: torch.Tensor, coords: torch.Tensor) -> None:
        self.feats = feats
        self.coords = coords

    def replace(self, new_feats: torch.Tensor) -> "_SimpleSparseTensor":
        return _SimpleSparseTensor(new_feats, self.coords)


def _sparse_item_set(st: _SimpleSparseTensor) -> set:
    c = st.coords.detach().cpu().numpy().astype(np.int64)
    f = st.feats.squeeze(-1).detach().cpu().numpy().astype(np.int64)
    return {(int(fid), int(x), int(y), int(z)) for fid, (_b, x, y, z) in zip(f.tolist(), c.tolist())}


def _load_corpus_part(path: Path) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Same on-disk format as build_qwen3vl_sft_3d_jsonl._load_corpus_part."""
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


def _discover_corpus_shards(corpus_prefix: Path) -> List[Path]:
    """corpus_prefix is the full path to out_jsonl; shards are <stem>.corpus.part{r}.npz."""
    base = str(corpus_prefix)
    paths: List[Path] = []
    r = 0
    while True:
        p = Path(f"{base}.corpus.part{r}.npz")
        if not p.is_file():
            break
        paths.append(p)
        r += 1
    return paths


_MESH_TAG_RE = re.compile(r"<mesh_(\d+)>")


def _mesh_ids_from_text(s: str) -> np.ndarray:
    ids = [int(m.group(1)) for m in _MESH_TAG_RE.finditer(s)]
    return np.asarray(ids, dtype=np.int64)


def _try_mesh_ids_from_sharegpt_row(row: Dict[str, Any]) -> Optional[np.ndarray]:
    """First message (in order) that contains <mesh_start> defines the mesh id sequence."""
    for m in row.get("messages") or []:
        content = m.get("content") or ""
        if "<mesh_start>" in content:
            return _mesh_ids_from_text(content)
    return None


def _records_per_caption(multiturn: bool, task: str) -> int:
    if multiturn:
        return 1
    if task == "both":
        return 2
    return 1


def _iter_jsonl_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def _iter_corpus_and_meta_shards(
    corpus_prefix: Path,
) -> Iterator[Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]]:
    base = str(corpus_prefix)
    r = 0
    while True:
        cpath = Path(f"{base}.corpus.part{r}.npz")
        mpath = Path(f"{base}.meta.part{r}.jsonl")
        if not cpath.is_file():
            break
        if not mpath.is_file():
            raise FileNotFoundError(f"Missing meta shard for corpus part {r}: {mpath}")
        shard, _ = _load_corpus_part(cpath)
        meta_lines = mpath.read_text(encoding="utf-8").splitlines()
        meta_objs = [json.loads(x) for x in meta_lines if x.strip()]
        if len(meta_objs) != len(shard):
            raise ValueError(
                f"meta/corpus mismatch shard {r}: meta={len(meta_objs)} corpus={len(shard)}"
            )
        yield shard, meta_objs
        r += 1


def run(args: argparse.Namespace) -> int:
    merge_path = Path(args.merge_table)
    if not merge_path.is_file():
        print(f"[verify_bpe] merge_table not found: {merge_path}", flush=True)
        return 2

    corpus_prefix = Path(args.corpus_prefix)
    if not _discover_corpus_shards(corpus_prefix):
        print(
            f"[verify_bpe] No corpus shards at {corpus_prefix}.corpus.part0.npz … "
            "(need Phase-1 cache next to the JSONL stem.)",
            flush=True,
        )
        return 2

    tok = BPE3DTokenizer.load(str(merge_path))
    device = torch.device("cpu")

    jsonl_orig = Path(args.jsonl_orig) if args.jsonl_orig else None
    jsonl_bpe = Path(args.jsonl_bpe) if args.jsonl_bpe else None
    do_jsonl = jsonl_orig is not None and jsonl_bpe is not None
    if (jsonl_orig is not None) ^ (jsonl_bpe is not None):
        print("[verify_bpe] Pass both --jsonl_orig and --jsonl_bpe, or neither.", flush=True)
        return 2

    jsonl_orig_it: Optional[Iterator[str]] = None
    jsonl_bpe_it: Optional[Iterator[str]] = None
    if do_jsonl:
        if not jsonl_orig.is_file() or not jsonl_bpe.is_file():
            print("[verify_bpe] jsonl path missing.", flush=True)
            return 2
        jsonl_orig_it = _iter_jsonl_lines(jsonl_orig)
        jsonl_bpe_it = _iter_jsonl_lines(jsonl_bpe)

    multiturn = bool(args.multiturn)
    task = str(args.task)

    n_ok = 0
    n_decode_mismatch = 0
    n_jsonl_orig_mismatch = 0
    n_jsonl_bpe_mismatch = 0
    n_done = 0
    stop_all = False

    for shard, meta_objs in _iter_corpus_and_meta_shards(corpus_prefix):
        if stop_all:
            break
        for local_i, (rec, meta) in enumerate(zip(shard, meta_objs)):
            if args.max_samples is not None and n_done >= int(args.max_samples):
                stop_all = True
                break
            tokens = np.asarray(rec["tokens"], dtype=np.int64, order="C")
            coords = np.asarray(rec["coords"], dtype=np.int64, order="C")

            macro_ids, macro_anchors = tok.encode_sparse_numpy(tokens, coords)
            dec = tok.decode_to_sparse(
                [{"ids": macro_ids, "anchors": macro_anchors}],
                device=device,
                sparse_tensor_cls=_SimpleSparseTensor,
            )
            orig_st = _SimpleSparseTensor(
                torch.tensor(tokens, dtype=torch.float32).unsqueeze(-1),
                torch.cat(
                    [
                        torch.zeros((tokens.shape[0], 1), dtype=torch.int32),
                        torch.tensor(coords, dtype=torch.int32),
                    ],
                    dim=1,
                ),
            )
            if _sparse_item_set(dec) != _sparse_item_set(orig_st):
                n_decode_mismatch += 1
                print(
                    f"[verify_bpe] decode multiset mismatch shard sample local_i={local_i}",
                    flush=True,
                )
            else:
                n_ok += 1

            if do_jsonl and jsonl_orig_it is not None and jsonl_bpe_it is not None:
                caps = meta.get("captions") or [""]
                if not isinstance(caps, list):
                    caps = [str(caps)]
                r_per = _records_per_caption(multiturn, task)
                n_lines = len(caps) * r_per
                try:
                    o_lines = [next(jsonl_orig_it) for _ in range(n_lines)]
                    b_lines = [next(jsonl_bpe_it) for _ in range(n_lines)]
                except StopIteration:
                    print(
                        f"[verify_bpe] JSONL exhausted early at corpus global sample ~{n_done}",
                        flush=True,
                    )
                    return 3

                for li, (lo, lb) in enumerate(zip(o_lines, b_lines)):
                    row_o = json.loads(lo)
                    row_b = json.loads(lb)
                    ids_o = _try_mesh_ids_from_sharegpt_row(row_o)
                    ids_b = _try_mesh_ids_from_sharegpt_row(row_b)
                    if ids_o is not None and not np.array_equal(ids_o, tokens):
                        n_jsonl_orig_mismatch += 1
                        print(
                            f"[verify_bpe] jsonl_orig mesh ids != corpus tokens at "
                            f"corpus_index={n_done} line_in_block={li}",
                            flush=True,
                        )
                    if ids_b is not None and not np.array_equal(ids_b, macro_ids):
                        n_jsonl_bpe_mismatch += 1
                        print(
                            f"[verify_bpe] jsonl_bpe mesh ids != BPE macro_ids at "
                            f"corpus_index={n_done} line_in_block={li}",
                            flush=True,
                        )

            n_done += 1

    print(
        f"[verify_bpe] samples_checked={n_ok + n_decode_mismatch} "
        f"decode_ok={n_ok} decode_mismatch={n_decode_mismatch} "
        f"jsonl_orig_mismatch={n_jsonl_orig_mismatch} jsonl_bpe_mismatch={n_jsonl_bpe_mismatch}",
        flush=True,
    )
    if n_decode_mismatch or n_jsonl_orig_mismatch or n_jsonl_bpe_mismatch:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify BPE decode round-trip on corpus + optional JSONL id checks."
    )
    p.add_argument(
        "--corpus_prefix",
        type=str,
        required=True,
        help="Path to the Phase-1 out_jsonl (used as prefix for .corpus.part*.npz / .meta.part*.jsonl).",
    )
    p.add_argument("--merge_table", type=str, required=True, help="BPE merge_table.json")
    p.add_argument("--max_samples", type=int, default=None, help="Stop after this many corpus samples.")
    p.add_argument(
        "--jsonl_orig",
        type=str,
        default=None,
        help="Merged original JSONL (same line layout as when built with meta shards).",
    )
    p.add_argument(
        "--jsonl_bpe",
        type=str,
        default=None,
        help="Merged BPE JSONL (e.g. qwen3vl_3d_sft_8cat_bpe_40W.jsonl).",
    )
    p.add_argument("--multiturn", action="store_true", help="Must match build_qwen3vl_sft_3d_jsonl.py flags.")
    p.add_argument(
        "--task",
        type=str,
        choices=("both", "caption_only", "reconstruct_only"),
        default="both",
        help="Must match build_qwen3vl_sft_3d_jsonl.py when not --multiturn (default: both).",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
