r'''
cd "e:\Python Project\3D-LLM-main\Med-3D-LLM-main"
python test_bpe_3d.py --analyze_bpe_outputs --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl --merge_table ./pad_outputs/merge_table_rel_offset.json --analysis_out_dir ./pad_outputs/bpe_analysis --plot
python test_bpe_3d.py --analyze_bpe_outputs --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl --merge_table ./pad_outputs/merge_table_rel_offset.json --analysis_out_dir ./pad_outputs/bpe_analysis --plot --bootstrap_n 1000 --bootstrap_seed 0 --reliability_check --viz_sample_index 0
python test_bpe_3d.py --sweep_D "0,32,128,512,2048,8192" --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl --merge_table ./pad_outputs/merge_table_rel_offset.json --analysis_out_dir ./pad_outputs/bpe_analysis --plot
python test_bpe_3d.py --robust all --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl --merge_table ./pad_outputs/merge_table_rel_offset.json --analysis_out_dir ./pad_outputs/bpe_analysis --plot

python test_bpe_3d.py --formal_lm_cost --formal_lm_sweep_K "0,32,128,512" --folds 5 --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl --merge_table ./pad_outputs/merge_table_rel_offset.json --analysis_out_dir ./pad_outputs/bpe_analysis --plot --max_samples 128
python test_bpe_3d.py --stability_panel --bootstrap_n 2000 --control_seeds 0,1,2,3,4 --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl --merge_table ./pad_outputs/merge_table_rel_offset.json --analysis_out_dir ./pad_outputs/bpe_analysis --plot --max_samples 128
python test_bpe_3d.py --motif_analysis --motif_controls token_shuffle,coord_shuffle,random_pair_merge --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl --merge_table ./pad_outputs/merge_table_rel_offset.json --analysis_out_dir ./cord_pad_outputs/bpe_analysis --plot --max_samples 128
python test_bpe_3d.py --baseline morton1d --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl --merge_table ./pad_outputs/merge_table_rel_offset.json --analysis_out_dir ./cord_pad_outputs/bpe_analysis --plot
python test_bpe_3d.py --baseline hilbert1d --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl --merge_table ./pad_outputs/merge_table_rel_offset.json --analysis_out_dir ./rerun/bpe_analysis --plot
python test_bpe_3d.py --baseline random3d --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl --merge_table ./pad_outputs/merge_table_rel_offset.json --analysis_out_dir ./cord_pad_outputs/bpe_analysis --plot

python test_bpe_3d.py --formal_lm_cost --stability_panel --motif_analysis --motif_controls token_shuffle,coord_shuffle,random_pair_merge --plot --corpus_prefix ./cord_pad_outputs/qwen3vl_3d_sft_8cat_bpe.jsonl --merge_table ./pad_outputs/merge_table_rel_offset.json --analysis_out_dir ./cord_pad_outputs/bpe_analysis

JSONL fallback: if no ``*.corpus.part*.npz`` exists, ``--corpus_prefix`` may point to a merged mesh JSONL file;
the script decodes ``<morton_*><mesh_*>`` pairs with ``--merge_table`` (same as training export). Pure ``<mesh_i>``-only rows cannot be decoded without anchors.
'''

from __future__ import annotations

import argparse
import copy
import csv
import heapq
import json
import math
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from bpe_3d import BPE3DTokenizer, parse_morton_mesh_pairs

try:
    from bpe_3d import morton3d_encode_xyz32 as _morton_encode_xyz32_strict
except ImportError:
    _morton_encode_xyz32_strict = None  # type: ignore[misc, assignment]


_ANALYSIS_DIRECTIONS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)

# Filled in child processes by ProcessPoolExecutor(initializer=...)
_WORKER_MERGE_TOK: Optional[BPE3DTokenizer] = None

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm is optional for this analysis script.
    _tqdm = None


def _progress(iterable: Iterable[Any], *, total: Optional[int] = None, desc: str = "") -> Iterable[Any]:
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)


class DummySparseTensor:
    def __init__(self, feats: torch.Tensor, coords: torch.Tensor) -> None:
        self.feats = feats
        self.coords = coords

    def replace(self, new_feats: torch.Tensor) -> "DummySparseTensor":
        return DummySparseTensor(new_feats, self.coords)


def _make_one_sample(
    rng: np.random.Generator,
    coord_max: int,
    points_per_sample: int,
    frequent_pairs: List[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    occ = set()
    pts: List[Tuple[int, int, int]] = []
    toks: List[int] = []

    inject_pairs = max(20, points_per_sample // 12)
    for _ in range(inject_pairs):
        ok = False
        for _retry in range(200):
            x = int(rng.integers(0, max(1, coord_max - 1)))
            y = int(rng.integers(0, coord_max))
            z = int(rng.integers(0, coord_max))
            a = (x, y, z)
            b = (x + 1, y, z)
            if b[0] >= coord_max:
                continue
            if a in occ or b in occ:
                continue
            pa, pb = frequent_pairs[int(rng.integers(0, len(frequent_pairs)))]
            pts.extend([a, b])
            toks.extend([pa, pb])
            occ.add(a)
            occ.add(b)
            ok = True
            break
        if not ok:
            break

    while len(pts) < points_per_sample:
        c = (
            int(rng.integers(0, coord_max)),
            int(rng.integers(0, coord_max)),
            int(rng.integers(0, coord_max)),
        )
        if c in occ:
            continue
        occ.add(c)
        pts.append(c)
        toks.append(int(rng.integers(0, 8192)))

    order = rng.permutation(len(pts))
    coords = np.asarray(pts, dtype=np.int64)[order]
    tokens = np.asarray(toks, dtype=np.int64)[order]
    return {"tokens": tokens, "coords": coords}


def build_corpus(
    n_samples: int,
    coord_max: int,
    points_per_sample: int,
    seed: int,
) -> List[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    hot_ids = rng.integers(0, 256, size=20, dtype=np.int64).tolist()
    frequent_pairs = [(int(hot_ids[i]), int(hot_ids[(i + 1) % len(hot_ids)])) for i in range(len(hot_ids))]
    return [
        _make_one_sample(
            rng=rng,
            coord_max=coord_max,
            points_per_sample=points_per_sample,
            frequent_pairs=frequent_pairs,
        )
        for _ in range(n_samples)
    ]


def build_structured_corpus() -> List[Dict[str, np.ndarray]]:
    """Cases that force multi-cell macro tokens to be merged again."""
    templates = [
        # Straight chain: old dir-only decode collapses the third/fourth cells.
        (
            [101, 102, 103, 104],
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
        ),
        # L / corner shape.
        (
            [201, 202, 203, 204],
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
        ),
        # Branch-like shape around a shared stem.
        (
            [301, 302, 303, 304, 305],
            [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1)],
        ),
    ]
    corpus: List[Dict[str, np.ndarray]] = []
    for shift in range(8):
        dx = shift * 5
        dy = shift % 3
        dz = shift % 2
        for tokens, coords in templates:
            xyz = np.asarray(
                [(x + dx, y + dy, z + dz) for x, y, z in coords],
                dtype=np.int64,
            )
            corpus.append(
                {
                    "tokens": np.asarray(tokens, dtype=np.int64),
                    "coords": xyz,
                }
            )
    return corpus


def to_sparse(sample: Dict[str, np.ndarray], sparse_cls) -> DummySparseTensor:
    coords_np = sample["coords"]
    tokens_np = sample["tokens"]
    n = coords_np.shape[0]
    batch_col = np.zeros((n, 1), dtype=np.int32)
    coords4 = np.concatenate([batch_col, coords_np.astype(np.int32)], axis=1)
    feats = torch.tensor(tokens_np, dtype=torch.float32).unsqueeze(-1)
    coords = torch.tensor(coords4, dtype=torch.int32)
    return sparse_cls(feats, coords)


def sparse_item_set(st: DummySparseTensor) -> set:
    c = st.coords.detach().cpu().numpy().astype(np.int64)
    f = st.feats.squeeze(-1).detach().cpu().numpy().astype(np.int64)
    return {(int(fid), int(x), int(y), int(z)) for fid, (_b, x, y, z) in zip(f.tolist(), c.tolist())}


def assert_roundtrip(tok: BPE3DTokenizer, corpus: List[Dict[str, np.ndarray]], sparse_cls) -> Tuple[int, int]:
    total_before = 0
    total_after = 0
    for sample in corpus:
        st = to_sparse(sample, sparse_cls)
        out = tok.encode_sparse(st, sparse_tensor_cls=sparse_cls)
        batch0 = out["batches"][0]
        total_before += int(sample["tokens"].shape[0])
        total_after += int(batch0["ids"].shape[0])

        dec = tok.decode_to_sparse(out["batches"], device=torch.device("cpu"), sparse_tensor_cls=sparse_cls)
        got = sparse_item_set(dec)
        expect = sparse_item_set(st)
        assert got == expect, "encode -> decode is not reversible on sample"
    return total_before, total_after


def _discover_corpus_shards(corpus_prefix: Path) -> List[Path]:
    paths: List[Path] = []
    r = 0
    while True:
        p = Path(f"{corpus_prefix}.corpus.part{r}.npz")
        if not p.is_file():
            break
        paths.append(p)
        r += 1
    return paths


def _load_corpus_part(path: Path) -> List[Dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    offsets = np.asarray(data["offsets"], dtype=np.int64)
    tokens = np.asarray(data["tokens"], dtype=np.int64)
    coords = np.asarray(data["coords"], dtype=np.int64)
    out: List[Dict[str, np.ndarray]] = []
    for i in range(len(offsets) - 1):
        a, b = int(offsets[i]), int(offsets[i + 1])
        out.append({"tokens": tokens[a:b], "coords": coords[a:b]})
    return out


def _first_mesh_content_from_sharegpt_row(row: Dict[str, Any]) -> Optional[str]:
    for m in row.get("messages") or []:
        content = m.get("content")
        if content and "<mesh_start>" in str(content):
            return str(content)
    return None


def _load_corpus_records_from_mesh_jsonl(
    jsonl_path: Path,
    tok: BPE3DTokenizer,
    max_samples: Optional[int],
) -> List[Dict[str, np.ndarray]]:
    """Recover leaf (tokens, coords) by decoding Morton/mesh pairs from JSONL (no npz required).

    Requires ``<morton_*><mesh_*>`` adjacency as produced by ``serialize_morton_mesh_pairs``.
    Legacy rows with only ``<mesh_i>`` cannot be decoded to anchors and are skipped.
    """
    out: List[Dict[str, np.ndarray]] = []
    n_skip_no_morton = 0
    n_skip_decode = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in _progress(f, desc="[CORPUS] decode jsonl"):
            if max_samples is not None and len(out) >= int(max_samples):
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = _first_mesh_content_from_sharegpt_row(row)
            if content is None:
                continue
            parsed = parse_morton_mesh_pairs(content, max_mesh_id=None)
            if int(parsed.ids.shape[0]) == 0:
                if re.search(r"<mesh_\d+>", content):
                    n_skip_no_morton += 1
                continue
            batches = [
                {
                    "ids": np.asarray(parsed.ids, dtype=np.int64),
                    "anchors": np.asarray(parsed.anchors, dtype=np.int64),
                }
            ]
            try:
                st = tok.decode_to_sparse(
                    batches, device=torch.device("cpu"), sparse_tensor_cls=DummySparseTensor
                )
            except Exception:
                n_skip_decode += 1
                continue
            c = st.coords.detach().cpu().numpy().astype(np.int64)
            feats = st.feats.squeeze(-1).detach().cpu().numpy().astype(np.int64)
            if c.shape[0] == 0:
                continue
            out.append({"tokens": feats, "coords": c[:, 1:4]})
    if not out:
        raise ValueError(
            f"jsonl {jsonl_path}: decoded 0 leaf samples. "
            f"Skipped ≈{n_skip_no_morton} lines with <mesh_*> but no valid <morton_*><mesh_*> pairs; "
            f"decode failures={n_skip_decode}. "
            "Use Phase-1 ``*.corpus.part*.npz`` next to the original --out_jsonl, or export JSONL with Morton/mesh pairs."
        )
    if n_skip_no_morton:
        print(
            f"[CORPUS] jsonl: skipped {n_skip_no_morton} lines (mesh tokens without adjacent morton pairs)",
            flush=True,
        )
    return out


def _entropy_bits(counter: Counter) -> float:
    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = float(c) / total
        ent -= p * math.log2(p)
    return ent


def _merge_counter(dst: Counter, src: Counter) -> None:
    for k, v in src.items():
        dst[k] += int(v)


def _token_counter(tokens: np.ndarray) -> Counter:
    return Counter(int(x) for x in np.asarray(tokens, dtype=np.int64).tolist())


def _adjacency_pair_counts(tokens: np.ndarray, coords: np.ndarray) -> Counter:
    tokens = np.asarray(tokens, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.int64)
    cell_to_idx: Dict[Tuple[int, int, int], int] = {}
    for i, xyz in enumerate(coords.tolist()):
        cell_to_idx[(int(xyz[0]), int(xyz[1]), int(xyz[2]))] = int(i)

    counts: Counter = Counter()
    for i, xyz in enumerate(coords.tolist()):
        x, y, z = int(xyz[0]), int(xyz[1]), int(xyz[2])
        a = int(tokens[i])
        for d, dv in enumerate(_ANALYSIS_DIRECTIONS):
            j = cell_to_idx.get((x + dv[0], y + dv[1], z + dv[2]))
            if j is None:
                continue
            counts[(a, int(tokens[j]), d)] += 1
    return counts


def _macro_footprint_offsets(tok: BPE3DTokenizer, token_id: int) -> List[Tuple[int, int, int]]:
    tid = int(token_id)
    if tid < int(tok.base_vocab_size):
        return [(0, 0, 0)]
    idx = tid - int(tok.base_vocab_size)
    if idx < 0 or idx >= len(tok.merge_table):
        raise ValueError(f"macro token id {tid} is outside merge_table range")
    offsets = tok.merge_table[idx].leaf_offsets
    if not offsets:
        raise ValueError(f"macro token id {tid} has empty leaf_offsets")
    return [(int(x), int(y), int(z)) for x, y, z in offsets]


def _macro_boundary_pair_counts(
    tok: BPE3DTokenizer,
    macro_ids: np.ndarray,
    macro_anchors: np.ndarray,
) -> Counter:
    macro_ids = np.asarray(macro_ids, dtype=np.int64)
    macro_anchors = np.asarray(macro_anchors, dtype=np.int64)
    cell_to_macro: Dict[Tuple[int, int, int], int] = {}
    for mi, (tid, anc) in enumerate(zip(macro_ids.tolist(), macro_anchors.tolist())):
        ax, ay, az = int(anc[0]), int(anc[1]), int(anc[2])
        for off in _macro_footprint_offsets(tok, int(tid)):
            c = (ax + off[0], ay + off[1], az + off[2])
            cell_to_macro[c] = int(mi)

    counts: Counter = Counter()
    for cell, mi in cell_to_macro.items():
        x, y, z = cell
        a = int(macro_ids[mi])
        for d, dv in enumerate(_ANALYSIS_DIRECTIONS):
            mj = cell_to_macro.get((x + dv[0], y + dv[1], z + dv[2]))
            if mj is None or mj == mi:
                continue
            counts[(a, int(macro_ids[mj]), d)] += 1
    return counts


def _directed_pair_stats(pair_counts: Counter) -> Dict[str, float]:
    pair_total = int(sum(pair_counts.values()))
    source_dir_counts: Counter = Counter()
    target_counts: Counter = Counter()
    direction_counts: Counter = Counter()
    for (a, b, d), c in pair_counts.items():
        source_dir_counts[(int(a), int(d))] += int(c)
        target_counts[int(b)] += int(c)
        direction_counts[int(d)] += int(c)

    pair_entropy = _entropy_bits(pair_counts)
    source_dir_entropy = _entropy_bits(source_dir_counts)
    target_entropy = _entropy_bits(target_counts)
    conditional_entropy = max(0.0, pair_entropy - source_dir_entropy)
    mutual_information = max(0.0, source_dir_entropy + target_entropy - pair_entropy)
    return {
        "edge_count": float(pair_total),
        "pair_entropy_bits": pair_entropy,
        "source_direction_entropy_bits": source_dir_entropy,
        "target_entropy_bits": target_entropy,
        "target_given_source_direction_entropy_bits": conditional_entropy,
        "source_direction_target_mutual_information_bits": mutual_information,
        "direction_entropy_bits": _entropy_bits(direction_counts),
    }


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) != 0.0 else 0.0


def _filter_pairs_by_direction(pair_counts: Counter, direction_index: int) -> Counter:
    out: Counter = Counter()
    for (a, b, d), c in pair_counts.items():
        if int(d) == int(direction_index):
            out[(int(a), int(b), int(d))] += int(c)
    return out


def _per_direction_block(
    leaf_pairs: Counter, macro_anchor_pairs: Counter, macro_boundary_pairs: Counter
) -> Dict[str, Any]:
    names = ("plus_x", "plus_y", "plus_z")
    out: Dict[str, Any] = {}
    for d, name in enumerate(names):
        out[name] = {
            "leaf": {"adjacency": _directed_pair_stats(_filter_pairs_by_direction(leaf_pairs, d))},
            "macro_anchor": {"adjacency": _directed_pair_stats(_filter_pairs_by_direction(macro_anchor_pairs, d))},
            "macro_boundary": {
                "adjacency": _directed_pair_stats(_filter_pairs_by_direction(macro_boundary_pairs, d))
            },
        }
    return out


def _macro_footprint_vocab_from_table(tok: BPE3DTokenizer) -> Dict[str, Any]:
    sizes: List[int] = []
    for entry in tok.merge_table:
        sizes.append(len(entry.leaf_offsets))
    if not sizes:
        return {"count": 0, "mean_cells": 0.0, "p50": 0.0, "p90": 0.0, "max_cells": 0}
    arr = np.asarray(sizes, dtype=np.float64)
    return {
        "merge_rules": int(len(sizes)),
        "mean_cells": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max_cells": int(np.max(arr)),
    }


def _merge_table_freq_metadata(merge_path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(merge_path.read_text(encoding="utf-8"))
    except OSError:
        return {"merge_freq_skipped": True, "merge_freq_skip_reason": "merge_table unreadable"}
    rows = data.get("merge_table", [])
    has_freq = any(isinstance(e, dict) and ("freq" in e or "frequency" in e) for e in rows)
    return {
        "merge_freq_skipped": not bool(has_freq),
        "merge_freq_skip_reason": None
        if has_freq
        else "merge_table entries have no freq/frequency field (training log not exported)",
    }


def _bootstrap_ci(
    per_sample_rows: Sequence[Dict[str, Any]],
    n_iter: int,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap 95% CI for ratios computable from per-sample aggregates only."""
    if not per_sample_rows or n_iter <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(per_sample_rows)
    tl_a = np.fromiter((int(r["leaf_tokens"]) for r in per_sample_rows), dtype=np.int64, count=n)
    tm_a = np.fromiter((int(r["macro_tokens"]) for r in per_sample_rows), dtype=np.int64, count=n)
    le_a = np.fromiter((int(r["leaf_edges"]) for r in per_sample_rows), dtype=np.int64, count=n)
    mb_a = np.fromiter((int(r["macro_boundary_edges"]) for r in per_sample_rows), dtype=np.int64, count=n)
    comp_parts: List[np.ndarray] = []
    edge_parts: List[np.ndarray] = []
    chunk = max(1, min(64, int(n_iter)))
    done = 0
    while done < int(n_iter):
        bs = min(chunk, int(n_iter) - done)
        idx = rng.integers(0, n, size=(bs, n))
        tl_s = tl_a[idx].sum(axis=1).astype(np.float64)
        tm_s = tm_a[idx].sum(axis=1).astype(np.float64)
        le_s = le_a[idx].sum(axis=1).astype(np.float64)
        mb_s = mb_a[idx].sum(axis=1).astype(np.float64)
        comp_parts.append(np.where(tl_s > 0.0, tm_s / tl_s, 0.0))
        edge_parts.append(np.where(le_s > 0.0, mb_s / le_s, 0.0))
        done += bs
    comp = np.concatenate(comp_parts)
    edge_r = np.concatenate(edge_parts)

    def _ci_arr(a: np.ndarray) -> Dict[str, float]:
        return {
            "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)),
            "mean": float(np.mean(a)),
        }

    return {
        "compression_ratio_total": _ci_arr(comp),
        "boundary_edge_ratio": _ci_arr(edge_r),
        "iterations": int(n_iter),
        "seed": int(seed),
    }


def _morton_codes_for_anchors(anchors: np.ndarray) -> np.ndarray:
    """Per-row Morton code; strict xyz32 when in range, else packed lex fallback."""
    anchors = np.asarray(anchors, dtype=np.int64)
    if anchors.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if (
        _morton_encode_xyz32_strict is not None
        and anchors.shape[1] == 3
        and anchors.min() >= 0
        and anchors.max() < 32
    ):
        try:
            return np.asarray(_morton_encode_xyz32_strict(anchors), dtype=np.int64).reshape(-1)
        except ValueError:
            pass
    cmax = int(np.max(anchors)) + 1 if anchors.size else 1
    return ((anchors[:, 0] * cmax + anchors[:, 1]) * cmax + anchors[:, 2]).astype(np.int64)


def _ordered_mesh_morton(
    ids: np.ndarray, anchors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    anchors = np.asarray(anchors, dtype=np.int64)
    if ids.shape[0] == 0:
        return ids, np.zeros((0,), dtype=np.int64)
    order = _sort_indices_packed(anchors)
    a = anchors[order]
    return ids[order], _morton_codes_for_anchors(a)


def _bigram_conditional_entropy_bits(seq: List[int]) -> float:
    if len(seq) < 2:
        return 0.0
    pair_c: Counter = Counter()
    uni_c: Counter = Counter()
    for i in range(len(seq) - 1):
        a, b = int(seq[i]), int(seq[i + 1])
        pair_c[(a, b)] += 1
        uni_c[a] += 1
    tot = float(sum(pair_c.values()))
    if tot <= 0:
        return 0.0
    h_pair = -sum((c / tot) * math.log2(c / tot) for c in pair_c.values())
    h_src = _entropy_bits(uni_c)
    return max(0.0, h_pair - h_src)


def _entropy_rate_ngram_bits(seq: List[int], n: int) -> float:
    """Conditional entropy H(X_n | X_{n-1}...X_1) estimated from empirical n-grams."""
    if n < 2 or len(seq) < n:
        return 0.0
    ctx_c: Counter = Counter()
    joint_c: Counter = Counter()
    for i in range(len(seq) - n + 1):
        ctx = tuple(int(seq[i + j]) for j in range(n - 1))
        x = int(seq[i + n - 1])
        ctx_c[ctx] += 1
        joint_c[(ctx, x)] += 1
    tot_ctx = float(sum(ctx_c.values()))
    if tot_ctx <= 0:
        return 0.0
    h_joint = -sum((c / tot_ctx) * math.log2(c / tot_ctx) for c in joint_c.values())
    h_ctx = -sum((c / tot_ctx) * math.log2(c / tot_ctx) for c in ctx_c.values())
    return max(0.0, h_joint - h_ctx)


def _repeat_mass_n(seq: List[int], n: int) -> float:
    if len(seq) < n:
        return 0.0
    ng: Counter = Counter()
    for i in range(len(seq) - n + 1):
        ng[tuple(int(seq[i + j]) for j in range(n))] += 1
    tot = float(sum(ng.values()))
    if tot <= 0:
        return 0.0
    rep = sum(float(c) for c in ng.values() if c > 1)
    return rep / tot


def _distinct_n_rate(seq: List[int], n: int) -> float:
    if len(seq) < n:
        return 0.0
    seen = set()
    for i in range(len(seq) - n + 1):
        seen.add(tuple(int(seq[i + j]) for j in range(n)))
    return float(len(seen)) / float(len(seq) - n + 1)


def _control_tokens_per_seq_overhead() -> float:
    return 2.0


def _corpus_mesh_sequences_for_tok(
    tok: BPE3DTokenizer, records: List[Dict[str, np.ndarray]]
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    mesh_seqs: List[List[int]] = []
    mort_seqs: List[List[int]] = []
    leaf_counts: List[int] = []
    for rec in _progress(records, total=len(records), desc="[FORMAL_LM] encode mesh"):
        t = np.asarray(rec["tokens"], dtype=np.int64)
        c = np.asarray(rec["coords"], dtype=np.int64)
        mid, ma = tok.encode_sparse_numpy(t, c)
        m_ids, m_codes = _ordered_mesh_morton(mid, ma)
        mesh_seqs.append([int(x) for x in m_ids.tolist()])
        mort_seqs.append([int(x) for x in m_codes.tolist()])
        leaf_counts.append(int(t.shape[0]))
    return mesh_seqs, mort_seqs, leaf_counts


def _merge_table_worker_init(merge_path: str) -> None:
    global _WORKER_MERGE_TOK
    _WORKER_MERGE_TOK = BPE3DTokenizer.load(merge_path)


def _analysis_workers_requested(args: argparse.Namespace) -> int:
    w = int(getattr(args, "analysis_num_workers", 0) or 0)
    if w <= 0:
        return max(1, min(os.cpu_count() or 1, 32))
    return max(1, w)


def _records_chunk_list(
    records: List[Dict[str, np.ndarray]], n_workers: int, min_samples: int = 200
) -> List[List[Dict[str, np.ndarray]]]:
    n = len(records)
    if n_workers <= 1 or n < min_samples:
        return [records]
    n_parts = min(n_workers, max(1, n // 100))
    if n_parts <= 1:
        return [records]
    chunk_sz = max(1, (n + n_parts - 1) // n_parts)
    return [records[i : i + chunk_sz] for i in range(0, n, chunk_sz)]


def _concat_mesh_sequence_parts(
    parts: Sequence[Tuple[List[List[int]], List[List[int]], List[int]]],
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    mesh_all: List[List[int]] = []
    mort_all: List[List[int]] = []
    leaf_all: List[int] = []
    for m, mo, lf in parts:
        mesh_all.extend(m)
        mort_all.extend(mo)
        leaf_all.extend(lf)
    return mesh_all, mort_all, leaf_all


def _mp_mesh_sequences_task(
    args: Tuple[int, List[Dict[str, np.ndarray]]],
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    K, chunk = args
    assert _WORKER_MERGE_TOK is not None
    tok_k = _truncate_tokenizer(_WORKER_MERGE_TOK, K)
    return _corpus_mesh_sequences_for_tok(tok_k, chunk)


def _mp_stability_chunk(
    chunk: List[Dict[str, np.ndarray]],
) -> Dict[str, List[Any]]:
    assert _WORKER_MERGE_TOK is not None
    tok_full = _WORKER_MERGE_TOK
    K_full = len(tok_full.merge_table)
    tok_bpe = _truncate_tokenizer(tok_full, K_full)
    tok_id = _truncate_tokenizer(tok_full, 0)
    d_comp: List[float] = []
    d_lm: List[float] = []
    d_edge: List[float] = []
    d_motif_cov: List[float] = []
    leaf_ns: List[int] = []
    bbox_vs: List[int] = []
    occs: List[float] = []
    uniq_types: List[int] = []
    clm_bpe: List[float] = []
    for rec in chunk:
        t = np.asarray(rec["tokens"], dtype=np.int64)
        c = np.asarray(rec["coords"], dtype=np.int64)
        m_b, a_b = tok_bpe.encode_sparse_numpy(t, c)
        m_l, a_l = tok_id.encode_sparse_numpy(t, c)
        lp = _adjacency_pair_counts(t, c)
        le = float(sum(lp.values()))
        mb_b = _macro_boundary_pair_counts(tok_bpe, m_b, a_b)
        mb_l = _macro_boundary_pair_counts(tok_id, m_l, a_l)
        r_b = _safe_ratio(int(m_b.shape[0]), int(t.shape[0]))
        r_l = _safe_ratio(int(m_l.shape[0]), int(t.shape[0]))
        edge_ratio_b = _safe_ratio(float(sum(mb_b.values())), le)
        edge_ratio_l = _safe_ratio(float(sum(mb_l.values())), le)
        d_comp.append(r_l - r_b)
        d_edge.append(edge_ratio_l - edge_ratio_b)
        sq_b, mort_b = _ordered_mesh_morton(m_b, a_b)
        sq_l, mort_l = _ordered_mesh_morton(m_l, a_l)
        pm_b = _per_sample_lm_proxy_cost(
            [int(x) for x in sq_b.tolist()], [int(x) for x in mort_b.tolist()], int(t.shape[0])
        )
        pm_l = _per_sample_lm_proxy_cost(
            [int(x) for x in sq_l.tolist()], [int(x) for x in mort_l.tolist()], int(t.shape[0])
        )
        d_lm.append(float(pm_l["clm_bits_per_leaf"] - pm_b["clm_bits_per_leaf"]))
        clm_bpe.append(float(pm_b["clm_bits_per_leaf"]))
        cov_b = (
            float(np.mean([len(_macro_footprint_offsets(tok_bpe, int(x))) for x in m_b.tolist()]))
            if m_b.shape[0]
            else 0.0
        )
        cov_l = (
            float(np.mean([len(_macro_footprint_offsets(tok_id, int(x))) for x in m_l.tolist()]))
            if m_l.shape[0]
            else 0.0
        )
        d_motif_cov.append(cov_l - cov_b)
        nl = int(t.shape[0])
        leaf_ns.append(nl)
        if c.size:
            cmin = c.min(axis=0)
            cmax = c.max(axis=0)
            bv = int(np.prod(np.maximum(cmax - cmin + 1, 1)))
            bbox_vs.append(bv)
            occs.append(float(t.shape[0]) / float(max(bv, 1)))
        else:
            bbox_vs.append(0)
            occs.append(0.0)
        uniq_types.append(int(len(set(t.tolist()))))
    return {
        "d_comp": d_comp,
        "d_lm": d_lm,
        "d_edge": d_edge,
        "d_motif_cov": d_motif_cov,
        "leaf_ns": leaf_ns,
        "bbox_vs": bbox_vs,
        "occs": occs,
        "uniq_types": uniq_types,
        "clm_bpe": clm_bpe,
    }


def _mp_motif_corpus_chunk(chunk: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    assert _WORKER_MERGE_TOK is not None
    tok = _truncate_tokenizer(_WORKER_MERGE_TOK, len(_WORKER_MERGE_TOK.merge_table))
    usage: Counter = Counter()
    corpus_size_hist: Counter = Counter()
    internal_dir = 0
    total_leaf_edges_dir = 0
    macro_usage: Counter = Counter()
    macro_leaf_cov: Counter = Counter()
    for rec in chunk:
        t = np.asarray(rec["tokens"], dtype=np.int64)
        c = np.asarray(rec["coords"], dtype=np.int64)
        mid, ma = tok.encode_sparse_numpy(t, c)
        cell_to_macro: Dict[Tuple[int, int, int], int] = {}
        for mi, (tid, anc) in enumerate(zip(mid.tolist(), ma.tolist())):
            ax, ay, az = int(anc[0]), int(anc[1]), int(anc[2])
            for ox, oy, oz in _macro_footprint_offsets(tok, int(tid)):
                cell_to_macro[(ax + ox, ay + oy, az + oz)] = mi
        lp = _adjacency_pair_counts(t, c)
        total_leaf_edges_dir += int(sum(lp.values()))
        for _, tid in enumerate(mid.tolist()):
            tid_i = int(tid)
            usage[tid_i] += 1
            sz = len(_macro_footprint_offsets(tok, tid_i))
            corpus_size_hist[sz] += 1
            macro_usage[tid_i] += 1
            macro_leaf_cov[tid_i] += sz
        for (x, y, z), mi in cell_to_macro.items():
            for dv in _ANALYSIS_DIRECTIONS:
                nj = cell_to_macro.get((x + dv[0], y + dv[1], z + dv[2]))
                if nj is None:
                    continue
                if nj == mi:
                    internal_dir += 1
    return {
        "usage": dict(usage),
        "corpus_size_hist": dict(corpus_size_hist),
        "internal_absorbed_dir": int(internal_dir),
        "total_leaf_edges_dir": int(total_leaf_edges_dir),
        "macro_usage": dict(macro_usage),
        "macro_leaf_cov": dict(macro_leaf_cov),
    }


def _mp_compression_chunk(
    payload: Tuple[
        List[Dict[str, np.ndarray]],
        Optional[Tuple[int, int, int]],
        Optional[Tuple[int, int, int]],
        float,
        float,
        int,
    ],
) -> Tuple[int, int]:
    chunk, coord_shift, axis_perm, noise_p, drop_p, seed = payload
    assert _WORKER_MERGE_TOK is not None
    tok = _WORKER_MERGE_TOK
    rng = np.random.default_rng(seed)
    total_leaf = 0
    total_macro = 0
    for rec in chunk:
        tokens = np.asarray(rec["tokens"], dtype=np.int64).copy()
        coords = np.asarray(rec["coords"], dtype=np.int64).copy()
        if coord_shift is not None:
            coords[:, 0] += coord_shift[0]
            coords[:, 1] += coord_shift[1]
            coords[:, 2] += coord_shift[2]
        if axis_perm is not None:
            coords = coords[:, list(axis_perm)]
        if noise_p > 0.0 and tokens.shape[0] > 0:
            mask = rng.random(tokens.shape[0]) < noise_p
            if np.any(mask):
                nrep = int(np.sum(mask))
                tokens[mask] = rng.integers(0, int(tok.base_vocab_size), size=nrep)
        if drop_p > 0.0 and tokens.shape[0] > 0:
            keep = rng.random(tokens.shape[0]) >= drop_p
            if not np.any(keep):
                continue
            tokens = tokens[keep]
            coords = coords[keep]
        if tokens.shape[0] == 0:
            continue
        macro_ids, _ = tok.encode_sparse_numpy(tokens, coords)
        total_leaf += int(tokens.shape[0])
        total_macro += int(macro_ids.shape[0])
    return total_leaf, total_macro


def _mp_control_shard(payload: Tuple[List[Dict[str, np.ndarray]], str]) -> Tuple[List[float], List[float], List[float]]:
    """Single-process shard for motif_controls inner loop (shuffle applied upstream per seed)."""
    chunk, _mode_tag = payload
    assert _WORKER_MERGE_TOK is not None
    tok_full = _WORKER_MERGE_TOK
    tok = _truncate_tokenizer(tok_full, len(tok_full.merge_table))
    tok0 = _truncate_tokenizer(tok_full, 0)
    ratios: List[float] = []
    cov: List[float] = []
    clm_delta: List[float] = []
    for rec in chunk:
        tt = np.asarray(rec["tokens"], dtype=np.int64)
        cc = np.asarray(rec["coords"], dtype=np.int64)
        mm, aa = tok.encode_sparse_numpy(tt, cc)
        mm0, aa0 = tok0.encode_sparse_numpy(tt, cc)
        ratios.append(_safe_ratio(int(mm.shape[0]), int(tt.shape[0])))
        cl = sum(len(_macro_footprint_offsets(tok, int(tid))) for tid in mm.tolist())
        cov.append(_safe_ratio(cl, int(tt.shape[0])))
        sq_b, mort_b = _ordered_mesh_morton(mm, aa)
        sq_l, mort_l = _ordered_mesh_morton(mm0, aa0)
        pm_b = _per_sample_lm_proxy_cost(
            [int(x) for x in sq_b.tolist()], [int(x) for x in mort_b.tolist()], int(tt.shape[0])
        )
        pm_l = _per_sample_lm_proxy_cost(
            [int(x) for x in sq_l.tolist()], [int(x) for x in mort_l.tolist()], int(tt.shape[0])
        )
        clm_delta.append(float(pm_l["clm_bits_per_leaf"] - pm_b["clm_bits_per_leaf"]))
    return ratios, cov, clm_delta


def _per_leaf_weighted_nll_plugin(
    seqs: List[List[int]], leaf_counts: List[int], global_counts: Counter
) -> float:
    vocab = max(len(global_counts), 1)
    total_toks = float(sum(global_counts.values()))
    if total_toks <= 0:
        return 0.0
    sm = float(vocab)
    acc = 0.0
    n_ok = 0
    for s, nl in zip(seqs, leaf_counts):
        if nl <= 0 or not s:
            continue
        nll = 0.0
        for x in s:
            cx = float(global_counts.get(int(x), 0)) + 1.0
            denom = total_toks + sm
            nll -= math.log2(cx / denom)
        acc += (float(len(s)) / float(nl)) * (nll / float(len(s)))
        n_ok += 1
    return acc / float(max(n_ok, 1))


def _kfold_unigram_nll_mean(
    seqs: List[List[int]], folds: int, seed: int
) -> Dict[str, float]:
    if folds < 2 or not seqs:
        return {"mean_nll_bits_per_symbol": 0.0, "folds": int(folds)}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(seqs))
    rng.shuffle(idx)
    fold_nlls: List[float] = []
    for f in range(folds):
        mask = np.ones(len(seqs), dtype=bool)
        mask[idx[f::folds]] = False
        train_c: Counter = Counter()
        for j in idx[mask]:
            train_c.update(seqs[int(j)])
        vocab = max(len(train_c), 1)
        tot_train = float(sum(train_c.values()))
        sm = float(vocab)
        nll_sum = 0.0
        n_tok = 0
        for j in idx[~mask]:
            for x in seqs[int(j)]:
                cx = float(train_c.get(int(x), 0)) + 1.0
                denom = tot_train + sm
                nll_sum -= math.log2(cx / denom)
                n_tok += 1
        fold_nlls.append(float(nll_sum / max(n_tok, 1)))
    return {
        "mean_nll_bits_per_mesh_symbol_kfold": float(np.mean(fold_nlls)),
        "std_nll_bits_per_mesh_symbol_kfold": float(np.std(fold_nlls)),
        "folds": int(folds),
        "seed": int(seed),
    }


def _heldout_nll_unigram(
    seqs: List[List[int]], train_frac: float, seed: int
) -> Dict[str, float]:
    if not seqs:
        return {"nll_bits_per_symbol_val": 0.0}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(seqs))
    rng.shuffle(idx)
    n = len(seqs)
    ntr = max(1, min(n - 1, int(round(float(train_frac) * n))))
    train_c: Counter = Counter()
    for j in idx[:ntr]:
        train_c.update(seqs[int(j)])
    vocab = max(len(train_c), 1)
    tot_train = float(sum(train_c.values()))
    sm = float(vocab)
    nll_sum = 0.0
    n_tok = 0
    for j in idx[ntr:]:
        for x in seqs[int(j)]:
            cx = float(train_c.get(int(x), 0)) + 1.0
            denom = tot_train + sm
            nll_sum -= math.log2(cx / denom)
            n_tok += 1
    return {
        "train_frac": float(train_frac),
        "nll_bits_per_mesh_symbol_heldout_val": float(nll_sum / max(n_tok, 1)),
        "val_samples": int(n - ntr),
        "seed": int(seed),
    }


def _load_corpus_records(
    corpus_prefix: Path,
    max_samples: Optional[int],
    *,
    tokenizer: Optional[BPE3DTokenizer] = None,
    verbose: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """Load leaf-level records from ``*.corpus.part*.npz`` or from a mesh JSONL file.

    JSONL path must be an existing file ending in ``.jsonl``; requires ``tokenizer`` to
    ``decode_to_sparse`` macro streams into leaf ``(tokens, coords)``.
    """
    shards = _discover_corpus_shards(corpus_prefix)
    if shards:
        records: List[Dict[str, np.ndarray]] = []
        for sp in _progress(shards, total=len(shards), desc="[CORPUS] load shards"):
            records.extend(_load_corpus_part(sp))
            if max_samples is not None and len(records) >= int(max_samples):
                return records[: int(max_samples)]
        return records
    p = Path(corpus_prefix)
    if p.is_file() and p.suffix.lower() == ".jsonl":
        if tokenizer is None:
            raise FileNotFoundError(
                f"no corpus shards at {corpus_prefix}.corpus.part*.npz. "
                f"To use JSONL {p}, pass the same --merge_table used to train/export mesh strings "
                f"so the script can decode <morton_*><mesh_*> to leaf voxels, "
                f"or re-run Phase-1 to materialize npz next to the original --out_jsonl path."
            )
        recs = _load_corpus_records_from_mesh_jsonl(p, tokenizer, max_samples)
        if verbose:
            print(f"[CORPUS] loaded {len(recs)} samples from jsonl via decode_to_sparse ({p})", flush=True)
        return recs
    raise FileNotFoundError(
        f"no corpus shards at {corpus_prefix}.corpus.part*.npz and path is not a .jsonl file: {p}"
    )


def _anchor_delta_entropy_bits(mort_seq: List[int]) -> float:
    if len(mort_seq) < 2:
        return 0.0
    deltas = [int(mort_seq[i] - mort_seq[i - 1]) for i in range(1, len(mort_seq))]
    return _entropy_bits(Counter(deltas))


def _per_sample_lm_proxy_cost(mesh_seq: List[int], mort_seq: List[int], leaf_n: int) -> Dict[str, float]:
    if leaf_n <= 0:
        return {"clm_bits_per_leaf": 0.0, "rlen": 0.0}
    rlen = float(len(mesh_seq)) / float(leaf_n)
    hm = _entropy_bits(Counter(mesh_seq))
    hd = _anchor_delta_entropy_bits(mort_seq)
    hb = _bigram_conditional_entropy_bits(mesh_seq)
    ctrl = _control_tokens_per_seq_overhead() * rlen / max(float(leaf_n), 1.0)
    clm = rlen * hm + rlen * hd + rlen * hb + ctrl
    return {
        "clm_bits_per_leaf": float(clm),
        "rlen": float(rlen),
        "H_mesh_micro_bits": float(hm),
        "H_anchor_delta_micro_bits": float(hd),
        "bigram_cond_micro_bits": float(hb),
        "control_proxy_per_leaf_bits": float(ctrl),
    }


def _sign_test_two_sided(pos: int, neg: int) -> float:
    """Exact two-sided binomial p-value for H0: P(+)=0.5 (ignores zeros)."""
    n = int(pos + neg)
    if n <= 0:
        return 1.0
    lo = min(int(pos), int(neg))
    p_one_tail = 0.0
    for k in range(0, lo + 1):
        p_one_tail += math.comb(n, k) * (0.5**n)
    return float(min(1.0, 2.0 * p_one_tail))


def _paired_bootstrap_mean_ci(
    deltas: List[float], n_iter: int, seed: int
) -> Dict[str, float]:
    if not deltas or n_iter <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(deltas)
    arr = np.asarray(deltas, dtype=np.float64)
    m_parts: List[np.ndarray] = []
    chunk = max(1, min(256, int(n_iter)))
    done = 0
    while done < int(n_iter):
        bs = min(chunk, int(n_iter) - done)
        idx = rng.integers(0, n, size=(bs, n))
        m_parts.append(np.mean(arr[idx], axis=1))
        done += bs
    m = np.concatenate(m_parts)
    pos = int(np.sum(arr > 0))
    neg = int(np.sum(arr < 0))
    return {
        "mean_delta": float(np.mean(arr)),
        "ci_lo": float(np.percentile(m, 2.5)),
        "ci_hi": float(np.percentile(m, 97.5)),
        "iterations": int(n_iter),
        "sign_test_p_value_two_sided": _sign_test_two_sided(pos, neg),
        "n_positive": pos,
        "n_negative": neg,
    }


def _motif_canonical_offsets(offsets: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    if not offsets:
        return []
    pts = [(int(a), int(b), int(c)) for a, b, c in offsets]
    mx = min(p[0] for p in pts)
    my = min(p[1] for p in pts)
    mz = min(p[2] for p in pts)
    canon = sorted({(p[0] - mx, p[1] - my, p[2] - mz) for p in pts})
    return canon


def _motif_graph_features(offsets: List[Tuple[int, int, int]]) -> Dict[str, Any]:
    canon = _motif_canonical_offsets(offsets)
    cells = set(canon)
    n = len(cells)
    if n == 0:
        return {"size": 0}
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    zs = [c[2] for c in cells]
    dx = max(xs) - min(xs) + 1
    dy = max(ys) - min(ys) + 1
    dz = max(zs) - min(zs) + 1
    bbox_vol = dx * dy * dz
    compact = float(n) / float(bbox_vol) if bbox_vol else 0.0
    internal_edges = 0
    dirs = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    for x, y, z in cells:
        for dx_, dy_, dz_ in dirs:
            if (x + dx_, y + dy_, z + dz_) in cells:
                internal_edges += 1
    internal_edges //= 2
    deg_hist: Counter = Counter()
    for x, y, z in cells:
        deg = 0
        for dx_, dy_, dz_ in dirs:
            if (x + dx_, y + dy_, z + dz_) in cells:
                deg += 1
        deg_hist[deg] += 1
    rank = "singleton"
    if n >= 2:
        if min(dx, dy, dz) == 1 and max(dx, dy, dz) == n:
            rank = "linear_chain_like"
        elif min(dx, dy, dz) == 1:
            rank = "planar_like"
        else:
            rank = "volumetric_like"
    branched = bool(any(d >= 3 for d in deg_hist))
    if branched and rank != "singleton":
        rank = rank + "_branched"
    return {
        "size": n,
        "bbox_dx": int(dx),
        "bbox_dy": int(dy),
        "bbox_dz": int(dz),
        "bbox_volume": int(bbox_vol),
        "compactness": compact,
        "internal_undirected_face_edges": int(internal_edges),
        "degree_histogram": {int(k): int(v) for k, v in sorted(deg_hist.items())},
        "shape_rank": rank,
    }


def _randomized_records(
    records: List[Dict[str, np.ndarray]], mode: str, seed: int
) -> List[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out: List[Dict[str, np.ndarray]] = []
    for rec in _progress(records, total=len(records), desc="[ANALYSIS] samples"):
        t = np.asarray(rec["tokens"], dtype=np.int64).copy()
        c = np.asarray(rec["coords"], dtype=np.int64).copy()
        if mode == "token_shuffle":
            order = rng.permutation(t.shape[0])
            t = t[order]
        elif mode == "coord_shuffle":
            order = rng.permutation(t.shape[0])
            c = c[order]
        out.append({"tokens": t, "coords": c})
    return out


def _random_pair_1d_metrics(
    records: List[Dict[str, np.ndarray]], tok_ref: BPE3DTokenizer, seed: int
) -> Dict[str, float]:
    tok0 = _truncate_tokenizer(tok_ref, 0)
    seqs_work: List[List[int]] = []
    leaf_total = 0
    for rec in records:
        t = np.asarray(rec["tokens"], dtype=np.int64)
        c = np.asarray(rec["coords"], dtype=np.int64)
        leaf_total += int(t.shape[0])
        mid, ma = tok0.encode_sparse_numpy(t, c)
        order = _sort_indices_packed(ma)
        seqs_work.append([int(x) for x in mid[order].tolist()])
    rng = np.random.default_rng(seed)
    merges, seqs_f = _train_1d_adjacent_bpe(
        seqs_work,
        len(tok_ref.merge_table),
        2,
        int(tok_ref.base_vocab_size),
        rng,
        True,
    )
    macro_total = sum(len(s) for s in seqs_f)
    return {
        "compression_ratio_total": float(_safe_ratio(macro_total, leaf_total)),
        "merges_1d": int(len(merges)),
    }


def _try_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _boxplot_with_labels(plt: Any, data: List[List[float]], labels: List[str]) -> None:
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)


def _interval_yerr_around_means(
    means: Sequence[float], lows: Sequence[float], highs: Sequence[float]
) -> np.ndarray:
    yerr_lo: List[float] = []
    yerr_hi: List[float] = []
    for m, lo, hi in zip(means, lows, highs):
        mf = float(m)
        lo_f = float(lo) if np.isfinite(lo) else mf
        hi_f = float(hi) if np.isfinite(hi) else mf
        interval_lo = min(lo_f, hi_f, mf)
        interval_hi = max(lo_f, hi_f, mf)
        yerr_lo.append(max(0.0, mf - interval_lo))
        yerr_hi.append(max(0.0, interval_hi - mf))
    return np.asarray([yerr_lo, yerr_hi], dtype=np.float64)


def _write_formal_lm_figures(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    plt = _try_plt()
    if plt is None or not rows:
        return
    ks = [int(r["K"]) for r in rows]
    clm = [float(r["C_LM_proxy_per_leaf_bits"]) for r in rows]
    h1pl = [float(r["H_mesh_unigram_bits"]) * float(r["compression_ratio_total"]) for r in rows]
    plt.figure(figsize=(8, 4))
    plt.plot(ks, clm, marker="o", label="C_LM proxy")
    plt.plot(ks, h1pl, marker="s", label="H_mesh(global)*R_len")
    plt.xlabel("K merges")
    plt.ylabel("bits / leaf (proxy)")
    plt.title("Formal LM cost vs K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figure_lm_cost_sweep.png", dpi=180)
    plt.close()
    r = rows[-1]
    parts = [
        float(r["compression_ratio_total"]) * float(r["H_mesh_unigram_bits"]),
        float(r["compression_ratio_total"]) * float(r["H_morton_unigram_bits"]),
        float(r["compression_ratio_total"]) * float(r["mean_H_anchor_delta_micro_bits"]),
        float(r["compression_ratio_total"]) * float(r["mean_bigram_cond_entropy_mesh_bits"]),
        float(r["control_proxy_per_leaf_bits"]),
    ]
    labels = ["mesh H1*R", "morton H1*R", "E[H(delta)]*R", "bigram*R", "control"]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, parts, color="#4C72B0")
    plt.ylabel("bits / leaf")
    plt.title("LM cost decomposition (last K in sweep)")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_lm_cost_decomposition.png", dpi=180)
    plt.close()
    r2m = [float(r["mean_repeat_mass_bigram"]) for r in rows]
    d2m = [float(r["mean_distinct_bigram_rate"]) for r in rows]
    plt.figure(figsize=(8, 4))
    plt.plot(ks, r2m, marker="o", label="repeat mass bi")
    plt.plot(ks, d2m, marker="s", label="distinct bi rate")
    plt.xlabel("K merges")
    plt.ylabel("mass / rate")
    plt.title("N-gram repeat mass & distinct-2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figure_ngram_repeat_mass.png", dpi=180)
    plt.close()


def _write_stability_figures(
    out_dir: Path,
    stability: Dict[str, Any],
    leaf_ns: List[int],
    d_comp: List[float],
    d_lm: List[float],
    d_motif_cov: List[float],
) -> None:
    plt = _try_plt()
    if plt is None:
        return
    pb = stability.get("paired_bootstrap", {})
    names: List[str] = []
    vals: List[float] = []
    errs: List[Tuple[float, float]] = []
    for key in pb:
        ci = pb[key]
        if not ci:
            continue
        names.append(key.replace("_", " ")[:28])
        vals.append(float(ci.get("mean_delta", 0.0)))
        lo = float(ci.get("ci_lo", 0.0))
        hi = float(ci.get("ci_hi", 0.0))
        yerr = _interval_yerr_around_means([vals[-1]], [lo], [hi])
        errs.append((float(yerr[0, 0]), float(yerr[1, 0])))
    if names:
        plt.figure(figsize=(9, 4))
        x = np.arange(len(names))
        yerr = np.asarray(errs, dtype=np.float64).T
        plt.errorbar(x, vals, yerr=yerr, fmt="o", capsize=4, color="#2ca02c")
        plt.axhline(0.0, color="gray", lw=1)
        plt.xticks(x, names, rotation=28, ha="right")
        plt.ylabel("paired mean delta")
        plt.title("Stability forest (paired bootstrap 95% CI)")
        plt.tight_layout()
        plt.savefig(out_dir / "figure_stability_forest.png", dpi=180)
        plt.close()
    if leaf_ns and d_comp:
        t1, t2 = np.percentile(leaf_ns, [33.3, 66.6])
        lows = [d_comp[i] for i, n in enumerate(leaf_ns) if n <= t1]
        mids = [d_comp[i] for i, n in enumerate(leaf_ns) if t1 < n <= t2]
        highs = [d_comp[i] for i, n in enumerate(leaf_ns) if n > t2]
        plt.figure(figsize=(7, 4))
        _boxplot_with_labels(plt, [lows, mids, highs], ["small leaf_n", "mid", "large"])
        plt.ylabel("delta compression (identity - full BPE)")
        plt.title("Stratified stability by leaf count")
        plt.tight_layout()
        plt.savefig(out_dir / "figure_stratified_stability.png", dpi=180)
        plt.close()
    if leaf_ns and d_lm:
        t1, t2 = np.percentile(leaf_ns, [33.3, 66.6])
        lows = [d_lm[i] for i, n in enumerate(leaf_ns) if n <= t1]
        mids = [d_lm[i] for i, n in enumerate(leaf_ns) if t1 < n <= t2]
        highs = [d_lm[i] for i, n in enumerate(leaf_ns) if n > t2]
        plt.figure(figsize=(7, 4))
        _boxplot_with_labels(plt, [lows, mids, highs], ["small", "mid", "large"])
        plt.ylabel("delta C_LM proxy (identity - full BPE)")
        plt.title("Stratified LM cost delta by leaf count")
        plt.tight_layout()
        plt.savefig(out_dir / "figure_stratified_stability_lm.png", dpi=180)
        plt.close()
    if leaf_ns and d_motif_cov:
        t1, t2 = np.percentile(leaf_ns, [33.3, 66.6])
        lows = [d_motif_cov[i] for i, n in enumerate(leaf_ns) if n <= t1]
        mids = [d_motif_cov[i] for i, n in enumerate(leaf_ns) if t1 < n <= t2]
        highs = [d_motif_cov[i] for i, n in enumerate(leaf_ns) if n > t2]
        plt.figure(figsize=(7, 4))
        _boxplot_with_labels(plt, [lows, mids, highs], ["small", "mid", "large"])
        plt.ylabel("delta motif coverage (mean footprint)")
        plt.title("Stratified motif coverage delta")
        plt.tight_layout()
        plt.savefig(out_dir / "figure_stratified_stability_motif.png", dpi=180)
        plt.close()


def _write_robustness_ci_figure(out_dir: Path, robust_block: Dict[str, Any]) -> None:
    plt = _try_plt()
    if plt is None or not robust_block:
        return
    names = list(robust_block.keys())
    means = [float(robust_block[k].get("mean", 0.0)) for k in names]
    lo = [float(robust_block[k].get("ci_lo", m)) for k, m in zip(names, means)]
    hi = [float(robust_block[k].get("ci_hi", m)) for k, m in zip(names, means)]
    yerr = _interval_yerr_around_means(means, lo, hi)
    plt.figure(figsize=(9, 4))
    x = np.arange(len(names))
    plt.errorbar(x, means, yerr=yerr, fmt="o", capsize=4, color="#d62728")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("compression_ratio_total")
    plt.title("Robustness multi-seed mean +/- 95% CI (orientation ablation = axis_perm)")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_robustness_ci.png", dpi=180)
    plt.close()


def _write_motif_figures(
    out_dir: Path,
    motif_rows: List[Dict[str, Any]],
    shape_hist: Counter,
    payload: Dict[str, Any],
    controls: Dict[str, Any],
    corpus_size_hist: Counter,
) -> None:
    plt = _try_plt()
    if plt is None:
        return
    if shape_hist:
        plt.figure(figsize=(7, 4))
        keys = list(shape_hist.keys())
        plt.bar(keys, [float(shape_hist[k]) for k in keys], color="#17becf")
        plt.title("Motif topology / shape_rank histogram")
        plt.xticks(rotation=22, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "figure_motif_topology.png", dpi=180)
        plt.close()
    if motif_rows:
        sizes = [int(r["size"]) for r in motif_rows if int(r.get("size", 0)) > 0]
        if sizes:
            plt.figure(figsize=(8, 4))
            plt.hist(sizes, bins=min(40, max(5, len(set(sizes)))), color="#4C72B0", edgecolor="black")
            plt.xlabel("motif size (cells) from merge_table")
            plt.ylabel("count")
            plt.title("Motif size distribution (learned rules)")
            plt.tight_layout()
            plt.savefig(out_dir / "figure_motif_size_coverage.png", dpi=180)
            plt.close()
    if corpus_size_hist:
        xs = sorted(corpus_size_hist.keys())
        ys = [float(corpus_size_hist[k]) for k in xs]
        plt.figure(figsize=(8, 4))
        plt.bar([str(k) for k in xs], ys, color="#ff7f0e")
        plt.xlabel("macro footprint size (corpus usage)")
        plt.ylabel("macro instance count")
        plt.title("Corpus macro usage by footprint size")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "figure_motif_corpus_size_usage.png", dpi=180)
        plt.close()
    ratio = float(payload.get("internalized_edge_ratio", 0.0))
    plt.figure(figsize=(5, 4))
    plt.bar(["internalized", "remaining"], [ratio, max(0.0, 1.0 - ratio)], color=["#2ca02c", "#98df8a"])
    plt.ylabel("fraction of undirected leaf face edges")
    plt.title("Internalized vs remaining boundary (leaf graph)")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_internalized_edges.png", dpi=180)
    plt.close()
    if controls:
        names = list(controls.keys())
        comp = [float(controls[k].get("mean_compression_ratio", 0.0)) for k in names]
        dlm = [
            float(controls[k].get("mean_delta_lm_proxy_per_leaf_identity_minus_bpe", 0.0))
            for k in names
        ]
        plt.figure(figsize=(9, 4))
        x = np.arange(len(names))
        w = 0.35
        plt.bar(x - w / 2, comp, width=w, label="compression ratio", color="#1f77b4")
        plt.bar(x + w / 2, dlm, width=w, label="mean Δ LM proxy (id−BPE)", color="#ff7f0e")
        plt.xticks(x, names, rotation=25, ha="right")
        plt.legend()
        plt.title("Motif vs random controls")
        plt.tight_layout()
        plt.savefig(out_dir / "figure_motif_vs_random_controls.png", dpi=180)
        plt.close()


def _write_motif_gallery_figure(out_dir: Path, gallery: List[Dict[str, Any]]) -> None:
    plt = _try_plt()
    if plt is None or not gallery:
        return
    n = min(9, len(gallery))
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes_flat = axes.flatten()
    for i in range(9):
        ax = axes_flat[i]
        if i >= n:
            ax.axis("off")
            continue
        g = gallery[i]
        offs = g.get("offsets_xy", [])
        if not offs:
            ax.axis("off")
            continue
        xs = [p[0] for p in offs]
        ys = [p[1] for p in offs]
        ax.scatter(xs, ys, s=120, c="#9467bd", edgecolors="k")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"id={g.get('macro_id')} u={g.get('usage_count')}", fontsize=8)
        ax.invert_yaxis()
    plt.suptitle("Motif gallery (xy projection of canonical offsets)")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_motif_gallery.png", dpi=180)
    plt.close()


def run_formal_lm_cost(
    args: argparse.Namespace,
    *,
    tok_full: Optional[BPE3DTokenizer] = None,
    records: Optional[List[Dict[str, np.ndarray]]] = None,
) -> None:
    corpus_prefix = Path(args.corpus_prefix)
    merge_table = Path(args.merge_table)
    out_dir = Path(args.analysis_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tok_full is None:
        if not merge_table.is_file():
            raise FileNotFoundError(f"merge table not found: {merge_table}")
        tok_full = BPE3DTokenizer.load(str(merge_table))
    if records is None:
        records = _load_corpus_records(corpus_prefix, args.max_samples, tokenizer=tok_full, verbose=True)
    sweep_k_str = str(getattr(args, "formal_lm_sweep_K", "") or "").strip()
    if sweep_k_str:
        ks = sorted({int(x.strip()) for x in sweep_k_str.split(",") if x.strip()})
    else:
        ks = [len(tok_full.merge_table)]
    folds = max(2, int(getattr(args, "folds", 5)))
    seed = int(getattr(args, "bootstrap_seed", 0))
    rows_out: List[Dict[str, Any]] = []
    nw = _analysis_workers_requested(args)
    merge_s = str(merge_table)
    chunks = _records_chunk_list(records, nw)
    use_mp = len(chunks) > 1
    mp_ex: Optional[ProcessPoolExecutor] = None
    if use_mp:
        print(
            f"[PARALLEL] formal_lm_cost: workers={nw}, chunks={len(chunks)}, samples={len(records)}",
            flush=True,
        )
        mp_ex = ProcessPoolExecutor(
            max_workers=nw,
            initializer=_merge_table_worker_init,
            initargs=(merge_s,),
        )
    try:
        for K in _progress(ks, total=len(ks), desc="[FORMAL_LM] K sweep"):
            if mp_ex is not None:
                tasks = [(K, ch) for ch in chunks]
                parts = list(
                    _progress(
                        mp_ex.map(_mp_mesh_sequences_task, tasks),
                        total=len(tasks),
                        desc=f"[FORMAL_LM] K={K} chunks",
                    )
                )
                mesh_seqs, mort_seqs, leaf_counts = _concat_mesh_sequence_parts(parts)
                print(f"[FORMAL_LM] K={K} mesh sequences aggregated", flush=True)
            else:
                tok_k = _truncate_tokenizer(tok_full, K)
                mesh_seqs, mort_seqs, leaf_counts = _corpus_mesh_sequences_for_tok(tok_k, records)
            glob_mesh: Counter = Counter()
            glob_mort: Counter = Counter()
            glob_delta: Counter = Counter()
            bigram_rates: List[float] = []
            tri_rates: List[float] = []
            r2: List[float] = []
            r3: List[float] = []
            r4: List[float] = []
            d2: List[float] = []
            mean_hd_micro: List[float] = []
            zip_lm = zip(mesh_seqs, mort_seqs)
            zip_lm = _progress(zip_lm, total=len(mesh_seqs), desc=f"[FORMAL_LM] K={K} per-sample LM stats")
            for s, ms in zip_lm:
                glob_mesh.update(s)
                glob_mort.update(ms)
                if len(ms) >= 2:
                    for i in range(1, len(ms)):
                        glob_delta[int(ms[i] - ms[i - 1])] += 1
                bigram_rates.append(_bigram_conditional_entropy_bits(s))
                tri_rates.append(_entropy_rate_ngram_bits(s, 3))
                r2.append(_repeat_mass_n(s, 2))
                r3.append(_repeat_mass_n(s, 3))
                r4.append(_repeat_mass_n(s, 4))
                d2.append(_distinct_n_rate(s, 2))
                mean_hd_micro.append(_anchor_delta_entropy_bits(ms))
            total_leaf = float(sum(leaf_counts))
            total_macro_toks = float(sum(len(s) for s in mesh_seqs))
            rlen_g = _safe_ratio(total_macro_toks, total_leaf)
            h_mesh = _entropy_bits(glob_mesh)
            h_mort = _entropy_bits(glob_mort)
            h_delta_global = _entropy_bits(glob_delta)
            cv_lens = float(np.std(leaf_counts) / max(np.mean(leaf_counts), 1e-9))
            eos_proxy = float(np.mean(leaf_counts)) + 2.0 * rlen_g
            ctrl_per_leaf = _control_tokens_per_seq_overhead() * rlen_g / max(
                np.mean(leaf_counts) if leaf_counts else 1.0, 1.0
            )
            c_lm_proxy = (
                rlen_g * h_mesh
                + rlen_g * float(np.mean(mean_hd_micro) if mean_hd_micro else 0.0)
                + rlen_g * float(np.mean(bigram_rates) if bigram_rates else 0.0)
                + ctrl_per_leaf
            )
            kfold = _kfold_unigram_nll_mean(mesh_seqs, folds, seed)
            held = _heldout_nll_unigram(mesh_seqs, 0.8, seed + 7)
            per_leaf_nll = _per_leaf_weighted_nll_plugin(mesh_seqs, leaf_counts, glob_mesh)
            rows_out.append(
                {
                    "K": int(K),
                    "compression_ratio_total": float(rlen_g),
                    "H_mesh_unigram_bits": float(h_mesh),
                    "H_morton_unigram_bits": float(h_mort),
                    "H_anchor_delta_global_bits": float(h_delta_global),
                    "mean_H_anchor_delta_micro_bits": float(np.mean(mean_hd_micro) if mean_hd_micro else 0.0),
                    "mean_bigram_cond_entropy_mesh_bits": float(np.mean(bigram_rates) if bigram_rates else 0.0),
                    "mean_trigram_entropy_rate_mesh_bits": float(np.mean(tri_rates) if tri_rates else 0.0),
                    "mean_repeat_mass_bigram": float(np.mean(r2) if r2 else 0.0),
                    "mean_repeat_mass_trigram": float(np.mean(r3) if r3 else 0.0),
                    "mean_repeat_mass_4gram": float(np.mean(r4) if r4 else 0.0),
                    "mean_distinct_bigram_rate": float(np.mean(d2) if d2 else 0.0),
                    "length_cv_leaf": float(cv_lens),
                    "eos_length_proxy_mean_leaf_plus_controls": float(eos_proxy),
                    "control_proxy_per_leaf_bits": float(ctrl_per_leaf),
                    "C_LM_proxy_per_leaf_bits": float(c_lm_proxy),
                    "per_leaf_weighted_mesh_nll_bits_plugin": float(per_leaf_nll),
                    "kfold_mesh_unigram_nll": kfold,
                    "heldout_train_80_val_20_mesh_unigram_nll": held,
                }
            )
    finally:
        if mp_ex is not None:
            mp_ex.shutdown(wait=True)
    payload = {
        "corpus_prefix": str(corpus_prefix),
        "merge_table": str(merge_table),
        "samples": len(records),
        "folds": folds,
        "parallel_mesh_encode_workers": int(nw) if use_mp else 1,
        "serialization_note": "mesh ids + anchors sorted by packed/Morton key; LM proxy uses H(mesh)+E_micro[H(delta)]+bigram_cond + control",
        "by_K": rows_out,
    }
    prefix = Path(getattr(args, "formal_out_prefix", "") or out_dir)
    prefix.mkdir(parents=True, exist_ok=True)
    out_json = prefix / "formal_lm_cost.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FORMAL_LM] wrote {out_json}", flush=True)
    if args.plot:
        _write_formal_lm_figures(prefix, rows_out)


def run_stability_panel(
    args: argparse.Namespace,
    *,
    tok_full: Optional[BPE3DTokenizer] = None,
    records: Optional[List[Dict[str, np.ndarray]]] = None,
) -> None:
    corpus_prefix = Path(args.corpus_prefix)
    merge_table = Path(args.merge_table)
    out_dir = Path(args.analysis_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tok_full is None:
        if not merge_table.is_file():
            raise FileNotFoundError(f"merge table not found: {merge_table}")
        tok_full = BPE3DTokenizer.load(str(merge_table))
    if records is None:
        records = _load_corpus_records(corpus_prefix, args.max_samples, tokenizer=tok_full, verbose=True)
    nw = _analysis_workers_requested(args)
    mp_path = str(merge_table)
    stab_chunks = _records_chunk_list(list(records), nw)
    use_mp_stab = nw > 1 and len(records) >= 200 and len(stab_chunks) > 1
    K_full = len(tok_full.merge_table)
    tok_bpe = _truncate_tokenizer(tok_full, K_full)
    tok_id = _truncate_tokenizer(tok_full, 0)
    n_iter = int(getattr(args, "bootstrap_n", 1000) or 1000)
    if n_iter <= 0:
        n_iter = 1000
    seed = int(getattr(args, "bootstrap_seed", 0))
    d_comp: List[float] = []
    d_lm: List[float] = []
    d_edge: List[float] = []
    d_motif_cov: List[float] = []
    leaf_ns: List[int] = []
    bbox_vs: List[int] = []
    occs: List[float] = []
    uniq_types: List[int] = []
    clm_bpe_all: List[float] = []
    if use_mp_stab:
        with ProcessPoolExecutor(
            max_workers=nw,
            initializer=_merge_table_worker_init,
            initargs=(mp_path,),
        ) as ex:
            parts = list(
                _progress(
                    ex.map(_mp_stability_chunk, stab_chunks),
                    total=len(stab_chunks),
                    desc="[STABILITY] chunks",
                )
            )
        for p in parts:
            d_comp.extend(p["d_comp"])
            d_lm.extend(p["d_lm"])
            d_edge.extend(p["d_edge"])
            d_motif_cov.extend(p["d_motif_cov"])
            leaf_ns.extend(p["leaf_ns"])
            bbox_vs.extend(p["bbox_vs"])
            occs.extend(p["occs"])
            uniq_types.extend(p["uniq_types"])
            clm_bpe_all.extend(p["clm_bpe"])
    else:
        for rec in _progress(records, total=len(records), desc="[STABILITY] samples"):
            t = np.asarray(rec["tokens"], dtype=np.int64)
            c = np.asarray(rec["coords"], dtype=np.int64)
            m_b, a_b = tok_bpe.encode_sparse_numpy(t, c)
            m_l, a_l = tok_id.encode_sparse_numpy(t, c)
            lp = _adjacency_pair_counts(t, c)
            le = float(sum(lp.values()))
            mb_b = _macro_boundary_pair_counts(tok_bpe, m_b, a_b)
            mb_l = _macro_boundary_pair_counts(tok_id, m_l, a_l)
            r_b = _safe_ratio(int(m_b.shape[0]), int(t.shape[0]))
            r_l = _safe_ratio(int(m_l.shape[0]), int(t.shape[0]))
            edge_ratio_b = _safe_ratio(float(sum(mb_b.values())), le)
            edge_ratio_l = _safe_ratio(float(sum(mb_l.values())), le)
            d_comp.append(r_l - r_b)
            d_edge.append(edge_ratio_l - edge_ratio_b)
            sq_b, mort_b = _ordered_mesh_morton(m_b, a_b)
            sq_l, mort_l = _ordered_mesh_morton(m_l, a_l)
            pm_b = _per_sample_lm_proxy_cost(
                [int(x) for x in sq_b.tolist()], [int(x) for x in mort_b.tolist()], int(t.shape[0])
            )
            pm_l = _per_sample_lm_proxy_cost(
                [int(x) for x in sq_l.tolist()], [int(x) for x in mort_l.tolist()], int(t.shape[0])
            )
            d_lm.append(float(pm_l["clm_bits_per_leaf"] - pm_b["clm_bits_per_leaf"]))
            clm_bpe_all.append(float(pm_b["clm_bits_per_leaf"]))
            cov_b = (
                float(np.mean([len(_macro_footprint_offsets(tok_bpe, int(x))) for x in m_b.tolist()]))
                if m_b.shape[0]
                else 0.0
            )
            cov_l = (
                float(np.mean([len(_macro_footprint_offsets(tok_id, int(x))) for x in m_l.tolist()]))
                if m_l.shape[0]
                else 0.0
            )
            d_motif_cov.append(cov_l - cov_b)
            nl = int(t.shape[0])
            leaf_ns.append(nl)
            if c.size:
                cmin = c.min(axis=0)
                cmax = c.max(axis=0)
                bv = int(np.prod(np.maximum(cmax - cmin + 1, 1)))
                bbox_vs.append(bv)
                occs.append(float(t.shape[0]) / float(max(bv, 1)))
            else:
                bbox_vs.append(0)
                occs.append(0.0)
            uniq_types.append(int(len(set(t.tolist()))))
    stability: Dict[str, Any] = {
        "paired_bootstrap": {
            "delta_compression_ratio_identity_minus_full_bpe": _paired_bootstrap_mean_ci(d_comp, n_iter, seed),
            "delta_lm_proxy_per_leaf_identity_minus_full_bpe": _paired_bootstrap_mean_ci(d_lm, n_iter, seed),
            "delta_boundary_edge_ratio_identity_minus_full_bpe": _paired_bootstrap_mean_ci(
                d_edge, n_iter, seed + 3
            ),
            "delta_mean_macro_footprint_size_identity_minus_full_bpe": _paired_bootstrap_mean_ci(
                d_motif_cov, n_iter, seed + 5
            ),
        },
        "note_axis_perm_orientation_ablation": (
            "axis permutation is an orientation ablation for the fixed merge table, not evidence of rotation invariance."
        ),
        "parallel_workers_requested": int(nw),
        "parallel_stability_chunking": bool(use_mp_stab),
    }

    def _stratify(label: str, keys: List[float], deltas: List[float]) -> None:
        if not keys:
            return
        t1, t2 = np.percentile(keys, [33.3, 66.6])
        buckets = {"low": [], "mid": [], "high": []}
        for k, d in zip(keys, deltas):
            if k <= t1:
                buckets["low"].append(d)
            elif k <= t2:
                buckets["mid"].append(d)
            else:
                buckets["high"].append(d)
        stability[f"stratified_delta_compression_{label}"] = {
            bk: {"mean": float(np.mean(v)), "n": len(v)} for bk, v in buckets.items() if v
        }

    _stratify("leaf_tokens", [float(x) for x in leaf_ns], d_comp)
    _stratify("bbox_volume", [float(x) for x in bbox_vs], d_comp)
    _stratify("occupancy", occs, d_comp)
    _stratify("leaf_unique_types", [float(x) for x in uniq_types], d_comp)

    cs_str = str(getattr(args, "control_seeds", "0") or "0").strip()
    ctrl_seeds = [int(x.strip()) for x in cs_str.split(",") if x.strip()]
    if not ctrl_seeds:
        ctrl_seeds = [0]

    def _rob_ci(name: str, vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {}
        a = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "ci_lo": float(np.percentile(a, 2.5)),
            "ci_hi": float(np.percentile(a, 97.5)),
            "seeds": len(vals),
        }

    robust_block: Dict[str, Any] = {}
    max_samples = args.max_samples
    for sd in _progress(ctrl_seeds, total=len(ctrl_seeds), desc="[STABILITY] translate seeds"):
        r_tr, _ = _compression_ratio_under_transform(
            tok_bpe,
            [],
            max_samples,
            records=records,
            coord_shift=(
                int(getattr(args, "robust_dx", 100)),
                int(getattr(args, "robust_dy", 100)),
                int(getattr(args, "robust_dz", 100)),
            ),
            mp_merge_path=mp_path,
            mp_workers=nw,
        )
        robust_block.setdefault("translate", []).append(float(r_tr))
    for sd in _progress(ctrl_seeds, total=len(ctrl_seeds), desc="[STABILITY] axis seeds"):
        r_ax, _ = _compression_ratio_under_transform(
            tok_bpe,
            [],
            max_samples,
            records=records,
            axis_perm=(1, 2, 0),
            mp_merge_path=mp_path,
            mp_workers=nw,
        )
        robust_block.setdefault("axis_perm_orientation_ablation_yzx", []).append(float(r_ax))
    for sd in _progress(ctrl_seeds, total=len(ctrl_seeds), desc="[STABILITY] noise seeds"):
        r_n, _ = _compression_ratio_under_transform(
            tok_bpe,
            [],
            max_samples,
            records=records,
            noise_p=float(getattr(args, "robust_noise_p", 0.05)),
            seed=int(sd),
            mp_merge_path=mp_path,
            mp_workers=nw,
        )
        robust_block.setdefault("noise", []).append(float(r_n))
    for sd in _progress(ctrl_seeds, total=len(ctrl_seeds), desc="[STABILITY] sparsity seeds"):
        r_sp, _ = _compression_ratio_under_transform(
            tok_bpe,
            [],
            max_samples,
            records=records,
            drop_p=float(getattr(args, "robust_drop_p", 0.1)),
            seed=int(sd) + 1000,
            mp_merge_path=mp_path,
            mp_workers=nw,
        )
        robust_block.setdefault("sparsity_drop", []).append(float(r_sp))

    robust_summary: Dict[str, Any] = {}
    for k, vals in robust_block.items():
        robust_summary[k] = _rob_ci(k, vals)
    stability["robustness_compression_ratio_multi_seed"] = robust_summary

    n = len(records)
    clm_train: List[float] = []
    clm_val: List[float] = []
    if n >= 4 and len(clm_bpe_all) == n:
        thr = int(0.8 * n)
        clm_train = [float(x) for x in clm_bpe_all[:thr]]
        clm_val = [float(x) for x in clm_bpe_all[thr:]]
        stability["held_out_lm_proxy_split"] = {
            "first_80pct_mean_clm_bits_per_leaf": float(np.mean(clm_train) if clm_train else 0.0),
            "last_20pct_mean_clm_bits_per_leaf": float(np.mean(clm_val) if clm_val else 0.0),
            "note": "descriptive split on sample index order (corpus shard order); not retuning BPE.",
        }

    prefix = Path(getattr(args, "formal_out_prefix", "") or out_dir)
    prefix.mkdir(parents=True, exist_ok=True)
    pth = prefix / "stability_metrics.json"
    pth.write_text(json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[STABILITY] wrote {pth}", flush=True)
    if args.plot:
        _write_stability_figures(prefix, stability, leaf_ns, d_comp, d_lm, d_motif_cov)
        _write_robustness_ci_figure(prefix, robust_summary)


def run_motif_analysis(
    args: argparse.Namespace,
    *,
    tok_full: Optional[BPE3DTokenizer] = None,
    records: Optional[List[Dict[str, np.ndarray]]] = None,
) -> None:
    corpus_prefix = Path(args.corpus_prefix)
    merge_table = Path(args.merge_table)
    out_dir = Path(args.analysis_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tok_full is None:
        if not merge_table.is_file():
            raise FileNotFoundError(f"merge table not found: {merge_table}")
        tok_full = BPE3DTokenizer.load(str(merge_table))
    if records is None:
        records = _load_corpus_records(corpus_prefix, args.max_samples, tokenizer=tok_full, verbose=True)
    nw_motif = _analysis_workers_requested(args)
    mp_path_motif = str(merge_table)
    motif_chunks = _records_chunk_list(list(records), nw_motif)
    use_mp_motif = nw_motif > 1 and len(records) >= 200 and len(motif_chunks) > 1
    tok = _truncate_tokenizer(tok_full, len(tok_full.merge_table))
    motif_rows: List[Dict[str, Any]] = []
    shape_hist: Counter = Counter()
    mt_enum = list(enumerate(tok.merge_table))
    for i, e in _progress(mt_enum, total=len(mt_enum), desc="[MOTIF] merge_table features"):
        canon = _motif_canonical_offsets([(int(x), int(y), int(z)) for x, y, z in e.leaf_offsets])
        feats = _motif_graph_features([(int(x), int(y), int(z)) for x, y, z in e.leaf_offsets])
        feats["macro_id"] = int(e.new_id)
        feats["merge_index"] = int(i)
        feats["canonical_offsets_json"] = json.dumps(canon, ensure_ascii=False)
        feats["usage_count"] = 0
        feats["leaf_coverage_tokens"] = 0
        motif_rows.append(feats)
        shape_hist[feats.get("shape_rank", "?")] += 1
    id_to_row = {int(r["macro_id"]): r for r in motif_rows}
    usage: Counter = Counter()
    corpus_size_hist: Counter = Counter()
    internal_absorbed = 0
    total_leaf_edges_dir = 0
    if use_mp_motif:
        internal_dir_sum = 0
        macro_usage_acc: Counter = Counter()
        macro_leaf_cov_acc: Counter = Counter()
        with ProcessPoolExecutor(
            max_workers=nw_motif,
            initializer=_merge_table_worker_init,
            initargs=(mp_path_motif,),
        ) as ex:
            parts = list(
                _progress(
                    ex.map(_mp_motif_corpus_chunk, motif_chunks),
                    total=len(motif_chunks),
                    desc="[MOTIF] corpus chunks",
                )
            )
        for p in parts:
            usage.update(Counter({int(k): int(v) for k, v in p["usage"].items()}))
            corpus_size_hist.update(Counter({int(k): int(v) for k, v in p["corpus_size_hist"].items()}))
            internal_dir_sum += int(p["internal_absorbed_dir"])
            total_leaf_edges_dir += int(p["total_leaf_edges_dir"])
            macro_usage_acc.update({int(k): int(v) for k, v in p["macro_usage"].items()})
            macro_leaf_cov_acc.update({int(k): int(v) for k, v in p["macro_leaf_cov"].items()})
        for tid_i, cnt in macro_usage_acc.items():
            if tid_i in id_to_row:
                id_to_row[tid_i]["usage_count"] = int(id_to_row[tid_i].get("usage_count", 0)) + int(cnt)
        for tid_i, cov in macro_leaf_cov_acc.items():
            if tid_i in id_to_row:
                id_to_row[tid_i]["leaf_coverage_tokens"] = int(id_to_row[tid_i].get("leaf_coverage_tokens", 0)) + int(
                    cov
                )
        internal_absorbed = internal_dir_sum // 2
    else:
        for rec in _progress(records, total=len(records), desc="[MOTIF] corpus samples"):
            t = np.asarray(rec["tokens"], dtype=np.int64)
            c = np.asarray(rec["coords"], dtype=np.int64)
            mid, ma = tok.encode_sparse_numpy(t, c)
            cell_to_macro: Dict[Tuple[int, int, int], int] = {}
            for mi, (tid, anc) in enumerate(zip(mid.tolist(), ma.tolist())):
                ax, ay, az = int(anc[0]), int(anc[1]), int(anc[2])
                for ox, oy, oz in _macro_footprint_offsets(tok, int(tid)):
                    cell_to_macro[(ax + ox, ay + oy, az + oz)] = mi
            lp = _adjacency_pair_counts(t, c)
            total_leaf_edges_dir += int(sum(lp.values()))
            for _, tid in enumerate(mid.tolist()):
                tid_i = int(tid)
                usage[tid_i] += 1
                sz = len(_macro_footprint_offsets(tok, tid_i))
                corpus_size_hist[sz] += 1
                if tid_i in id_to_row:
                    id_to_row[tid_i]["usage_count"] = int(id_to_row[tid_i].get("usage_count", 0)) + 1
                    id_to_row[tid_i]["leaf_coverage_tokens"] = int(id_to_row[tid_i].get("leaf_coverage_tokens", 0)) + sz
            for (x, y, z), mi in cell_to_macro.items():
                for dv in _ANALYSIS_DIRECTIONS:
                    nj = cell_to_macro.get((x + dv[0], y + dv[1], z + dv[2]))
                    if nj is None:
                        continue
                    if nj == mi:
                        internal_absorbed += 1
        internal_absorbed //= 2
    total_undirected = total_leaf_edges_dir // 2
    boundary_remain = max(0, total_undirected - internal_absorbed)
    controls_payload: Dict[str, Any] = {}
    ctrl_modes = str(getattr(args, "motif_controls", "") or "").strip()
    seeds = str(getattr(args, "control_seeds", "0") or "0").strip()
    ctrl_seed_list = [int(x.strip()) for x in seeds.split(",") if x.strip()]
    if not ctrl_seed_list:
        ctrl_seed_list = [0]
    modes_list = [x.strip() for x in ctrl_modes.split(",") if x.strip()] if ctrl_modes else []
    for mode in _progress(modes_list, total=len(modes_list), desc="[MOTIF] controls"):
        comp_vals: List[float] = []
        cov_vals: List[float] = []
        cdelta_vals: List[float] = []
        for sd in _progress(ctrl_seed_list, total=len(ctrl_seed_list), desc=f"[MOTIF] {mode} seeds"):
            if mode == "random_pair_merge":
                m = _random_pair_1d_metrics(records, tok_full, sd)
                comp_vals.append(float(m["compression_ratio_total"]))
                cov_vals.append(1.0)
                cdelta_vals.append(0.0)
                continue
            r_ctrl = _randomized_records(records, mode, sd)
            ctrl_chunks = _records_chunk_list(list(r_ctrl), nw_motif)
            use_mp_ctrl = nw_motif > 1 and len(r_ctrl) >= 200 and len(ctrl_chunks) > 1
            ratios = []
            cov: List[float] = []
            clm_delta_list: List[float] = []
            if use_mp_ctrl:
                with ProcessPoolExecutor(
                    max_workers=nw_motif,
                    initializer=_merge_table_worker_init,
                    initargs=(mp_path_motif,),
                ) as ex:
                    ctrl_tasks = [(ch, mode) for ch in ctrl_chunks]
                    shard_results = list(
                        _progress(
                            ex.map(_mp_control_shard, ctrl_tasks),
                            total=len(ctrl_tasks),
                            desc=f"[MOTIF] {mode} chunks",
                        )
                    )
                for rs, cv, cd in shard_results:
                    ratios.extend(rs)
                    cov.extend(cv)
                    clm_delta_list.extend(cd)
            else:
                tok0 = _truncate_tokenizer(tok, 0)
                for rec in _progress(r_ctrl, total=len(r_ctrl), desc=f"[MOTIF] {mode} samples"):
                    tt = np.asarray(rec["tokens"], dtype=np.int64)
                    cc = np.asarray(rec["coords"], dtype=np.int64)
                    mm, aa = tok.encode_sparse_numpy(tt, cc)
                    mm0, aa0 = tok0.encode_sparse_numpy(tt, cc)
                    ratios.append(_safe_ratio(int(mm.shape[0]), int(tt.shape[0])))
                    cl = 0
                    for tid in mm.tolist():
                        cl += len(_macro_footprint_offsets(tok, int(tid)))
                    cov.append(_safe_ratio(cl, int(tt.shape[0])))
                    sq_b, mort_b = _ordered_mesh_morton(mm, aa)
                    sq_l, mort_l = _ordered_mesh_morton(mm0, aa0)
                    pm_b = _per_sample_lm_proxy_cost(
                        [int(x) for x in sq_b.tolist()], [int(x) for x in mort_b.tolist()], int(tt.shape[0])
                    )
                    pm_l = _per_sample_lm_proxy_cost(
                        [int(x) for x in sq_l.tolist()], [int(x) for x in mort_l.tolist()], int(tt.shape[0])
                    )
                    clm_delta_list.append(float(pm_l["clm_bits_per_leaf"] - pm_b["clm_bits_per_leaf"]))
            comp_vals.append(float(np.mean(ratios) if ratios else 0.0))
            cov_vals.append(float(np.mean(cov) if cov else 0.0))
            cdelta_vals.append(float(np.mean(clm_delta_list) if clm_delta_list else 0.0))
        controls_payload[mode] = {
            "mean_compression_ratio": float(np.mean(comp_vals) if comp_vals else 0.0),
            "std_compression_ratio": float(np.std(comp_vals) if comp_vals else 0.0),
            "mean_leaf_coverage_per_macro_instance": float(np.mean(cov_vals) if cov_vals else 0.0),
            "mean_delta_lm_proxy_per_leaf_identity_minus_bpe": float(np.mean(cdelta_vals) if cdelta_vals else 0.0),
            "seeds_used": list(ctrl_seed_list),
            **(
                {"note_lm_delta_omitted": "random_pair_merge uses 1D adjacent BPE — LM proxy not aligned with 3D serialization"}
                if mode == "random_pair_merge"
                else {}
            ),
        }
    gallery_sorted: List[Dict[str, Any]] = []
    for r in sorted(
        motif_rows,
        key=lambda row: (
            -int(row.get("usage_count", 0)),
            -int(row.get("size", 0)),
            -float(row.get("compactness", 0.0)),
        ),
    )[:9]:
        try:
            canon = json.loads(str(r.get("canonical_offsets_json", "[]")))
        except json.JSONDecodeError:
            canon = []
        gallery_sorted.append(
            {
                "macro_id": int(r.get("macro_id", -1)),
                "size": int(r.get("size", 0)),
                "usage_count": int(r.get("usage_count", 0)),
                "offsets_xy": [(int(p[0]), int(p[1])) for p in canon],
            }
        )
    payload = {
        "macro_footprint_vocab_stats": _macro_footprint_vocab_from_table(tok),
        "motif_shape_histogram": dict(shape_hist),
        "corpus_macro_usage_by_footprint_size": dict(corpus_size_hist),
        "internalized_undirected_leaf_face_edges": int(internal_absorbed),
        "total_undirected_leaf_face_edges": int(total_undirected),
        "boundary_undirected_leaf_face_edges_remaining": int(boundary_remain),
        "internalized_edge_ratio": _safe_ratio(float(internal_absorbed), float(max(total_undirected, 1))),
        "internalized_mi_proxy_same_as_ratio_note": "ratio internal/total on undirected leaf face adjacency",
        "macro_occurrence_total": int(sum(usage.values())),
        "controls": controls_payload,
        "parallel_motif_workers": int(nw_motif),
        "parallel_motif_corpus_chunking": bool(use_mp_motif),
    }
    prefix = Path(getattr(args, "formal_out_prefix", "") or out_dir)
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "motif_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if motif_rows:
        with (prefix / "motif_instances.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(motif_rows[0].keys()))
            w.writeheader()
            w.writerows(motif_rows)
    print(f"[MOTIF] wrote {prefix / 'motif_metrics.json'}", flush=True)
    if args.plot:
        _write_motif_figures(prefix, motif_rows, shape_hist, payload, controls_payload, corpus_size_hist)
        _write_motif_gallery_figure(prefix, gallery_sorted)


def _pair_counts_adjacent_1d(seqs: List[List[int]]) -> Counter:
    c: Counter = Counter()
    for s in seqs:
        for i in range(len(s) - 1):
            c[(int(s[i]), int(s[i + 1]))] += 1
    return c


def _apply_merge_step_1d(seqs: List[List[int]], a: int, b: int, new_id: int) -> None:
    for si, s in enumerate(seqs):
        if len(s) < 2:
            continue
        out: List[int] = []
        i = 0
        while i < len(s):
            if i + 1 < len(s) and int(s[i]) == a and int(s[i + 1]) == b:
                out.append(int(new_id))
                i += 2
            else:
                out.append(int(s[i]))
                i += 1
        seqs[si] = out


def _order_indices_for_baseline(tokens: np.ndarray, coords: np.ndarray, order: str) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.int64)
    if order == "morton":
        return _sort_indices_packed(coords)
    if order == "hilbert":
        return _sort_indices_hilbert(coords)
    if order == "random":
        rng = np.random.default_rng(hash(tuple(tokens.tolist()[:8])) % (2**31))
        return rng.permutation(tokens.shape[0])
    return np.arange(tokens.shape[0])


def _percentile_or_zero(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize_values(values: Sequence[float], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": _mean_or_zero(values),
        f"{prefix}_p50": _percentile_or_zero(values, 50),
        f"{prefix}_p90": _percentile_or_zero(values, 90),
        f"{prefix}_p99": _percentile_or_zero(values, 99),
    }


def _face_edge_index_pairs(coords: np.ndarray) -> List[Tuple[int, int]]:
    coords = np.asarray(coords, dtype=np.int64)
    cell_to_idx: Dict[Tuple[int, int, int], int] = {}
    for i, xyz in enumerate(coords.tolist()):
        cell_to_idx[(int(xyz[0]), int(xyz[1]), int(xyz[2]))] = int(i)
    edges: List[Tuple[int, int]] = []
    for i, xyz in enumerate(coords.tolist()):
        x, y, z = int(xyz[0]), int(xyz[1]), int(xyz[2])
        for dv in _ANALYSIS_DIRECTIONS:
            j = cell_to_idx.get((x + dv[0], y + dv[1], z + dv[2]))
            if j is not None:
                edges.append((int(i), int(j)))
    return edges


def _sequence_adjacent_face_ratio(coords_ordered: np.ndarray) -> float:
    coords_ordered = np.asarray(coords_ordered, dtype=np.int64)
    if coords_ordered.shape[0] < 2:
        return 0.0
    face_adj = 0
    for i in range(coords_ordered.shape[0] - 1):
        d = np.abs(coords_ordered[i + 1] - coords_ordered[i])
        if int(np.sum(d)) == 1:
            face_adj += 1
    return _safe_ratio(face_adj, coords_ordered.shape[0] - 1)


def _sequence_face_locality_stats(coords: np.ndarray, order_idx: np.ndarray) -> Dict[str, float]:
    coords = np.asarray(coords, dtype=np.int64)
    order_idx = np.asarray(order_idx, dtype=np.int64)
    if coords.shape[0] == 0:
        return {
            "face_edge_count": 0.0,
            "face_edge_rank_gap_mean": 0.0,
            "face_edge_rank_gap_p50": 0.0,
            "face_edge_rank_gap_p90": 0.0,
            "face_edge_rank_gap_p99": 0.0,
            "face_edge_rank_gap_eq_1_ratio": 0.0,
            "sequence_adjacent_face_ratio": 0.0,
        }
    pos = np.empty(coords.shape[0], dtype=np.int64)
    pos[order_idx] = np.arange(order_idx.shape[0], dtype=np.int64)
    gaps: List[float] = []
    for i, j in _face_edge_index_pairs(coords):
        gaps.append(float(abs(int(pos[i]) - int(pos[j]))))
    stats = _summarize_values(gaps, "face_edge_rank_gap")
    stats["face_edge_count"] = float(len(gaps))
    stats["face_edge_rank_gap_eq_1_ratio"] = _safe_ratio(sum(1 for g in gaps if int(g) == 1), len(gaps))
    stats["sequence_adjacent_face_ratio"] = _sequence_adjacent_face_ratio(coords[order_idx])
    return stats


class _EligiblePairIndex:
    """O(1) 随机抽取 / 维护：频次 >= min_freq 的相邻 token 对（与 naive 算法中 eligible 集合一致）。"""

    __slots__ = ("min_freq", "_items", "_pos")

    def __init__(self, min_freq: int) -> None:
        self.min_freq = int(min_freq)
        self._items: List[Tuple[int, int]] = []
        self._pos: Dict[Tuple[int, int], int] = {}

    def sync(self, pair: Tuple[int, int], count: int) -> None:
        ok = count >= self.min_freq
        has = pair in self._pos
        if ok and not has:
            self._pos[pair] = len(self._items)
            self._items.append(pair)
        elif not ok and has:
            idx = self._pos.pop(pair)
            last = self._items.pop()
            if idx < len(self._items):
                self._items[idx] = last
                self._pos[last] = idx

    def pick(self, rng: np.random.Generator) -> Tuple[int, int]:
        return self._items[int(rng.integers(0, len(self._items)))]

    def __bool__(self) -> bool:
        return bool(self._items)


def _train_1d_adjacent_bpe(
    sequences: List[List[int]],
    num_merges: int,
    min_freq: int,
    base_vocab_size: int,
    rng: Optional[np.random.Generator],
    random_pair: bool,
) -> Tuple[List[Tuple[int, int, int]], List[List[int]]]:
    """Greedy adjacent-pair BPE on 1D sequences. Returns merges and final sequences."""
    if random_pair:
        rng = rng if rng is not None else np.random.default_rng(0)
        vals = [[int(x) for x in s] for s in sequences]
        prevs: List[List[int]] = []
        nexts: List[List[int]] = []
        alive: List[List[bool]] = []
        pair_locs: Dict[Tuple[int, int], set] = {}
        eligible_reg = _EligiblePairIndex(min_freq)

        def _pair_at(si: int, i: int) -> Optional[Tuple[int, int]]:
            if i < 0 or not alive[si][i]:
                return None
            j = nexts[si][i]
            if j < 0 or not alive[si][j]:
                return None
            return int(vals[si][i]), int(vals[si][j])

        def _add_occ(si: int, i: int) -> None:
            pair = _pair_at(si, i)
            if pair is None:
                return
            locs = pair_locs.setdefault(pair, set())
            locs.add((si, i))
            eligible_reg.sync(pair, len(locs))

        def _remove_occ(si: int, i: int) -> None:
            pair = _pair_at(si, i)
            if pair is None:
                return
            locs = pair_locs.get(pair)
            if locs is None:
                return
            locs.discard((si, i))
            if not locs:
                pair_locs.pop(pair, None)
                eligible_reg.sync(pair, 0)
            else:
                eligible_reg.sync(pair, len(locs))

        for si, seq in enumerate(vals):
            n = len(seq)
            prevs.append(([-1] + list(range(0, n - 1))) if n else [])
            nexts.append(list(range(1, n)) + [-1] if n else [])
            alive.append([True] * n)
            for i in range(max(0, n - 1)):
                _add_occ(si, i)

        merges: List[Tuple[int, int, int]] = []
        next_id = base_vocab_size
        for _ in _progress(range(num_merges), total=num_merges, desc="[BASELINE] train random 1D BPE"):
            if not eligible_reg:
                break
            chosen = eligible_reg.pick(rng)
            chosen_locs = pair_locs.pop(chosen, None)
            if not chosen_locs:
                continue
            eligible_reg.sync(chosen, 0)
            a, b = chosen
            for si, i in sorted(chosen_locs):
                if _pair_at(si, i) != chosen:
                    continue
                j = nexts[si][i]
                l = prevs[si][i]
                r = nexts[si][j]
                _remove_occ(si, l)
                _remove_occ(si, i)
                _remove_occ(si, j)
                vals[si][i] = int(next_id)
                alive[si][j] = False
                nexts[si][i] = r
                if r >= 0:
                    prevs[si][r] = i
                _add_occ(si, l)
                _add_occ(si, i)
            merges.append((int(a), int(b), int(next_id)))
            next_id += 1

        seqs: List[List[int]] = []
        for si, seq in enumerate(vals):
            if not seq:
                seqs.append([])
                continue
            cur = 0
            while cur >= 0 and not alive[si][cur]:
                cur += 1
                if cur >= len(seq):
                    cur = -1
                    break
            out: List[int] = []
            while cur >= 0:
                out.append(int(vals[si][cur]))
                cur = nexts[si][cur]
            seqs.append(out)
        return merges, seqs

    vals: List[List[int]] = [[int(x) for x in s] for s in sequences]
    prevs: List[List[int]] = []
    nexts: List[List[int]] = []
    alive: List[List[bool]] = []
    pair_locs: Dict[Tuple[int, int], set] = {}
    heap: List[Tuple[int, int, int, Tuple[int, int]]] = []

    def _pair_at(si: int, i: int) -> Optional[Tuple[int, int]]:
        if i < 0 or not alive[si][i]:
            return None
        j = nexts[si][i]
        if j < 0 or not alive[si][j]:
            return None
        return int(vals[si][i]), int(vals[si][j])

    def _push_pair(pair: Tuple[int, int]) -> None:
        locs = pair_locs.get(pair)
        if locs:
            heapq.heappush(heap, (-len(locs), -int(pair[0]), -int(pair[1]), pair))

    def _add_occ(si: int, i: int) -> None:
        pair = _pair_at(si, i)
        if pair is None:
            return
        locs = pair_locs.setdefault(pair, set())
        locs.add((si, i))
        _push_pair(pair)

    def _remove_occ(si: int, i: int) -> None:
        pair = _pair_at(si, i)
        if pair is None:
            return
        locs = pair_locs.get(pair)
        if locs is not None:
            locs.discard((si, i))
            if not locs:
                pair_locs.pop(pair, None)

    for si, seq in enumerate(vals):
        n = len(seq)
        prevs.append(([-1] + list(range(0, n - 1))) if n else [])
        nexts.append(list(range(1, n)) + [-1] if n else [])
        alive.append([True] * n)
        for i in range(max(0, n - 1)):
            _add_occ(si, i)

    merges: List[Tuple[int, int, int]] = []
    next_id = base_vocab_size
    for _ in _progress(range(num_merges), total=num_merges, desc="[BASELINE] train 1D BPE"):
        chosen: Optional[Tuple[int, int]] = None
        chosen_locs: set = set()
        while heap:
            neg_count, _neg_a, _neg_b, pair = heapq.heappop(heap)
            locs = pair_locs.get(pair)
            if not locs:
                continue
            if len(locs) != -int(neg_count):
                heapq.heappush(heap, (-len(locs), -int(pair[0]), -int(pair[1]), pair))
                continue
            if len(locs) >= min_freq:
                chosen = pair
                chosen_locs = set(locs)
                break
        if chosen is None:
            break
        a, b = chosen
        pair_locs.pop(chosen, None)
        for si, i in sorted(chosen_locs):
            if _pair_at(si, i) != chosen:
                continue
            j = nexts[si][i]
            l = prevs[si][i]
            r = nexts[si][j]
            _remove_occ(si, l)
            _remove_occ(si, i)
            _remove_occ(si, j)
            vals[si][i] = int(next_id)
            alive[si][j] = False
            nexts[si][i] = r
            if r >= 0:
                prevs[si][r] = i
            _add_occ(si, l)
            _add_occ(si, i)
        merges.append((int(a), int(b), int(next_id)))
        next_id += 1
    seqs: List[List[int]] = []
    for si, seq in enumerate(vals):
        if not seq:
            seqs.append([])
            continue
        cur = 0
        while cur >= 0 and not alive[si][cur]:
            cur += 1
            if cur >= len(seq):
                cur = -1
                break
        out: List[int] = []
        while cur >= 0:
            out.append(int(vals[si][cur]))
            cur = nexts[si][cur]
        seqs.append(out)
    return merges, seqs


def _encode_with_merges_1d(seq: List[int], merges: List[Tuple[int, int, int]]) -> List[int]:
    ids, _, _ = _encode_with_merges_1d_traced(seq, np.zeros((len(seq), 3), dtype=np.int64), merges)
    return ids


def _min_l1_between_footprints(
    left: Sequence[Tuple[int, int, int]], right: Sequence[Tuple[int, int, int]]
) -> int:
    best: Optional[int] = None
    for ax, ay, az in left:
        for bx, by, bz in right:
            d = abs(int(ax) - int(bx)) + abs(int(ay) - int(by)) + abs(int(az) - int(bz))
            if best is None or d < best:
                best = d
                if best == 0:
                    return 0
    return int(best if best is not None else 0)


def _encode_with_merges_1d_traced(
    tokens_ordered: Sequence[int],
    coords_ordered: np.ndarray,
    merges: List[Tuple[int, int, int]],
) -> Tuple[List[int], List[Tuple[Tuple[int, int, int], ...]], List[int]]:
    ids = [int(x) for x in tokens_ordered]
    coords_ordered = np.asarray(coords_ordered, dtype=np.int64)
    footprints: List[Tuple[Tuple[int, int, int], ...]] = [
        ((int(x), int(y), int(z)),) for x, y, z in coords_ordered.tolist()
    ]
    merge_distances: List[int] = []
    n = len(ids)
    if n < 2 or not merges:
        return ids, footprints, merge_distances
    rank: Dict[Tuple[int, int], Tuple[int, int]] = {
        (int(a), int(b)): (ri, int(nid)) for ri, (a, b, nid) in enumerate(merges)
    }
    prev = ([-1] + list(range(0, n - 1))) if n else []
    nxt = list(range(1, n)) + [-1] if n else []
    alive = [True] * n
    heap: List[Tuple[int, int, int]] = []

    def _push(i: int) -> None:
        if i < 0 or not alive[i]:
            return
        j = nxt[i]
        if j < 0 or not alive[j]:
            return
        info = rank.get((int(ids[i]), int(ids[j])))
        if info is not None:
            heapq.heappush(heap, (int(info[0]), i, int(info[1])))

    for i in range(n - 1):
        _push(i)
    rank_floor = -1
    while heap:
        ri, i, nid = heapq.heappop(heap)
        if int(ri) < rank_floor:
            continue
        if i < 0 or not alive[i]:
            continue
        j = nxt[i]
        if j < 0 or not alive[j]:
            continue
        info = rank.get((int(ids[i]), int(ids[j])))
        if info is None or int(info[0]) != int(ri) or int(info[1]) != int(nid):
            continue
        rank_floor = int(ri)
        left = footprints[i]
        right = footprints[j]
        merge_distances.append(_min_l1_between_footprints(left, right))
        ids[i] = int(nid)
        footprints[i] = tuple(list(left) + list(right))
        alive[j] = False
        r = nxt[j]
        nxt[i] = r
        if r >= 0:
            prev[r] = i
        _push(prev[i])
        _push(i)

    out_ids: List[int] = []
    out_fp: List[Tuple[Tuple[int, int, int], ...]] = []
    cur = 0
    while cur >= 0:
        if alive[cur]:
            out_ids.append(int(ids[cur]))
            out_fp.append(footprints[cur])
        cur = nxt[cur]
    return out_ids, out_fp, merge_distances


def _internalized_face_edge_ratio_from_footprints(
    coords: np.ndarray, footprints: Sequence[Sequence[Tuple[int, int, int]]]
) -> float:
    edges = _face_edge_index_pairs(coords)
    if not edges:
        return 0.0
    cell_to_macro: Dict[Tuple[int, int, int], int] = {}
    for mi, fp in enumerate(footprints):
        for cell in fp:
            cell_to_macro[(int(cell[0]), int(cell[1]), int(cell[2]))] = int(mi)
    internal = 0
    coords = np.asarray(coords, dtype=np.int64)
    for i, j in edges:
        a = tuple(int(v) for v in coords[i].tolist())
        b = tuple(int(v) for v in coords[j].tolist())
        if cell_to_macro.get(a) == cell_to_macro.get(b):
            internal += 1
    return _safe_ratio(internal, len(edges))


def _internalized_face_edge_ratio_3d(
    tok: BPE3DTokenizer, coords: np.ndarray, macro_ids: np.ndarray, macro_anchors: np.ndarray
) -> float:
    footprints: List[List[Tuple[int, int, int]]] = []
    for tid, anc in zip(np.asarray(macro_ids, dtype=np.int64).tolist(), np.asarray(macro_anchors, dtype=np.int64).tolist()):
        ax, ay, az = int(anc[0]), int(anc[1]), int(anc[2])
        fp = [(ax + ox, ay + oy, az + oz) for ox, oy, oz in _macro_footprint_offsets(tok, int(tid))]
        footprints.append(fp)
    return _internalized_face_edge_ratio_from_footprints(coords, footprints)


def _sequence_entropy_summary(seqs: Sequence[Sequence[int]], total_leaf: int) -> Dict[str, float]:
    glob: Counter = Counter()
    bigram_rates: List[float] = []
    trigram_rates: List[float] = []
    for seq in seqs:
        s = [int(x) for x in seq]
        glob.update(s)
        bigram_rates.append(_bigram_conditional_entropy_bits(s))
        trigram_rates.append(_entropy_rate_ngram_bits(s, 3))
    total_tokens = int(sum(len(s) for s in seqs))
    ratio = _safe_ratio(total_tokens, total_leaf)
    h1 = _entropy_bits(glob)
    return {
        "total_tokens": float(total_tokens),
        "compression_ratio_total": ratio,
        "unigram_entropy_bits": h1,
        "unigram_entropy_bits_per_leaf": h1 * ratio,
        "mean_bigram_cond_entropy_bits": _mean_or_zero(bigram_rates),
        "mean_trigram_entropy_rate_bits": _mean_or_zero(trigram_rates),
    }


def _macro_footprints_for_3d(
    tok: BPE3DTokenizer, macro_ids: np.ndarray, macro_anchors: np.ndarray
) -> List[Tuple[Tuple[int, int, int], ...]]:
    out: List[Tuple[Tuple[int, int, int], ...]] = []
    for tid, anc in zip(np.asarray(macro_ids, dtype=np.int64).tolist(), np.asarray(macro_anchors, dtype=np.int64).tolist()):
        ax, ay, az = int(anc[0]), int(anc[1]), int(anc[2])
        out.append(tuple((ax + ox, ay + oy, az + oz) for ox, oy, oz in _macro_footprint_offsets(tok, int(tid))))
    return out


def _baseline_metrics_for_records(
    tok_ref: BPE3DTokenizer,
    records: Sequence[Dict[str, np.ndarray]],
    order_key: str,
    merges: List[Tuple[int, int, int]],
    sample_offset: int = 0,
    *,
    show_progress: bool = False,
) -> Dict[str, Any]:
    seqs_1d_final: List[List[int]] = []
    seqs_3d_final: List[List[int]] = []
    locality_rows: List[Dict[str, Any]] = []
    rank_gaps_all: List[float] = []
    seq_adj_face_ratios: List[float] = []
    internal_1d: List[float] = []
    internal_3d: List[float] = []
    merge_distances_all: List[float] = []
    total_3d = 0
    iterable: Iterable[Tuple[int, Dict[str, np.ndarray]]] = enumerate(records)
    if show_progress:
        iterable = _progress(iterable, total=len(records), desc="[BASELINE] sample metrics")

    for local_index, rec in iterable:
        sample_index = int(sample_offset + local_index)
        tokens = np.asarray(rec["tokens"], dtype=np.int64)
        coords = np.asarray(rec["coords"], dtype=np.int64)
        order_idx = _order_indices_for_baseline(tokens, coords, order_key)
        tokens_ordered = [int(x) for x in tokens[order_idx].tolist()]
        coords_ordered = coords[order_idx]

        seq_1d, fp_1d, merge_distances = _encode_with_merges_1d_traced(
            tokens_ordered, coords_ordered, merges
        )
        seqs_1d_final.append(seq_1d)
        merge_distances_all.extend(float(x) for x in merge_distances)

        macro_ids, macro_anchors = tok_ref.encode_sparse_numpy(tokens, coords)
        macro_ordered, _ = _ordered_mesh_morton(macro_ids, macro_anchors)
        seqs_3d_final.append([int(x) for x in macro_ordered.tolist()])
        total_3d += int(macro_ids.shape[0])

        loc = _sequence_face_locality_stats(coords, order_idx)
        gaps_for_sample: List[float] = []
        pos = np.empty(coords.shape[0], dtype=np.int64)
        pos[order_idx] = np.arange(order_idx.shape[0], dtype=np.int64)
        for i, j in _face_edge_index_pairs(coords):
            gaps_for_sample.append(float(abs(int(pos[i]) - int(pos[j]))))
        rank_gaps_all.extend(gaps_for_sample)
        seq_adj_face_ratios.append(float(loc["sequence_adjacent_face_ratio"]))

        int_1d = _internalized_face_edge_ratio_from_footprints(coords, fp_1d)
        int_3d = _internalized_face_edge_ratio_3d(tok_ref, coords, macro_ids, macro_anchors)
        internal_1d.append(int_1d)
        internal_3d.append(int_3d)

        locality_rows.append(
            {
                "sample_index": int(sample_index),
                "leaf_tokens": int(tokens.shape[0]),
                "macro_tokens_3d": int(macro_ids.shape[0]),
                "macro_tokens_1d": int(len(seq_1d)),
                "compression_ratio_3d": _safe_ratio(int(macro_ids.shape[0]), int(tokens.shape[0])),
                "compression_ratio_1d": _safe_ratio(int(len(seq_1d)), int(tokens.shape[0])),
                "face_edge_count": int(loc["face_edge_count"]),
                "face_edge_rank_gap_mean": float(loc["face_edge_rank_gap_mean"]),
                "face_edge_rank_gap_p50": float(loc["face_edge_rank_gap_p50"]),
                "face_edge_rank_gap_p90": float(loc["face_edge_rank_gap_p90"]),
                "face_edge_rank_gap_p99": float(loc["face_edge_rank_gap_p99"]),
                "face_edge_rank_gap_eq_1_ratio": float(loc["face_edge_rank_gap_eq_1_ratio"]),
                "sequence_adjacent_face_ratio": float(loc["sequence_adjacent_face_ratio"]),
                "internalized_face_edge_ratio_1d": float(int_1d),
                "internalized_face_edge_ratio_3d": float(int_3d),
                "merge_min_l1_mean_1d": _mean_or_zero([float(x) for x in merge_distances]),
                "nonlocal_merge_rate_1d": _safe_ratio(
                    sum(1 for x in merge_distances if int(x) > 1), len(merge_distances)
                ),
            }
        )

    return {
        "seqs_1d_final": seqs_1d_final,
        "seqs_3d_final": seqs_3d_final,
        "locality_rows": locality_rows,
        "rank_gaps_all": rank_gaps_all,
        "seq_adj_face_ratios": seq_adj_face_ratios,
        "internal_1d": internal_1d,
        "internal_3d": internal_3d,
        "merge_distances_all": merge_distances_all,
        "total_3d": int(total_3d),
    }


def _mp_baseline_metrics_chunk(
    payload: Tuple[int, List[Dict[str, np.ndarray]], str, List[Tuple[int, int, int]]]
) -> Dict[str, Any]:
    assert _WORKER_MERGE_TOK is not None
    sample_offset, chunk, order_key, merges = payload
    return _baseline_metrics_for_records(
        _WORKER_MERGE_TOK,
        chunk,
        order_key,
        merges,
        int(sample_offset),
        show_progress=False,
    )


def _write_baseline_comparison_figure(out_dir: Path, payload: Dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[BASELINE] figure_baseline_comparison skipped: {exc}", flush=True)
        return

    ref = payload.get("reference_3d", {})
    base = payload.get("baseline_1d", {})
    loc = payload.get("locality", {})
    merge_geom = payload.get("merge_geometry", {})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].bar(
        ["3D BPE", "1D BPE"],
        [
            float(ref.get("compression_ratio_total", 0.0)),
            float(base.get("compression_ratio_total", 0.0)),
        ],
        color=["#1f77b4", "#bc80bd"],
    )
    axes[0, 0].set_title("Compression ratio")
    axes[0, 0].set_ylabel("macro / leaf")

    axes[0, 1].bar(
        ["3D BPE", "1D BPE"],
        [
            float(ref.get("unigram_entropy_bits_per_leaf", 0.0)),
            float(base.get("unigram_entropy_bits_per_leaf", 0.0)),
        ],
        color=["#1f77b4", "#bc80bd"],
    )
    axes[0, 1].set_title("H1 per leaf")
    axes[0, 1].set_ylabel("bits / leaf")

    axes[1, 0].bar(
        ["1D nonlocal", "1D face-touch"],
        [
            float(merge_geom.get("nonlocal_merge_rate", 0.0)),
            float(merge_geom.get("face_touch_merge_rate", 0.0)),
        ],
        color=["#d62728", "#2ca02c"],
    )
    axes[1, 0].set_title("1D merge geometry")
    axes[1, 0].set_ylim(0.0, 1.0)

    axes[1, 1].bar(
        ["rank gap p90", "seq-adj face"],
        [
            float(loc.get("face_edge_rank_gap_p90", 0.0)),
            float(loc.get("sequence_adjacent_face_ratio", 0.0)),
        ],
        color=["#ff7f0e", "#2ca02c"],
    )
    axes[1, 1].set_title("1D ordering locality")
    axes[1, 1].set_ylabel("gap / ratio")

    fig.suptitle("3D BPE vs 1D adjacent BPE baseline")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_baseline_comparison.png", dpi=180)
    plt.close(fig)


# Cached HilbertCurve instances: constructing one per sample (e.g. 18k×) dominated baseline hilbert1d.
_HILBERT_CURVE_CACHE: Dict[Tuple[int, int], object] = {}


def _get_hilbert_curve_cached(bits: int, dim: int = 3) -> Optional[object]:
    key = (bits, dim)
    hc = _HILBERT_CURVE_CACHE.get(key)
    if hc is not None:
        return hc
    try:
        from hilbertcurve.hilbertcurve import HilbertCurve
    except ImportError:
        return None
    hc = HilbertCurve(bits, dim)
    _HILBERT_CURVE_CACHE[key] = hc
    return hc


def _hilbert_distances_for_scaled_points(
    hc: object, px: np.ndarray, py: np.ndarray, pz: np.ndarray
) -> np.ndarray:
    """1D Hilbert index per voxel; compatible with hilbertcurve v1 and v2."""
    n = int(px.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    # v2+: batch API (preferred)
    if hasattr(hc, "distances_from_points"):
        pts = np.column_stack([px, py, pz]).astype(np.int64, copy=False).tolist()
        return np.asarray(list(hc.distances_from_points(pts)), dtype=np.int64)
    # v1 legacy name
    if hasattr(hc, "distance_from_coordinates"):
        dist_fn = hc.distance_from_coordinates
        dists = np.empty(n, dtype=np.int64)
        for i in range(n):
            dists[i] = int(dist_fn([int(px[i]), int(py[i]), int(pz[i])]))
        return dists
    # v2 single-point fallback
    if hasattr(hc, "distance_from_point"):
        dists = np.empty(n, dtype=np.int64)
        for i in range(n):
            dists[i] = int(hc.distance_from_point([int(px[i]), int(py[i]), int(pz[i])]))
        return dists
    raise RuntimeError(
        "hilbertcurve.HilbertCurve has no distances_from_points / "
        "distance_from_coordinates / distance_from_point; upgrade or pin hilbertcurve."
    )


def _sort_indices_packed(coords: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=np.int64)
    cmax = int(np.max(c)) + 1 if c.size else 1
    codes = (c[:, 0] * cmax + c[:, 1]) * cmax + c[:, 2]
    return np.argsort(codes, kind="stable")


def _sort_indices_hilbert(coords: np.ndarray, bits: int = 8) -> np.ndarray:
    hc = _get_hilbert_curve_cached(bits, 3)
    if hc is None:
        return _sort_indices_packed(coords)
    c = np.asarray(coords, dtype=np.int64)
    n = c.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    mx = int(np.max(c)) + 1 if c.size else 1
    lim = int((1 << bits) - 1)
    mx_d = float(max(mx, 1))
    lim_f = float(lim)
    # Match legacy per-voxel scaling: int(x * lim / mx) then min(..., lim)
    px = np.minimum((c[:, 0].astype(np.float64) * lim_f / mx_d).astype(np.int64), lim)
    py = np.minimum((c[:, 1].astype(np.float64) * lim_f / mx_d).astype(np.int64), lim)
    pz = np.minimum((c[:, 2].astype(np.float64) * lim_f / mx_d).astype(np.int64), lim)
    dists = _hilbert_distances_for_scaled_points(hc, px, py, pz)
    return np.argsort(dists, kind="stable")


def _hilbert_order_chunk_worker(chunk: List[Dict[str, np.ndarray]]) -> Tuple[List[List[int]], int]:
    """ProcessPool worker: Hilbert-order tokens for a list of samples (hilbert1d baseline)."""
    seqs: List[List[int]] = []
    total_leaf = 0
    for rec in chunk:
        tok = np.asarray(rec["tokens"], dtype=np.int64)
        xyz = np.asarray(rec["coords"], dtype=np.int64)
        order_idx = _sort_indices_hilbert(xyz)
        seqs.append([int(x) for x in tok[order_idx].tolist()])
        total_leaf += int(tok.shape[0])
    return seqs, total_leaf


def _corpus_sequences_ordered(
    records: Iterable[Dict[str, np.ndarray]],
    order: str,
    *,
    parallel_hilbert_workers: int = 1,
) -> Tuple[List[List[int]], int, int]:
    records_list = list(records) if not isinstance(records, list) else records
    n_rec = len(records_list)
    nw_h = int(parallel_hilbert_workers)

    if (
        order == "hilbert"
        and nw_h > 1
        and n_rec >= 200
        and _get_hilbert_curve_cached(8, 3) is not None
    ):
        chunks = _records_chunk_list(records_list, nw_h, min_samples=50)
        if len(chunks) > 1:
            seqs_all: List[List[int]] = []
            total_leaf = 0
            with ProcessPoolExecutor(max_workers=nw_h) as ex:
                results = list(
                    _progress(
                        ex.map(_hilbert_order_chunk_worker, chunks),
                        total=len(chunks),
                        desc="[BASELINE] Hilbert order (parallel)",
                    )
                )
            for part_seqs, part_tl in results:
                seqs_all.extend(part_seqs)
                total_leaf += part_tl
            return seqs_all, total_leaf, total_leaf

    seqs: List[List[int]] = []
    total_leaf = 0
    total_after_identity = 0
    total = n_rec
    for rec in _progress(records_list, total=total, desc="[BASELINE] order sequences"):
        tok = np.asarray(rec["tokens"], dtype=np.int64)
        xyz = np.asarray(rec["coords"], dtype=np.int64)
        if order == "morton":
            order_idx = _sort_indices_packed(xyz)
        elif order == "hilbert":
            order_idx = _sort_indices_hilbert(xyz)
        elif order == "random":
            rng = np.random.default_rng(hash(tuple(tok.tolist()[:8])) % (2**31))
            order_idx = rng.permutation(tok.shape[0])
        else:
            order_idx = np.arange(tok.shape[0])
        seq = tok[order_idx].tolist()
        seqs.append([int(x) for x in seq])
        total_leaf += len(seq)
        total_after_identity += len(seq)
    return seqs, total_leaf, total_after_identity


def _truncate_tokenizer(tok: BPE3DTokenizer, k: int) -> BPE3DTokenizer:
    t2 = BPE3DTokenizer(base_vocab_size=int(tok.base_vocab_size), directions=tuple(tok.directions))
    t2.merge_table = list(tok.merge_table[: int(k)])
    return t2


def run_reliability_encode_decode(
    tok: BPE3DTokenizer,
    shards: List[Path],
    max_samples: Optional[int],
    print_every: int,
    *,
    records: Optional[Sequence[Dict[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    sparse_cls = DummySparseTensor
    failures = 0
    checked = 0
    sample_index = 0

    def _iter_samples() -> Iterable[Dict[str, np.ndarray]]:
        if records is not None:
            yield from records
            return
        for shard_path in shards:
            yield from _load_corpus_part(shard_path)

    total_records = len(records) if records is not None else None
    for rec in _progress(_iter_samples(), total=total_records, desc="[RELIABILITY] roundtrip"):
        if max_samples is not None and sample_index >= int(max_samples):
            break
        sample = {
            "tokens": np.asarray(rec["tokens"], dtype=np.int64),
            "coords": np.asarray(rec["coords"], dtype=np.int64),
        }
        try:
            st = to_sparse(sample, sparse_cls)
            out = tok.encode_sparse(st, sparse_tensor_cls=sparse_cls)
            dec = tok.decode_to_sparse(out["batches"], device=torch.device("cpu"), sparse_tensor_cls=sparse_cls)
            if sparse_item_set(dec) != sparse_item_set(st):
                failures += 1
        except Exception:
            failures += 1
        checked += 1
        sample_index += 1
        if print_every and sample_index % print_every == 0:
            print(f"[RELIABILITY] checked={sample_index} failures={failures}", flush=True)
    return {
        "encode_decode_roundtrip_checked": int(checked),
        "encode_decode_failures": int(failures),
        "encode_decode_success_rate": _safe_ratio(checked - failures, checked),
    }


def run_reliability_save_load(
    tok: BPE3DTokenizer,
    shards: List[Path],
    max_samples: Optional[int],
    *,
    records: Optional[Sequence[Dict[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "merge_table_reload.json"
        tok.save(str(p))
        tok2 = BPE3DTokenizer.load(str(p))
    mismatches = 0
    checked = 0
    sample_index = 0

    def _iter_samples() -> Iterable[Dict[str, np.ndarray]]:
        if records is not None:
            yield from records
            return
        for shard_path in shards:
            yield from _load_corpus_part(shard_path)

    total_records = len(records) if records is not None else None
    for rec in _progress(_iter_samples(), total=total_records, desc="[RELIABILITY] save/load"):
        if max_samples is not None and sample_index >= int(max_samples):
            break
        tokens = np.asarray(rec["tokens"], dtype=np.int64)
        coords = np.asarray(rec["coords"], dtype=np.int64)
        a1, b1 = tok.encode_sparse_numpy(tokens, coords)
        a2, b2 = tok2.encode_sparse_numpy(tokens, coords)
        if not (np.array_equal(a1, a2) and np.array_equal(b1, b2)):
            mismatches += 1
        checked += 1
        sample_index += 1
    return {
        "save_load_encode_checked": int(checked),
        "save_load_id_mismatches": int(mismatches),
        "save_load_match_rate": _safe_ratio(checked - mismatches, checked),
    }


def _write_analysis_plots(
    out_dir: Path,
    per_sample_rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[ANALYSIS] matplotlib unavailable; skip plots: {exc}", flush=True)
        return

    ratios = [float(r["compression_ratio"]) for r in per_sample_rows]
    if ratios:
        plt.figure(figsize=(7, 4))
        plt.hist(ratios, bins=40, color="#4C72B0", edgecolor="black", alpha=0.85)
        plt.xlabel("macro tokens / leaf tokens")
        plt.ylabel("sample count")
        plt.title("3D BPE Compression Ratio")
        plt.tight_layout()
        plt.savefig(out_dir / "compression_ratio.png", dpi=180)
        plt.close()

    leaf = metrics["leaf"]
    macro_boundary = metrics["macro_boundary"]
    macro_anchor = metrics["macro_anchor"]

    plt.figure(figsize=(8, 4))
    labels = ["H1", "H_pair", "H_cond"]
    before = [
        float(leaf["unigram_entropy_bits"]),
        float(leaf["adjacency"]["pair_entropy_bits"]),
        float(leaf["adjacency"]["target_given_source_direction_entropy_bits"]),
    ]
    after = [
        float(metrics["macro"]["unigram_entropy_bits"]),
        float(macro_boundary["adjacency"]["pair_entropy_bits"]),
        float(macro_boundary["adjacency"]["target_given_source_direction_entropy_bits"]),
    ]
    x = np.arange(len(labels))
    width = 0.36
    plt.bar(x - width / 2, before, width, label="leaf")
    plt.bar(x + width / 2, after, width, label="macro boundary")
    plt.xticks(x, labels)
    plt.ylabel("bits")
    plt.title("Entropy Before and After 3D BPE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    mi_values = [
        float(leaf["adjacency"]["source_direction_target_mutual_information_bits"]),
        float(macro_anchor["adjacency"]["source_direction_target_mutual_information_bits"]),
        float(macro_boundary["adjacency"]["source_direction_target_mutual_information_bits"]),
    ]
    plt.bar(["leaf", "macro anchor", "macro boundary"], mi_values)
    plt.ylabel("bits")
    plt.title("Directed 3D Adjacency Mutual Information")
    plt.tight_layout()
    plt.savefig(out_dir / "mutual_information_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    ratio_labels = [
        "boundary MI ratio",
        "boundary edge ratio",
        "weighted boundary MI",
    ]
    ratio_values = [
        float(metrics["deltas"]["boundary_mi_ratio"]),
        float(metrics["deltas"]["boundary_edge_ratio"]),
        float(metrics["deltas"]["weighted_boundary_mi_ratio"]),
    ]
    plt.bar(
        ratio_labels,
        ratio_values,
    )
    plt.ylim(0, max(1.0, max(ratio_values) * 1.1))
    plt.ylabel("ratio")
    plt.title("Residual Boundary Dependency After 3D BPE")
    plt.tight_layout()
    plt.savefig(out_dir / "boundary_mi_ratio.png", dpi=180)
    plt.close()


def _edge_counts_by_direction(leaf_pairs: Counter) -> Tuple[int, int, int]:
    ex = ey = ez = 0
    for (a, b, d), c in leaf_pairs.items():
        di = int(d)
        if di == 0:
            ex += int(c)
        elif di == 1:
            ey += int(c)
        else:
            ez += int(c)
    return ex, ey, ez


def _finalize_bpe_metrics(
    corpus_prefix: Path,
    merge_table: Path,
    sample_index: int,
    total_leaf: int,
    total_macro: int,
    leaf_token_counts: Counter,
    macro_token_counts: Counter,
    leaf_adj_counts: Counter,
    macro_anchor_adj_counts: Counter,
    macro_boundary_adj_counts: Counter,
    tok: BPE3DTokenizer,
) -> Dict[str, Any]:
    leaf_adj_stats = _directed_pair_stats(leaf_adj_counts)
    macro_anchor_adj_stats = _directed_pair_stats(macro_anchor_adj_counts)
    macro_boundary_adj_stats = _directed_pair_stats(macro_boundary_adj_counts)
    leaf_mi = leaf_adj_stats["source_direction_target_mutual_information_bits"]
    boundary_mi = macro_boundary_adj_stats["source_direction_target_mutual_information_bits"]
    leaf_unigram_entropy = _entropy_bits(leaf_token_counts)
    macro_unigram_entropy = _entropy_bits(macro_token_counts)
    compression_ratio = _safe_ratio(total_macro, total_leaf)
    boundary_mi_ratio = _safe_ratio(boundary_mi, leaf_mi)
    boundary_edge_ratio = _safe_ratio(
        macro_boundary_adj_stats["edge_count"],
        leaf_adj_stats["edge_count"],
    )
    macro_unigram_entropy_per_leaf = macro_unigram_entropy * compression_ratio
    weighted_boundary_mi_ratio = boundary_mi_ratio * boundary_edge_ratio

    metrics: Dict[str, Any] = {
        "corpus_prefix": str(corpus_prefix),
        "merge_table": str(merge_table),
        "samples": int(sample_index),
        "total_leaf_tokens": int(total_leaf),
        "total_macro_tokens": int(total_macro),
        "compression_ratio_total": compression_ratio,
        "compression_rate_percent": 100.0 * (1.0 - compression_ratio),
        "leaf": {
            "vocab_observed": int(len(leaf_token_counts)),
            "unigram_entropy_bits": leaf_unigram_entropy,
            "adjacency": leaf_adj_stats,
        },
        "macro": {
            "vocab_observed": int(len(macro_token_counts)),
            "unigram_entropy_bits": macro_unigram_entropy,
            "unigram_entropy_bits_per_leaf": macro_unigram_entropy_per_leaf,
        },
        "macro_anchor": {
            "adjacency": macro_anchor_adj_stats,
        },
        "macro_boundary": {
            "adjacency": macro_boundary_adj_stats,
        },
        "per_direction": _per_direction_block(
            leaf_adj_counts, macro_anchor_adj_counts, macro_boundary_adj_counts
        ),
        "macro_footprint_vocab": _macro_footprint_vocab_from_table(tok),
        "merge_table_metadata": _merge_table_freq_metadata(Path(merge_table)),
        "deltas": {
            "unigram_entropy_bits": leaf_unigram_entropy - macro_unigram_entropy,
            "unigram_entropy_bits_per_leaf": (
                leaf_unigram_entropy - macro_unigram_entropy_per_leaf
            ),
            "unigram_entropy_per_leaf_ratio": _safe_ratio(
                macro_unigram_entropy_per_leaf,
                leaf_unigram_entropy,
            ),
            "boundary_pair_entropy_bits": (
                leaf_adj_stats["pair_entropy_bits"] - macro_boundary_adj_stats["pair_entropy_bits"]
            ),
            "boundary_conditional_entropy_bits": (
                leaf_adj_stats["target_given_source_direction_entropy_bits"]
                - macro_boundary_adj_stats["target_given_source_direction_entropy_bits"]
            ),
            "boundary_mutual_information_bits": leaf_mi - boundary_mi,
            "boundary_mi_ratio": boundary_mi_ratio,
            "boundary_edge_ratio": boundary_edge_ratio,
            "weighted_boundary_mi_ratio": weighted_boundary_mi_ratio,
        },
        "paper_reporting": {
            "headline_bits_per_leaf_before_H1_X": leaf_unigram_entropy,
            "headline_bits_per_leaf_after_H1_B_times_R_len": macro_unigram_entropy_per_leaf,
            "weighted_boundary_mi_ratio": weighted_boundary_mi_ratio,
            "per_edge_boundary_mi_ratio": boundary_mi_ratio,
            "note_per_edge_mi_survivor_bias": (
                "per-edge MI ratio can exceed 1 after BPE because high-MI adjacent pairs "
                "are absorbed into macros; report weighted_boundary_mi_ratio for total MI flux."
            ),
        },
    }
    return metrics


def _write_extended_figures(
    out_dir: Path,
    leaf_token_counts: Counter,
    macro_token_counts: Counter,
    leaf_adj_counts: Counter,
    macro_boundary_adj_counts: Counter,
    viz_coords: Optional[np.ndarray],
    viz_macro_ids: Optional[np.ndarray],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[ANALYSIS] extended figures skipped: {exc}", flush=True)
        return

    # Zipf: rank vs log2 count
    def _zipf_xy(ct: Counter) -> Tuple[np.ndarray, np.ndarray]:
        vals = sorted(int(v) for v in ct.values() if v > 0)
        if not vals:
            return np.asarray([]), np.asarray([])
        vals.sort(reverse=True)
        ranks = np.arange(1, len(vals) + 1, dtype=np.float64)
        counts = np.log2(np.asarray(vals, dtype=np.float64))
        return ranks, counts

    rx, cx = _zipf_xy(leaf_token_counts)
    rm, cm = _zipf_xy(macro_token_counts)
    if rx.size > 0:
        plt.figure(figsize=(7, 4))
        plt.plot(rx, cx, label="leaf tokens", alpha=0.85)
        if rm.size > 0:
            plt.plot(rm, cm, label="macro tokens", alpha=0.85)
        plt.xlabel("rank")
        plt.ylabel("log2 count")
        plt.title("Zipf-style rank vs frequency (leaf vs macro)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "figure_zipf.png", dpi=180)
        plt.close()

    # PMI proxy: top directed pairs by count (same as high PMI mass under uniform marginals)
    top_leaf = leaf_adj_counts.most_common(40)
    top_b = macro_boundary_adj_counts.most_common(40)
    if top_leaf or top_b:
        plt.figure(figsize=(9, 5))
        if top_leaf:
            labs = [f"{a}|{b}|d{d}" for (a, b, d), _ in top_leaf[:25]]
            plt.subplot(1, 2, 1)
            plt.barh(range(len(labs)), [c for _, c in top_leaf[:25]][::-1])
            plt.yticks(range(len(labs)), labs[::-1], fontsize=6)
            plt.xlabel("count")
            plt.title("Top leaf directed pairs")
        if top_b:
            labs2 = [f"{a}|{b}|d{d}" for (a, b, d), _ in top_b[:25]]
            plt.subplot(1, 2, 2)
            plt.barh(range(len(labs2)), [c for _, c in top_b[:25]][::-1])
            plt.yticks(range(len(labs2)), labs2[::-1], fontsize=6)
            plt.xlabel("count")
            plt.title("Top macro-boundary directed pairs")
        plt.tight_layout()
        plt.savefig(out_dir / "figure_pmi_topk.png", dpi=180)
        plt.close()

    if viz_coords is not None and viz_macro_ids is not None and viz_coords.shape[0] > 0:
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            mids = viz_macro_ids.astype(np.int64) % 20
            sc = ax.scatter(
                viz_coords[:, 0],
                viz_coords[:, 1],
                viz_coords[:, 2],
                c=mids,
                cmap="tab20",
                s=8,
                alpha=0.85,
            )
            plt.colorbar(sc, ax=ax, shrink=0.6, label="macro_id mod 20")
            ax.set_title("Voxel motif (one sample; color = macro id mod 20)")
            plt.tight_layout()
            plt.savefig(out_dir / "figure_voxel_motif.png", dpi=180)
            plt.close()
        except Exception as exc:
            print(f"[ANALYSIS] voxel plot skipped: {exc}", flush=True)


def _write_merge_freq_figure(out_dir: Path, merge_table_path: Path) -> None:
    meta = _merge_table_freq_metadata(merge_table_path)
    if not meta.get("merge_freq_skipped"):
        # If freq fields appear in future JSON, extend parsing here (still single-file).
        pass
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(6, 3))
    plt.text(
        0.5,
        0.5,
        meta.get("merge_freq_skip_reason") or "merge frequency not in merge_table JSON",
        ha="center",
        va="center",
        wrap=True,
    )
    plt.axis("off")
    plt.title("Merge frequency curve (skipped)")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_merge_freq.png", dpi=180)
    plt.close()


def _write_figure_d_sweep(out_dir: Path, sweep_rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ANALYSIS] D-sweep plot skipped: {exc}", flush=True)
        return
    if not sweep_rows:
        return
    ks = [int(r["K"]) for r in sweep_rows]
    rlen = [float(r["compression_ratio_total"]) for r in sweep_rows]
    h1pl = [float(r["macro"]["unigram_entropy_bits_per_leaf"]) for r in sweep_rows]
    rmi = [float(r["deltas"]["boundary_mi_ratio"]) for r in sweep_rows]
    wmi = [float(r["deltas"]["weighted_boundary_mi_ratio"]) for r in sweep_rows]
    redge = [float(r["deltas"]["boundary_edge_ratio"]) for r in sweep_rows]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].plot(ks, rlen, marker="o")
    axs[0, 0].set_xlabel("K merges")
    axs[0, 0].set_ylabel("R_len")
    axs[0, 0].set_title("Compression ratio vs K")
    axs[0, 0].set_xscale("symlog", linthresh=1)

    axs[0, 1].plot(ks, h1pl, marker="o", label="H1(B)·R_len")
    if sweep_rows:
        h1x = float(sweep_rows[0]["leaf"]["unigram_entropy_bits"])
        axs[0, 1].axhline(h1x, color="gray", linestyle="--", label="H1(X) leaf")
    axs[0, 1].set_xlabel("K merges")
    axs[0, 1].set_ylabel("bits / leaf")
    axs[0, 1].set_title("Per-leaf unigram cost")
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].set_xscale("symlog", linthresh=1)

    axs[1, 0].plot(ks, rmi, marker="o", label="R_MI per edge")
    axs[1, 0].plot(ks, wmi, marker="s", label="weighted MI flux")
    axs[1, 0].set_xlabel("K merges")
    axs[1, 0].set_ylabel("ratio")
    axs[1, 0].set_title("Boundary MI metrics")
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].set_xscale("symlog", linthresh=1)

    axs[1, 1].plot(ks, redge, marker="o", color="#2ca02c")
    axs[1, 1].set_xlabel("K merges")
    axs[1, 1].set_ylabel("R_edge")
    axs[1, 1].set_title("Boundary edge ratio")
    axs[1, 1].set_xscale("symlog", linthresh=1)

    plt.tight_layout()
    plt.savefig(out_dir / "figure_D_sweep.png", dpi=180)
    plt.close()


def _write_figure_robustness(out_dir: Path, panel: Dict[str, float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ANALYSIS] robustness plot skipped: {exc}", flush=True)
        return
    names = list(panel.keys())
    vals = [float(panel[k]) for k in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, vals, color="#9467bd")
    plt.ylabel("compression_ratio_total")
    plt.title("Robustness panel (compression under perturbations)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_robustness.png", dpi=180)
    plt.close()


def run_bpe_output_analysis(args: argparse.Namespace) -> None:
    corpus_prefix = Path(args.corpus_prefix)
    merge_table = Path(args.merge_table)
    out_dir = Path(args.analysis_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not merge_table.is_file():
        raise FileNotFoundError(f"merge table not found: {merge_table}")
    tok = BPE3DTokenizer.load(str(merge_table))
    max_samples: Optional[int] = args.max_samples
    records = _load_corpus_records(corpus_prefix, max_samples, tokenizer=tok)
    leaf_token_counts = Counter()
    macro_token_counts = Counter()
    leaf_adj_counts = Counter()
    macro_anchor_adj_counts = Counter()
    macro_boundary_adj_counts = Counter()
    per_sample_rows: List[Dict[str, Any]] = []

    total_leaf = 0
    total_macro = 0
    sample_index = 0
    print_every = max(1, int(args.analysis_print_every))
    viz_idx = int(getattr(args, "viz_sample_index", 0))
    viz_coords: Optional[np.ndarray] = None
    viz_macro_ids: Optional[np.ndarray] = None

    for rec in records:
        tokens = np.asarray(rec["tokens"], dtype=np.int64, order="C")
        coords = np.asarray(rec["coords"], dtype=np.int64, order="C")
        macro_ids, macro_anchors = tok.encode_sparse_numpy(tokens, coords)

        leaf_pairs = _adjacency_pair_counts(tokens, coords)
        macro_anchor_pairs = _adjacency_pair_counts(macro_ids, macro_anchors)
        macro_boundary_pairs = _macro_boundary_pair_counts(tok, macro_ids, macro_anchors)

        _merge_counter(leaf_token_counts, _token_counter(tokens))
        _merge_counter(macro_token_counts, _token_counter(macro_ids))
        _merge_counter(leaf_adj_counts, leaf_pairs)
        _merge_counter(macro_anchor_adj_counts, macro_anchor_pairs)
        _merge_counter(macro_boundary_adj_counts, macro_boundary_pairs)

        n_leaf = int(tokens.shape[0])
        n_macro = int(macro_ids.shape[0])
        total_leaf += n_leaf
        total_macro += n_macro
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        bbox_vol = int(np.prod(np.maximum(cmax - cmin + 1, 1)))
        ex, ey, ez = _edge_counts_by_direction(leaf_pairs)
        row = {
            "sample_index": sample_index,
            "leaf_tokens": n_leaf,
            "macro_tokens": n_macro,
            "compression_ratio": _safe_ratio(n_macro, n_leaf),
            "leaf_edges": int(sum(leaf_pairs.values())),
            "macro_anchor_edges": int(sum(macro_anchor_pairs.values())),
            "macro_boundary_edges": int(sum(macro_boundary_pairs.values())),
            "bbox_volume": bbox_vol,
            "leaf_unique_types": int(len(set(tokens.tolist()))),
            "macro_unique_types": int(len(set(macro_ids.tolist()))),
            "leaf_edges_plus_x": ex,
            "leaf_edges_plus_y": ey,
            "leaf_edges_plus_z": ez,
        }
        per_sample_rows.append(row)
        if sample_index == viz_idx and n_leaf > 0:
            expanded_xyz: List[np.ndarray] = []
            expanded_mid: List[int] = []
            for mi in range(int(macro_ids.shape[0])):
                tid = int(macro_ids[mi])
                ax, ay, az = (
                    int(macro_anchors[mi, 0]),
                    int(macro_anchors[mi, 1]),
                    int(macro_anchors[mi, 2]),
                )
                for ox, oy, oz in _macro_footprint_offsets(tok, tid):
                    expanded_xyz.append(
                        np.asarray([ax + ox, ay + oy, az + oz], dtype=np.int64)
                    )
                    expanded_mid.append(tid)
            if expanded_xyz:
                viz_coords = np.stack(expanded_xyz, axis=0)
                viz_macro_ids = np.asarray(expanded_mid, dtype=np.int64)
        sample_index += 1
        if sample_index % print_every == 0:
            print(f"[ANALYSIS] processed samples={sample_index}", flush=True)

    metrics = _finalize_bpe_metrics(
        corpus_prefix,
        merge_table,
        sample_index,
        total_leaf,
        total_macro,
        leaf_token_counts,
        macro_token_counts,
        leaf_adj_counts,
        macro_anchor_adj_counts,
        macro_boundary_adj_counts,
        tok,
    )

    leaf_mi = float(metrics["leaf"]["adjacency"]["source_direction_target_mutual_information_bits"])
    boundary_mi = float(
        metrics["macro_boundary"]["adjacency"]["source_direction_target_mutual_information_bits"]
    )

    bootstrap_n = int(getattr(args, "bootstrap_n", 0) or 0)
    if bootstrap_n > 0 and per_sample_rows:
        metrics["bootstrap_ci"] = _bootstrap_ci(
            per_sample_rows,
            bootstrap_n,
            int(getattr(args, "bootstrap_seed", 0)),
        )

    rel: Dict[str, Any] = {}
    if getattr(args, "reliability_check", False):
        rel_pe = int(getattr(args, "reliability_print_every", 1000) or 1000)
        rel_max = getattr(args, "reliability_max_samples", None)
        rel.update(
            run_reliability_encode_decode(
                tok, [], rel_max, rel_pe if rel_pe > 0 else 10**9, records=records
            )
        )
        rel.update(run_reliability_save_load(tok, [], rel_max, records=records))
    if rel:
        metrics["reliability"] = rel

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "per_sample.csv"
    if per_sample_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample_rows)

    if args.plot:
        _write_analysis_plots(out_dir, per_sample_rows, metrics)
        _write_extended_figures(
            out_dir,
            leaf_token_counts,
            macro_token_counts,
            leaf_adj_counts,
            macro_boundary_adj_counts,
            viz_coords,
            viz_macro_ids,
        )
        _write_merge_freq_figure(out_dir, merge_table)

    print(
        "[ANALYSIS] "
        f"samples={sample_index} compression={metrics['compression_ratio_total']:.6f} "
        f"H1_leaf={metrics['leaf']['unigram_entropy_bits']:.6f} "
        f"H1_macro={metrics['macro']['unigram_entropy_bits']:.6f} "
        f"H1_macro_per_leaf={metrics['macro']['unigram_entropy_bits_per_leaf']:.6f} "
        f"MI_leaf={leaf_mi:.6f} MI_boundary={boundary_mi:.6f} "
        f"boundary_mi_ratio={metrics['deltas']['boundary_mi_ratio']:.6f} "
        f"weighted_boundary_mi_ratio={metrics['deltas']['weighted_boundary_mi_ratio']:.6f}",
        flush=True,
    )
    print(f"[ANALYSIS] wrote {metrics_path}", flush=True)


def _compression_ratio_under_transform(
    tok: BPE3DTokenizer,
    shards: List[Path],
    max_samples: Optional[int],
    *,
    records: Optional[Sequence[Dict[str, np.ndarray]]] = None,
    coord_shift: Optional[Tuple[int, int, int]] = None,
    axis_perm: Optional[Tuple[int, int, int]] = None,
    noise_p: float = 0.0,
    drop_p: float = 0.0,
    seed: int = 0,
    sample_range: Optional[Tuple[int, int]] = None,
    mp_merge_path: Optional[str] = None,
    mp_workers: int = 1,
) -> Tuple[float, int]:
    if (
        records is not None
        and mp_merge_path
        and mp_workers > 1
        and sample_range is None
    ):
        rec_list = list(records)
        if max_samples is not None:
            rec_list = rec_list[: int(max_samples)]
        if len(rec_list) >= 200:
            chunks = _records_chunk_list(rec_list, mp_workers)
            if len(chunks) > 1:
                tasks = [
                    (ch, coord_shift, axis_perm, noise_p, drop_p, seed + i)
                    for i, ch in enumerate(chunks)
                ]
                with ProcessPoolExecutor(
                    max_workers=mp_workers,
                    initializer=_merge_table_worker_init,
                    initargs=(mp_merge_path,),
                ) as ex:
                    results = list(
                        _progress(
                            ex.map(_mp_compression_chunk, tasks),
                            total=len(tasks),
                            desc="[ROBUST] transform chunks",
                        )
                    )
                tl = sum(int(r[0]) for r in results)
                tm = sum(int(r[1]) for r in results)
                return _safe_ratio(tm, tl), sum(len(ch) for ch in chunks)

    rng = np.random.default_rng(seed)
    total_leaf = 0
    total_macro = 0
    sample_index = 0

    def _iter_samples() -> Iterable[Dict[str, np.ndarray]]:
        if records is not None:
            yield from records
            return
        for shard_path in shards:
            yield from _load_corpus_part(shard_path)

    total_records = len(records) if records is not None and sample_range is None else None
    for rec in _progress(_iter_samples(), total=total_records, desc="[ROBUST] transform samples"):
        if max_samples is not None and sample_index >= int(max_samples):
            break
        if sample_range is not None:
            lo, hi = sample_range
            if sample_index < lo or sample_index >= hi:
                sample_index += 1
                continue
        tokens = np.asarray(rec["tokens"], dtype=np.int64).copy()
        coords = np.asarray(rec["coords"], dtype=np.int64).copy()
        if coord_shift is not None:
            coords[:, 0] += coord_shift[0]
            coords[:, 1] += coord_shift[1]
            coords[:, 2] += coord_shift[2]
        if axis_perm is not None:
            coords = coords[:, list(axis_perm)]
        if noise_p > 0.0 and tokens.shape[0] > 0:
            mask = rng.random(tokens.shape[0]) < noise_p
            if np.any(mask):
                nrep = int(np.sum(mask))
                tokens[mask] = rng.integers(0, int(tok.base_vocab_size), size=nrep)
        if drop_p > 0.0 and tokens.shape[0] > 0:
            keep = rng.random(tokens.shape[0]) >= drop_p
            if not np.any(keep):
                sample_index += 1
                continue
            tokens = tokens[keep]
            coords = coords[keep]
        if tokens.shape[0] == 0:
            sample_index += 1
            continue
        macro_ids, _ = tok.encode_sparse_numpy(tokens, coords)
        total_leaf += int(tokens.shape[0])
        total_macro += int(macro_ids.shape[0])
        sample_index += 1
    return _safe_ratio(total_macro, total_leaf), sample_index


def run_d_sweep(args: argparse.Namespace) -> None:
    corpus_prefix = Path(args.corpus_prefix)
    merge_table = Path(args.merge_table)
    out_dir = Path(args.analysis_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not merge_table.is_file():
        raise FileNotFoundError(f"merge table not found: {merge_table}")
    ks = sorted({int(x.strip()) for x in str(args.sweep_D).split(",") if x.strip()})
    tok_full = BPE3DTokenizer.load(str(merge_table))
    max_samples: Optional[int] = args.max_samples
    records = _load_corpus_records(corpus_prefix, max_samples, tokenizer=tok_full)
    print_every = max(1, int(args.analysis_print_every))
    sweep_rows: List[Dict[str, Any]] = []
    for K in _progress(ks, total=len(ks), desc="[SWEEP] K values"):
        tok_k = _truncate_tokenizer(tok_full, K)
        leaf_token_counts = Counter()
        macro_token_counts = Counter()
        leaf_adj_counts = Counter()
        macro_anchor_adj_counts = Counter()
        macro_boundary_adj_counts = Counter()
        total_leaf = 0
        total_macro = 0
        sample_index = 0
        for rec in _progress(records, total=len(records), desc=f"[SWEEP K={K}] samples"):
            tokens = np.asarray(rec["tokens"], dtype=np.int64, order="C")
            coords = np.asarray(rec["coords"], dtype=np.int64, order="C")
            macro_ids, macro_anchors = tok_k.encode_sparse_numpy(tokens, coords)
            leaf_pairs = _adjacency_pair_counts(tokens, coords)
            macro_anchor_pairs = _adjacency_pair_counts(macro_ids, macro_anchors)
            macro_boundary_pairs = _macro_boundary_pair_counts(tok_k, macro_ids, macro_anchors)
            _merge_counter(leaf_token_counts, _token_counter(tokens))
            _merge_counter(macro_token_counts, _token_counter(macro_ids))
            _merge_counter(leaf_adj_counts, leaf_pairs)
            _merge_counter(macro_anchor_adj_counts, macro_anchor_pairs)
            _merge_counter(macro_boundary_adj_counts, macro_boundary_pairs)
            total_leaf += int(tokens.shape[0])
            total_macro += int(macro_ids.shape[0])
            sample_index += 1
            if sample_index % print_every == 0:
                print(f"[SWEEP K={K}] processed samples={sample_index}", flush=True)
        m = _finalize_bpe_metrics(
            corpus_prefix,
            merge_table,
            sample_index,
            total_leaf,
            total_macro,
            leaf_token_counts,
            macro_token_counts,
            leaf_adj_counts,
            macro_anchor_adj_counts,
            macro_boundary_adj_counts,
            tok_k,
        )
        m["K"] = int(K)
        sweep_rows.append(m)
        out_k = out_dir / f"metrics_K{K}.json"
        out_k.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SWEEP] wrote {out_k}", flush=True)
    summary = {
        "Ks": ks,
        "compression_by_K": [
            {"K": int(r["K"]), "compression_ratio_total": r["compression_ratio_total"]} for r in sweep_rows
        ],
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.plot:
        _write_figure_d_sweep(out_dir, sweep_rows)
    print(f"[SWEEP] done; summary={out_dir / 'sweep_summary.json'}", flush=True)


def run_baseline_analysis(args: argparse.Namespace) -> None:
    corpus_prefix = Path(args.corpus_prefix)
    merge_table = Path(args.merge_table)
    out_dir = Path(args.analysis_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_ref = BPE3DTokenizer.load(str(merge_table))
    records = _load_corpus_records(corpus_prefix, args.max_samples, tokenizer=tok_ref)
    mode = str(args.baseline)
    if mode == "morton1d":
        order_key = "morton"
    elif mode == "hilbert1d":
        order_key = "hilbert"
    else:
        order_key = "random"
    nw = _analysis_workers_requested(args)
    seqs_orig, tl, _ = _corpus_sequences_ordered(
        records,
        order_key,
        parallel_hilbert_workers=(nw if mode == "hilbert1d" else 1),
    )
    seqs_work = copy.deepcopy(seqs_orig)
    rng = np.random.default_rng(int(getattr(args, "baseline_seed", 0)))
    merges, _ = _train_1d_adjacent_bpe(
        seqs_work,
        len(tok_ref.merge_table),
        int(getattr(args, "baseline_min_freq", 2)),
        int(tok_ref.base_vocab_size),
        rng,
        random_pair=(mode == "random3d"),
    )
    seqs_1d_final: List[List[int]] = []
    seqs_3d_final: List[List[int]] = []
    locality_rows: List[Dict[str, Any]] = []
    rank_gaps_all: List[float] = []
    seq_adj_face_ratios: List[float] = []
    internal_1d: List[float] = []
    internal_3d: List[float] = []
    merge_distances_all: List[float] = []
    total_3d = 0
    chunks = _records_chunk_list(list(records), nw)
    use_mp_baseline = nw > 1 and len(records) >= 200 and len(chunks) > 1
    if use_mp_baseline:
        offsets: List[int] = []
        cur = 0
        for ch in chunks:
            offsets.append(cur)
            cur += len(ch)
        print(
            f"[PARALLEL] baseline metrics: workers={nw}, chunks={len(chunks)}, samples={len(records)}",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=nw,
            initializer=_merge_table_worker_init,
            initargs=(str(merge_table),),
        ) as ex:
            parts = list(
                _progress(
                    ex.map(
                        _mp_baseline_metrics_chunk,
                        [(off, ch, order_key, merges) for off, ch in zip(offsets, chunks)],
                    ),
                    total=len(chunks),
                    desc="[BASELINE] metric chunks",
                )
            )
    else:
        parts = [
            _baseline_metrics_for_records(
                tok_ref,
                records,
                order_key,
                merges,
                0,
                show_progress=True,
            )
        ]

    for part in parts:
        seqs_1d_final.extend(part["seqs_1d_final"])
        seqs_3d_final.extend(part["seqs_3d_final"])
        locality_rows.extend(part["locality_rows"])
        rank_gaps_all.extend(part["rank_gaps_all"])
        seq_adj_face_ratios.extend(part["seq_adj_face_ratios"])
        internal_1d.extend(part["internal_1d"])
        internal_3d.extend(part["internal_3d"])
        merge_distances_all.extend(part["merge_distances_all"])
        total_3d += int(part["total_3d"])

    total_after = sum(len(s) for s in seqs_1d_final)
    ratio = _safe_ratio(total_after, tl)
    reference_3d = _sequence_entropy_summary(seqs_3d_final, tl)
    baseline_1d = _sequence_entropy_summary(seqs_1d_final, tl)
    gap_stats = _summarize_values(rank_gaps_all, "face_edge_rank_gap")
    locality = {
        "face_edge_count": int(len(rank_gaps_all)),
        **gap_stats,
        "face_edge_rank_gap_eq_1_ratio": _safe_ratio(
            sum(1 for g in rank_gaps_all if int(g) == 1), len(rank_gaps_all)
        ),
        "sequence_adjacent_face_ratio": _mean_or_zero(seq_adj_face_ratios),
        "internalized_face_edge_ratio_1d": _mean_or_zero(internal_1d),
        "internalized_face_edge_ratio_3d": _mean_or_zero(internal_3d),
        "internalized_face_edge_ratio_delta_3d_minus_1d": _mean_or_zero(internal_3d)
        - _mean_or_zero(internal_1d),
    }
    merge_dist_stats = _summarize_values(merge_distances_all, "merge_min_l1")
    merge_geometry = {
        "merge_events_observed": int(len(merge_distances_all)),
        **merge_dist_stats,
        "nonlocal_merge_rate": _safe_ratio(
            sum(1 for x in merge_distances_all if int(x) > 1), len(merge_distances_all)
        ),
        "face_touch_merge_rate": _safe_ratio(
            sum(1 for x in merge_distances_all if int(x) == 1), len(merge_distances_all)
        ),
    }
    reference_3d["internalized_face_edge_ratio"] = locality["internalized_face_edge_ratio_3d"]
    baseline_1d["internalized_face_edge_ratio"] = locality["internalized_face_edge_ratio_1d"]
    payload = {
        "baseline": mode,
        "compression_ratio_total": ratio,
        "merges_learned": len(merges),
        "merge_rules_3d_reference": len(tok_ref.merge_table),
        "reference_3d": reference_3d,
        "baseline_1d": baseline_1d,
        "locality": locality,
        "entropy": {
            "unigram_entropy_bits_per_leaf_delta_1d_minus_3d": float(
                baseline_1d["unigram_entropy_bits_per_leaf"]
                - reference_3d["unigram_entropy_bits_per_leaf"]
            ),
            "bigram_cond_entropy_delta_1d_minus_3d": float(
                baseline_1d["mean_bigram_cond_entropy_bits"]
                - reference_3d["mean_bigram_cond_entropy_bits"]
            ),
        },
        "merge_geometry": merge_geometry,
        "interpretation": {
            "rank_gap": "High face-edge rank gaps mean true 3D neighbors are far apart after 1D ordering.",
            "sequence_adjacent_face_ratio": "Low values mean ordinary 1D BPE sees many adjacent pairs that are not 3D face neighbors.",
            "nonlocal_merge_rate": "High values mean 1D BPE actually merges sequence neighbors whose 3D footprints do not touch by a face.",
            "internalized_face_edge_ratio": "Higher 3D than 1D means 3D BPE absorbs more real spatial edges into macro tokens.",
        },
        "note": (
            "1D adjacent BPE on Morton/Hilbert/random vertex order, compared with the loaded 3D BPE merge table "
            "on the same records and merge budget."
        ),
    }
    per_mode_path = out_dir / f"baseline_metrics_{mode}.json"
    per_mode_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    outp = out_dir / "baseline_metrics.json"
    if mode == "morton1d":
        # Legacy single-file name for backward compatibility (plot scripts / docs).
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        outp = per_mode_path
    csv_path = out_dir / "baseline_per_sample.csv"
    if locality_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(locality_rows[0].keys()))
            writer.writeheader()
            writer.writerows(locality_rows)
    print(
        f"[BASELINE] wrote {per_mode_path}"
        + (f" and {out_dir / 'baseline_metrics.json'}" if mode == "morton1d" else "")
        + f" compression_1d={ratio:.6f} "
        f"compression_3d={_safe_ratio(total_3d, tl):.6f} "
        f"nonlocal_merge_rate={merge_geometry['nonlocal_merge_rate']:.6f}",
        flush=True,
    )
    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(5, 4))
            plt.bar([mode], [ratio], color="#bc80bd")
            plt.ylabel("compression_ratio_total")
            plt.title("1D adjacent BPE baseline (length ratio)")
            plt.tight_layout()
            plt.savefig(out_dir / "figure_baseline.png", dpi=180)
            plt.close()
        except Exception as exc:
            print(f"[BASELINE] figure_baseline skipped: {exc}", flush=True)
        _write_baseline_comparison_figure(out_dir, payload)


def run_robust_analysis(args: argparse.Namespace) -> None:
    merge_table = Path(args.merge_table)
    corpus_prefix = Path(args.corpus_prefix)
    out_dir = Path(args.analysis_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = BPE3DTokenizer.load(str(merge_table))
    max_samples = args.max_samples
    records = _load_corpus_records(corpus_prefix, max_samples, tokenizer=tok)
    nw = _analysis_workers_requested(args)
    mp_path = str(merge_table)
    mode = str(args.robust)
    panel = {}
    seed = int(getattr(args, "robust_seed", 0))
    run_all = mode == "all"
    if run_all or mode == "translate":
        r, _ = _compression_ratio_under_transform(
            tok,
            [],
            max_samples,
            records=records,
            coord_shift=(
                int(getattr(args, "robust_dx", 100)),
                int(getattr(args, "robust_dy", 100)),
                int(getattr(args, "robust_dz", 100)),
            ),
            mp_merge_path=mp_path,
            mp_workers=nw,
        )
        panel["translate"] = float(r)
    if run_all or mode == "axis_perm":
        r, _ = _compression_ratio_under_transform(
            tok,
            [],
            max_samples,
            records=records,
            axis_perm=(1, 2, 0),
            mp_merge_path=mp_path,
            mp_workers=nw,
        )
        panel["axis_perm_yzx"] = float(r)
    if run_all or mode == "noise":
        r, _ = _compression_ratio_under_transform(
            tok,
            [],
            max_samples,
            records=records,
            noise_p=float(getattr(args, "robust_noise_p", 0.05)),
            seed=seed,
            mp_merge_path=mp_path,
            mp_workers=nw,
        )
        panel["noise"] = float(r)
    if run_all or mode == "sparsity":
        r, _ = _compression_ratio_under_transform(
            tok,
            [],
            max_samples,
            records=records,
            drop_p=float(getattr(args, "robust_drop_p", 0.1)),
            seed=seed + 1,
            mp_merge_path=mp_path,
            mp_workers=nw,
        )
        panel["sparsity_drop"] = float(r)
    if run_all or mode == "split":
        n = len(records)
        half = max(n // 2, 1)
        r1, _ = _compression_ratio_under_transform(
            tok, [], max_samples, records=records, sample_range=(0, half)
        )
        r2, _ = _compression_ratio_under_transform(
            tok, [], max_samples, records=records, sample_range=(half, n)
        )
        panel["split_first_half"] = float(r1)
        panel["split_second_half"] = float(r2)

    base_r, _ = _compression_ratio_under_transform(
        tok, [], max_samples, records=records, mp_merge_path=mp_path, mp_workers=nw
    )
    out_payload = {
        "baseline_compression": float(base_r),
        "perturbed": panel,
        "parallel_compression_workers": int(nw),
    }
    (out_dir / "robustness_metrics.json").write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.plot:
        _write_figure_robustness(out_dir, panel)
    print(f"[ROBUST] wrote {out_dir / 'robustness_metrics.json'}", flush=True)


def run(args: argparse.Namespace) -> None:
    formal_blocks = bool(getattr(args, "formal_lm_cost", False))
    stability_blocks = bool(getattr(args, "stability_panel", False))
    motif_blocks = bool(getattr(args, "motif_analysis", False))
    if formal_blocks or stability_blocks or motif_blocks:
        merge_table = Path(args.merge_table)
        if not merge_table.is_file():
            raise FileNotFoundError(f"merge table not found: {merge_table}")
        tok_shared = BPE3DTokenizer.load(str(merge_table))
        corpus_prefix = Path(args.corpus_prefix)
        stages = [
            name
            for name, on in (
                ("formal_lm_cost", formal_blocks),
                ("stability_panel", stability_blocks),
                ("motif_analysis", motif_blocks),
            )
            if on
        ]
        print(f"[CORPUS] loading samples once for: {', '.join(stages)} ...", flush=True)
        records_shared = _load_corpus_records(
            corpus_prefix, args.max_samples, tokenizer=tok_shared, verbose=True
        )
        if formal_blocks:
            run_formal_lm_cost(args, tok_full=tok_shared, records=records_shared)
            print("[PIPELINE] formal_lm_cost done", flush=True)
        if stability_blocks:
            run_stability_panel(args, tok_full=tok_shared, records=records_shared)
            print("[PIPELINE] stability_panel done", flush=True)
        if motif_blocks:
            run_motif_analysis(args, tok_full=tok_shared, records=records_shared)
            print("[PIPELINE] motif_analysis done", flush=True)
        return

    if args.sweep_D:
        run_d_sweep(args)
        return
    if args.baseline:
        run_baseline_analysis(args)
        return
    if args.robust:
        run_robust_analysis(args)
        return
    if args.analyze_bpe_outputs:
        run_bpe_output_analysis(args)
        return

    sparse_cls = DummySparseTensor
    if args.with_trellis:
        from trellis.modules import sparse as sp

        sparse_cls = sp.SparseTensor

    structured = build_structured_corpus()
    tok_struct = BPE3DTokenizer(base_vocab_size=8192)
    tok_struct.train(structured, num_merges=32, min_freq=2, verbose=True)
    assert len(tok_struct.merge_table) > 0, "structured merge_table is empty"
    assert_roundtrip(tok_struct, structured, sparse_cls)

    corpus = build_corpus(
        n_samples=args.n_samples,
        coord_max=args.coord_max,
        points_per_sample=args.points_per_sample,
        seed=args.seed,
    )
    tok = BPE3DTokenizer(base_vocab_size=8192)
    tok.train(corpus, num_merges=args.num_merges, min_freq=args.min_freq, verbose=True)
    assert len(tok.merge_table) > 0, "merge_table is empty; expected at least one learned merge"

    total_before, total_after = assert_roundtrip(tok, corpus, sparse_cls)

    ratio = (total_after / max(total_before, 1)) if total_before > 0 else 1.0
    print(f"[TEST] compression ratio={ratio:.6f} ({total_after}/{total_before})")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "merge_table.json"
        tok.save(str(p))
        tok2 = BPE3DTokenizer.load(str(p))

        for sample in corpus[: min(5, len(corpus))]:
            st = to_sparse(sample, sparse_cls)
            out1 = tok.encode_sparse(st, sparse_tensor_cls=sparse_cls)
            out2 = tok2.encode_sparse(st, sparse_tensor_cls=sparse_cls)
            ids1 = out1["batches"][0]["ids"]
            ids2 = out2["batches"][0]["ids"]
            anc1 = out1["batches"][0]["anchors"]
            anc2 = out2["batches"][0]["anchors"]
            assert np.array_equal(ids1, ids2), "save/load mismatch on ids"
            assert np.array_equal(anc1, anc2), "save/load mismatch on anchors"

            dec2 = tok2.decode_to_sparse(out2["batches"], device=torch.device("cpu"), sparse_tensor_cls=sparse_cls)
            assert sparse_item_set(dec2) == sparse_item_set(st), "loaded tokenizer decode mismatch"

    print("[TEST] PASS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic end-to-end test for 3D BPE tokenizer")
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--points_per_sample", type=int, default=300)
    p.add_argument("--num_merges", type=int, default=128)
    p.add_argument("--min_freq", type=int, default=5)
    p.add_argument("--coord_max", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with_trellis", action="store_true")
    p.add_argument("--analyze_bpe_outputs", action="store_true")
    p.add_argument(
        "--corpus_prefix",
        type=str,
        default="./pad_outputs/qwen3vl_3d_sft_8cat.jsonl",
        help="Phase-1 prefix for *.corpus.part*.npz shards, OR path to a merged mesh .jsonl when npz is absent (needs --merge_table for decode).",
    )
    p.add_argument("--merge_table", type=str, default="./pad_outputs/merge_table_rel_offset.json")
    p.add_argument("--analysis_out_dir", type=str, default="./pad_outputs/bpe_analysis")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument(
        "--analysis_num_workers",
        type=int,
        default=0,
        help="并行 worker 数（stability / motif / robust 压缩统计）；0 表示自动 min(CPU,32)",
    )
    p.add_argument("--plot", action="store_true")
    p.add_argument("--analysis_print_every", type=int, default=1000)
    # Extended analysis (compatible numpy+matplotlib recommended for --plot)
    p.add_argument("--bootstrap_n", type=int, default=0, help="bootstrap iterations for CI (0=skip)")
    p.add_argument("--bootstrap_seed", type=int, default=0)
    p.add_argument("--reliability_check", action="store_true", help="encode/decode roundtrip + save/load id check")
    p.add_argument("--reliability_max_samples", type=int, default=None)
    p.add_argument("--reliability_print_every", type=int, default=1000)
    p.add_argument("--viz_sample_index", type=int, default=0, help="sample index for voxel motif figure")
    p.add_argument(
        "--sweep_D",
        type=str,
        default="",
        metavar="K_LIST",
        help="comma-separated merge counts; runs D-sweep only (writes metrics_K*.json, figure_D_sweep.png)",
    )
    p.add_argument(
        "--baseline",
        type=str,
        default="",
        metavar="MODE",
        help="morton1d | hilbert1d | random3d (1D adjacent BPE baseline); runs baseline only",
    )
    p.add_argument("--baseline_seed", type=int, default=0)
    p.add_argument("--baseline_min_freq", type=int, default=2)
    p.add_argument(
        "--robust",
        type=str,
        default="",
        metavar="MODE",
        help="translate | axis_perm | noise | sparsity | split | all",
    )
    p.add_argument("--robust_seed", type=int, default=0)
    p.add_argument("--robust_noise_p", type=float, default=0.05)
    p.add_argument("--robust_drop_p", type=float, default=0.1)
    p.add_argument("--robust_dx", type=int, default=100)
    p.add_argument("--robust_dy", type=int, default=100)
    p.add_argument("--robust_dz", type=int, default=100)
    p.add_argument(
        "--formal_lm_cost",
        action="store_true",
        help="write formal_lm_cost.json (+ figures): LM proxy, n-grams, k-fold/held-out NLL",
    )
    p.add_argument(
        "--stability_panel",
        action="store_true",
        help="write stability_metrics.json (+ figures): paired bootstrap, strata, multi-seed robustness CI",
    )
    p.add_argument(
        "--motif_analysis",
        action="store_true",
        help="write motif_metrics.json, motif_instances.csv (+ figures): footprints, coverage, controls",
    )
    p.add_argument(
        "--formal_lm_sweep_K",
        type=str,
        default="",
        metavar="K_LIST",
        help="comma merge counts for formal LM sweep (empty => single K=full merge_table)",
    )
    p.add_argument(
        "--formal_out_prefix",
        type=str,
        default="",
        help="directory for formal/stability/motif outputs (empty uses --analysis_out_dir)",
    )
    p.add_argument("--folds", type=int, default=5, help="K-fold for mesh unigram NLL in formal_lm_cost")
    p.add_argument(
        "--control_seeds",
        type=str,
        default="0,1,2,3,4",
        help="comma seeds for stability robustness panel & motif random controls",
    )
    p.add_argument(
        "--motif_controls",
        type=str,
        default="",
        help="comma modes among token_shuffle,coord_shuffle,random_pair_merge (empty skips)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
