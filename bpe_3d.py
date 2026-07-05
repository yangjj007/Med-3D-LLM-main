from __future__ import annotations

"""
3D BPE tokenizer for SparseSDFVQVAE block-level Encode outputs.

Expected VQVAE output:
- coords: [N, 4] as (batch_idx, bx, by, bz)
- feats:  [N, 1] codebook ids

Typical wiring:
1) LLM input side
    enc_st = vqvae.Encode(batch)
    out = tokenizer.encode_sparse(enc_st)
    # out["batches"][b]["ids"] -> "<mesh_start>...<mesh_end>"

2) LLM output side
    # parse ids + anchors back from generation
    leaf_st = tokenizer.decode_to_sparse(batches, device=device)
    recon = vqvae.Decode(leaf_st)

Notes:
- BPE is learned in block-coordinate space.
- Adjacency uses directed +x/+y/+z.
- New ids are assigned from base_vocab_size upward.

Performance notes:
- Default training uses incremental NumPy-friendly updates (train_mode=incremental).
- Legacy multiprocess path (train_mode=legacy): num_workers=0 → auto
  min(os.cpu_count(), 8) capped to avoid massive IPC overhead; 1 = single-threaded.
- On Linux/macOS uses "fork" (near-zero startup cost, copy-on-write memory).
- On Windows uses "spawn" (one-time pickle overhead per worker chunk at startup).
- Double-traversal eliminated in legacy: pair matches computed once per iteration
  per sample and reused in the merge step.

Tie-breaking: among equal global pair frequency, the pair with lexicographically
smallest (a_tok, b_tok, rel_x, rel_y, rel_z) is chosen (deterministic; replaces Counter.most_common).
"""

from dataclasses import dataclass, asdict
import heapq
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

_logger = logging.getLogger(__name__)

_DIR_TO_VEC: Dict[int, Tuple[int, int, int]] = {
    0: (1, 0, 0),  # +x
    1: (0, 1, 0),  # +y
    2: (0, 0, 1),  # +z
}

MORTON_GRID_SIZE = 32
MORTON_BITS = 5
MORTON_MAX_CODE = (1 << (MORTON_BITS * 3)) - 1
MORTON_TOKEN_RE = re.compile(r"<morton_(\d+)>")
MESH_TOKEN_RE = re.compile(r"<mesh_(\d+)>")
MORTON_MESH_PAIR_RE = re.compile(r"<morton_(\d+)>\s*<mesh_(\d+)>")


# Heuristic RAM model for incremental BPE (graph build dominates). Calibrated on large corpora.
_BYTES_PER_POINT_CORPUS = 32  # int64 token + int64[3] coord per cell
_BYTES_PER_POINT_GRAPH_LOW = 220
_BYTES_PER_POINT_GRAPH_TYPICAL = 380
_BYTES_PER_POINT_GRAPH_HIGH = 620
_BYTES_PER_SAMPLE_OVERHEAD = 8192
_MERGE_PHASE_OVERHEAD_RATIO = 0.25
_LEGACY_GRAPH_MULTIPLIER = 1.8
_MEMORY_WARN_AVAILABLE_RATIO = 0.85


def _bpe_defer_heap_enabled() -> bool:
    """Defer pair-heap updates during graph build / merge apply (flush dirty keys per iter)."""
    return os.environ.get("BPE3D_DEFER_HEAP", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _bpe_heap_flush_rebuild() -> bool:
    """If True, end each deferred merge iter with full heapify (debug / regression fallback)."""
    return os.environ.get("BPE3D_HEAP_FLUSH", "dirty").strip().lower() == "rebuild"


@dataclass
class BPEMemoryEstimate:
    """Heuristic RAM estimate for BPE training (GiB)."""

    n_samples: int
    total_points: int
    train_mode: str
    corpus_array_gib: float
    graph_build_gib_low: float
    graph_build_gib_typical: float
    graph_build_gib_high: float
    merge_overhead_gib_typical: float
    total_typical_gib: float
    total_low_gib: float
    total_high_gib: float
    bytes_per_point_graph: int = _BYTES_PER_POINT_GRAPH_TYPICAL


def _bytes_to_gib(n_bytes: float) -> float:
    return float(n_bytes) / (1024.0**3)


def _format_gib(gib: Optional[float]) -> str:
    if gib is None:
        return "n/a"
    if gib >= 1024.0:
        return f"{gib / 1024.0:.2f} TiB"
    return f"{gib:.2f} GiB"


def _format_gib_compact(gib: Optional[float]) -> str:
    """Short GiB/TiB label for tqdm postfix."""
    if gib is None:
        return "?"
    if gib >= 1024.0:
        return f"{gib / 1024.0:.2f}T"
    if gib >= 100.0:
        return f"{gib:.0f}G"
    return f"{gib:.1f}G"


def get_process_rss_bytes() -> Optional[int]:
    """Current process RSS in bytes."""
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    if sys.platform != "win32":
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # Linux reports VmRSS in kB
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    return None


def get_process_rss_gib() -> Optional[float]:
    rss_b = get_process_rss_bytes()
    if rss_b is None:
        return None
    return _bytes_to_gib(float(rss_b))


def estimate_rss_from_cells_progress(
    cells_built: int,
    total_cells: int,
    rss_gib: Optional[float],
    *,
    merge_overhead_ratio: float = _MERGE_PHASE_OVERHEAD_RATIO,
) -> Dict[str, Optional[float]]:
    """
    Linear extrapolation: process RSS grows ~proportionally to cells built during graph build.

    Returns rss_gib, est_graph_end_gib (projected RSS when all cells are built),
    est_peak_gib (graph end + merge overhead), bytes_per_cell_observed.
    """
    cells_built = int(cells_built)
    total_cells = int(total_cells)
    out: Dict[str, Optional[float]] = {
        "rss_gib": rss_gib,
        "est_graph_end_gib": None,
        "est_peak_gib": None,
        "bytes_per_cell": None,
    }
    if rss_gib is None or cells_built <= 0 or total_cells <= 0:
        return out
    frac = float(cells_built) / float(total_cells)
    if frac <= 0.0:
        return out
    out["bytes_per_cell"] = float(rss_gib) * (1024.0**3) / float(cells_built)
    out["est_graph_end_gib"] = float(rss_gib) / frac
    out["est_peak_gib"] = out["est_graph_end_gib"] * (1.0 + float(merge_overhead_ratio))
    return out


def format_graph_build_memory_postfix(
    cells_built: int,
    total_cells: int,
    *,
    merge_overhead_ratio: float = _MERGE_PHASE_OVERHEAD_RATIO,
) -> Dict[str, str]:
    """tqdm postfix: cells progress + measured RSS + extrapolated graph/peak RSS."""
    cells_built = int(cells_built)
    total_cells = int(total_cells)
    post: Dict[str, str] = {
        "cells": (
            f"{cells_built / 1e6:.1f}/{total_cells / 1e6:.0f}M"
            if total_cells > 0
            else f"{cells_built / 1e6:.1f}M"
        ),
    }
    rss = get_process_rss_gib()
    if rss is not None:
        post["rss"] = _format_gib_compact(rss)
    min_cells = max(500_000, total_cells // 2000) if total_cells > 0 else 500_000
    if rss is not None and cells_built >= min_cells and total_cells > 0:
        est = estimate_rss_from_cells_progress(
            cells_built,
            total_cells,
            rss,
            merge_overhead_ratio=merge_overhead_ratio,
        )
        if est["est_graph_end_gib"] is not None:
            post["~graph"] = _format_gib_compact(est["est_graph_end_gib"])
        if est["est_peak_gib"] is not None:
            post["~peak"] = _format_gib_compact(est["est_peak_gib"])
    return post


def get_system_memory_info() -> Dict[str, Any]:
    """Return host RAM stats in GiB. Uses psutil when installed, else /proc or WinAPI."""
    try:
        import psutil  # type: ignore

        v = psutil.virtual_memory()
        return {
            "total_gib": _bytes_to_gib(v.total),
            "available_gib": _bytes_to_gib(v.available),
            "used_gib": _bytes_to_gib(v.used),
            "percent": float(v.percent),
            "source": "psutil",
        }
    except ImportError:
        pass

    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total = _bytes_to_gib(stat.ullTotalPhys)
                avail = _bytes_to_gib(stat.ullAvailPhys)
                used = max(0.0, total - avail)
                return {
                    "total_gib": total,
                    "available_gib": avail,
                    "used_gib": used,
                    "percent": float(stat.dwMemoryLoad),
                    "source": "GlobalMemoryStatusEx",
                }
        except Exception:
            pass

    if sys.platform != "win32":
        try:
            meminfo: Dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    key, rest = line.split(":", 1)
                    meminfo[key.strip()] = int(rest.split()[0])
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            total = total_kb / (1024.0**2)
            avail = avail_kb / (1024.0**2)
            used = max(0.0, total - avail)
            pct = 100.0 * used / total if total > 0 else 0.0
            return {
                "total_gib": total,
                "available_gib": avail,
                "used_gib": used,
                "percent": pct,
                "source": "/proc/meminfo",
            }
        except OSError:
            pass

    return {
        "total_gib": None,
        "available_gib": None,
        "used_gib": None,
        "percent": None,
        "source": "unknown",
    }


def estimate_bpe_training_memory(
    n_samples: int,
    total_points: int,
    train_mode: str = "incremental",
) -> BPEMemoryEstimate:
    """Heuristic peak-RAM estimate for BPE training (graph build is the dominant term)."""
    n_samples = int(n_samples)
    total_points = int(total_points)
    mode = str(train_mode).strip().lower()
    graph_low_b = _BYTES_PER_POINT_GRAPH_LOW
    graph_typ_b = _BYTES_PER_POINT_GRAPH_TYPICAL
    graph_high_b = _BYTES_PER_POINT_GRAPH_HIGH
    if mode == "legacy":
        graph_low_b = int(graph_low_b * _LEGACY_GRAPH_MULTIPLIER)
        graph_typ_b = int(graph_typ_b * _LEGACY_GRAPH_MULTIPLIER)
        graph_high_b = int(graph_high_b * _LEGACY_GRAPH_MULTIPLIER)

    corpus_b = (
        total_points * _BYTES_PER_POINT_CORPUS + n_samples * _BYTES_PER_SAMPLE_OVERHEAD
    )
    graph_low_b_total = total_points * graph_low_b + n_samples * _BYTES_PER_SAMPLE_OVERHEAD
    graph_typ_b_total = total_points * graph_typ_b + n_samples * _BYTES_PER_SAMPLE_OVERHEAD
    graph_high_b_total = total_points * graph_high_b + n_samples * _BYTES_PER_SAMPLE_OVERHEAD
    merge_typ_b = graph_typ_b_total * _MERGE_PHASE_OVERHEAD_RATIO

    total_typ = corpus_b + graph_typ_b_total + merge_typ_b
    total_low = corpus_b + graph_low_b_total + graph_low_b_total * _MERGE_PHASE_OVERHEAD_RATIO
    total_high = corpus_b + graph_high_b_total + graph_high_b_total * _MERGE_PHASE_OVERHEAD_RATIO

    return BPEMemoryEstimate(
        n_samples=n_samples,
        total_points=total_points,
        train_mode=mode,
        corpus_array_gib=_bytes_to_gib(corpus_b),
        graph_build_gib_low=_bytes_to_gib(graph_low_b_total),
        graph_build_gib_typical=_bytes_to_gib(graph_typ_b_total),
        graph_build_gib_high=_bytes_to_gib(graph_high_b_total),
        merge_overhead_gib_typical=_bytes_to_gib(merge_typ_b),
        total_typical_gib=_bytes_to_gib(total_typ),
        total_low_gib=_bytes_to_gib(total_low),
        total_high_gib=_bytes_to_gib(total_high),
        bytes_per_point_graph=graph_typ_b,
    )


def print_bpe_memory_report(
    estimate: BPEMemoryEstimate,
    mem_info: Optional[Dict[str, Any]] = None,
    *,
    verbose: bool = True,
    abort_if_insufficient: Optional[bool] = None,
) -> bool:
    """
    Print system RAM + heuristic BPE RAM estimate. Returns True if likely sufficient.

    Set env BPE_MEMORY_ABORT=1 to raise SystemExit when typical estimate exceeds available RAM.
    """
    if mem_info is None:
        mem_info = get_system_memory_info()
    if abort_if_insufficient is None:
        abort_if_insufficient = os.environ.get("BPE_MEMORY_ABORT", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    total_gib = mem_info.get("total_gib")
    avail_gib = mem_info.get("available_gib")
    used_gib = mem_info.get("used_gib")
    pct = mem_info.get("percent")
    src = mem_info.get("source", "unknown")

    lines = [
        "[BPE3D][memory] ---- host RAM ----",
        (
            f"  total={_format_gib(total_gib)}  available={_format_gib(avail_gib)}  "
            f"used={_format_gib(used_gib)}"
            + (f"  load={pct:.1f}%" if pct is not None else "")
            + f"  (source: {src})"
        ),
        "[BPE3D][memory] ---- corpus ----",
        (
            f"  samples={estimate.n_samples:,}  points={estimate.total_points:,}  "
            f"mode={estimate.train_mode}"
        ),
        f"[BPE3D][memory] ---- estimated peak RAM ({estimate.train_mode} BPE) ----",
        f"  corpus arrays (in memory):  {_format_gib(estimate.corpus_array_gib)}",
        (
            f"  graph build (typical):      {_format_gib(estimate.graph_build_gib_typical)}  "
            f"[{_format_gib(estimate.graph_build_gib_low)} – "
            f"{_format_gib(estimate.graph_build_gib_high)}]"
        ),
        f"  merge-train overhead (est): {_format_gib(estimate.merge_overhead_gib_typical)}",
        (
            f"  total peak (typical):       {_format_gib(estimate.total_typical_gib)}  "
            f"[{_format_gib(estimate.total_low_gib)} – {_format_gib(estimate.total_high_gib)}]"
        ),
        (
            f"  (static heuristic: ~{estimate.bytes_per_point_graph} B/cell graph + "
            f"{_BYTES_PER_POINT_CORPUS} B/cell corpus; often underestimates Python graphs)"
        ),
        "  (live: graph-build tqdm shows rss / ~graph / ~peak from RSS×cells extrapolation)",
    ]

    likely_ok = True
    if avail_gib is not None:
        warn_thresh = float(avail_gib) * _MEMORY_WARN_AVAILABLE_RATIO
        if estimate.total_typical_gib > avail_gib:
            likely_ok = False
            lines.append(
                "[BPE3D][memory] WARNING: typical estimate exceeds currently available RAM."
            )
        elif estimate.total_typical_gib > warn_thresh:
            lines.append(
                "[BPE3D][memory] WARNING: typical estimate is close to available RAM "
                f"(>{_MEMORY_WARN_AVAILABLE_RATIO:.0%} of available)."
            )
        if estimate.total_high_gib > (total_gib or float("inf")):
            lines.append(
                "[BPE3D][memory] WARNING: high estimate exceeds physical RAM — OOM is likely."
            )
            likely_ok = False

    if not likely_ok:
        lines.extend(
            [
                "[BPE3D][memory] Suggestions:",
                "  - Train merge_table on a subset (train_bpe_3d.py --corpus_cache subset.npz),",
                "    then run build_qwen3vl_sft_3d_jsonl.py Phase-3 with the saved merge_table.",
                "  - Use a machine with more RAM, or add swap (slow).",
                "  - Set BPE_MEMORY_ABORT=1 to stop instead of continuing.",
            ]
        )

    text = "\n".join(lines)
    if verbose:
        print(text, flush=True)
    _logger.info("%s", text.replace("\n", " | "))

    if not likely_ok and abort_if_insufficient:
        raise SystemExit(
            "[BPE3D][memory] Aborting: insufficient RAM (BPE_MEMORY_ABORT=1). "
            "See suggestions above."
        )
    return likely_ok


@dataclass
class MergeEntry:
    new_id: int
    pair: Tuple[int, int]
    direction: int
    rel_offset: Tuple[int, int, int]
    leaf_tokens: List[int]
    leaf_offsets: List[Tuple[int, int, int]]


@dataclass
class _TokenSlot:
    token_id: int
    cells: List[Tuple[int, int, int]]
    anchor: Tuple[int, int, int]


@dataclass
class MortonMeshParseResult:
    ids: np.ndarray
    anchors: np.ndarray
    dropped_count: int = 0


def _as_int64_array(x: Any, name: str, ndim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape={arr.shape}")
    return arr


def morton3d_encode_xyz32(coords: Any) -> Any:
    """Encode block coordinates ``[0, 31]^3`` with fixed x/y/z bit interleave.

    Bit order is fixed for the whole project:
    ``x_i -> bit 3*i``, ``y_i -> bit 3*i+1``, ``z_i -> bit 3*i+2``.
    """
    arr = np.asarray(coords, dtype=np.int64)
    single = arr.ndim == 1
    if single:
        if arr.shape[0] != 3:
            raise ValueError(f"coords must be [3] or [N,3], got shape={arr.shape}")
        arr2 = arr.reshape(1, 3)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        arr2 = arr
    else:
        raise ValueError(f"coords must be [3] or [N,3], got shape={arr.shape}")
    if arr2.size and ((arr2 < 0).any() or (arr2 >= MORTON_GRID_SIZE).any()):
        raise ValueError(
            f"Morton coordinates must be in [0, {MORTON_GRID_SIZE - 1}], "
            f"got min={arr2.min()} max={arr2.max()}"
        )
    x = arr2[:, 0]
    y = arr2[:, 1]
    z = arr2[:, 2]
    code = np.zeros(arr2.shape[0], dtype=np.int64)
    for bit in range(MORTON_BITS):
        code |= ((x >> bit) & 1) << (3 * bit)
        code |= ((y >> bit) & 1) << (3 * bit + 1)
        code |= ((z >> bit) & 1) << (3 * bit + 2)
    return int(code[0]) if single else code


def morton3d_decode_xyz32(codes: Any) -> Any:
    """Decode fixed 15-bit Morton code(s) back to block coordinates."""
    arr = np.asarray(codes, dtype=np.int64)
    single = arr.ndim == 0
    arr1 = arr.reshape(-1)
    if arr1.size and ((arr1 < 0).any() or (arr1 > MORTON_MAX_CODE).any()):
        raise ValueError(
            f"Morton codes must be in [0, {MORTON_MAX_CODE}], "
            f"got min={arr1.min()} max={arr1.max()}"
        )
    coords = np.zeros((arr1.shape[0], 3), dtype=np.int64)
    for bit in range(MORTON_BITS):
        coords[:, 0] |= ((arr1 >> (3 * bit)) & 1) << bit
        coords[:, 1] |= ((arr1 >> (3 * bit + 1)) & 1) << bit
        coords[:, 2] |= ((arr1 >> (3 * bit + 2)) & 1) << bit
    return coords[0] if single else coords


def serialize_morton_mesh_pairs(ids: Sequence[int], anchors: Any) -> str:
    """Serialize BPE/VQ ids with explicit block-anchor Morton tokens."""
    ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
    anchors_np = _as_int64_array(anchors, "anchors", 2)
    if anchors_np.shape[1] != 3:
        raise ValueError(f"anchors must be [K,3], got {anchors_np.shape}")
    if ids_np.shape[0] != anchors_np.shape[0]:
        raise ValueError("ids and anchors length mismatch")
    if ids_np.size == 0:
        return "<mesh_start><mesh_end>"
    morton_codes = morton3d_encode_xyz32(anchors_np)
    parts = [
        f"<morton_{m}><mesh_{t}>"
        for m, t in zip(morton_codes.tolist(), ids_np.tolist())
    ]
    return "<mesh_start>" + "".join(parts) + "<mesh_end>"


def parse_morton_mesh_pairs(text: str, max_mesh_id: Optional[int] = None) -> MortonMeshParseResult:
    """Parse valid ``<morton_*><mesh_*>`` pairs, dropping damaged fragments.

    Only adjacent Morton/Mesh pairs are accepted. Lone Morton tokens, lone mesh
    tokens, out-of-range Morton codes, and mesh ids >= ``max_mesh_id`` are
    skipped and counted in ``dropped_count``.
    """
    ids: List[int] = []
    morton_codes: List[int] = []
    used_morton_spans = set()
    used_mesh_spans = set()
    dropped = 0

    for match in MORTON_MESH_PAIR_RE.finditer(text):
        morton = int(match.group(1))
        mesh_id = int(match.group(2))
        used_morton_spans.add((match.start(1), match.end(1)))
        used_mesh_spans.add((match.start(2), match.end(2)))
        if morton < 0 or morton > MORTON_MAX_CODE:
            dropped += 1
            continue
        if max_mesh_id is not None and mesh_id >= int(max_mesh_id):
            dropped += 1
            continue
        morton_codes.append(morton)
        ids.append(mesh_id)

    total_morton = len(list(MORTON_TOKEN_RE.finditer(text)))
    total_mesh = len(list(MESH_TOKEN_RE.finditer(text)))
    dropped += max(0, total_morton - len(used_morton_spans))
    dropped += max(0, total_mesh - len(used_mesh_spans))

    if ids:
        anchors = morton3d_decode_xyz32(np.asarray(morton_codes, dtype=np.int64))
        ids_np = np.asarray(ids, dtype=np.int64)
    else:
        anchors = np.zeros((0, 3), dtype=np.int64)
        ids_np = np.zeros((0,), dtype=np.int64)
    return MortonMeshParseResult(ids=ids_np, anchors=anchors, dropped_count=dropped)


def _morton_order(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.array([], dtype=np.int64)
    cmax = int(np.max(coords))
    return _morton_sort_indices(coords.astype(np.int64), coord_max=max(cmax + 1, 1))


def _morton_sort_indices(coords_xyz: np.ndarray, coord_max: int) -> np.ndarray:
    n = coords_xyz.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    bits = max(1, int(np.ceil(np.log2(coord_max + 1))))
    x = coords_xyz[:, 0].astype(np.int64)
    y = coords_xyz[:, 1].astype(np.int64)
    z = coords_xyz[:, 2].astype(np.int64)
    bit_idx = np.arange(bits, dtype=np.int64)
    x_bits = ((x[:, None] >> bit_idx) & 1) << (3 * bit_idx)
    y_bits = ((y[:, None] >> bit_idx) & 1) << (3 * bit_idx + 1)
    z_bits = ((z[:, None] >> bit_idx) & 1) << (3 * bit_idx + 2)
    code = x_bits.sum(axis=1) + y_bits.sum(axis=1) + z_bits.sum(axis=1)
    return np.argsort(code, kind="stable")


def _morton_min_cell(cells: Sequence[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    coords = np.asarray(cells, dtype=np.int64)
    order = _morton_order(coords)
    return tuple(int(v) for v in coords[order[0]].tolist())


def _canonical_min_cell(cells: Sequence[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Translation-invariant anchor for a macro token."""
    return min((int(c[0]), int(c[1]), int(c[2])) for c in cells)


def _canonical_min_offset(
    offsets: Sequence[Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    return min((int(c[0]), int(c[1]), int(c[2])) for c in offsets)


def _add3(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub3(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


PairKey = Tuple[int, int, int, int, int]


def _make_pair_key(
    a_tok: int,
    b_tok: int,
    a_anchor: Sequence[int],
    b_anchor: Sequence[int],
) -> PairKey:
    return (
        int(a_tok),
        int(b_tok),
        int(b_anchor[0]) - int(a_anchor[0]),
        int(b_anchor[1]) - int(a_anchor[1]),
        int(b_anchor[2]) - int(a_anchor[2]),
    )


def _pair_key_parts(key: PairKey) -> Tuple[int, int, Tuple[int, int, int]]:
    return int(key[0]), int(key[1]), (int(key[2]), int(key[3]), int(key[4]))


# Default cap when num_workers=0 (avoids hundreds of tiny chunks + Queue IPC).
_DEFAULT_BPE_WORKER_CAP = 8


def _encode_pair_int(V: int, a: int, b: int, d: int) -> int:
    return ((int(a) * int(V) + int(b)) * 3) + int(d)


def _decode_pair_int(pk: int, V: int) -> Tuple[int, int, int]:
    d = int(pk % 3)
    x = pk // 3
    b = int(x % V)
    a = int(x // V)
    return a, b, d


def _best_pair_tuple_from_counter(
    pair_freq: Dict[PairKey, int],
) -> Tuple[PairKey, int]:
    if not pair_freq:
        raise ValueError("empty pair_freq")
    best_cnt = max(pair_freq.values())
    cands = [k for k, v in pair_freq.items() if v == best_cnt]
    best_k = min(cands)  # lex (a,b,d)
    return best_k, best_cnt


def _best_pair_int_from_freq(pair_freq: Dict[int, int], V: int) -> Tuple[int, int]:
    if not pair_freq:
        raise ValueError("empty pair_freq")
    best_cnt = max(pair_freq.values())
    cands = [k for k, v in pair_freq.items() if v == best_cnt]
    best_pk = min(cands, key=lambda pk: _decode_pair_int(pk, V))
    return best_pk, best_cnt


# ---------------------------------------------------------------------------
# Module-level parallel-safe functions (must be top-level for pickle / spawn)
# ---------------------------------------------------------------------------

def _collect_pairs_fn(
    slots: List[_TokenSlot],
    directions: Tuple[int, ...],
) -> Dict[PairKey, List[Tuple[int, int]]]:
    """Collect adjacent pair matches for one sample.  Module-level so it is
    picklable when used in worker processes."""
    cell_to_slot: Dict[Tuple[int, int, int], int] = {}
    for idx, slot in enumerate(slots):
        for c in slot.cells:
            cell_to_slot[c] = idx

    pair_to_matches: Dict[PairKey, List[Tuple[int, int]]] = {}
    pair_seen: Dict[PairKey, set] = {}
    for a_idx, a_slot in enumerate(slots):
        for c in a_slot.cells:
            for d in directions:
                dv = _DIR_TO_VEC[d]
                n = (c[0] + dv[0], c[1] + dv[1], c[2] + dv[2])
                b_idx = cell_to_slot.get(n)
                if b_idx is None or b_idx == a_idx:
                    continue
                b_slot = slots[b_idx]
                key = _make_pair_key(
                    a_slot.token_id, b_slot.token_id, a_slot.anchor, b_slot.anchor
                )
                slot_pair = (a_idx, b_idx)
                if key not in pair_seen:
                    pair_seen[key] = set()
                if slot_pair in pair_seen[key]:
                    continue
                pair_seen[key].add(slot_pair)
                if key not in pair_to_matches:
                    pair_to_matches[key] = []
                pair_to_matches[key].append(slot_pair)
    return pair_to_matches


def _apply_merge_fn(
    slots: List[_TokenSlot],
    pair_to_matches: Dict[PairKey, List[Tuple[int, int]]],
    target_key: PairKey,
    new_id: int,
) -> List[_TokenSlot]:
    """Apply merge using pre-computed pair_to_matches (no redundant traversal)."""
    candidates = pair_to_matches.get(target_key, [])
    if not candidates:
        return slots

    if len(candidates) > 1:
        a_anchors = np.asarray([slots[p[0]].anchor for p in candidates], dtype=np.int64)
        order = _morton_order(a_anchors)
        candidates = [candidates[int(i)] for i in order.tolist()]

    used: set = set()
    replacement: Dict[int, _TokenSlot] = {}
    skip: set = set()
    for a_idx, b_idx in candidates:
        if a_idx in used or b_idx in used:
            continue
        used.add(a_idx)
        used.add(b_idx)
        merged_cells = sorted(set(slots[a_idx].cells) | set(slots[b_idx].cells))
        replacement[a_idx] = _TokenSlot(
            token_id=new_id,
            cells=merged_cells,
            anchor=_canonical_min_cell(merged_cells),
        )
        skip.add(b_idx)

    out: List[_TokenSlot] = []
    for idx, slot in enumerate(slots):
        if idx in skip:
            continue
        out.append(replacement[idx] if idx in replacement else slot)
    return out


def _worker_loop(
    samples_chunk: List[List[_TokenSlot]],
    directions: Tuple[int, ...],
    cmd_queue: Any,
    res_queue: Any,
) -> None:
    """Persistent BPE worker process.

    Protocol (repeated for each BPE iteration):
      recv ("count",)           → compute pair frequencies → send Counter
      recv ("merge", key, id)   → apply merge reusing cached pairs → send "done"
      recv None                 → exit

    Pair matches computed in the "count" step are cached and reused in the
    immediately following "merge" step, eliminating double traversal entirely.
    """
    chunk = list(samples_chunk)
    cached_pairs: Optional[List[Dict]] = None

    while True:
        try:
            cmd = cmd_queue.get()
        except Exception:
            break
        if cmd is None:
            break

        op = cmd[0]
        if op == "count":
            freq: Counter = Counter()
            local_cache: List[Dict] = []
            for slots in chunk:
                pairs = _collect_pairs_fn(slots, directions)
                local_cache.append(pairs)
                for k, v in pairs.items():
                    freq[k] += len(v)
            cached_pairs = local_cache
            res_queue.put(freq)

        elif op == "merge":
            _, target_key, new_id = cmd
            if cached_pairs is not None and len(cached_pairs) == len(chunk):
                for i, slots in enumerate(chunk):
                    chunk[i] = _apply_merge_fn(slots, cached_pairs[i], target_key, new_id)
            else:
                for i, slots in enumerate(chunk):
                    pairs = _collect_pairs_fn(slots, directions)
                    chunk[i] = _apply_merge_fn(slots, pairs, target_key, new_id)
            cached_pairs = None
            res_queue.put("done")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_chunks(lst: List, n: int) -> List[List]:
    """Split list into at most n roughly equal non-empty chunks."""
    if not lst:
        return []
    k = min(n, len(lst))
    size = len(lst)
    chunks = []
    for i in range(k):
        a = size * i // k
        b = size * (i + 1) // k
        if a < b:
            chunks.append(lst[a:b])
    return chunks


def _resolve_workers(num_workers: int, n_samples: int) -> int:
    """Resolve effective worker count.

    0 → min(os.cpu_count(), 8, n_samples) by default (see _DEFAULT_BPE_WORKER_CAP).
    """
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if num_workers == 0:
        available = os.cpu_count() or 1
        capped = min(available, _DEFAULT_BPE_WORKER_CAP)
        return min(capped, max(1, n_samples))
    return min(num_workers, max(1, n_samples))


# ---------------------------------------------------------------------------
# Incremental BPE training (default)
# ---------------------------------------------------------------------------


class _IncSampleState:
    """Mutable per-sample state for incremental BPE (slot graph on a fixed cell set)."""

    __slots__ = (
        "sample_id",
        "M",
        "cell_xyz",
        "directions",
        "coord_to_idx",
        "slot_of_cell",
        "token_of_slot",
        "slot_anchor",
        "cells_by_slot",
        "pair_slot_pairs",
        "incidence",
    )

    def __init__(
        self,
        sample_id: int,
        tokens: np.ndarray,
        coords: np.ndarray,
        directions: Tuple[int, ...],
        trainer: "_IncrementalBPETrainer",
    ) -> None:
        self.sample_id = sample_id
        self.M = int(tokens.shape[0])
        self.cell_xyz = coords.astype(np.int64, copy=False)
        self.directions = directions
        self.coord_to_idx: Dict[Tuple[int, int, int], int] = {}
        for i in range(self.M):
            t = (
                int(self.cell_xyz[i, 0]),
                int(self.cell_xyz[i, 1]),
                int(self.cell_xyz[i, 2]),
            )
            self.coord_to_idx[t] = i

        self.slot_of_cell = np.arange(self.M, dtype=np.int64)
        self.token_of_slot = tokens.astype(np.int64, copy=False).copy()
        self.slot_anchor = self.cell_xyz.copy()
        self.cells_by_slot: Dict[int, np.ndarray] = {
            i: np.array([i], dtype=np.int64) for i in range(self.M)
        }
        self.pair_slot_pairs: Dict[PairKey, set] = {}
        self.incidence: Dict[int, Dict[PairKey, Dict[int, int]]] = defaultdict(dict)

        # Vectorized initial +d neighbor discovery (same directed edges as legacy loop).
        if self.M > 0:
            cmax = int(np.max(self.cell_xyz))
            K = int(cmax) + 2
            xyz = self.cell_xyz
            codes = ((xyz[:, 0] * K + xyz[:, 1]) * K + xyz[:, 2]).astype(np.int64)
            order = np.argsort(codes, kind="stable")
            sorted_codes = codes[order]
            tok = self.token_of_slot
            for d in directions:
                dv = np.asarray(_DIR_TO_VEC[int(d)], dtype=np.int64)
                nbr = xyz + dv
                nc = ((nbr[:, 0] * K + nbr[:, 1]) * K + nbr[:, 2]).astype(np.int64)
                idx = np.searchsorted(sorted_codes, nc)
                hit = (idx < self.M) & (sorted_codes[np.minimum(idx, self.M - 1)] == nc)
                a_cell = np.nonzero(hit)[0]
                if a_cell.size == 0:
                    continue
                b_cell = order[idx[hit]]
                mask = b_cell != a_cell
                a_cell = a_cell[mask]
                b_cell = b_cell[mask]
                for k in range(int(a_cell.shape[0])):
                    i = int(a_cell[k])
                    j = int(b_cell[k])
                    key = _make_pair_key(
                        int(tok[i]), int(tok[j]), self.slot_anchor[i], self.slot_anchor[j]
                    )
                    trainer.register_directed(self, key, i, j)

    def remove_all_incident(self, trainer: "_IncrementalBPETrainer", slot: int) -> None:
        for key, others in list(self.incidence[slot].items()):
            for other, role in list(others.items()):
                if int(role) == 0:
                    trainer.remove_directed(self, key, int(slot), int(other))
                else:
                    trainer.remove_directed(self, key, int(other), int(slot))

    def merge_slots(self, sa: int, sb: int, new_id: int) -> None:
        if sa == sb:
            return
        ca = self.cells_by_slot[sa]
        cb = self.cells_by_slot[sb]
        merged = np.concatenate([ca, cb], axis=0)
        self.cells_by_slot[sa] = merged
        del self.cells_by_slot[sb]
        for idx in cb.tolist():
            self.slot_of_cell[int(idx)] = sa
        self.token_of_slot[sa] = int(new_id)
        self.token_of_slot[sb] = -1
        if tuple(int(v) for v in self.slot_anchor[sb].tolist()) < tuple(
            int(v) for v in self.slot_anchor[sa].tolist()
        ):
            self.slot_anchor[sa] = self.slot_anchor[sb]

    def add_edges_for_slot(self, trainer: "_IncrementalBPETrainer", slot: int) -> None:
        if int(self.token_of_slot[slot]) < 0:
            return
        ta = int(self.token_of_slot[slot])
        cells = self.cells_by_slot[slot]
        for ci in cells.tolist():
            c = (
                int(self.cell_xyz[int(ci), 0]),
                int(self.cell_xyz[int(ci), 1]),
                int(self.cell_xyz[int(ci), 2]),
            )
            for d in self.directions:
                dv = _DIR_TO_VEC[int(d)]
                # Forward (+d): same as legacy iterating from this slot's cell.
                n = (c[0] + dv[0], c[1] + dv[1], c[2] + dv[2])
                j = self.coord_to_idx.get(n)
                if j is not None:
                    sj = int(self.slot_of_cell[int(j)])
                    if sj != int(slot):
                        tb = int(self.token_of_slot[sj])
                        if tb >= 0:
                            key = _make_pair_key(ta, tb, self.slot_anchor[slot], self.slot_anchor[sj])
                            trainer.register_directed(self, key, int(slot), sj)
                # Inverse: neighbor at c-dv reaches this cell via +d (legacy counts from neighbor).
                n_sub = (c[0] - dv[0], c[1] - dv[1], c[2] - dv[2])
                k = self.coord_to_idx.get(n_sub)
                if k is not None:
                    sk = int(self.slot_of_cell[int(k)])
                    if sk != int(slot):
                        t_left = int(self.token_of_slot[sk])
                        if t_left >= 0:
                            key2 = _make_pair_key(
                                t_left, ta, self.slot_anchor[sk], self.slot_anchor[slot]
                            )
                            trainer.register_directed(self, key2, sk, int(slot))


class _IncrementalBPETrainer:
    """Global pair counts + per-sample slot graph; incremental edge updates per merge."""

    def __init__(
        self,
        tokenizer: "BPE3DTokenizer",
        corpus: List[Dict[str, np.ndarray]],
        num_merges: int,
        min_freq: int,
        verbose: bool,
        progress: bool,
    ) -> None:
        self.tok = tokenizer
        self.num_merges = int(num_merges)
        self.min_freq = int(min_freq)
        self.verbose = verbose
        self.progress = progress
        self.directions = tokenizer.directions
        self.pair_freq: Dict[PairKey, int] = {}
        self.pair_heap: List[Tuple[int, PairKey]] = []
        self.pair_to_samples: Dict[PairKey, set] = defaultdict(set)
        self.samples: List[_IncSampleState] = []
        self._defer_heap = False
        self._dirty_pair_keys: set = set()
        self._build_sample_graphs(corpus)

    def _build_sample_graphs(self, corpus: List[Dict[str, np.ndarray]]) -> None:
        """Build per-sample slot graphs (dominant RAM + CPU cost before merge iterations)."""
        n_total = len(corpus)
        total_cells = sum(
            int(_as_int64_array(rec["tokens"], "tokens", 1).shape[0]) for rec in corpus
        )
        show_progress = bool(self.progress) and not (
            os.environ.get("BPE3D_GRAPH_PROGRESS", "1").strip().lower()
            in ("0", "false", "no", "off")
        )
        t0 = time.perf_counter()
        last_postfix_ts = 0.0
        postfix_interval_s = float(os.environ.get("BPE3D_GRAPH_POSTFIX_INTERVAL", "0.5"))
        self._defer_heap = _bpe_defer_heap_enabled()
        n_points_built = 0

        try:
            from tqdm import tqdm as _tqdm

            _has_tqdm = True
        except ImportError:
            _tqdm = None  # type: ignore
            _has_tqdm = False

        def _refresh_graph_postfix(force: bool = False) -> None:
            nonlocal last_postfix_ts
            if pbar is None:
                return
            now = time.perf_counter()
            if not force and (now - last_postfix_ts) < postfix_interval_s:
                return
            last_postfix_ts = now
            pbar.set_postfix(
                format_graph_build_memory_postfix(n_points_built, total_cells),
                refresh=False,
            )

        sample_iter: Iterable[Tuple[int, Dict[str, np.ndarray]]] = enumerate(corpus)
        pbar = None
        if show_progress and _has_tqdm and _tqdm is not None and n_total > 0:
            pbar = _tqdm(
                enumerate(corpus),
                total=n_total,
                desc="BPE3D build graph",
                unit="sample",
                dynamic_ncols=True,
                leave=True,
                file=sys.stderr,
            )
            sample_iter = pbar
            _refresh_graph_postfix(force=True)
        elif self.verbose and n_total > 0:
            print(
                f"[BPE3D] Building slot graphs for {n_total:,} samples "
                f"({total_cells:,} cells; install tqdm for a progress bar)...",
                flush=True,
            )

        try:
            for sid, rec in sample_iter:
                tokens = _as_int64_array(rec["tokens"], "tokens", 1)
                coords = _as_int64_array(rec["coords"], "coords", 2)
                n_points_built += int(tokens.shape[0])
                self.samples.append(
                    _IncSampleState(sid, tokens, coords, self.directions, self)
                )
                _refresh_graph_postfix()
        finally:
            if pbar is not None:
                _refresh_graph_postfix(force=True)
                pbar.close()

        self._defer_heap = False
        if _bpe_defer_heap_enabled():
            t_heap = time.perf_counter()
            self._rebuild_pair_heap()
            if self.verbose:
                print(
                    f"[BPE3D] Rebuilt pair heap: keys={len(self.pair_freq):,} "
                    f"elapsed={time.perf_counter() - t_heap:.1f}s",
                    flush=True,
                )

        elapsed = time.perf_counter() - t0
        n_pairs = len(self.pair_freq)
        rss_final = get_process_rss_gib()
        est_final = estimate_rss_from_cells_progress(
            n_points_built, max(total_cells, 1), rss_final
        )
        if self.verbose:
            print(
                f"[BPE3D] Graph build done: samples={len(self.samples):,} "
                f"cells={n_points_built:,} unique_pairs={n_pairs:,} "
                f"elapsed={elapsed:.1f}s"
                + (
                    f" rss={_format_gib(rss_final)}"
                    if rss_final is not None
                    else ""
                ),
                flush=True,
            )
            bpc = est_final.get("bytes_per_cell")
            if bpc is not None and bpc > 0:
                print(
                    f"[BPE3D][memory] observed during graph build: "
                    f"{bpc/1024:.1f} KiB/cell (RSS/cells_built); "
                    f"use progress bar ~peak for extrapolated merge peak",
                    flush=True,
                )
        _logger.info(
            "[BPE3D] graph build done samples=%s cells=%s pairs=%s elapsed=%.1fs rss=%s",
            len(self.samples),
            n_points_built,
            n_pairs,
            elapsed,
            rss_final,
        )

    def _rebuild_pair_heap(self) -> None:
        """Rebuild min-heap from pair_freq in O(n) (graph-build end or BPE3D_HEAP_FLUSH=rebuild)."""
        self.pair_heap = [
            (-int(freq), key) for key, freq in self.pair_freq.items() if int(freq) > 0
        ]
        heapq.heapify(self.pair_heap)
        self._dirty_pair_keys.clear()

    def _mark_dirty_pair_key(self, key: PairKey) -> None:
        if self._defer_heap:
            self._dirty_pair_keys.add(key)

    def _flush_dirty_heap(self) -> int:
        """Push one heap entry per dirty key (lazy best_pair skips stale entries). Returns |dirty|."""
        dirty = self._dirty_pair_keys
        self._dirty_pair_keys = set()
        n_pushed = 0
        for key in dirty:
            freq = int(self.pair_freq.get(key, 0))
            if freq > 0:
                heapq.heappush(self.pair_heap, (-freq, key))
                n_pushed += 1
        return n_pushed

    def _finish_deferred_heap_updates(self) -> Tuple[str, int]:
        """Apply deferred heap updates after a merge iteration. Returns (mode, count)."""
        if _bpe_heap_flush_rebuild():
            n = len(self.pair_freq)
            self._rebuild_pair_heap()
            return ("rebuild", n)
        n = self._flush_dirty_heap()
        return ("dirty", n)

    def _push_heap(self, key: PairKey) -> None:
        if self._defer_heap:
            self._mark_dirty_pair_key(key)
            return
        freq = int(self.pair_freq.get(key, 0))
        if freq > 0:
            heapq.heappush(self.pair_heap, (-freq, key))

    def best_pair(self) -> Tuple[PairKey, int]:
        while self.pair_heap:
            neg_freq, key = heapq.heappop(self.pair_heap)
            freq = int(self.pair_freq.get(key, 0))
            if freq > 0 and -int(neg_freq) == freq:
                return key, freq
        raise ValueError("empty pair_freq")

    def register_directed(self, st: _IncSampleState, key: PairKey, a: int, b: int) -> None:
        bucket = st.pair_slot_pairs.setdefault(key, set())
        if (a, b) in bucket:
            return
        bucket.add((a, b))
        st.incidence[a].setdefault(key, {})[b] = 0
        st.incidence[b].setdefault(key, {})[a] = 1
        self.pair_freq[key] = self.pair_freq.get(key, 0) + 1
        self._push_heap(key)
        self.pair_to_samples[key].add(st.sample_id)

    def remove_directed(self, st: _IncSampleState, key: PairKey, a: int, b: int) -> None:
        bucket = st.pair_slot_pairs.get(key)
        if bucket is None or (a, b) not in bucket:
            return
        bucket.discard((a, b))
        if not bucket:
            del st.pair_slot_pairs[key]
        if key in st.incidence[a]:
            st.incidence[a][key].pop(b, None)
            if not st.incidence[a][key]:
                del st.incidence[a][key]
        if key in st.incidence[b]:
            st.incidence[b][key].pop(a, None)
            if not st.incidence[b][key]:
                del st.incidence[b][key]
        self.pair_freq[key] = int(self.pair_freq.get(key, 0)) - 1
        if self.pair_freq[key] <= 0:
            del self.pair_freq[key]
        else:
            self._push_heap(key)
        if key not in st.pair_slot_pairs:
            self.pair_to_samples[key].discard(st.sample_id)
            if not self.pair_to_samples[key]:
                del self.pair_to_samples[key]

    def run(self) -> None:
        self.tok.merge_table = []
        t_start = time.perf_counter()
        if self.verbose:
            if not _bpe_defer_heap_enabled():
                defer_note = "off (eager heappush)"
            elif _bpe_heap_flush_rebuild():
                defer_note = "on, full heapify/iter (BPE3D_HEAP_FLUSH=rebuild)"
            else:
                defer_note = "on, dirty-key flush/iter (default)"
            print(
                "[BPE3D] merge phase: incremental trainer uses a single CPU thread. "
                f"Heap updates: {defer_note}. "
                "For multi-core training use train_mode=legacy with --workers "
                "(higher RAM; only in train_bpe_3d / --bpe_train_mode legacy).",
                flush=True,
            )
        try:
            from tqdm import tqdm as _tqdm

            _has_tqdm = True
        except ImportError:
            _tqdm = None  # type: ignore
            _has_tqdm = False

        pbar = None
        if self.progress and _has_tqdm and _tqdm is not None:
            pbar = _tqdm(
                total=self.num_merges,
                desc="BPE3D train (incremental)",
                unit="merge",
                dynamic_ncols=True,
                leave=True,
                file=sys.stderr,
            )

        try:
            for it in range(self.num_merges):
                t_iter = time.perf_counter()
                if not self.pair_freq:
                    if self.verbose:
                        print(f"[BPE3D] stop at iter={it}: no valid pair", flush=True)
                    _logger.info("[BPE3D] stop at iter=%s: no valid pair", it)
                    break

                best_pk, best_cnt = self.best_pair()
                if best_cnt < self.min_freq:
                    if self.verbose:
                        print(
                            f"[BPE3D] stop at iter={it}: best_freq={best_cnt} < min_freq={self.min_freq}",
                            flush=True,
                        )
                    _logger.info(
                        "[BPE3D] stop at iter=%s: best_freq=%s < min_freq=%s",
                        it,
                        best_cnt,
                        self.min_freq,
                    )
                    break

                a_tok, b_tok, rel_offset = _pair_key_parts(best_pk)
                new_id = int(self.tok.base_vocab_size) + len(self.tok.merge_table)

                self._defer_heap = _bpe_defer_heap_enabled()
                sids = sorted(self.pair_to_samples.get(best_pk, ()))
                for sid in sids:
                    st = self.samples[sid]
                    if best_pk not in st.pair_slot_pairs:
                        continue
                    snapshot = list(st.pair_slot_pairs[best_pk])
                    pairs = [
                        (int(x), int(y))
                        for x, y in snapshot
                        if int(st.token_of_slot[int(x)]) == a_tok
                        and int(st.token_of_slot[int(y)]) == b_tok
                    ]
                    if not pairs:
                        continue
                    if len(pairs) > 1:
                        anchors = np.stack([st.slot_anchor[pa] for pa, _ in pairs])
                        order = _morton_order(anchors)
                        pairs = [pairs[int(i)] for i in order.tolist()]
                    used: set = set()
                    for sa, sb in pairs:
                        if sa in used or sb in used:
                            continue
                        if (
                            int(st.token_of_slot[int(sa)]) < 0
                            or int(st.token_of_slot[int(sb)]) < 0
                        ):
                            continue
                        if int(st.token_of_slot[int(sa)]) != a_tok or int(
                            st.token_of_slot[int(sb)]
                        ) != b_tok:
                            continue
                        st.remove_all_incident(self, sa)
                        st.remove_all_incident(self, sb)
                        st.merge_slots(sa, sb, new_id)
                        st.add_edges_for_slot(self, sa)
                        used.add(sa)
                        used.add(sb)

                self.tok.merge_table.append(
                    MergeEntry(
                        new_id=new_id,
                        pair=(int(a_tok), int(b_tok)),
                        direction=0,
                        rel_offset=rel_offset,
                        leaf_tokens=[],
                        leaf_offsets=[],
                    )
                )

                if _bpe_defer_heap_enabled():
                    self._defer_heap = False
                    t_heap = time.perf_counter()
                    heap_mode, heap_n = self._finish_deferred_heap_updates()
                    heap_elapsed = time.perf_counter() - t_heap
                    if self.verbose and (it % 100 == 0 or it < 3):
                        print(
                            f"[BPE3D] heap {heap_mode}: "
                            f"n={heap_n:,} elapsed={heap_elapsed:.3f}s",
                            flush=True,
                        )

                t_elapsed = time.perf_counter() - t_start
                iter_time = time.perf_counter() - t_iter
                if self.verbose and (it % 10 == 0 or it == self.num_merges - 1):
                    msg = (
                        f"[BPE3D] iter={it+1:04d}"
                        f" pair=({a_tok},{b_tok},rel={rel_offset})"
                        f" freq={best_cnt} new_id={new_id}"
                        f" iter={iter_time:.3f}s elapsed={t_elapsed:.1f}s"
                    )
                    print(msg, flush=True)
                    _logger.info(msg)
                if self.verbose and (it % 100 == 0 and it > 0):
                    alive = sum(len(st.cells_by_slot) for st in self.samples)
                    print(
                        f"[BPE3D] progress it={it} pair_keys={len(self.pair_freq)} "
                        f"alive_slots={alive}",
                        flush=True,
                    )

                if pbar is not None:
                    its = 1.0 / max(iter_time, 1e-9)
                    pbar.set_postfix(
                        {
                            "pair": f"({a_tok},{b_tok},rel{rel_offset})",
                            "freq": best_cnt,
                            "it/s": f"{its:.2f}",
                        },
                        refresh=False,
                    )
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()


# ---------------------------------------------------------------------------
# Incremental BPE encode (replay merge_table; O(merges + edges) not O(merges * N))
# ---------------------------------------------------------------------------


class _IncEncodeState:
    """Per-sample slot graph for fast BPE encode (replays ``merge_table`` in order)."""

    __slots__ = (
        "tok",
        "M",
        "cell_xyz",
        "directions",
        "coord_to_idx",
        "slot_of_cell",
        "token_of_slot",
        "slot_anchor",
        "cells_by_slot",
        "pair_slot_pairs",
        "incidence",
    )

    def __init__(
        self,
        tok: "BPE3DTokenizer",
        tokens: np.ndarray,
        coords: np.ndarray,
    ) -> None:
        self.tok = tok
        self.M = int(tokens.shape[0])
        self.cell_xyz = coords.astype(np.int64, copy=False)
        self.directions = tok.directions
        self.coord_to_idx: Dict[Tuple[int, int, int], int] = {}
        for i in range(self.M):
            t = (
                int(self.cell_xyz[i, 0]),
                int(self.cell_xyz[i, 1]),
                int(self.cell_xyz[i, 2]),
            )
            self.coord_to_idx[t] = i

        self.slot_of_cell = np.arange(self.M, dtype=np.int64)
        self.token_of_slot = tokens.astype(np.int64, copy=False).copy()
        self.slot_anchor = self.cell_xyz.copy()
        self.cells_by_slot: Dict[int, np.ndarray] = {
            i: np.array([i], dtype=np.int64) for i in range(self.M)
        }
        self.pair_slot_pairs: Dict[PairKey, set] = {}
        self.incidence: Dict[int, Dict[PairKey, Dict[int, int]]] = defaultdict(dict)

        if self.M > 0:
            cmax = int(np.max(self.cell_xyz))
            K = int(cmax) + 2
            xyz = self.cell_xyz
            codes = ((xyz[:, 0] * K + xyz[:, 1]) * K + xyz[:, 2]).astype(np.int64)
            order = np.argsort(codes, kind="stable")
            sorted_codes = codes[order]
            tokv = self.token_of_slot
            for d in self.directions:
                dv = np.asarray(_DIR_TO_VEC[int(d)], dtype=np.int64)
                nbr = xyz + dv
                nc = ((nbr[:, 0] * K + nbr[:, 1]) * K + nbr[:, 2]).astype(np.int64)
                idx = np.searchsorted(sorted_codes, nc)
                hit = (idx < self.M) & (sorted_codes[np.minimum(idx, self.M - 1)] == nc)
                a_cell = np.nonzero(hit)[0]
                if a_cell.size == 0:
                    continue
                b_cell = order[idx[hit]]
                mask = b_cell != a_cell
                a_cell = a_cell[mask]
                b_cell = b_cell[mask]
                for k in range(int(a_cell.shape[0])):
                    i = int(a_cell[k])
                    j = int(b_cell[k])
                    self.register_directed(
                        _make_pair_key(
                            int(tokv[i]), int(tokv[j]), self.slot_anchor[i], self.slot_anchor[j]
                        ),
                        int(i),
                        int(j),
                    )

    def register_directed(
        self, key: PairKey, a_slot: int, b_slot: int
    ) -> None:
        bucket = self.pair_slot_pairs.setdefault(key, set())
        if (a_slot, b_slot) in bucket:
            return
        bucket.add((a_slot, b_slot))
        self.incidence[a_slot].setdefault(key, {})[b_slot] = 0
        self.incidence[b_slot].setdefault(key, {})[a_slot] = 1

    def remove_directed(
        self, key: PairKey, a_slot: int, b_slot: int
    ) -> None:
        bucket = self.pair_slot_pairs.get(key)
        if bucket is None or (a_slot, b_slot) not in bucket:
            return
        bucket.discard((a_slot, b_slot))
        if not bucket:
            del self.pair_slot_pairs[key]
        if key in self.incidence[a_slot]:
            self.incidence[a_slot][key].pop(b_slot, None)
            if not self.incidence[a_slot][key]:
                del self.incidence[a_slot][key]
        if key in self.incidence[b_slot]:
            self.incidence[b_slot][key].pop(a_slot, None)
            if not self.incidence[b_slot][key]:
                del self.incidence[b_slot][key]

    def remove_all_incident(self, slot: int) -> None:
        for key, others in list(self.incidence[slot].items()):
            for other, role in list(others.items()):
                if int(role) == 0:
                    self.remove_directed(key, int(slot), int(other))
                else:
                    self.remove_directed(key, int(other), int(slot))

    def merge_slots(self, sa: int, sb: int, new_id: int) -> None:
        if sa == sb:
            return
        ca = self.cells_by_slot[sa]
        cb = self.cells_by_slot[sb]
        merged = np.concatenate([ca, cb], axis=0)
        self.cells_by_slot[sa] = merged
        del self.cells_by_slot[sb]
        for idx in cb.tolist():
            self.slot_of_cell[int(idx)] = sa
        self.token_of_slot[sa] = int(new_id)
        self.token_of_slot[sb] = -1
        if tuple(int(v) for v in self.slot_anchor[sb].tolist()) < tuple(
            int(v) for v in self.slot_anchor[sa].tolist()
        ):
            self.slot_anchor[sa] = self.slot_anchor[sb]

    def add_edges_for_slot(self, slot: int) -> None:
        if int(self.token_of_slot[slot]) < 0:
            return
        ta = int(self.token_of_slot[slot])
        cells = self.cells_by_slot[slot]
        for ci in cells.tolist():
            c = (
                int(self.cell_xyz[int(ci), 0]),
                int(self.cell_xyz[int(ci), 1]),
                int(self.cell_xyz[int(ci), 2]),
            )
            for d in self.directions:
                dv = _DIR_TO_VEC[int(d)]
                n = (c[0] + dv[0], c[1] + dv[1], c[2] + dv[2])
                j = self.coord_to_idx.get(n)
                if j is not None:
                    sj = int(self.slot_of_cell[int(j)])
                    if sj != int(slot):
                        tb = int(self.token_of_slot[sj])
                        if tb >= 0:
                            self.register_directed(
                                _make_pair_key(ta, tb, self.slot_anchor[slot], self.slot_anchor[sj]),
                                int(slot),
                                sj,
                            )
                n_sub = (c[0] - dv[0], c[1] - dv[1], c[2] - dv[2])
                k = self.coord_to_idx.get(n_sub)
                if k is not None:
                    sk = int(self.slot_of_cell[int(k)])
                    if sk != int(slot):
                        t_left = int(self.token_of_slot[sk])
                        if t_left >= 0:
                            self.register_directed(
                                _make_pair_key(
                                    t_left, ta, self.slot_anchor[sk], self.slot_anchor[slot]
                                ),
                                sk,
                                int(slot),
                            )

    def apply_merge_table(self) -> None:
        for entry in self.tok.merge_table:
            a_tok, b_tok = int(entry.pair[0]), int(entry.pair[1])
            rel_offset = tuple(int(v) for v in entry.rel_offset)
            new_id = int(entry.new_id)
            key = (a_tok, b_tok, rel_offset[0], rel_offset[1], rel_offset[2])
            if key not in self.pair_slot_pairs:
                continue
            snapshot = list(self.pair_slot_pairs[key])
            pairs = [
                (int(x), int(y))
                for x, y in snapshot
                if int(self.token_of_slot[int(x)]) == a_tok
                and int(self.token_of_slot[int(y)]) == b_tok
            ]
            if not pairs:
                continue
            if len(pairs) > 1:
                anchors = np.stack([self.slot_anchor[pa] for pa, _ in pairs])
                order = _morton_order(anchors)
                pairs = [pairs[int(i)] for i in order.tolist()]
            used: set = set()
            for sa, sb in pairs:
                if sa in used or sb in used:
                    continue
                if int(self.token_of_slot[int(sa)]) < 0 or int(
                    self.token_of_slot[int(sb)]
                ) < 0:
                    continue
                if int(self.token_of_slot[int(sa)]) != a_tok or int(
                    self.token_of_slot[int(sb)]
                ) != b_tok:
                    continue
                self.remove_all_incident(int(sa))
                self.remove_all_incident(int(sb))
                self.merge_slots(int(sa), int(sb), new_id)
                self.add_edges_for_slot(int(sa))
                used.add(sa)
                used.add(sb)

    def final_ids_anchors(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.M == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=np.int64)
        alive_slots = [s for s in self.cells_by_slot if int(self.token_of_slot[s]) >= 0]
        if not alive_slots:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=np.int64)
        ids = np.asarray(
            [int(self.token_of_slot[s]) for s in alive_slots], dtype=np.int64
        )
        anchors = np.stack([self.slot_anchor[s] for s in alive_slots]).astype(
            np.int64, copy=False
        )
        order = _morton_order(anchors)
        return ids[order], anchors[order]


# ---------------------------------------------------------------------------
# BPE3DTokenizer
# ---------------------------------------------------------------------------

class BPE3DTokenizer:
    def __init__(self, base_vocab_size: int = 8192, directions: Tuple[int, ...] = (0, 1, 2)) -> None:
        self.base_vocab_size = int(base_vocab_size)
        self.directions = tuple(int(d) for d in directions)
        for d in self.directions:
            if d not in _DIR_TO_VEC:
                raise ValueError(f"unsupported direction={d}, expected one of {tuple(_DIR_TO_VEC.keys())}")
        self.merge_table: List[MergeEntry] = []

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.merge_table)

    def _sample_to_slots(self, tokens: np.ndarray, coords: np.ndarray) -> List[_TokenSlot]:
        slots: List[_TokenSlot] = []
        for tok, xyz in zip(tokens.tolist(), coords.tolist()):
            c = (int(xyz[0]), int(xyz[1]), int(xyz[2]))
            slots.append(_TokenSlot(token_id=int(tok), cells=[c], anchor=c))
        return slots

    def _collect_pair_matches(
        self, slots: List[_TokenSlot]
    ) -> Dict[PairKey, List[Tuple[int, int]]]:
        """Instance-method wrapper kept for backward compatibility."""
        return _collect_pairs_fn(slots, self.directions)

    def _apply_merge_on_sample(
        self,
        slots: List[_TokenSlot],
        target_key: PairKey,
        new_id: int,
    ) -> List[_TokenSlot]:
        """Instance-method wrapper kept for backward compatibility (encode path)."""
        pair_to_matches = _collect_pairs_fn(slots, self.directions)
        return _apply_merge_fn(slots, pair_to_matches, target_key, new_id)

    def _build_flat_leaf_templates(self) -> None:
        flat: Dict[int, Tuple[List[int], List[Tuple[int, int, int]]]] = {
            i: ([i], [(0, 0, 0)]) for i in range(self.base_vocab_size)
        }
        for entry in self.merge_table:
            a, b = entry.pair
            shift = tuple(int(v) for v in entry.rel_offset)
            a_leaf_ids, a_leaf_off = flat[a]
            b_leaf_ids, b_leaf_off = flat[b]
            leaf_ids = list(a_leaf_ids)
            leaf_offs = list(a_leaf_off)
            for t, off in zip(b_leaf_ids, b_leaf_off):
                leaf_ids.append(int(t))
                leaf_offs.append(_add3(off, shift))
            anchor_shift = _canonical_min_offset(leaf_offs)
            leaf_offs = [_sub3(off, anchor_shift) for off in leaf_offs]
            entry.leaf_tokens = leaf_ids
            entry.leaf_offsets = leaf_offs
            flat[entry.new_id] = (leaf_ids, leaf_offs)

    # ------------------------------------------------------------------
    # train – legacy single-threaded path (full recount each iteration)
    # ------------------------------------------------------------------

    def _train_legacy_single(
        self,
        samples: List[List[_TokenSlot]],
        num_merges: int,
        min_freq: int,
        verbose: bool,
        progress: bool,
        t_start: float,
    ) -> None:
        try:
            from tqdm import tqdm as _tqdm
            _has_tqdm = True
        except ImportError:
            _has_tqdm = False

        pbar = None
        if progress and _has_tqdm:
            pbar = _tqdm(
                total=num_merges,
                desc="BPE3D train (legacy)",
                unit="merge",
                dynamic_ncols=True,
                leave=True,
                file=sys.stderr,
            )

        try:
            for it in range(int(num_merges)):
                t_iter = time.perf_counter()

                # Count pairs AND cache per-sample matches for reuse in apply step
                pair_freq: Counter = Counter()
                per_sample_pairs: List[Dict] = []
                for slots in samples:
                    pairs = _collect_pairs_fn(slots, self.directions)
                    per_sample_pairs.append(pairs)
                    for k, v in pairs.items():
                        pair_freq[k] += len(v)

                if not pair_freq:
                    msg = f"[BPE3D] stop at iter={it}: no valid pair"
                    if verbose:
                        print(msg, flush=True)
                    _logger.info(msg)
                    break

                target_key, best_cnt = _best_pair_tuple_from_counter(dict(pair_freq))
                best_a, best_b, rel_offset = _pair_key_parts(target_key)
                if best_cnt < int(min_freq):
                    msg = f"[BPE3D] stop at iter={it}: best_freq={best_cnt} < min_freq={min_freq}"
                    if verbose:
                        print(msg, flush=True)
                    _logger.info(msg)
                    break

                new_id = self.base_vocab_size + len(self.merge_table)

                # Apply merge reusing cached pair matches (no double traversal)
                for si in range(len(samples)):
                    samples[si] = _apply_merge_fn(
                        samples[si], per_sample_pairs[si], target_key, new_id
                    )

                self.merge_table.append(
                    MergeEntry(
                        new_id=new_id,
                        pair=(int(best_a), int(best_b)),
                        direction=0,
                        rel_offset=rel_offset,
                        leaf_tokens=[],
                        leaf_offsets=[],
                    )
                )

                t_elapsed = time.perf_counter() - t_start
                iter_time = time.perf_counter() - t_iter
                total_slots = sum(len(s) for s in samples)

                if verbose and (it % 10 == 0 or it == num_merges - 1):
                    msg = (
                        f"[BPE3D] iter={it+1:04d}"
                        f" pair=({best_a},{best_b},rel={rel_offset})"
                        f" freq={best_cnt} new_id={new_id}"
                        f" slots={total_slots}"
                        f" iter={iter_time:.3f}s elapsed={t_elapsed:.1f}s"
                    )
                    print(msg, flush=True)
                    _logger.info(msg)

                if pbar is not None:
                    its = 1.0 / max(iter_time, 1e-9)
                    pbar.set_postfix(
                        {
                            "pair": f"({best_a},{best_b},rel{rel_offset})",
                            "freq": best_cnt,
                            "slots": total_slots,
                            "it/s": f"{its:.2f}",
                        },
                        refresh=False,
                    )
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

    # ------------------------------------------------------------------
    # train – legacy parallel path (persistent workers, minimal IPC per iteration)
    # ------------------------------------------------------------------

    def _train_legacy_parallel(
        self,
        samples: List[List[_TokenSlot]],
        num_merges: int,
        min_freq: int,
        verbose: bool,
        progress: bool,
        num_workers: int,
        t_start: float,
    ) -> None:
        chunks = _split_chunks(samples, num_workers)
        actual_workers = len(chunks)

        start_method = "fork" if sys.platform != "win32" else "spawn"
        ctx = mp.get_context(start_method)

        msg = (
            f"[BPE3D] parallel train: {actual_workers} workers"
            f" ({start_method}), {len(samples)} samples"
        )
        if verbose:
            print(msg, flush=True)
        _logger.info(msg)

        cmd_qs = [ctx.Queue(maxsize=2) for _ in range(actual_workers)]
        res_qs = [ctx.Queue(maxsize=2) for _ in range(actual_workers)]
        procs = []
        for chunk, cmd_q, res_q in zip(chunks, cmd_qs, res_qs):
            p = ctx.Process(
                target=_worker_loop,
                args=(chunk, self.directions, cmd_q, res_q),
                daemon=True,
            )
            p.start()
            procs.append(p)

        try:
            from tqdm import tqdm as _tqdm
            _has_tqdm = True
        except ImportError:
            _has_tqdm = False

        pbar = None
        if progress and _has_tqdm:
            pbar = _tqdm(
                total=num_merges,
                desc=f"BPE3D train (legacy ×{actual_workers})",
                unit="merge",
                dynamic_ncols=True,
                leave=True,
                file=sys.stderr,
            )

        try:
            for it in range(int(num_merges)):
                t_iter = time.perf_counter()

                for q in cmd_qs:
                    q.put(("count",))

                pair_freq: Counter = Counter()
                for q in res_qs:
                    pair_freq.update(q.get())

                if not pair_freq:
                    msg = f"[BPE3D] stop at iter={it}: no valid pair"
                    if verbose:
                        print(msg, flush=True)
                    _logger.info(msg)
                    break

                target_key, best_cnt = _best_pair_tuple_from_counter(dict(pair_freq))
                best_a, best_b, rel_offset = _pair_key_parts(target_key)
                if best_cnt < int(min_freq):
                    msg = f"[BPE3D] stop at iter={it}: best_freq={best_cnt} < min_freq={min_freq}"
                    if verbose:
                        print(msg, flush=True)
                    _logger.info(msg)
                    break

                new_id = self.base_vocab_size + len(self.merge_table)

                for q in cmd_qs:
                    q.put(("merge", target_key, new_id))
                for q in res_qs:
                    q.get()  # "done"

                self.merge_table.append(
                    MergeEntry(
                        new_id=new_id,
                        pair=(int(best_a), int(best_b)),
                        direction=0,
                        rel_offset=rel_offset,
                        leaf_tokens=[],
                        leaf_offsets=[],
                    )
                )

                t_elapsed = time.perf_counter() - t_start
                iter_time = time.perf_counter() - t_iter

                if verbose and (it % 10 == 0 or it == num_merges - 1):
                    msg = (
                        f"[BPE3D] iter={it+1:04d}"
                        f" pair=({best_a},{best_b},rel={rel_offset})"
                        f" freq={best_cnt} new_id={new_id}"
                        f" iter={iter_time:.3f}s elapsed={t_elapsed:.1f}s"
                    )
                    print(msg, flush=True)
                    _logger.info(msg)

                if pbar is not None:
                    its = 1.0 / max(iter_time, 1e-9)
                    pbar.set_postfix(
                        {
                            "pair": f"({best_a},{best_b},rel{rel_offset})",
                            "freq": best_cnt,
                            "it/s": f"{its:.2f}",
                        },
                        refresh=False,
                    )
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()
            for q in cmd_qs:
                try:
                    q.put(None)
                except Exception:
                    pass
            for p in procs:
                p.join(timeout=15)
                if p.is_alive():
                    p.terminate()

    # ------------------------------------------------------------------
    # Public train API
    # ------------------------------------------------------------------

    def train(
        self,
        corpus: Iterable[Dict[str, np.ndarray]],
        num_merges: int,
        min_freq: int = 2,
        verbose: bool = True,
        num_workers: int = 0,
        progress: bool = True,
        train_mode: Optional[str] = None,
    ) -> None:
        """Train BPE merge table.

        Args:
            corpus:      Iterable of {"tokens": np.int64[M], "coords": np.int64[M,3]}.
            num_merges:  Maximum number of merges.
            min_freq:    Stop when the best pair frequency drops below this.
            verbose:     Print per-iteration stats to stdout.
            num_workers: Parallel worker processes (legacy train_mode only).
                         0 (default) = min(os.cpu_count(), 8) capped (see _DEFAULT_BPE_WORKER_CAP).
                         1 = single-threaded (no subprocess overhead, good for small corpora).
                         N > 1 = exactly N workers.
            progress:    Show tqdm progress bar on stderr.
            train_mode:  "incremental" (default) or "legacy". If None, use env BPE_TRAIN_MODE
                         or "incremental".
        """
        corpus_list: List[Dict[str, np.ndarray]] = []
        for rec in corpus:
            tokens = _as_int64_array(rec["tokens"], "tokens", 1)
            coords = _as_int64_array(rec["coords"], "coords", 2)
            if coords.shape[1] != 3:
                raise ValueError(f"coords must be shape [N,3], got {coords.shape}")
            if tokens.shape[0] != coords.shape[0]:
                raise ValueError("tokens and coords must have the same length")
            corpus_list.append({"tokens": tokens, "coords": coords})

        n_samples = len(corpus_list)
        total_points = sum(int(rec["tokens"].shape[0]) for rec in corpus_list)
        mode_raw = train_mode if train_mode is not None else os.environ.get("BPE_TRAIN_MODE", "incremental")
        mode = str(mode_raw).strip().lower()
        if mode not in ("incremental", "legacy"):
            raise ValueError(
                f"train_mode must be 'incremental' or 'legacy', got {mode_raw!r} "
                f"(env BPE_TRAIN_MODE)"
            )

        msg = (
            f"[BPE3D] train: mode={mode} samples={n_samples}, points={total_points},"
            f" num_merges={num_merges}, min_freq={min_freq}"
        )
        if mode == "legacy":
            msg += f", num_workers={num_workers}"
        if verbose:
            print(msg, flush=True)
        _logger.info(msg)

        mem_est = estimate_bpe_training_memory(n_samples, total_points, train_mode=mode)
        print_bpe_memory_report(mem_est, verbose=verbose)

        self.merge_table = []
        t_start = time.perf_counter()

        if mode == "incremental":
            _IncrementalBPETrainer(
                self, corpus_list, num_merges, min_freq, verbose, progress
            ).run()
        else:
            samples = [
                self._sample_to_slots(rec["tokens"], rec["coords"]) for rec in corpus_list
            ]
            effective_workers = _resolve_workers(num_workers, len(samples))
            if effective_workers <= 1:
                self._train_legacy_single(
                    samples, num_merges, min_freq, verbose, progress, t_start
                )
            else:
                self._train_legacy_parallel(
                    samples, num_merges, min_freq, verbose, progress, effective_workers, t_start
                )

        self._build_flat_leaf_templates()

        elapsed = time.perf_counter() - t_start
        msg = (
            f"[BPE3D] done: merges={len(self.merge_table)},"
            f" vocab_size={self.vocab_size}, elapsed={elapsed:.1f}s"
        )
        if verbose:
            print(msg, flush=True)
        _logger.info(msg)

    def _encode_single_legacy(self, tokens: np.ndarray, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Full-graph scan each merge (slow; kept for regression / BPE3D_ENCODE_VERIFY)."""
        slots = self._sample_to_slots(tokens, coords)
        for entry in self.merge_table:
            rel = tuple(int(v) for v in entry.rel_offset)
            slots = self._apply_merge_on_sample(
                slots,
                (
                    int(entry.pair[0]),
                    int(entry.pair[1]),
                    rel[0],
                    rel[1],
                    rel[2],
                ),
                entry.new_id,
            )
        if not slots:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=np.int64)
        ids = np.asarray([s.token_id for s in slots], dtype=np.int64)
        anchors = np.asarray([s.anchor for s in slots], dtype=np.int64)
        order = _morton_order(anchors)
        return ids[order], anchors[order]

    def _encode_single_incremental(
        self, tokens: np.ndarray, coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Replay ``merge_table`` with local edge updates (fast encode path)."""
        tokens = _as_int64_array(tokens, "tokens", 1)
        coords = _as_int64_array(coords, "coords", 2)
        if tokens.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=np.int64)
        st = _IncEncodeState(self, tokens, coords)
        st.apply_merge_table()
        return st.final_ids_anchors()

    def _encode_single(self, tokens: np.ndarray, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._encode_single_incremental(tokens, coords)

    def encode_sparse_numpy(
        self, tokens: np.ndarray, coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode one mesh sample on CPU only (no per-point ``torch.tensor``).

        Returns ``(macro_ids [K], macro_anchors [K,3])`` in Morton order of anchors.
        """
        return self._encode_single(tokens, coords)

    def encode_sparse(self, enc_st: Any, sparse_tensor_cls: Optional[Any] = None) -> Dict[str, Any]:
        coords_t = enc_st.coords
        feats_t = enc_st.feats
        coords_np = coords_t.detach().cpu().numpy().astype(np.int64)
        ids_np = feats_t.squeeze(-1).detach().cpu().numpy().astype(np.int64)
        if coords_np.shape[1] != 4:
            raise ValueError(f"enc_st.coords must be [N,4], got {coords_np.shape}")

        batch_ids = np.unique(coords_np[:, 0]).tolist()
        batch_ids = sorted(int(x) for x in batch_ids)
        batches: List[Dict[str, np.ndarray]] = []
        all_coords: List[List[int]] = []
        all_ids: List[int] = []

        for b in batch_ids:
            mask = coords_np[:, 0] == b
            tok_b = ids_np[mask]
            xyz_b = coords_np[mask][:, 1:4]
            macro_ids, macro_anchors = self._encode_single(tok_b, xyz_b)
            batches.append({"ids": macro_ids, "anchors": macro_anchors})
            for mid, anc in zip(macro_ids.tolist(), macro_anchors.tolist()):
                all_ids.append(int(mid))
                all_coords.append([b, int(anc[0]), int(anc[1]), int(anc[2])])

        device = feats_t.device
        if all_ids:
            out_feats = torch.tensor(all_ids, dtype=torch.float32, device=device).unsqueeze(-1)
            out_coords = torch.tensor(all_coords, dtype=torch.int32, device=device)
        else:
            out_feats = torch.zeros((0, 1), dtype=torch.float32, device=device)
            out_coords = torch.zeros((0, 4), dtype=torch.int32, device=device)

        if sparse_tensor_cls is None:
            from trellis.modules import sparse as sp  # delayed import
            sparse_tensor_cls = sp.SparseTensor
        macro_sparse = sparse_tensor_cls(out_feats, out_coords)
        return {"macro_sparse": macro_sparse, "batches": batches}

    def decode_to_sparse(
        self,
        batches: Sequence[Dict[str, np.ndarray]],
        device: Optional[torch.device] = None,
        sparse_tensor_cls: Optional[Any] = None,
    ) -> Any:
        if device is None:
            device = torch.device("cpu")
        all_coords: List[List[int]] = []
        all_ids: List[int] = []

        for batch_idx, rec in enumerate(batches):
            ids = _as_int64_array(rec["ids"], "ids", 1)
            anchors = _as_int64_array(rec["anchors"], "anchors", 2)
            if anchors.shape[1] != 3:
                raise ValueError(f"anchors must be [K,3], got {anchors.shape}")
            if ids.shape[0] != anchors.shape[0]:
                raise ValueError("ids and anchors length mismatch")

            for tid, anc in zip(ids.tolist(), anchors.tolist()):
                aid = int(tid)
                a = (int(anc[0]), int(anc[1]), int(anc[2]))
                if aid < self.base_vocab_size:
                    all_ids.append(aid)
                    all_coords.append([batch_idx, a[0], a[1], a[2]])
                    continue
                idx = aid - self.base_vocab_size
                if idx < 0 or idx >= len(self.merge_table):
                    raise ValueError(f"token id {aid} out of merge table range")
                entry = self.merge_table[idx]
                for leaf_id, off in zip(entry.leaf_tokens, entry.leaf_offsets):
                    c = _add3(a, off)
                    all_ids.append(int(leaf_id))
                    all_coords.append([batch_idx, c[0], c[1], c[2]])

        if all_ids:
            feats = torch.tensor(all_ids, dtype=torch.float32, device=device).unsqueeze(-1)
            coords = torch.tensor(all_coords, dtype=torch.int32, device=device)
        else:
            feats = torch.zeros((0, 1), dtype=torch.float32, device=device)
            coords = torch.zeros((0, 4), dtype=torch.int32, device=device)
        if sparse_tensor_cls is None:
            from trellis.modules import sparse as sp  # delayed import
            sparse_tensor_cls = sp.SparseTensor
        return sparse_tensor_cls(feats, coords)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_vocab_size": self.base_vocab_size,
            "directions": list(self.directions),
            "merge_table": [asdict(m) for m in self.merge_table],
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BPE3DTokenizer":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        tok = cls(
            base_vocab_size=int(data["base_vocab_size"]),
            directions=tuple(int(x) for x in data.get("directions", [0, 1, 2])),
        )
        table: List[MergeEntry] = []
        for r in data["merge_table"]:
            if "rel_offset" not in r:
                raise ValueError(
                    "This merge_table.json was created by an older non-reversible "
                    "3D BPE format without rel_offset. Please retrain the BPE merge "
                    "table before using decode_to_sparse."
                )
            table.append(
                MergeEntry(
                    new_id=int(r["new_id"]),
                    pair=(int(r["pair"][0]), int(r["pair"][1])),
                    direction=int(r["direction"]),
                    rel_offset=tuple(int(v) for v in r["rel_offset"]),
                    leaf_tokens=[int(x) for x in r.get("leaf_tokens", [])],
                    leaf_offsets=[tuple(int(v) for v in t) for t in r.get("leaf_offsets", [])],
                )
            )
        tok.merge_table = table
        if len(tok.merge_table) > 0 and len(tok.merge_table[0].leaf_tokens) == 0:
            tok._build_flat_leaf_templates()
        return tok
