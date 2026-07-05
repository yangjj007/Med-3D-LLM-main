#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Morton 排序前后：双 VQVAE（SparseSDFVQVAE vs ShapeLLM VQVAE3D）的熵与空间局部性分析。

对 **VQVAE 量化后的离散 token 序列**及其对应的 **latent 整数格点 (x,y,z)** 做第二种序列重排，与 ``pre``（编码器输出的序列顺序）对比：

- **ShapeLLM**：``post`` 为 Morton（Z-order，``variable_length_3d.morton_sort_indices``）。
- **our（SparseSDFVQVAE）**：``post`` 默认为与 ShapeLLM **同构的线性体素索引排序**（``x`` 最慢变、``z`` 最快变：
  ``idx = x·(ny·nz) + y·nz + z``，对 64³ 即 ``idx = x·4096 + y·64 + z``）；可用 ``--our_post_sort morton`` 恢复原先 Morton。

**指标在什么阶段算**：一律在 **VQ Encode 之后、已得到离散 codebook 索引（token）与每个 token 的 (x,y,z) 格坐标** 上算；**没有**对连续 latent 向量做「每 8 个拼成一大向量」的拼接——``cube8_*`` 只看 **序列上每 8 个 token 的 (x,y,z) 是否构成轴对齐 2×2×2 角点**。

比较 ``pre`` / ``post`` 的 1-gram Shannon、2-Renyi、bigram Shannon，并输出显式熵差 ``d_* = post - pre``。

另输出 **码本使用**：每条样本 ``n_unique_codes``、``unique_over_seq_len``（= 不同 id 数 / 序列长，jsonl 中 ``unique_pct_of_seq_len`` 为其百分数）；
全语料上 ``codebook_freq_our_sparse_sdf_vqvae.csv`` / ``codebook_freq_shapellm_vqvae3d.csv`` 给出各 id 出现次数及占全部分配次数的百分比（``--codebook_size``，默认 8192）。

**排序后序列相邻性**（第 i 与 i+1 个 token 的 latent 体素）：
- **6-邻（面邻接）**：|dx|+|dy|+|dz|=1（上下左右前后，体素边长为 1 的网格）。
- **26-邻**：L∞ 距离 ≤ 1 且非同一格点。

**归一化几何距离**（体素中心在 ``[-0.5, 0.5]^3`` 内，与常见 mesh/SDF 归一化一致）：
- **our（latent 64³）**：各轴等距，体素中心 ``(c+0.5)/64 - 0.5``，面邻接两中心欧氏距离 = **1/64**。
- **ShapeLLM（latent 8×8×16）**：中心 ``((lx+0.5)/8-0.5, (ly+0.5)/8-0.5, (lz+0.5)/16-0.5)``，
  面邻沿 x/y 为 **1/8**，沿 z 为 **1/16**（对应 64³ 上 x/y 步 8 格、z 步 4 格）。

**连续 8 个一组（2×2×2 latent 块）**：按序列下标非重叠分组 ``[0:8],[8:16],...``，
判断每组 8 个 latent 整数坐标是否**恰好**为某个轴对齐 ``[x0,x0+1]×[y0,y0+1]×[z0,z0+1]`` 的 **8 个角点**
（与 ShapeLLM 式「每 8 token 拼成一块」的理想几何一致）。输出 ``cube8_exact_frac = 命中组数 / 完整组数``，
以及滑动窗口 ``cube8_sliding_exact_frac``（起点 ``i=0..L-8`` 中命中比例，用于对照非对齐分割）。

用法（在 Med-3D-LLM-main 根目录；若 TRELLIS-500K 在仓库内请用 ``./TRELLIS-500K/...``，勿用 ``../`` 指到仓库外）::

  python analyze_morton_entropy.py \
    --input_dir ./TRELLIS-500K/ObjaverseXL/raw/hf-objaverse-v1 \
    --our_vae_config configs/vae/sdf_vqvae_stage2.json \
    --our_vae_ckpt outputs/sdf_vqvae_stage2_512_0.1-amp/ckpts/vqvae_step0000300.pt \
    --our_vae_impl auto \
    --n_samples 1000 --seed 42 --device cuda --output_dir morton_analysis_output

``--our_vae_impl auto``：旧权重（VQ 码本维 = ``latent_channels``）会自动选用 ``ss_vqvae_backup``；新权重（维 = ``latent_channels * vq_group_size``）用当前 ``ss_vqvae``。

若自动查找 ``metadata.csv`` 仍失败，可显式指定::

  --metadata_csv ./TRELLIS-500K/ObjaverseXL/metadata.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh

# -----------------------------------------------------------------------------
# 路径与后端
# -----------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent  # 3D-LLM-main
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

os.environ.setdefault("SPARSE_BACKEND", "torchsparse")
os.environ.setdefault("SPCONV_ALGO", "native")


@contextlib.contextmanager
def _stdio_suppressed() -> Any:
    """静默 C 扩展 stdout/stderr（与 build_qwen3vl_sft_3d_jsonl 一致）。"""
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


def _load_shapellm_module():
    """动态加载 3D-Alpaca/match_vqvae_tokens.py（VQVAE3D、load_vertices、mesh_path_to_occupancy_tensor）。"""
    path = _REPO_ROOT / "3D-Alpaca" / "match_vqvae_tokens.py"
    if not path.is_file():
        raise FileNotFoundError(f"未找到 ShapeLLM 辅助模块: {path}")
    spec = importlib.util.spec_from_file_location("match_vqvae_tokens_shim", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _peek_vq_embedding_dim_from_ckpt(vae_ckpt_path: str) -> Optional[int]:
    """读取 checkpoint 里 vq.embeddings.weight 的第二维；失败则 None。"""
    try:
        ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    except Exception:
        return None
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        return None
    w = ckpt.get("vq.embeddings.weight")
    if w is None:
        return None
    return int(w.shape[1])


def _filter_vae_args_for_backup(vae_args: Dict[str, Any]) -> Dict[str, Any]:
    """ss_vqvae_backup 无 vq_group_size（块内通道拼接），需从配置里去掉以免 TypeError。"""
    out = dict(vae_args)
    out.pop("vq_group_size", None)
    return out


def load_our_vae_from_config(
    vae_config_path: str,
    vae_ckpt_path: Optional[str],
    device: torch.device,
    *,
    impl: str = "auto",
) -> torch.nn.Module:
    """
    加载 SparseSDFVQVAE。当前 trellis 实现含 vq_group_size，VQ 码本维 = latent_channels * vq_group_size；
    旧 checkpoint（如 vqvae_step0000300.pt）为每格 latent_channels 维，需用 ss_vqvae_backup 加载。

    impl: ``auto`` 根据 checkpoint 的 vq.embeddings.weight 形状与配置自动选择；
    ``current`` / ``backup`` 强制使用对应实现。
    """
    with open(vae_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    vae_args: Dict[str, Any] = dict(config["models"]["vqvae"]["args"])

    lat = vae_args.get("latent_channels")
    if lat is None:
        lat = vae_args.get("embed_dim")
    lat_i = int(lat) if lat is not None else 0
    vq_g = int(vae_args.get("vq_group_size", 8))
    expect_current_vq_emb = lat_i * vq_g

    use_backup: bool
    if impl == "backup":
        use_backup = True
    elif impl == "current":
        use_backup = False
    elif impl == "auto":
        use_backup = False
        if vae_ckpt_path and os.path.isfile(vae_ckpt_path):
            d_ckpt = _peek_vq_embedding_dim_from_ckpt(vae_ckpt_path)
            if d_ckpt is not None:
                if d_ckpt == expect_current_vq_emb:
                    use_backup = False
                elif d_ckpt == lat_i:
                    use_backup = True
                else:
                    raise RuntimeError(
                        f"无法自动匹配 VQVAE 实现：checkpoint vq.embeddings.weight 维度为 {d_ckpt}，"
                        f"而配置 latent_channels={lat_i}、vq_group_size={vq_g} 时当前实现期望 {expect_current_vq_emb}，"
                        f"备份实现期望 {lat_i}。请显式指定 --our_vae_impl current 或 backup，或检查配置与权重。"
                    )
    else:
        raise ValueError(f"unknown --our_vae_impl: {impl}")

    if use_backup:
        from trellis.models.autoencoders import ss_vqvae_backup

        cls = ss_vqvae_backup.SparseSDFVQVAE
        build_args = _filter_vae_args_for_backup(vae_args)
        tag = "backup (ss_vqvae_backup.py, 无 vq_group_size)"
    else:
        from trellis.models import SparseSDFVQVAE

        cls = SparseSDFVQVAE
        build_args = vae_args
        tag = "current (trellis ss_vqvae.py)"

    model = cls(**build_args)
    if vae_ckpt_path and os.path.isfile(vae_ckpt_path):
        ckpt = torch.load(vae_ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
    print(f"[init] our SparseSDFVQVAE 实现: {tag}")
    model = model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Morton（与 variable_length_3d 一致）
# -----------------------------------------------------------------------------

from vae_qwen3vl.variable_length_3d import morton_sort_indices  # noqa: E402


def apply_morton_order(
    tokens: np.ndarray,
    coords_xyz: np.ndarray,
    coord_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 Morton 重排后的 (tokens_sorted, coords_sorted)。"""
    if len(tokens) == 0:
        return tokens, coords_xyz
    order = morton_sort_indices(coords_xyz.astype(np.int64), coord_max=coord_max)
    return tokens[order], coords_xyz[order]


def linear_raster_index(
    coords_xyz: np.ndarray,
    _dim_x: int,
    dim_y: int,
    dim_z: int,
) -> np.ndarray:
    """
    与 ShapeLLM 8×8×16 展开同构：x 最慢变、y 次之、z 最快变。
    idx = x * (dim_y * dim_z) + y * dim_z + z
    """
    c = coords_xyz.astype(np.int64)
    return c[:, 0] * (dim_y * dim_z) + c[:, 1] * dim_z + c[:, 2]


def apply_linear_raster_order(
    tokens: np.ndarray,
    coords_xyz: np.ndarray,
    dim_x: int,
    dim_y: int,
    dim_z: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """按线性体素索引（x 慢 → z 快）升序重排；同 idx 时保持稳定顺序。"""
    if len(tokens) == 0:
        return tokens, coords_xyz
    keys = linear_raster_index(coords_xyz, dim_x, dim_y, dim_z)
    order = np.argsort(keys, kind="stable")
    return tokens[order], coords_xyz[order]


# -----------------------------------------------------------------------------
# 熵（对齐 q3REST/mesh_token_entropy_stats 思路：自然对数 / nats）
# -----------------------------------------------------------------------------


def shannon_unigram_entropy(tokens: np.ndarray) -> float:
    if len(tokens) == 0:
        return float("nan")
    cnt = Counter(int(t) for t in tokens.tolist())
    total = sum(cnt.values())
    h = 0.0
    for c in cnt.values():
        p = c / total
        h -= p * math.log(p)
    return h


def renyi2_entropy(tokens: np.ndarray) -> float:
    if len(tokens) == 0:
        return float("nan")
    cnt = Counter(int(t) for t in tokens.tolist())
    total = sum(cnt.values())
    s = sum((c / total) ** 2 for c in cnt.values())
    if s <= 0:
        return float("nan")
    return -math.log(s)


def shannon_bigram_entropy(tokens: np.ndarray) -> float:
    if len(tokens) < 2:
        return float("nan")
    pairs = list(zip(tokens[:-1].tolist(), tokens[1:].tolist()))
    cnt: Counter[Tuple[int, int]] = Counter(pairs)
    total = len(pairs)
    h = 0.0
    for c in cnt.values():
        p = c / total
        h -= p * math.log(p)
    return h


def compute_entropy_bundle(tokens: np.ndarray) -> Dict[str, float]:
    return {
        "h1_nat": shannon_unigram_entropy(tokens),
        "h2_nat": renyi2_entropy(tokens),
        "h_bigram_nat": shannon_bigram_entropy(tokens),
    }


# -----------------------------------------------------------------------------
# Latent 体素中心（归一化 [-0.5, 0.5]^3）与序列相邻对统计
# -----------------------------------------------------------------------------


def latent_centers_normalized(
    coords_xyz: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """
    将 latent 整数格点映射为体素中心坐标，落在 [-0.5, 0.5]^3。
    our: 64^3 各向同性；shapellm: 8×8×16 各向异性。
    """
    c = coords_xyz.astype(np.float64)
    if "our" in model_name:
        return (c + 0.5) / 64.0 - 0.5
    # shapellm_vqvae3d: latent 8×8×16
    x = (c[:, 0] + 0.5) / 8.0 - 0.5
    y = (c[:, 1] + 0.5) / 8.0 - 0.5
    z = (c[:, 2] + 0.5) / 16.0 - 0.5
    return np.stack([x, y, z], axis=1)


def _index_l1(a: np.ndarray, b: np.ndarray) -> int:
    d = np.abs(a.astype(np.int64) - b.astype(np.int64))
    return int(np.sum(d))


def _index_linf(a: np.ndarray, b: np.ndarray) -> int:
    d = np.abs(a.astype(np.int64) - b.astype(np.int64))
    return int(np.max(d))


def consecutive_latent_pair_stats(
    coords_xyz: np.ndarray,
    model_name: str,
) -> Dict[str, float]:
    """
    对当前序列顺序，统计相邻 token (i, i+1) 在 latent 索引上的邻接比例，
    以及在归一化 [-0.5,0.5]^3 中体素中心之间的欧氏距离。
    """
    L = coords_xyz.shape[0]
    if L < 2:
        return {
            "seq_face_adj_frac": float("nan"),
            "seq_26_adj_frac": float("nan"),
            "seq_consec_eucl_mean": float("nan"),
            "seq_consec_eucl_median": float("nan"),
        }
    centers = latent_centers_normalized(coords_xyz, model_name)
    n_pairs = L - 1
    n_face = 0
    n_26 = 0
    dists: List[float] = []
    for i in range(n_pairs):
        ca = coords_xyz[i]
        cb = coords_xyz[i + 1]
        l1 = _index_l1(ca, cb)
        linf = _index_linf(ca, cb)
        if l1 == 1:
            n_face += 1
        # 26-邻：Chebyshev ≤1 且非同一格点
        if linf <= 1 and l1 > 0:
            n_26 += 1
        dists.append(float(np.linalg.norm(centers[i + 1] - centers[i])))
    arr = np.array(dists, dtype=np.float64)
    return {
        "seq_face_adj_frac": n_face / n_pairs,
        "seq_26_adj_frac": n_26 / n_pairs,
        "seq_consec_eucl_mean": float(arr.mean()),
        "seq_consec_eucl_median": float(np.median(arr)),
    }


# -----------------------------------------------------------------------------
# 非重叠每 8 个 token：latent 是否构成轴对齐 2×2×2 角点集合
# -----------------------------------------------------------------------------


def is_axis_aligned_2x2x2_corner_set(coords8: np.ndarray) -> bool:
    """
    8 个 latent 整数格点是否恰好为 [x0,x0+1]×[y0,y0+1]×[z0,z0+1] 的 8 个顶点（轴对齐 2×2×2）。
    """
    if coords8.shape != (8, 3):
        return False
    pts = coords8.astype(np.int64)
    if np.unique(pts, axis=0).shape[0] != 8:
        return False
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    if np.any(mx - mn != 1):
        return False
    x0, y0, z0 = int(mn[0]), int(mn[1]), int(mn[2])
    corners = {
        (x0 + dx, y0 + dy, z0 + dz)
        for dx in (0, 1)
        for dy in (0, 1)
        for dz in (0, 1)
    }
    got = {tuple(int(t) for t in row) for row in pts}
    return corners == got


def cube8_block_stats(
    coords_xyz: np.ndarray,
    group_size: int = 8,
) -> Dict[str, float]:
    """
    - cube8_n_groups: floor(L / group_size)（仅完整组）
    - cube8_exact_count: 其中几何上为轴对齐 2×2×2 角点集合的组数
    - cube8_exact_frac: exact_count / n_groups（无完整组时为 nan）
    - cube8_sliding_exact_frac: 所有起点 i∈[0, L-group_size] 的滑动窗中，命中比例（L<group_size 为 nan）
    """
    L = int(coords_xyz.shape[0])
    if L < group_size:
        return {
            "cube8_n_groups": 0.0,
            "cube8_exact_count": 0.0,
            "cube8_exact_frac": float("nan"),
            "cube8_sliding_exact_frac": float("nan"),
        }
    n_g = L // group_size
    cnt = 0
    for k in range(n_g):
        blk = coords_xyz[k * group_size : (k + 1) * group_size]
        if is_axis_aligned_2x2x2_corner_set(blk):
            cnt += 1
    exact_frac = float(cnt) / float(n_g) if n_g > 0 else float("nan")
    n_slide = L - group_size + 1
    hit_slide = 0
    for i in range(n_slide):
        if is_axis_aligned_2x2x2_corner_set(coords_xyz[i : i + group_size]):
            hit_slide += 1
    slide_frac = float(hit_slide) / float(n_slide) if n_slide > 0 else float("nan")
    return {
        "cube8_n_groups": float(n_g),
        "cube8_exact_count": float(cnt),
        "cube8_exact_frac": exact_frac,
        "cube8_sliding_exact_frac": slide_frac,
    }


# -----------------------------------------------------------------------------
# 空间局部性：26-邻域在序列中的 index gap
# -----------------------------------------------------------------------------

_NEIGH_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def compute_spatial_locality_gaps(
    coords_xyz: np.ndarray,
) -> Tuple[float, float, float]:
    """
    coords_xyz: [L, 3] int，与序列顺序一致。
    对每个体素，对每个在网格内的 26-邻域若也存在，记录 |idx_neighbor - idx_self|。
    返回 (median, mean, p90)；若无任何邻接对则 nan。
    """
    L = coords_xyz.shape[0]
    if L == 0:
        return float("nan"), float("nan"), float("nan")
    pts = coords_xyz.astype(np.int64)
    key_to_idx: Dict[Tuple[int, int, int], int] = {
        (int(pts[i, 0]), int(pts[i, 1]), int(pts[i, 2])): i for i in range(L)
    }
    gaps: List[int] = []
    for i in range(L):
        x, y, z = int(pts[i, 0]), int(pts[i, 1]), int(pts[i, 2])
        for dx, dy, dz in _NEIGH_OFFSETS:
            nk = (x + dx, y + dy, z + dz)
            j = key_to_idx.get(nk)
            if j is not None and j != i:
                gaps.append(abs(j - i))
    if not gaps:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(gaps, dtype=np.float64)
    return (
        float(np.median(arr)),
        float(arr.mean()),
        float(np.percentile(arr, 90)),
    )


# -----------------------------------------------------------------------------
# 两条编码管线
# -----------------------------------------------------------------------------

def glb_to_our_tokens(
    glb_path: str,
    vae: torch.nn.Module,
    device: torch.device,
    resolution: int,
    threshold_factor: float,
    silent: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """返回 (tokens int64 [N], coords int64 [N,3]) 按 SparseTensor 行顺序；失败返回 None。"""
    from trellis.utils.mesh_utils import mesh2sparse_sdf

    try:
        mesh = trimesh.load(glb_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, "geometry") and len(mesh.geometry) > 0:
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            else:
                return None
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None
        sdf_data = mesh2sparse_sdf(
            mesh,
            resolution=resolution,
            threshold_factor=threshold_factor,
            normalize=True,
            scale=0.95,
            watertight=False,
        )
        sparse_sdf = torch.from_numpy(sdf_data["sparse_sdf"]).float()
        sparse_index = torch.from_numpy(sdf_data["sparse_index"]).long()
        if len(sparse_sdf) == 0:
            return None
        batch = {
            "sparse_sdf": sparse_sdf,
            "sparse_index": sparse_index,
            "batch_idx": torch.zeros(len(sparse_sdf), dtype=torch.long),
        }
        ctx = _stdio_suppressed() if silent else contextlib.nullcontext()
        with ctx:
            vae_f = vae.float() if next(vae.parameters()).dtype != torch.float32 else vae
            x = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if isinstance(x.get("sparse_sdf"), torch.Tensor) and x["sparse_sdf"].is_floating_point():
                x["sparse_sdf"] = x["sparse_sdf"].to(dtype=torch.float32)
            with torch.no_grad():
                enc = vae_f.Encode(x)
            feats = enc.feats.squeeze(-1).long().cpu().numpy()
            coords = enc.coords.cpu().numpy()
        mask = coords[:, 0] == 0
        if not np.any(mask):
            return None
        xyz = coords[mask][:, 1:4].astype(np.int64)
        tok = feats[mask].astype(np.int64)
        return tok, xyz
    except Exception:
        return None


def shapellm_latent_index_to_xyz(seq_len: int = 1024) -> np.ndarray:
    """
    VQVAE3D: 扁平位置 i∈[0,1024) 与格点 (x,y,z) 的对应（与线性 idx 互逆）：
    ``i = x·(8·16) + y·16 + z``，即 ``x=i//128``, ``y=(i%128)//16``, ``z=i%16``（x 最慢, z 最快）。
    用于把 Encode 输出的 [1024] 顺序映射到 latent 坐标以计算空间指标。
    """
    out = np.zeros((seq_len, 3), dtype=np.int64)
    for i in range(seq_len):
        out[i, 0] = i // 128
        out[i, 1] = (i % 128) // 16
        out[i, 2] = i % 16
    return out


def _unique_code_stats(tokens: np.ndarray) -> Tuple[int, int, float]:
    """
    编码器输出的离散码本索引序列统计。
    返回 (seq_len, n_unique_codes, n_unique / seq_len)；seq_len==0 时第三项为 nan。
    """
    flat = np.asarray(tokens).ravel()
    n = int(flat.size)
    if n == 0:
        return 0, 0, float("nan")
    n_u = int(np.unique(flat).size)
    return n, n_u, float(n_u) / float(n)


def write_codebook_frequency_csv(
    path: Path,
    counts: Counter,
    codebook_size: int,
) -> None:
    """
    每个码本 id 的出现次数及占「全部分配次数」的百分比（未出现为 0%）。
    """
    total = sum(counts.values())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code_id", "count", "pct_of_all_assignments"])
        for code_id in range(codebook_size):
            c = int(counts.get(code_id, 0))
            pct = (100.0 * c / total) if total > 0 else 0.0
            w.writerow([code_id, c, f"{pct:.8f}"])


def glb_to_shapellm_tokens(
    glb_path: str,
    vae: torch.nn.Module,
    device: torch.device,
    sh_mod: Any,
    silent: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """返回 (tokens [1024], coords [1024,3]) 原始 raster 顺序。"""
    try:
        pos = sh_mod.load_vertices(glb_path)
        ss = sh_mod.mesh_path_to_occupancy_tensor(pos, device)
        ctx = _stdio_suppressed() if silent else contextlib.nullcontext()
        with ctx:
            with torch.no_grad():
                enc = vae.Encode(ss)
        if enc.dim() != 2 or enc.shape[0] != 1:
            enc = enc.view(1, -1)
        tok = enc[0].detach().cpu().numpy().astype(np.int64)
        coords = shapellm_latent_index_to_xyz(len(tok))
        return tok, coords
    except Exception:
        return None


# -----------------------------------------------------------------------------
# 数据抽样
# -----------------------------------------------------------------------------

def sample_glb_rows(
    input_dir: str,
    n_samples: int,
    seed: int,
    metadata_csv: Optional[str] = None,
) -> List[Dict[str, str]]:
    from dataset_toolkits.sdf_voxelize import load_trellis500k_metadata

    df = load_trellis500k_metadata(input_dir, metadata_csv=metadata_csv)
    if "glb_path" not in df.columns:
        raise ValueError("metadata 缺少 glb_path 列")
    exists = df["glb_path"].astype(str).map(os.path.isfile)
    df = df[exists].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("没有磁盘上存在的 glb_path")
    rng = np.random.default_rng(seed)
    k = min(n_samples, len(df))
    idx = rng.choice(len(df), size=k, replace=False)
    rows = []
    for i in idx:
        r = df.iloc[int(i)]
        sha = str(r.get("sha256", i)).strip()
        rows.append({"sha256": sha, "glb_path": str(r["glb_path"])})
    return rows


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Morton 排序前后双 VQVAE 熵与空间局部性")
    ap.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="ObjaverseXL raw/hf-objaverse-v1（含 object-paths.json）。若在仓库内请用 ./TRELLIS-500K/...",
    )
    ap.add_argument(
        "--metadata_csv",
        type=str,
        default="",
        help="可选：显式指定 metadata.csv（当自动沿父目录查找失败时使用）",
    )
    ap.add_argument(
        "--our_vae_config",
        type=str,
        default=str(_THIS_DIR / "configs" / "vae" / "sdf_vqvae_stage2.json"),
    )
    ap.add_argument(
        "--our_vae_ckpt",
        type=str,
        default=str(_THIS_DIR / "outputs" / "sdf_vqvae_stage2_512_0.1-amp" / "ckpts" / "vqvae_step0000300.pt"),
    )
    ap.add_argument(
        "--our_vae_impl",
        type=str,
        choices=("auto", "current", "backup"),
        default="auto",
        help=(
            "SparseSDFVQVAE 实现：auto 根据权重中 vq.embeddings.weight 维数与配置自动选择；"
            "current=trellis ss_vqvae（vq_group_size>1 时码本维=latent×group）；"
            "backup=ss_vqvae_backup（旧 ckpt 常为每格 latent 维，如 16 而非 128）"
        ),
    )
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--output_dir",
        type=str,
        default=str(_THIS_DIR / "morton_analysis_output"),
    )
    ap.add_argument("--sdf_resolution", type=int, default=512)
    ap.add_argument("--threshold_factor", type=float, default=0.1)
    ap.add_argument("--coord_max_our", type=int, default=64, help="Morton coord_max for latent 64^3")
    ap.add_argument("--coord_max_shapellm", type=int, default=16, help="Morton: max index is 15 for z; use 16")
    ap.add_argument(
        "--our_post_sort",
        type=str,
        choices=("linear", "morton"),
        default="linear",
        help=(
            "our 第二种序列：linear=ShapeLLM 同构的线性体素索引序 (idx=x·ny·nz+y·nz+z，64³ 默认)；"
            "morton=Z-order（morton_sort_indices）。ShapeLLM 侧始终为 morton。"
        ),
    )
    ap.add_argument("--silent_vqvae", action="store_true", help="静默 VQVAE 与扩展的 stdout/stderr")
    ap.add_argument(
        "--shapellm_weights",
        type=str,
        default="",
        help="本地 3DVQVAE.bin；空则 hf_hub_download(yejunliang23/3DVQVAE)",
    )
    ap.add_argument(
        "--codebook_size",
        type=int,
        default=8192,
        help="码本大小（写频率表 0..K-1；our 与 shapellm 默认均为 8192）",
    )
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] device={device} output_dir={out_dir}")
    meta_override = args.metadata_csv.strip() or None
    rows_meta = sample_glb_rows(
        args.input_dir, args.n_samples, args.seed, metadata_csv=meta_override
    )
    print(f"[data] sampled {len(rows_meta)} meshes")

    print("[init] loading our SparseSDFVQVAE...")
    our_vae = load_our_vae_from_config(
        args.our_vae_config,
        args.our_vae_ckpt or None,
        device,
        impl=str(args.our_vae_impl),
    )

    print("[init] loading ShapeLLM VQVAE3D...")
    sh_mod = _load_shapellm_module()
    weights_path = args.shapellm_weights.strip() or None
    sh_vae = sh_mod.load_vqvae(device, weights_path=weights_path)
    sh_vae.eval()

    csv_path = out_dir / "entropy_stats.csv"
    jsonl_path = out_dir / "per_sample.jsonl"
    summary_path = out_dir / "summary.txt"

    csv_fields = [
        "model",
        "sort_order",
        "sha256",
        "seq_len",
        "n_unique_codes",
        "unique_over_seq_len",
        "h1_nat",
        "h2_nat",
        "h_bigram_nat",
        "d_h1_nat",
        "d_h2_nat",
        "d_h_bigram_nat",
        "spatial_gap_median",
        "spatial_gap_mean",
        "spatial_gap_p90",
        "seq_face_adj_frac",
        "seq_26_adj_frac",
        "seq_consec_eucl_mean",
        "seq_consec_eucl_median",
        "cube8_n_groups",
        "cube8_exact_count",
        "cube8_exact_frac",
        "cube8_sliding_exact_frac",
    ]

    agg: Dict[str, List[float]] = {}

    def _agg_add(key: str, val: float) -> None:
        if not math.isnan(val):
            agg.setdefault(key, []).append(val)

    our_code_global: Counter = Counter()
    sh_code_global: Counter = Counter()

    with open(jsonl_path, "w", encoding="utf-8") as jf, open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()

        for rec in rows_meta:
            sha = rec["sha256"]
            glb = rec["glb_path"]

            def one_model(
                name: str,
                tokens_pre: np.ndarray,
                coords_pre: np.ndarray,
                coord_max: int,
                *,
                post_mode: str,
                linear_dims: Tuple[int, int, int],
            ) -> None:
                if tokens_pre is None or len(tokens_pre) == 0:
                    return
                seq_len_u, n_unique_u, unique_ratio = _unique_code_stats(tokens_pre)
                ent_pre = compute_entropy_bundle(tokens_pre)
                seq_pre = consecutive_latent_pair_stats(coords_pre, name)
                if post_mode == "linear":
                    tokens_post, coords_post = apply_linear_raster_order(
                        tokens_pre, coords_pre, *linear_dims
                    )
                    sort_post_key = "post_linear_raster"
                elif post_mode == "morton":
                    tokens_post, coords_post = apply_morton_order(
                        tokens_pre, coords_pre, coord_max
                    )
                    sort_post_key = "post_morton"
                else:
                    raise ValueError(f"unknown post_mode: {post_mode}")
                ent_post = compute_entropy_bundle(tokens_post)
                seq_post = consecutive_latent_pair_stats(coords_post, name)
                d_h1 = ent_post["h1_nat"] - ent_pre["h1_nat"]
                d_h2 = ent_post["h2_nat"] - ent_pre["h2_nat"]
                d_hb = ent_post["h_bigram_nat"] - ent_pre["h_bigram_nat"]

                for sort_label, toks, crd, ent, seq, d_ent in (
                    (
                        "pre_morton",
                        tokens_pre,
                        coords_pre,
                        ent_pre,
                        seq_pre,
                        (float("nan"), float("nan"), float("nan")),
                    ),
                    (
                        sort_post_key,
                        tokens_post,
                        coords_post,
                        ent_post,
                        seq_post,
                        (d_h1, d_h2, d_hb),
                    ),
                ):
                    smed, smean, sp90 = compute_spatial_locality_gaps(crd)
                    cube = cube8_block_stats(crd, group_size=8)
                    row = {
                        "model": name,
                        "sort_order": sort_label,
                        "sha256": sha,
                        "seq_len": len(toks),
                        "n_unique_codes": n_unique_u,
                        "unique_over_seq_len": unique_ratio,
                        "h1_nat": ent["h1_nat"],
                        "h2_nat": ent["h2_nat"],
                        "h_bigram_nat": ent["h_bigram_nat"],
                        "d_h1_nat": d_ent[0],
                        "d_h2_nat": d_ent[1],
                        "d_h_bigram_nat": d_ent[2],
                        "spatial_gap_median": smed,
                        "spatial_gap_mean": smean,
                        "spatial_gap_p90": sp90,
                        "seq_face_adj_frac": seq["seq_face_adj_frac"],
                        "seq_26_adj_frac": seq["seq_26_adj_frac"],
                        "seq_consec_eucl_mean": seq["seq_consec_eucl_mean"],
                        "seq_consec_eucl_median": seq["seq_consec_eucl_median"],
                        "cube8_n_groups": cube["cube8_n_groups"],
                        "cube8_exact_count": cube["cube8_exact_count"],
                        "cube8_exact_frac": cube["cube8_exact_frac"],
                        "cube8_sliding_exact_frac": cube["cube8_sliding_exact_frac"],
                    }
                    writer.writerow(row)
                    _agg_add(f"{name}_{sort_label}_h1", ent["h1_nat"])
                    _agg_add(f"{name}_{sort_label}_h2", ent["h2_nat"])
                    _agg_add(f"{name}_{sort_label}_hbg", ent["h_bigram_nat"])
                    _agg_add(f"{name}_{sort_label}_smed", smed)
                    _agg_add(f"{name}_{sort_label}_smean", smean)
                    _agg_add(f"{name}_{sort_label}_sp90", sp90)
                    _agg_add(f"{name}_{sort_label}_face", seq["seq_face_adj_frac"])
                    _agg_add(f"{name}_{sort_label}_26", seq["seq_26_adj_frac"])
                    _agg_add(f"{name}_{sort_label}_euclm", seq["seq_consec_eucl_mean"])
                    _agg_add(f"{name}_{sort_label}_c8", cube["cube8_exact_frac"])
                    _agg_add(f"{name}_{sort_label}_c8s", cube["cube8_sliding_exact_frac"])
                    if sort_label == sort_post_key:
                        _agg_add(f"{name}_d_h1", d_ent[0])
                        _agg_add(f"{name}_d_h2", d_ent[1])
                        _agg_add(f"{name}_d_hbg", d_ent[2])
                    # 与排序无关；仅在 pre 行汇总，避免同一 mesh 重复计入均值
                    if sort_label == "pre_morton":
                        _agg_add(f"{name}_n_unique", float(n_unique_u))
                        _agg_add(f"{name}_unique_over_len", unique_ratio)

                    mesh_str = (
                        "<mesh_start>"
                        + "".join(f"<mesh_{int(v)}>" for v in toks.tolist())
                        + "<mesh_end>"
                    )
                    jf.write(
                        json.dumps(
                            {
                                "sha256": sha,
                                "glb_path": glb,
                                "model": name,
                                "sort_order": sort_label,
                                "n_unique_codes": n_unique_u,
                                "unique_over_seq_len": unique_ratio,
                                "unique_pct_of_seq_len": (
                                    round(100.0 * unique_ratio, 6)
                                    if not math.isnan(unique_ratio)
                                    else None
                                ),
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": f"[{name}/{sort_label}]\n{mesh_str}",
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            our_pack = glb_to_our_tokens(
                glb,
                our_vae,
                device,
                resolution=args.sdf_resolution,
                threshold_factor=args.threshold_factor,
                silent=args.silent_vqvae,
            )
            if our_pack is not None:
                t_o, c_o = our_pack
                our_code_global.update(int(x) for x in np.asarray(t_o).ravel().tolist())
                one_model(
                    "our_sparse_sdf_vqvae",
                    t_o,
                    c_o,
                    args.coord_max_our,
                    post_mode=str(args.our_post_sort),
                    linear_dims=(64, 64, 64),
                )

            sh_pack = glb_to_shapellm_tokens(
                glb, sh_vae, device, sh_mod, silent=args.silent_vqvae
            )
            if sh_pack is not None:
                t_s, c_s = sh_pack
                sh_code_global.update(int(x) for x in np.asarray(t_s).ravel().tolist())
                one_model(
                    "shapellm_vqvae3d",
                    t_s,
                    c_s,
                    args.coord_max_shapellm,
                    post_mode="morton",
                    linear_dims=(8, 8, 16),
                )

    k_cb = int(args.codebook_size)
    freq_our_path = out_dir / "codebook_freq_our_sparse_sdf_vqvae.csv"
    freq_sh_path = out_dir / "codebook_freq_shapellm_vqvae3d.csv"
    write_codebook_frequency_csv(freq_our_path, our_code_global, k_cb)
    write_codebook_frequency_csv(freq_sh_path, sh_code_global, k_cb)

    # summary
    our_post_key = "post_linear_raster" if args.our_post_sort == "linear" else "post_morton"
    lines = [
        "=== latent (x,y,z) 序列重排：熵、熵差、序列相邻空间关系 ===",
        "",
        f"our 第二种序列：{our_post_key}（--our_post_sort={args.our_post_sort}）；shapellm 第二种序列：post_morton（Z-order）。",
        "",
        "几何约定：体素中心在归一化 [-0.5, 0.5]^3。",
        "  our_sparse_sdf_vqvae：latent 64^3，面邻接两中心欧氏距离 = 1/64。",
        "  shapellm_vqvae3d：latent 8×8×16，面邻沿 x/y 中心距 1/8，沿 z 为 1/16。",
        "",
        "seq_face_adj_frac：序列相邻 (i,i+1) 在 latent 索引上为 6-邻（L1=1）的比例。",
        "seq_26_adj_frac：序列相邻为 26-邻（L∞≤1 且非同格）的比例。",
        "d_h*_nat：仅 post 行有意义，= post - pre（H1/H2 重排不变故应≈0）。",
        "",
        "cube8_exact_frac：非重叠每 8 个 latent 坐标恰为轴对齐 2×2×2 角点集合的组比例。",
        "cube8_sliding_exact_frac：滑动窗（步长 1）同上判定的命中比例。",
        "",
        "n_unique_codes：该样本 Encode 输出序列中**不同**码本 id 的个数（与 Morton/linear 重排无关）。",
        "unique_over_seq_len：n_unique_codes / seq_len，即「类型数 / 序列长度」；",
        "  unique_pct_of_seq_len（jsonl）= 该值×100，可理解为与 debug 中 762/1024 同类的**多样性比例**（非「仅出现一次的 token 占比」）。",
        "",
        f"codebook_freq_*.csv：全数据集上每个码本 id 的出现次数及占**全部量化分配次数**的百分比（codebook_size={k_cb}）。",
        "",
    ]
    for name in ("our_sparse_sdf_vqvae", "shapellm_vqvae3d"):
        post_key = our_post_key if name == "our_sparse_sdf_vqvae" else "post_morton"
        for sort_label in ("pre_morton", post_key):
            lines.append(f"## {name} / {sort_label}")
            for metric, label in (
                ("h1", "H1 unigram Shannon (nats) mean"),
                ("h2", "H2 Renyi-2 (nats) mean"),
                ("hbg", "H bigram Shannon (nats) mean"),
                ("smed", "spatial gap median mean (26-neigh in seq)"),
                ("smean", "spatial gap mean mean"),
                ("sp90", "spatial gap p90 mean"),
                ("face", "seq adjacent face-neighbor (L1=1) frac"),
                ("26", "seq adjacent 26-neighbor frac"),
                ("euclm", "seq adjacent center euclidean mean"),
                ("c8", "nonoverlap 8-token 2x2x2 exact-cube frac"),
                ("c8s", "sliding 8-token 2x2x2 exact-cube frac"),
            ):
                key = f"{name}_{sort_label}_{metric}"
                vals = agg.get(key, [])
                if vals:
                    lines.append(f"  {label}: {float(np.mean(vals)):.6g} (n={len(vals)})")
                else:
                    lines.append(f"  {label}: n/a")
            lines.append("")
        # 码本使用（按 mesh、仅 pre 行聚合，各样本一次）
        lines.append(f"## {name} / code usage (per mesh, encoder indices)")
        for metric, label in (
            ("n_unique", "mean n_unique_codes"),
            ("unique_over_len", "mean unique_over_seq_len (= n_unique/len)"),
        ):
            key = f"{name}_{metric}"
            vals = agg.get(key, [])
            if vals:
                lines.append(f"  {label}: {float(np.mean(vals)):.6g} (n={len(vals)})")
            else:
                lines.append(f"  {label}: n/a")
        ctr = our_code_global if name == "our_sparse_sdf_vqvae" else sh_code_global
        tot_tok = sum(ctr.values())
        used_ids = sum(1 for i in range(k_cb) if ctr.get(i, 0) > 0)
        lines.append(
            f"  corpus: total_quant_assignments={tot_tok}  distinct_ids_used={used_ids}/{k_cb}"
        )
        if tot_tok > 0:
            top5 = ctr.most_common(5)
            lines.append(
                "  top5 codes by count: "
                + ", ".join(f"id {a}={b} ({100.0*b/tot_tok:.4f}%)" for a, b in top5)
            )
        lines.append("")
        lines.append(f"## {name} / post_minus_pre entropy (per mesh, mean over samples)")
        for metric, label in (
            ("d_h1", "mean d_h1_nat"),
            ("d_h2", "mean d_h2_nat"),
            ("d_hbg", "mean d_h_bigram_nat"),
        ):
            key = f"{name}_{metric}"
            vals = agg.get(key, [])
            if vals:
                lines.append(f"  {label}: {float(np.mean(vals)):.6g} (n={len(vals)})")
            else:
                lines.append(f"  {label}: n/a")
        lines.append("")

    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(
        f"[done] wrote {csv_path} , {jsonl_path} , {summary_path} , "
        f"{freq_our_path.name} , {freq_sh_path.name}"
    )


if __name__ == "__main__":
    main()
