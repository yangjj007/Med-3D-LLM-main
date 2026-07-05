"""
Smoke tests for sparse SDF → Marching Cubes batch grouping.

训练快照曾错误地将多轮 ``batch_idx`` 全为 0 的样本 ``cat`` 合并，导致
``sparse_sample_dict_to_trimeshes`` 把多个物体混进同一体素网格。
本文件验证：不同 ``batch_idx`` 必须产生独立 mesh 列表项。

运行（在 Med-3D-LLM-main 根目录）::

    pytest tests/test_sparse_sdf_marching_cubes_batch.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_mc_path = _ROOT / "trellis" / "utils" / "sparse_sdf_marching_cubes.py"
_spec = importlib.util.spec_from_file_location(
    "_sparse_sdf_marching_cubes_testonly", str(_mc_path)
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
sparse_sample_dict_to_trimeshes = _mod.sparse_sample_dict_to_trimeshes


def _tiny_signed_shell(center: tuple[int, int, int], sdf_neg: float = -0.35, sdf_pos: float = 0.12):
    """少量体素：中心为负、三轴邻居为正，便于 MC 产生小曲面。"""
    cx, cy, cz = center
    idx = []
    s = []
    idx.append((cx, cy, cz))
    s.append(sdf_neg)
    for d in (-1, 1):
        idx.append((cx + d, cy, cz))
        s.append(sdf_pos)
        idx.append((cx, cy + d, cz))
        s.append(sdf_pos)
        idx.append((cx, cy, cz + d))
        s.append(sdf_pos)
    return idx, s


def test_sparse_sample_dict_two_batches_separate_meshes():
    """两个 batch、空间分离的小壳层 → 两条 mesh，且质心距离足够大。"""
    R = 32
    idx0, s0 = _tiny_signed_shell((6, 6, 6))
    idx1, s1 = _tiny_signed_shell((22, 22, 22))

    sparse_index = torch.tensor(idx0 + idx1, dtype=torch.long)
    sparse_sdf = torch.tensor([[v] for v in s0 + s1], dtype=torch.float32)
    batch_idx = torch.tensor([0] * len(idx0) + [1] * len(idx1), dtype=torch.long)

    sample = {
        "sparse_sdf": sparse_sdf,
        "sparse_index": sparse_index,
        "batch_idx": batch_idx,
    }
    meshes = sparse_sample_dict_to_trimeshes(sample, R, mc_threshold=0.0)
    assert len(meshes) == 2, f"expected 2 mesh slots, got {len(meshes)}"
    assert meshes[0] is not None and meshes[1] is not None
    c0 = meshes[0].vertices.mean(axis=0)
    c1 = meshes[1].vertices.mean(axis=0)
    dist = float(np.linalg.norm(c0 - c1))
    assert dist > 0.5, f"expected separated surfaces, centroid dist={dist}"


def test_wrong_all_zero_batch_idx_collapses_to_one_volume():
    """回归对照：若两团点都标为 batch 0，只会得到一条合并后的 mesh。"""
    R = 32
    idx0, s0 = _tiny_signed_shell((6, 6, 6))
    idx1, s1 = _tiny_signed_shell((22, 22, 22))

    sparse_index = torch.tensor(idx0 + idx1, dtype=torch.long)
    sparse_sdf = torch.tensor([[v] for v in s0 + s1], dtype=torch.float32)
    batch_idx = torch.zeros(len(idx0) + len(idx1), dtype=torch.long)

    sample = {
        "sparse_sdf": sparse_sdf,
        "sparse_index": sparse_index,
        "batch_idx": batch_idx,
    }
    meshes = sparse_sample_dict_to_trimeshes(sample, R, mc_threshold=0.0)
    assert len(meshes) == 1
    assert meshes[0] is not None


def test_combined_dict_max_batch_plus_one_equals_slots():
    """合并后 ``max(batch_idx)+1`` 应等于 mesh 槽位数（与 run_snapshot 全局偏移一致）。"""
    R = 16
    # 模拟两轮各 1 个样本、局部 batch_idx 均为 0，合并后应偏移为 0 与 1
    a_idx, a_s = _tiny_signed_shell((4, 4, 4))
    b_idx, b_s = _tiny_signed_shell((12, 12, 12))

    s_a = torch.tensor([[v] for v in a_s], dtype=torch.float32)
    i_a = torch.tensor(a_idx, dtype=torch.long)
    b_a = torch.zeros(len(a_idx), dtype=torch.long)

    s_b = torch.tensor([[v] for v in b_s], dtype=torch.float32)
    i_b = torch.tensor(b_idx, dtype=torch.long)
    b_b = torch.zeros(len(b_idx), dtype=torch.long) + 1  # 第二轮全局偏移

    combined = {
        "sparse_sdf": torch.cat([s_a, s_b], dim=0),
        "sparse_index": torch.cat([i_a, i_b], dim=0),
        "batch_idx": torch.cat([b_a, b_b], dim=0),
    }
    meshes = sparse_sample_dict_to_trimeshes(combined, R, 0.0)
    B = int(combined["batch_idx"].max().item()) + 1
    assert B == 2
    assert len(meshes) == B
