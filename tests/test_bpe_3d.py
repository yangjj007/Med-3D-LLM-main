"""
Lightweight tests for BPE3DTokenizer + Morton+mesh pair 文本（无 trellis SparseTensor 依赖）。

运行方式（请在 Med-3D-LLM-main 项目根目录下执行，以便正确导入 bpe_3d）：

  # 使用 pytest
  pytest tests/test_bpe_3d.py -q

  # 直接运行本文件（会打印随机 roundtrip 的前后序列对比）
  python tests/test_bpe_3d.py

  # 打印 pytest 中的详细输出
  pytest tests/test_bpe_3d.py -q -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bpe_3d import (  # noqa: E402
    BPE3DTokenizer,
    MESH_TOKEN_RE,
    morton3d_encode_xyz32,
    parse_morton_mesh_pairs,
    serialize_morton_mesh_pairs,
)


class DummySparseTensor:
    def __init__(self, feats: torch.Tensor, coords: torch.Tensor) -> None:
        self.feats = feats
        self.coords = coords

    def replace(self, new_feats: torch.Tensor) -> "DummySparseTensor":
        return DummySparseTensor(new_feats, self.coords)


def _leaf_set_from_sparse(st: DummySparseTensor) -> set[tuple[int, int, int, int]]:
    feats = st.feats.detach().cpu().numpy().astype(np.int64).reshape(-1)
    coords = st.coords.detach().cpu().numpy().astype(np.int64)
    out: set[tuple[int, int, int, int]] = set()
    for i in range(len(feats)):
        b, x, y, z = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]), int(coords[i, 3])
        out.add((int(feats[i]), b, x, y, z))
    return out


def _make_enc_st(tokens: np.ndarray, coords_xyz: np.ndarray) -> DummySparseTensor:
    m = int(tokens.shape[0])
    feats = torch.tensor(tokens, dtype=torch.float32).unsqueeze(-1)
    batch_col = torch.zeros((m, 1), dtype=torch.int32)
    xyz = torch.tensor(coords_xyz.astype(np.int64), dtype=torch.int32)
    coords4 = torch.cat([batch_col, xyz], dim=1)
    return DummySparseTensor(feats, coords4)


def test_roundtrip_single_merge() -> None:
    base = 128
    t0, t1 = 3, 7
    tokens = np.asarray([t0, t1], dtype=np.int64)
    coords = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    corpus = [{"tokens": tokens, "coords": coords}]

    tok = BPE3DTokenizer(base_vocab_size=base)
    tok.train(corpus, num_merges=1, min_freq=1, verbose=False)

    st_in = _make_enc_st(tokens, coords)
    out = tok.encode_sparse(st_in, sparse_tensor_cls=DummySparseTensor)
    dec = tok.decode_to_sparse(out["batches"], device=torch.device("cpu"), sparse_tensor_cls=DummySparseTensor)

    expected = _leaf_set_from_sparse(st_in)
    got = _leaf_set_from_sparse(dec)
    assert got == expected


def test_nonoverlap_merge() -> None:
    base = 128
    t0, t1 = 5, 9
    tokens = np.asarray([t0, t1, t0, t1], dtype=np.int64)
    coords = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.int64)
    corpus = [{"tokens": tokens, "coords": coords}]

    tok = BPE3DTokenizer(base_vocab_size=base)
    tok.train(corpus, num_merges=1, min_freq=1, verbose=False)

    st_in = _make_enc_st(tokens, coords)
    out = tok.encode_sparse(st_in, sparse_tensor_cls=DummySparseTensor)
    b0 = out["batches"][0]
    assert b0["ids"].shape[0] == 2
    anc = b0["anchors"]
    anchors = {(int(anc[i, 0]), int(anc[i, 1]), int(anc[i, 2])) for i in range(anc.shape[0])}
    assert anchors == {(0, 0, 0), (2, 0, 0)}


def test_macro_id_range() -> None:
    base = 128
    t0, t1 = 11, 22
    tokens = np.asarray([t0, t1, t0, t1], dtype=np.int64)
    coords = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.int64)
    corpus = [{"tokens": tokens, "coords": coords}]

    tok = BPE3DTokenizer(base_vocab_size=base)
    tok.train(corpus, num_merges=1, min_freq=1, verbose=False)
    assert tok.vocab_size == base + 1

    st_in = _make_enc_st(tokens, coords)
    out = tok.encode_sparse(st_in, sparse_tensor_cls=DummySparseTensor)
    macro_ids = out["batches"][0]["ids"]
    assert int(macro_ids.max()) <= tok.vocab_size - 1


def test_morton_mesh_serialize_parse_identity() -> None:
    """serialize -> parse 在合法 macro 上应完全还原 ids 与 anchors。"""
    ids = np.asarray([1234, 5], dtype=np.int64)
    anchors = np.asarray([[10, 12, 14], [0, 0, 0]], dtype=np.int64)
    text = serialize_morton_mesh_pairs(ids, anchors)
    parsed = parse_morton_mesh_pairs(text)
    assert parsed.dropped_count == 0
    np.testing.assert_array_equal(parsed.ids, ids)
    np.testing.assert_array_equal(parsed.anchors, anchors)


def test_morton_mesh_string_full_pipeline_random() -> None:
    """
    随机叶子 (token, block_xyz) -> BPE encode -> LLM 风格 Morton+mesh 串
    -> parse -> decode：叶子集合应与原始一致（宏序列长度可能变短，但几何可完全复原）。
    """
    rng = np.random.default_rng(42)
    base_vocab = 256
    n = 24
    # 在 [0,31]^3 内随机不重复坐标
    coords_list = []
    seen: set[tuple[int, int, int]] = set()
    while len(coords_list) < n:
        c = tuple(int(x) for x in rng.integers(0, 32, size=3))
        if c not in seen:
            seen.add(c)
            coords_list.append(c)
    coords = np.asarray(coords_list, dtype=np.int64)
    tokens = rng.integers(0, base_vocab, size=n, dtype=np.int64)

    # 用随机子集 + 邻接对构造语料，训练少量 merge，使 encode 后可能出现宏 token
    n_corpus = min(n, 12)
    corpus_tokens = tokens[:n_corpus].copy()
    corpus_coords = coords[:n_corpus].copy()
    corpus = [{"tokens": corpus_tokens, "coords": corpus_coords}]

    tok = BPE3DTokenizer(base_vocab_size=base_vocab)
    tok.train(corpus, num_merges=4, min_freq=1, verbose=False)

    st_in = _make_enc_st(tokens, coords)
    out = tok.encode_sparse(st_in, sparse_tensor_cls=DummySparseTensor)
    b0 = out["batches"][0]
    macro_ids = b0["ids"]
    macro_anchors = b0["anchors"]

    # BPE 前：按输入顺序的叶子 Morton+mesh 串（对数 = n）
    before_str = serialize_morton_mesh_pairs(tokens, coords)
    assert len(MESH_TOKEN_RE.findall(before_str)) == n
    # BPE 后：宏 token + 代表 anchor
    after_str = serialize_morton_mesh_pairs(macro_ids, macro_anchors)

    assert macro_ids.shape[0] == macro_anchors.shape[0]
    assert macro_ids.shape[0] <= tokens.shape[0]
    assert len(MESH_TOKEN_RE.findall(after_str)) == int(macro_ids.shape[0])

    parsed = parse_morton_mesh_pairs(after_str, max_mesh_id=tok.vocab_size)
    assert parsed.dropped_count == 0
    np.testing.assert_array_equal(parsed.ids, macro_ids)
    np.testing.assert_array_equal(parsed.anchors, macro_anchors)

    batch = [{"ids": parsed.ids, "anchors": parsed.anchors}]
    dec = tok.decode_to_sparse(batch, device=torch.device("cpu"), sparse_tensor_cls=DummySparseTensor)
    assert _leaf_set_from_sparse(dec) == _leaf_set_from_sparse(st_in)


def _synthetic_corpus(
    rng: np.random.Generator,
    *,
    n_samples: int = 10,
    points_per_sample: int = 96,
    coord_max: int = 28,
) -> list[dict[str, np.ndarray]]:
    frequent_pairs = [(3, 7), (11, 22), (5, 9)]
    corpus: list[dict[str, np.ndarray]] = []
    for _ in range(n_samples):
        occ: set[tuple[int, int, int]] = set()
        pts: list[tuple[int, int, int]] = []
        toks: list[int] = []
        for _inj in range(max(12, points_per_sample // 10)):
            x = int(rng.integers(0, max(1, coord_max - 1)))
            y = int(rng.integers(0, coord_max))
            z = int(rng.integers(0, coord_max))
            a = (x, y, z)
            b = (x + 1, y, z)
            if b[0] >= coord_max or a in occ or b in occ:
                continue
            pa, pb = frequent_pairs[int(rng.integers(0, len(frequent_pairs)))]
            pts.extend([a, b])
            toks.extend([pa, pb])
            occ.add(a)
            occ.add(b)
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
            toks.append(int(rng.integers(0, 64)))
        corpus.append(
            {
                "tokens": np.asarray(toks, dtype=np.int64),
                "coords": np.asarray(pts, dtype=np.int64),
            }
        )
    return corpus


def _merge_tables_equal(a: BPE3DTokenizer, b: BPE3DTokenizer) -> bool:
    if len(a.merge_table) != len(b.merge_table):
        return False
    for ea, eb in zip(a.merge_table, b.merge_table):
        if ea.new_id != eb.new_id or ea.pair != eb.pair:
            return False
        if tuple(ea.rel_offset) != tuple(eb.rel_offset):
            return False
    return True


def test_incremental_matches_legacy_merge_table() -> None:
    """incremental (dirty heap) and legacy single-thread must pick the same merges."""
    corpus = _synthetic_corpus(np.random.default_rng(99))
    base = 64
    num_merges = 16
    min_freq = 2

    os.environ["BPE3D_DEFER_HEAP"] = "1"
    os.environ["BPE3D_HEAP_FLUSH"] = "dirty"
    tok_inc = BPE3DTokenizer(base_vocab_size=base)
    tok_inc.train(
        corpus,
        num_merges=num_merges,
        min_freq=min_freq,
        verbose=False,
        train_mode="incremental",
    )

    tok_leg = BPE3DTokenizer(base_vocab_size=base)
    tok_leg.train(
        corpus,
        num_merges=num_merges,
        min_freq=min_freq,
        verbose=False,
        train_mode="legacy",
        num_workers=1,
    )
    assert _merge_tables_equal(tok_inc, tok_leg)

    os.environ["BPE3D_HEAP_FLUSH"] = "rebuild"
    tok_rebuild = BPE3DTokenizer(base_vocab_size=base)
    tok_rebuild.train(
        corpus,
        num_merges=num_merges,
        min_freq=min_freq,
        verbose=False,
        train_mode="incremental",
    )
    assert _merge_tables_equal(tok_inc, tok_rebuild)


def test_morton_mesh_string_after_merge_roundtrip() -> None:
    """与 test_roundtrip_single_merge 相同几何，再走一遍文本 parse 链路。"""
    base = 128
    t0, t1 = 3, 7
    tokens = np.asarray([t0, t1], dtype=np.int64)
    coords = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    corpus = [{"tokens": tokens, "coords": coords}]

    tok = BPE3DTokenizer(base_vocab_size=base)
    tok.train(corpus, num_merges=1, min_freq=1, verbose=False)

    st_in = _make_enc_st(tokens, coords)
    out = tok.encode_sparse(st_in, sparse_tensor_cls=DummySparseTensor)
    b0 = out["batches"][0]
    text = serialize_morton_mesh_pairs(b0["ids"], b0["anchors"])
    parsed = parse_morton_mesh_pairs(text, max_mesh_id=tok.vocab_size)
    assert parsed.dropped_count == 0
    dec = tok.decode_to_sparse(
        [{"ids": parsed.ids, "anchors": parsed.anchors}],
        device=torch.device("cpu"),
        sparse_tensor_cls=DummySparseTensor,
    )
    assert _leaf_set_from_sparse(dec) == _leaf_set_from_sparse(st_in)


def _print_random_roundtrip_demo() -> None:
    rng = np.random.default_rng(2026)
    base_vocab = 128
    n = 8
    coords_list = []
    seen: set[tuple[int, int, int]] = set()
    while len(coords_list) < n:
        c = tuple(int(x) for x in rng.integers(0, 32, size=3))
        if c not in seen:
            seen.add(c)
            coords_list.append(c)
    coords = np.asarray(coords_list, dtype=np.int64)
    tokens = rng.integers(0, base_vocab, size=n, dtype=np.int64)

    tok = BPE3DTokenizer(base_vocab_size=base_vocab)
    tok.train([{"tokens": tokens.copy(), "coords": coords.copy()}], num_merges=2, min_freq=1, verbose=False)

    st_in = _make_enc_st(tokens, coords)
    out = tok.encode_sparse(st_in, sparse_tensor_cls=DummySparseTensor)
    macro_ids = out["batches"][0]["ids"]
    macro_anchors = out["batches"][0]["anchors"]

    before_str = serialize_morton_mesh_pairs(tokens, coords)
    after_str = serialize_morton_mesh_pairs(macro_ids, macro_anchors)

    print("=== Morton+mesh 序列：BPE 前（叶子，按输入顺序）===")
    print(before_str[:500] + ("..." if len(before_str) > 500 else ""))
    print(f"  叶子对数={n}, 字符长度={len(before_str)}")
    print("=== Morton+mesh 序列：BPE 后（宏 token + 代表 anchor）===")
    print(after_str[:500] + ("..." if len(after_str) > 500 else ""))
    print(f"  宏对数={macro_ids.shape[0]}, 字符长度={len(after_str)}")
    print("=== 叶子 (token, bx,by,bz) 预览（前 5 个）===")
    for i in range(min(5, n)):
        m = int(morton3d_encode_xyz32(coords[i]))
        print(f"  leaf: morton={m} mesh={int(tokens[i])} xyz={tuple(coords[i].tolist())}")

    parsed = parse_morton_mesh_pairs(after_str, max_mesh_id=tok.vocab_size)
    dec = tok.decode_to_sparse(
        [{"ids": parsed.ids, "anchors": parsed.anchors}],
        device=torch.device("cpu"),
        sparse_tensor_cls=DummySparseTensor,
    )
    ok = _leaf_set_from_sparse(dec) == _leaf_set_from_sparse(st_in)
    print(f"=== parse+decode 后叶子集合与原始一致: {ok} (dropped_pairs={parsed.dropped_count}) ===")


if __name__ == "__main__":
    _print_random_roundtrip_demo()
    # 无 pytest 时至少跑关键断言
    test_morton_mesh_serialize_parse_identity()
    test_morton_mesh_string_full_pipeline_random()
    test_morton_mesh_string_after_merge_roundtrip()
    print("manual checks: ok")
