"""
从本地 TRELLIS / ObjaverseXL（hf-objaverse-v1）目录中随机抽取若干已存在的 .glb，
并复制到新目录，同时写出对应的 metadata.csv 子集与 object-paths.json 子集。

默认 **完全扁平**（与 ShapeLLM-Omni 等用法一致）：所有模型为 ``<sha256>.glb``，与
``metadata.csv``、``object-paths.json`` 同在 ``output_dir`` 根目录。

可选 ``--nested`` 恢复原先分层目录（``glbs/000-xxx/...``），便于直接作为
``sdf_voxelize --format trellis500k`` 的 ``--input_dir``（见 SDF 文档）。

**注意：** 扁平模式请使用**专用空目录**（例如 ``./ObjaverseXL_flat_5k``），不要把 ``--output_dir``
指到已存在 ``glbs/000-xxx`` 下载树的项目 ``data/`` 根目录，否则会与旧目录混杂；脚本会检测并拒绝。

**子集模式：** 传入 ``--subset_metadata`` 时，只读取该 CSV（须含 ``file_identifier``、``sha256``），
用其中的 path_key 在 ``input_dir`` 的 ``object-paths.json`` 与磁盘上做对齐；**不再**依赖全量
TRELLIS ``metadata.csv``。随后仅在「子集内已在本地存在的 glb」中随机抽 ``--num_samples`` 条。

**对齐键：** ``metadata`` 与 ``object-paths.json`` 使用同一套规范化键匹配。非 URL 的
``file_identifier`` 按**完整相对路径**（posix、去首尾斜杠、压 ``//``）对齐，避免仅 basename
相同时错配；若以 ``http://`` / ``https://`` 开头则取 URL 最后一段（与 ``sdf_voxelize`` 中
ObjaverseXL 合并逻辑一致），便于 Sketchfab 类 key 对齐。

使用示例::

 python dataset_toolkits/sample_objaverse_glb_subset.py \
        --input_dir ./TRELLIS-500K/ObjaverseXL/raw/hf-objaverse-v1 \
        --output_dir ./ObjaverseXL_flat_5k \
        --num_samples 5000 \
        --seed 42 \
        --max_workers 32

 仅从你已有的 SDF/筛选 metadata 子集中再随机抽样（需在 input_dir 中已下载对应 glb）::

 python dataset_toolkits/sample_objaverse_glb_subset.py \
        --input_dir ./TRELLIS-500K/ObjaverseXL/raw/hf-objaverse-v1 \
        --subset_metadata ./train_sdf_dataset/res512_thre0.5/metadata.csv \
        --output_dir ./eval_data_100 \
        --num_samples 100 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def _find_metadata_csv(input_dir: str) -> Path:
    """与 sdf_voxelize.load_trellis500k_metadata 一致：在上一级或上上一级找 metadata.csv。"""
    p = Path(input_dir).resolve()
    parent = p.parent
    for _ in range(2):
        cand = parent / "metadata.csv"
        if cand.is_file():
            return cand
        parent = parent.parent
    raise FileNotFoundError(
        "未找到 metadata.csv。已尝试:\n"
        f"  {p.parent / 'metadata.csv'}\n"
        f"  {p.parent.parent / 'metadata.csv'}"
    )


def _looks_like_objaverse_bucket_glbs_dir(glbs_dir: Path) -> bool:
    """是否为 Objaverse 常见的 glbs/000-000 分桶目录。"""
    if not glbs_dir.is_dir():
        return False
    try:
        for child in glbs_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            # 例如 000-000、127-255
            if len(name) == 7 and name[3] == "-" and name.replace("-", "").isdigit():
                return True
    except OSError:
        return False
    return False


def _flat_output_dir_guard(output_root: Path) -> None:
    """
    扁平模式下禁止写到已含 ``glbs/000-xxx`` 的目录，避免与旧数据混杂。
    请使用专用子目录，例如 ``--output_dir ./data/ObjaverseXL_flat_5k``。
    """
    glbs_child = output_root / "glbs"
    if _looks_like_objaverse_bucket_glbs_dir(glbs_child):
        raise RuntimeError(
            f"扁平输出目录下已存在分桶目录: {glbs_child}\n"
            "这会导致 <sha256>.glb 与 glbs/000-xxx 混在一起。\n"
            "请换用空目录或新子目录，例如:\n"
            f"  --output_dir {output_root / 'ObjaverseXL_flat_subset'}\n"
            "若你确认要保留该 glbs/，可再加 --allow_mixed_output_dir（不推荐）。"
        )


def _infer_data_relpath_from_input(input_dir: str) -> Path:
    """
    若路径中包含 ``raw``，则从 ``raw`` 起保留后缀（如 raw/hf-objaverse-v1）；
    否则使用 raw/<input最后一级目录名>。
    """
    p = Path(input_dir).resolve()
    parts = p.parts
    try:
        i = parts.index("raw")
        return Path(*parts[i:])
    except ValueError:
        return Path("raw") / p.name


def _canonical_match_key(s: Any) -> str:
    """
    metadata / object-paths 共用的匹配键。

    - ``https?://...``：取路径最后一段（与 sdf_voxelize._load_objaversexl_paths 一致），
      便于 ``object-paths`` 仅以 Sketchfab id 等为 key 时仍能合并。
    - 否则：完整相对路径规范化，避免不同目录下同名 ``*.glb`` 的 basename 碰撞。
    """
    raw = str(s).strip().replace("\\", "/")
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw.split("/")[-1]
    p = raw.lstrip("/")
    while "//" in p:
        p = p.replace("//", "/")
    return p


def _path_key_to_sha256_map(metadata_df: pd.DataFrame, path_key_series: pd.Series) -> Dict[str, str]:
    """每个规范化匹配键 -> sha256（小写十六进制字符串）。"""
    t = pd.DataFrame(
        {
            "_path_key": path_key_series.astype(str),
            "sha256": metadata_df["sha256"].astype(str).str.strip().str.lower(),
        }
    )
    t = t.drop_duplicates(subset=["_path_key"], keep="first")
    return dict(zip(t["_path_key"], t["sha256"]))


def _default_max_workers() -> int:
    n = os.cpu_count() or 8
    return max(8, min(128, n * 4))


def _build_candidates(
    input_dir: Path,
    object_paths: Dict[str, str],
    path_key_set: set[str],
    path_key_to_sha256: Mapping[str, str],
    max_workers: int,
    stat_batch_size: int,
    show_scan_progress: bool,
) -> List[Tuple[str, str, Path, str, str]]:
    """
    返回 [(path_key, rel_path, abs_glb_path, object_paths_key, sha256), ...]，
    仅包含磁盘上存在且在 metadata 中通过 ``_canonical_match_key(file_identifier)`` 能匹配的行。

    按 ``object_paths`` 迭代顺序，同一 ``path_key`` 在首次 ``is_file`` 为真时收录；
    ``is_file`` 按批并行以加速大量路径探测。
    """
    rows: List[Tuple[str, str, Path, str, str]] = []
    seen_path_key: set[str] = set()
    buffer: List[Tuple[str, str, Path, str]] = []

    def flush(executor: ThreadPoolExecutor) -> None:
        if not buffer:
            return
        paths = [b[2] for b in buffer]
        exists = list(executor.map(Path.is_file, paths))
        for (path_key, rel_path, abs_glb, op_key), ok in zip(buffer, exists):
            if not ok:
                continue
            if path_key in seen_path_key:
                continue
            sha = path_key_to_sha256.get(path_key)
            if not sha:
                continue
            seen_path_key.add(path_key)
            rows.append((path_key, rel_path, abs_glb.resolve(), op_key, sha))
        buffer.clear()

    n_op = len(object_paths)
    use_bar = show_scan_progress and tqdm is not None and n_op > 500
    iterator = object_paths.items()
    if use_bar:
        iterator = tqdm(  # type: ignore[assignment]
            iterator,
            total=n_op,
            desc="扫描 object-paths（匹配 metadata + is_file）",
            unit="key",
            mininterval=0.5,
        )

    workers = max(1, max_workers)
    t0 = time.perf_counter()
    scanned = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for file_identifier, rel_path in iterator:
            scanned += 1
            if (
                not use_bar
                and show_scan_progress
                and scanned % 100_000 == 0
            ):
                elapsed = time.perf_counter() - t0
                print(
                    f"  …已遍历 object-paths {scanned}/{n_op} 条（{elapsed:.0f}s），"
                    f"当前已命中磁盘上存在 {len(rows)} 个",
                    flush=True,
                )
            path_key = _canonical_match_key(file_identifier)
            if path_key in seen_path_key or path_key not in path_key_set:
                continue
            abs_glb = input_dir / rel_path
            buffer.append((path_key, rel_path, abs_glb, str(file_identifier)))
            if len(buffer) >= stat_batch_size:
                flush(executor)
        flush(executor)

    return rows


def _invert_object_paths_for_allowlist(
    object_paths: Dict[str, str],
    path_key_allowlist: set[str],
    show_scan_progress: bool,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    单次遍历 ``object_paths``：仅当 ``_canonical_match_key(op_key)`` 落在 allowlist 内时，
    记录 ``path_key -> [(object_paths 原始 key, rel_path), ...]``（保留 ``object_paths`` 迭代顺序）。
    """
    inv: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    n_op = len(object_paths)
    it = object_paths.items()
    use_bar = show_scan_progress and tqdm is not None and n_op > 500
    if use_bar:
        it = tqdm(  # type: ignore[assignment]
            it,
            total=n_op,
            desc="索引 object-paths（仅子集 path_key）",
            unit="key",
            mininterval=0.5,
        )
    scanned = 0
    t0 = time.perf_counter()
    for op_key, rel_path in it:
        scanned += 1
        if (
            not use_bar
            and show_scan_progress
            and scanned % 100_000 == 0
        ):
            elapsed = time.perf_counter() - t0
            print(
                f"  …已遍历 object-paths {scanned}/{n_op} 条（{elapsed:.0f}s），"
                f"子集 path_key 命中索引 {len(inv)} 个",
                flush=True,
            )
        pk = _canonical_match_key(op_key)
        if pk in path_key_allowlist:
            inv[pk].append((str(op_key), rel_path))
    return inv


def _resolve_candidates_subset(
    input_dir: Path,
    inv: Mapping[str, List[Tuple[str, str]]],
    path_key_ordered: Sequence[str],
    path_key_to_sha256: Mapping[str, str],
    max_workers: int,
    stat_batch_size: int,
    show_scan_progress: bool,
) -> List[Tuple[str, str, Path, str, str]]:
    """
    对子集内的每个 path_key，按 ``inv`` 中顺序尝试 ``is_file``，首个存在的 glb 记为候选。
    返回与 ``_build_candidates`` 相同结构的列表。
    """

    def try_one_pk(pk: str) -> Optional[Tuple[str, str, Path, str, str]]:
        pairs = inv.get(pk)
        if not pairs:
            return None
        sha = path_key_to_sha256.get(pk)
        if not sha:
            return None
        for op_key, rel_path in pairs:
            abs_glb = input_dir / rel_path
            if abs_glb.is_file():
                return (pk, rel_path, abs_glb.resolve(), op_key, sha)
        return None

    pks = list(path_key_ordered)
    rows: List[Tuple[str, str, Path, str, str]] = []
    workers = max(1, max_workers)
    batch = max(1, stat_batch_size)
    use_bar = show_scan_progress and tqdm is not None and len(pks) > 500
    outer = range(0, len(pks), batch)
    if use_bar:
        outer = tqdm(  # type: ignore[assignment]
            outer,
            total=(len(pks) + batch - 1) // batch,
            desc="探测子集 glb（is_file）",
            unit="batch",
            mininterval=0.3,
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in outer:
            chunk = pks[start : start + batch]
            for r in executor.map(try_one_pk, chunk):
                if r is not None:
                    rows.append(r)
    return rows


def _parallel_copy(
    jobs: Sequence[Tuple[Path, Path]],
    max_workers: int,
    desc: str,
    show_progress: bool,
) -> None:
    """并行复制 (src, dst)，目标父目录在各自任务内创建。"""

    def one(job: Tuple[Path, Path]) -> None:
        src, dst = job
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    workers = max(1, max_workers)
    if len(jobs) == 0:
        return
    if len(jobs) == 1:
        one(jobs[0])
        return

    use_bar = show_progress and tqdm is not None and len(jobs) > 10
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, j) for j in jobs]
        if use_bar:
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="file"):
                fut.result()
        else:
            for fut in futures:
                fut.result()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="随机抽取 ObjaverseXL 本地 GLB + 对应 metadata / object-paths 子集"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="含 glbs/ 与 object-paths.json 的目录（如 .../raw/hf-objaverse-v1）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出根目录；默认扁平布局下 *.glb / object-paths.json / metadata.csv 均在此目录",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="随机抽取条数。全量模式为「全库已下载」中抽取；--subset_metadata 时为「子集内已下载」中抽取。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，便于复现",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help="可选：显式指定全量 TRELLIS metadata.csv；默认按 sdf_voxelize 规则自动查找。"
             "若同时指定 --subset_metadata，则忽略本项。",
    )
    parser.add_argument(
        "--subset_metadata",
        type=str,
        default=None,
        help="仅使用该 CSV 中的记录作为候选池（须含列 file_identifier、sha256；可有 aesthetic_score 等"
             "额外列）。在 input_dir 的 object-paths.json 与磁盘上对齐后，仅在子集内已下载的 glb 中"
             "再随机抽取 --num_samples 条。指定后不再读取全量 TRELLIS metadata.csv。",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=0,
        help="并行线程数：用于批量 is_file 与复制；0 表示自动。远程盘/NFS 上 is_file 过多并发易慢或抖动，可改为 8～16。",
    )
    parser.add_argument(
        "--stat_batch_size",
        type=int,
        default=8192,
        help="每批并行 is_file 的路径条数（默认 8192）。网络盘上可试 2048～4096 并配合较小 --max_workers。",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="禁用 tqdm：全量扫描 / 子集 is_file / 复制 GLB 等阶段的进度条",
    )
    parser.add_argument(
        "--nested",
        action="store_true",
        help="保留 glbs/000-xxx/ 分层目录；默认关闭，即输出为扁平的 <sha256>.glb",
    )
    parser.add_argument(
        "--allow_mixed_output_dir",
        action="store_true",
        help="扁平模式下跳过「output_dir 下已有 glbs/000-xxx」检查（易混杂，一般勿用）",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    op_json = input_dir / "object-paths.json"
    if not op_json.is_file():
        raise FileNotFoundError(f"缺少 object-paths.json: {op_json}")

    if args.subset_metadata:
        if args.metadata_csv:
            print(
                "注意: 已指定 --subset_metadata，将忽略 --metadata_csv，仅以子集 CSV 为准。",
                flush=True,
            )
        metadata_path = Path(args.subset_metadata).resolve()
        if not metadata_path.is_file():
            raise FileNotFoundError(f"subset_metadata 不存在: {metadata_path}")
    elif args.metadata_csv:
        metadata_path = Path(args.metadata_csv).resolve()
    else:
        metadata_path = _find_metadata_csv(str(input_dir))
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv 不存在: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path, low_memory=False)
    if "file_identifier" not in metadata_df.columns or "sha256" not in metadata_df.columns:
        raise ValueError("metadata.csv 需包含列: file_identifier, sha256")

    path_key_series = metadata_df["file_identifier"].map(_canonical_match_key)
    path_key_set: set[str] = set(path_key_series)
    path_key_to_sha256 = _path_key_to_sha256_map(metadata_df, path_key_series)

    max_workers = args.max_workers if args.max_workers > 0 else _default_max_workers()
    stat_batch = max(256, args.stat_batch_size)

    with open(op_json, "r", encoding="utf-8") as f:
        object_paths = json.load(f)

    n_op = len(object_paths)
    n_meta_keys = len(path_key_set)
    print(
        f"并行参数: max_workers={max_workers}, stat_batch_size={stat_batch} "
        f"(可用 --max_workers / --stat_batch_size 调整)",
        flush=True,
    )
    if args.subset_metadata:
        print(
            f"子集模式：子集 CSV 共 {len(metadata_df)} 行，唯一 path_key {n_meta_keys} 个；"
            f"object-paths 共 {n_op} 条。将先索引命中的 path_key，再对子集做 is_file（不扫全库键空间）。",
            flush=True,
        )
    else:
        print(
            f"开始扫描磁盘：object-paths 共 {n_op} 条，metadata 中可匹配 path_key 共 {n_meta_keys} 个。\n"
            "全量 ObjaverseXL / TRELLIS 常为数十万级；需逐条过滤并对已下载路径做 is_file，"
            "在机械盘、网络盘上可能持续数分钟至更久，并非死机。",
            flush=True,
        )

    if args.subset_metadata:
        inv = _invert_object_paths_for_allowlist(
            object_paths,
            path_key_set,
            show_scan_progress=not args.no_progress,
        )
        path_key_ordered: List[str] = []
        _seen_pk: set[str] = set()
        for pk in path_key_series.astype(str):
            if pk in _seen_pk:
                continue
            _seen_pk.add(pk)
            path_key_ordered.append(pk)
        candidates = _resolve_candidates_subset(
            input_dir,
            inv,
            path_key_ordered,
            path_key_to_sha256,
            max_workers,
            stat_batch,
            show_scan_progress=not args.no_progress,
        )
    else:
        candidates = _build_candidates(
            input_dir,
            object_paths,
            path_key_set,
            path_key_to_sha256,
            max_workers,
            stat_batch,
            show_scan_progress=not args.no_progress,
        )
    n_avail = len(candidates)
    print(
        f"扫描完成：磁盘上已下载且可对齐的样本共 {n_avail} 个。",
        flush=True,
    )
    if n_avail == 0:
        if args.subset_metadata:
            raise RuntimeError(
                "没有可用样本：子集 CSV 中的 path_key 在 object-paths.json 中无映射，"
                "或 input_dir 下对应 glb 尚未下载。请核对 file_identifier 是否与 TRELLIS 一致。"
            )
        raise RuntimeError(
            "没有可用样本：请确认 glb 已下载，且 metadata 与 object-paths 的 file_identifier 可对齐。"
        )

    k = min(args.num_samples, n_avail)
    if k < args.num_samples:
        print(f"警告: 仅找到 {n_avail} 个可用样本，少于请求的 {args.num_samples}，将使用全部 {k} 个。")

    if args.seed is not None:
        random.seed(args.seed)

    chosen = random.sample(candidates, k)
    chosen_path_keys = {t[0] for t in chosen}

    use_nested = args.nested
    if use_nested:
        data_relpath = _infer_data_relpath_from_input(str(input_dir))
        out_data_dir = output_root / data_relpath
    else:
        if not args.allow_mixed_output_dir:
            _flat_output_dir_guard(output_root)
        out_data_dir = output_root
    out_data_dir.mkdir(parents=True, exist_ok=True)

    subset_paths: Dict[str, str] = {}
    copy_jobs: List[Tuple[Path, Path]] = []
    copied_records: List[Dict[str, str]] = []

    for _path_key, rel_path, abs_glb, op_key, sha256 in chosen:
        if use_nested:
            dest_rel = rel_path
            dest = out_data_dir / rel_path
        else:
            dest_name = f"{sha256}.glb"
            dest_rel = dest_name
            dest = out_data_dir / dest_name
        subset_paths[op_key] = dest_rel
        copy_jobs.append((abs_glb, dest))
        if not use_nested:
            copied_records.append(
                {
                    "sha256": sha256,
                    "source_relative": rel_path,
                    "dest_filename": dest.name,
                }
            )

    _parallel_copy(
        copy_jobs,
        max_workers=max_workers,
        desc="复制 GLB",
        show_progress=not args.no_progress,
    )

    out_op = out_data_dir / "object-paths.json"
    with open(out_op, "w", encoding="utf-8") as f:
        json.dump(subset_paths, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if copied_records:
        with open(out_data_dir / "copied_files_record.json", "w", encoding="utf-8") as f:
            json.dump(copied_records, f, indent=2, ensure_ascii=False)
            f.write("\n")

    meta_sub = metadata_df[path_key_series.isin(chosen_path_keys)].copy()
    meta_sub["_path_key"] = meta_sub["file_identifier"].map(_canonical_match_key)
    meta_sub = meta_sub.drop_duplicates(subset=["_path_key"], keep="first")
    meta_sub = meta_sub.drop(columns=["_path_key"])
    out_meta = output_root / "metadata.csv"
    meta_sub.to_csv(out_meta, index=False)

    manifest = {
        "input_dir": str(input_dir),
        "source_metadata_csv": str(metadata_path),
        "subset_metadata_csv": str(metadata_path) if args.subset_metadata else None,
        "source_object_paths": str(op_json),
        "output_root": str(output_root),
        "output_data_dir": str(out_data_dir),
        "layout": "nested" if use_nested else "flat",
        "num_requested": args.num_samples,
        "num_copied": k,
        "num_available_matched_on_disk": n_avail,
        "seed": args.seed,
        "max_workers": max_workers,
        "stat_batch_size": stat_batch,
    }
    with open(output_root / "sample_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"已复制 {k} 个 GLB 到: {out_data_dir} （布局: {'nested' if use_nested else 'flat'}）")
    print(f"object-paths.json: {out_op}")
    print(f"metadata.csv: {out_meta}")
    if use_nested:
        print(
            "后续 SDF 处理示例:\n"
            f"  python dataset_toolkits/sdf_voxelize.py --format trellis500k "
            f"--input_dir {out_data_dir} --output_dir <sdf_out> ..."
        )
    else:
        print(
            "扁平布局：所有 .glb 与 object-paths.json、metadata.csv、sample_manifest.json 均在同一目录。\n"
            "请勿把 --output_dir 指到已含 glbs/000-xxx 下载树的项目目录；应使用专用子目录。"
        )


if __name__ == "__main__":
    main()
