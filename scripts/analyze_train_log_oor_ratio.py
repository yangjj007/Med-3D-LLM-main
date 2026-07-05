#!/usr/bin/env python3
"""
从 train*.log 中解析 [LOSS-SUMMARY] 行，计算 oor_ratio = n_oor / M_out，
其中 M_out = n_in + n_sharp + n_extra + n_oor（与训练代码中 decoder 输出体素数一致）。

用法:
  python scripts/analyze_train_log_oor_ratio.py path/to/train2.log
  python scripts/analyze_train_log_oor_ratio.py path/to/train2.log --per-step max
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np

# [LOSS-SUMMARY] step=61 | n_in=82566 | n_sharp=12898 | n_extra=375113 | n_oor=20433519 | ...
LOSS_SUMMARY_RE = re.compile(
    r"\[LOSS-SUMMARY\]\s*step=(?P<step>\d+)\s*\|\s*"
    r"n_in=(?P<n_in>\d+)\s*\|\s*n_sharp=(?P<n_sharp>\d+)\s*\|\s*"
    r"n_extra=(?P<n_extra>\d+)\s*\|\s*n_oor=(?P<n_oor>\d+)"
)

# 可选：与 recon 行交叉校验
RECON_NOUT_RE = re.compile(r"recon:\s*N_out=(?P<n_out>\d+)")


def parse_log(path: Path) -> Tuple[List[dict], List[Tuple[int, int]]]:
    """返回 (records, recon_n_out_sequence 仅顺序列表 (line_hint, n_out))。"""
    records: List[dict] = []
    recon_seq: List[Tuple[int, int]] = []

    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for lineno, line in enumerate(text, start=1):
        m = RECON_NOUT_RE.search(line)
        if m:
            recon_seq.append((lineno, int(m.group("n_out"))))

        m = LOSS_SUMMARY_RE.search(line)
        if not m:
            continue
        n_in = int(m.group("n_in"))
        n_sharp = int(m.group("n_sharp"))
        n_extra = int(m.group("n_extra"))
        n_oor = int(m.group("n_oor"))
        m_out = n_in + n_sharp + n_extra + n_oor
        if m_out <= 0:
            continue
        oor_ratio = n_oor / m_out
        records.append(
            {
                "line": lineno,
                "step": int(m.group("step")),
                "n_in": n_in,
                "n_sharp": n_sharp,
                "n_extra": n_extra,
                "n_oor": n_oor,
                "M_out": m_out,
                "oor_ratio": oor_ratio,
            }
        )
    return records, recon_seq


def summarize(name: str, ratios: np.ndarray) -> None:
    if ratios.size == 0:
        print(f"\n=== {name} ===\n  (无数据)")
        return
    r = np.sort(ratios.astype(np.float64))
    mean = float(r.mean())
    std = float(r.std(ddof=0))
    var = float(r.var(ddof=0))
    med = float(np.median(r))
    p25, p75 = float(np.percentile(r, 25)), float(np.percentile(r, 75))
    p90 = float(np.percentile(r, 90))
    p95 = float(np.percentile(r, 95))
    p99 = float(np.percentile(r, 99))
    mn, mx = float(r.min()), float(r.max())
    print(f"\n=== {name} (N={r.size}) ===")
    print(f"  min      : {mn:.6f}")
    print(f"  max      : {mx:.6f}")
    print(f"  mean     : {mean:.6f}")
    print(f"  std      : {std:.6f}")
    print(f"  variance : {var:.6f}")
    print(f"  median   : {med:.6f}")
    print(f"  p25 / p75: {p25:.6f} / {p75:.6f}")
    print(f"  p90      : {p90:.6f}")
    print(f"  p95      : {p95:.6f}")
    print(f"  p99      : {p99:.6f}")


def aggregate_per_step(
    records: List[dict], how: Literal["max", "min", "mean"]
) -> np.ndarray:
    by_step: dict = defaultdict(list)
    for rec in records:
        by_step[rec["step"]].append(rec["oor_ratio"])
    out: List[float] = []
    for step in sorted(by_step):
        arr = np.array(by_step[step], dtype=np.float64)
        if how == "max":
            out.append(float(arr.max()))
        elif how == "min":
            out.append(float(arr.min()))
        else:
            out.append(float(arr.mean()))
    return np.array(out, dtype=np.float64)


def recommend_threshold(ratios: np.ndarray, per_step_max: np.ndarray | None) -> None:
    print("\n=== 配置 extra_oor_ratio_threshold 建议 ===\n")
    print(
        "说明: 训练中当 oor_ratio > 阈值时，policy=sample 会整步跳过 backward/optimizer；\n"
        "DDP 下使用「任一 rank 超过阈值则全员跳过」，因此与 **每步各 rank 的 max(oor_ratio)** 最相关。\n"
    )
    if per_step_max is not None and per_step_max.size > 0:
        r = np.sort(per_step_max)
        for label, q in [
            ("约跳过最少 ~1% 的训练步 (p99)", float(np.percentile(r, 99))),
            ("约跳过 ~5% 的步 (p95)", float(np.percentile(r, 95))),
            ("约跳过 ~10% 的步 (p90)", float(np.percentile(r, 90))),
            ("约跳过 ~25% 的步 (p75)", float(np.percentile(r, 75))),
            ("约跳过 ~50% 的步 (median)", float(np.median(r))),
        ]:
            thr = q
            print(f"  - {label}: 阈值约 {thr:.4f}  (严格 > 该值才跳过; 参考分位数)")
        print(
            "\n  实操: 若主要想 **避免极端 OOM 步** 且尽量少跳步，可取 **p90～p95** 略加一点余量；\n"
            "  若当前日志里大量步 oor_ratio 都很高、宁可多跳步保显存，可取 **p50～p75**。\n"
            "  阈值略 **低于** 你想拦截的分位数，会跳得更多。"
        )
    if ratios.size:
        print(
            f"\n  全 rank 原始记录参考: 全体 mean={ratios.mean():.4f}, "
            f"median={float(np.median(ratios)):.4f}。"
        )

    if per_step_max is not None and per_step_max.size > 0:
        lo = float(per_step_max.min())
        hi = float(per_step_max.max())
        print(
            f"\n注意: 按 step 的 max(oor_ratio) 范围为 [{lo:.4f}, {hi:.4f}]。\n"
            f"若 extra_oor_ratio_threshold 设得 **低于** {lo:.4f}，则几乎 **每一步** 都会触发跳过；\n"
            f"若希望只在「更差的一小撮步」上跳过，阈值应接近上界（例如 p90~p99 给出的值）。"
        )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="分析训练日志中的 oor_ratio 分布")
    ap.add_argument("log_path", type=Path, help="train2.log 等日志路径")
    ap.add_argument(
        "--per-step",
        choices=("max", "min", "mean", "none"),
        default="max",
        help="按全局 step 聚合多卡记录（DDP 下跳过逻辑与 max 最相关；none 不打印聚合）",
    )
    ap.add_argument("--csv", type=Path, default=None, help="可选：导出每条 LOSS-SUMMARY 为 CSV")
    args = ap.parse_args()

    if not args.log_path.is_file():
        print(f"文件不存在: {args.log_path}", file=sys.stderr)
        return 1

    records, recon_seq = parse_log(args.log_path)
    if not records:
        print("未匹配到任何 [LOSS-SUMMARY] 行（请确认日志格式）。", file=sys.stderr)
        return 2

    ratios = np.array([r["oor_ratio"] for r in records], dtype=np.float64)
    summarize("全记录 (每条 LOSS-SUMMARY，通常含多 GPU)", ratios)

    per_step = None
    if args.per_step != "none":
        per_step = aggregate_per_step(records, args.per_step)  # type: ignore[arg-type]
        summarize(f"按 step 聚合: {args.per_step}(各 step 上各 rank 的 oor_ratio)", per_step)

    recommend_threshold(ratios, per_step)

    if args.csv:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["line", "step", "n_in", "n_sharp", "n_extra", "n_oor", "M_out", "oor_ratio"],
            )
            w.writeheader()
            for r in records:
                w.writerow({k: r[k] for k in w.fieldnames})
        print(f"\n已写入 CSV: {args.csv.resolve()}")

    # 与 recon 行数量对比（仅提示，因日志交错不保证一一对应）
    if len(recon_seq) and len(records):
        print(
            f"\n(参考) 日志中 recon:N_out 行数={len(recon_seq)}, LOSS-SUMMARY 行数={len(records)}。"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
