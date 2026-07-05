"""
Offline 3D BPE training: VQVAE Encode -> block (token, coord) corpus -> BPE3DTokenizer.train -> merge_table.json.

Reuses load_vae_from_config, collate_inputs_3d, _stdio_suppressed from build_qwen3vl_sft_3d_jsonl.py.

Usage:
  python dataset_toolkits/train_bpe_3d.py \\
    --sdf_dir ./train_sdf_dataset/res512_thre0.1 \\
    --vae_config ./configs/vae/sdf_vqvae_stage2.json \\
    --vae_ckpt ./outputs/.../vqvae_step0000100.pt \\
    --out_merge_table ./merge_table.json

  # Use corpus cache to skip VAE Encode on subsequent runs:
  python dataset_toolkits/train_bpe_3d.py ... --corpus_cache ./bpe_corpus.npz

  # From cached corpus (no VAE needed):
  python dataset_toolkits/train_bpe_3d.py \\
    --corpus_cache ./outputs/bpe_corpus_watertight.npz \\
    --out_merge_table ./outputs/merge_table_watertight.json \\
    --num_merges 4096 --min_freq 2

  # Legacy mode only: parallel workers (default auto = min(cpu,8)):
  python dataset_toolkits/train_bpe_3d.py ... --bpe_train_mode legacy --workers 0

  # Capture full log to file (in addition to stdout):
  python dataset_toolkits/train_bpe_3d.py ... 2>&1 | tee train_bpe.log
  # Or use the built-in --log_file (default ON, auto-named):
  python dataset_toolkits/train_bpe_3d.py ... --log_file ./outputs/train_bpe.log
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

os.environ.setdefault("SPARSE_BACKEND", "torchsparse")

_TOOLKIT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOLKIT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_build_module():
    path = _TOOLKIT_DIR / "build_qwen3vl_sft_3d_jsonl.py"
    spec = importlib.util.spec_from_file_location("build_qwen3vl_sft_3d_jsonl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


from bpe_3d import BPE3DTokenizer  # noqa: E402

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _setup_logging(log_file: Optional[str], verbose: bool) -> logging.Logger:
    """Configure root logger with console handler + optional file handler.

    The 'bpe_3d' logger (used inside BPE3DTokenizer) is also captured.
    """
    root = logging.getLogger()
    root.handlers.clear()
    level = logging.DEBUG if verbose else logging.INFO
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(level)
    root.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        root.addHandler(fh)

    logger = logging.getLogger("train_bpe_3d")
    return logger


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu" or not torch.cuda.is_available():
        if device_str != "cpu" and not torch.cuda.is_available():
            logging.getLogger("train_bpe_3d").warning("CUDA not available; using CPU.")
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda:0")
    return torch.device(device_str)


def _encode_batch_to_corpus_entries(
    vae: torch.nn.Module,
    collated: Dict[str, torch.Tensor],
    device: torch.device,
    silent: bool,
    inputs_3d_to_device,
    stdio_suppressed,
) -> List[Dict[str, np.ndarray]]:
    def _forward() -> List[Dict[str, np.ndarray]]:
        with torch.no_grad():
            x = inputs_3d_to_device(collated, device)
            vae_f = vae.float() if next(vae.parameters()).dtype != torch.float32 else vae
            enc_st = vae_f.Encode(x)
        coords_t = enc_st.coords
        feats_t = enc_st.feats
        coords_np = coords_t.detach().cpu().numpy().astype(np.int64)
        ids_np = feats_t.squeeze(-1).detach().cpu().numpy().astype(np.int64)
        if coords_np.shape[1] != 4:
            raise ValueError(f"Encode coords must be [N,4], got {coords_np.shape}")

        batch_ids = sorted(int(x) for x in np.unique(coords_np[:, 0]).tolist())
        out: List[Dict[str, np.ndarray]] = []
        for b in batch_ids:
            mask = coords_np[:, 0] == b
            tok_b = ids_np[mask].astype(np.int64, copy=False)
            xyz_b = coords_np[mask][:, 1:4].astype(np.int64, copy=False)
            out.append({"tokens": tok_b, "coords": xyz_b})
        return out

    if silent:
        with stdio_suppressed():
            return _forward()
    return _forward()


def _save_corpus_npz(
    path: Path,
    corpus: List[Dict[str, np.ndarray]],
    base_vocab_size: int,
    logger: logging.Logger,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    tok_chunks: List[np.ndarray] = []
    coord_chunks: List[np.ndarray] = []
    for rec in corpus:
        t = rec["tokens"].astype(np.int64, copy=False)
        c = rec["coords"].astype(np.int64, copy=False)
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
    msg = f"Wrote corpus cache: {path}  samples={len(corpus)}  points={tokens.shape[0]}"
    print(f"[train_bpe_3d] {msg}", flush=True)
    logger.info(msg)


def _load_corpus_npz(
    path: Path, logger: logging.Logger
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    data = np.load(path, allow_pickle=False)
    offsets = data["offsets"]
    tokens = data["tokens"]
    coords = data["coords"]
    base_vocab_size = int(data["base_vocab_size"][0])
    corpus: List[Dict[str, np.ndarray]] = []
    for i in range(len(offsets) - 1):
        a, b = int(offsets[i]), int(offsets[i + 1])
        corpus.append(
            {
                "tokens": np.asarray(tokens[a:b], dtype=np.int64),
                "coords": np.asarray(coords[a:b], dtype=np.int64),
            }
        )
    msg = (
        f"Loaded corpus cache: {path}"
        f"  samples={len(corpus)}"
        f"  points={tokens.shape[0]}"
        f"  base_vocab_size={base_vocab_size}"
    )
    print(f"[train_bpe_3d] {msg}", flush=True)
    logger.info(msg)
    return corpus, base_vocab_size


class _DummySparseTensor:
    """Minimal stand-in for trellis SparseTensor (feats [M,1], coords [M,4])."""

    def __init__(self, feats: torch.Tensor, coords: torch.Tensor) -> None:
        self.feats = feats
        self.coords = coords

    def replace(self, new_feats: torch.Tensor) -> "_DummySparseTensor":
        return _DummySparseTensor(new_feats, self.coords)


def _macro_token_count_encode(tok: BPE3DTokenizer, rec: Dict[str, np.ndarray]) -> int:
    m = int(rec["tokens"].shape[0])
    if m == 0:
        return 0
    feats = torch.tensor(rec["tokens"], dtype=torch.float32).unsqueeze(-1)
    coords = torch.cat(
        [
            torch.zeros((m, 1), dtype=torch.int32),
            torch.tensor(rec["coords"], dtype=torch.int32),
        ],
        dim=1,
    )
    st = _DummySparseTensor(feats, coords)
    out = tok.encode_sparse(st, sparse_tensor_cls=_DummySparseTensor)
    return int(out["batches"][0]["ids"].shape[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train 3D BPE merge table from VQVAE-encoded SDF dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / VAE args
    p.add_argument("--sdf_dir", type=str, default=None,
                   help="SDF dataset dir (not needed when corpus_cache already exists).")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--min_points", type=int, default=100)
    p.add_argument("--max_points", type=int, default=500000)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--vae_config", type=str, default=None,
                   help="VAE JSON config (provides num_embeddings; needed for Encode).")
    p.add_argument("--vae_ckpt", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")

    # BPE training args
    p.add_argument("--num_merges", type=int, default=4096,
                   help="Maximum number of BPE merge operations.")
    p.add_argument("--min_freq", type=int, default=2,
                   help="Stop when best pair frequency drops below this.")
    p.add_argument("--num_embeddings", type=int, default=None,
                   help="Override base codebook size (default: read from vae_config or corpus cache).")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallel workers for legacy BPE only (--bpe_train_mode legacy). "
            "0 = auto min(os.cpu_count(), 8); 1 = single-threaded. "
            "Default incremental is single-threaded; BPE3D_DEFER_HEAP=1 uses dirty-key "
            "heap flush/iter (BPE3D_HEAP_FLUSH=rebuild for full heapify fallback)."
        ),
    )
    p.add_argument(
        "--bpe_train_mode",
        type=str,
        default=None,
        choices=("incremental", "legacy"),
        help="incremental (default) or legacy. If omitted, uses env BPE_TRAIN_MODE or incremental.",
    )

    # I/O
    p.add_argument(
        "--corpus_cache",
        type=str,
        default=None,
        help=(
            "Optional single-file .npz corpus cache. "
            "Loaded if the file exists (skips VAE Encode); "
            "saved after collecting when it does not exist. "
            "Mutually exclusive with --from_build_out."
        ),
    )
    p.add_argument(
        "--from_build_out",
        type=str,
        default=None,
        metavar="OUT_JSONL",
        help=(
            "Load corpus from per-shard .npz files produced by build_qwen3vl_sft_3d_jsonl.py "
            "--enable_bpe auto-train mode. Pass the same value that was used as --out_jsonl "
            "in that script (e.g. ./outputs/qwen3vl_3d_sft.jsonl). "
            "The script will discover and load "
            "<OUT_JSONL>.corpus.part0.npz, <OUT_JSONL>.corpus.part1.npz, … "
            "in ascending order."
        ),
    )
    p.add_argument("--out_merge_table", type=str, default="merge_table.json",
                   help="Output path for the trained BPE merge table.")

    # Logging
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help=(
            "Path to log file. "
            "Default (None): auto-generate <out_merge_table stem>_train.log. "
            "Pass an empty string '' to disable file logging."
        ),
    )
    p.add_argument(
        "--no_log_file",
        action="store_true",
        help="Disable file logging (equivalent to --log_file '').",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Emit detailed per-iteration log lines (default ON).",
    )
    p.add_argument(
        "--no_verbose",
        action="store_true",
        help="Suppress per-iteration stdout prints (progress bar still shown).",
    )
    p.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    p.add_argument(
        "--vqvae_verbose",
        action="store_true",
        help="Do not suppress stdout/stderr during VQVAE Encode.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = args.verbose and not args.no_verbose
    progress = not args.no_progress

    # Resolve log file path
    log_file: Optional[str]
    if args.no_log_file or args.log_file == "":
        log_file = None
    elif args.log_file is not None:
        log_file = args.log_file
    else:
        stem = Path(args.out_merge_table).stem
        log_file = str(Path(args.out_merge_table).parent / f"{stem}_train.log")

    logger = _setup_logging(log_file, verbose)

    if log_file:
        logger.info(f"Log file: {log_file}")
        print(f"[train_bpe_3d] Log file: {log_file}", flush=True)

    t_total_start = time.perf_counter()
    logger.info(
        f"train_bpe_3d started | "
        f"out_merge_table={args.out_merge_table} "
        f"num_merges={args.num_merges} "
        f"min_freq={args.min_freq} "
        f"workers={args.workers}"
    )

    if args.corpus_cache and args.from_build_out:
        raise SystemExit(
            "[train_bpe_3d] --corpus_cache and --from_build_out are mutually exclusive."
        )

    cache_path = Path(args.corpus_cache) if args.corpus_cache else None

    base_vocab_size: int
    corpus: List[Dict[str, np.ndarray]]

    # ── Branch A: load per-shard corpus from build_qwen3vl_sft_3d_jsonl.py ────
    if args.from_build_out:
        import glob as _glob
        base = args.from_build_out
        pattern = base + ".corpus.part*.npz"
        part_paths = sorted(
            _glob.glob(pattern),
            key=lambda p: int(
                Path(p).name.split(".corpus.part")[-1].split(".npz")[0]
            ),
        )
        if not part_paths:
            raise SystemExit(
                f"[train_bpe_3d] No corpus part files found matching: {pattern}\n"
                f"  Make sure build_qwen3vl_sft_3d_jsonl.py was run with --enable_bpe "
                f"and produced corpus files next to --out_jsonl."
            )
        msg = f"Loading {len(part_paths)} corpus shard(s) from --from_build_out={base}"
        print(f"[train_bpe_3d] {msg}", flush=True)
        logger.info(msg)
        corpus = []
        base_vocab_size = 0
        for p in part_paths:
            shard, bvs = _load_corpus_npz(Path(p), logger)
            corpus.extend(shard)
            base_vocab_size = bvs  # all shards share same base_vocab_size
        if args.num_embeddings is not None:
            base_vocab_size = int(args.num_embeddings)
            msg2 = f"Overriding base_vocab_size from CLI: {base_vocab_size}"
            print(f"[train_bpe_3d] {msg2}", flush=True)
            logger.info(msg2)

    # ── Branch B: load single-file cache (old behaviour) ──────────────────────
    elif cache_path is not None and cache_path.is_file():
        corpus, base_vocab_size = _load_corpus_npz(cache_path, logger)
        if args.num_embeddings is not None:
            base_vocab_size = int(args.num_embeddings)
            msg = f"Overriding base_vocab_size from CLI: {base_vocab_size}"
            print(f"[train_bpe_3d] {msg}", flush=True)
            logger.info(msg)

    # ── Branch C: encode from VAE ──────────────────────────────────────────────
    else:
        if not args.sdf_dir or not args.vae_config:
            raise SystemExit(
                "[train_bpe_3d] Need --sdf_dir and --vae_config when corpus cache is"
                " missing or not found."
            )
        with open(args.vae_config, "r", encoding="utf-8") as f:
            vae_cfg = json.load(f)
        base_vocab_size = (
            int(args.num_embeddings)
            if args.num_embeddings is not None
            else int(vae_cfg["models"]["vqvae"]["args"].get("num_embeddings", 8192))
        )

        mod = _load_build_module()
        load_vae_from_config = mod.load_vae_from_config
        collate_inputs_3d = mod.collate_inputs_3d
        stdio_suppressed = mod._stdio_suppressed
        inputs_3d_to_device = mod._inputs_3d_to_device

        from vae_qwen3vl.dataset_sdf_caption import SDF3DCaptionDataset

        device = _resolve_device(args.device)
        vae = load_vae_from_config(args.vae_config, args.vae_ckpt, device)
        ds = SDF3DCaptionDataset(
            sdf_dir=args.sdf_dir,
            resolution=args.resolution,
            min_points=args.min_points,
            max_points=args.max_points,
            max_samples=args.max_samples,
        )
        silent = not bool(args.vqvae_verbose)
        corpus = []
        n = len(ds)
        batch_size = max(1, int(args.batch_size))

        msg = f"Collecting corpus via VAE Encode: {n} samples, batch_size={batch_size}"
        print(f"[train_bpe_3d] {msg}", flush=True)
        logger.info(msg)

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None  # type: ignore

        indices = list(range(n))
        it = range(0, len(indices), batch_size)
        if progress and tqdm is not None:
            it = tqdm(it, desc="VQ Encode → corpus", unit="batch",
                      dynamic_ncols=True, file=sys.stderr)

        n_ok = 0
        for b0 in it:
            chunk_idx = indices[b0 : b0 + batch_size]
            batch_samples: List[Dict[str, Any]] = []
            for idx in chunk_idx:
                try:
                    batch_samples.append(ds[idx])
                except Exception as e:
                    logger.warning(f"skip idx={idx}: {e}")
            if not batch_samples:
                continue
            try:
                collated = collate_inputs_3d(batch_samples)
                entries = _encode_batch_to_corpus_entries(
                    vae, collated, device, silent=silent,
                    inputs_3d_to_device=inputs_3d_to_device,
                    stdio_suppressed=stdio_suppressed,
                )
                corpus.extend(entries)
                n_ok += len(entries)
            except Exception as e:
                logger.warning(f"batch failed, fallback per-sample: {e}")
                for sample in batch_samples:
                    try:
                        collated = collate_inputs_3d([sample])
                        entries = _encode_batch_to_corpus_entries(
                            vae, collated, device, silent=silent,
                            inputs_3d_to_device=inputs_3d_to_device,
                            stdio_suppressed=stdio_suppressed,
                        )
                        corpus.extend(entries)
                        n_ok += len(entries)
                    except Exception as e2:
                        logger.warning(f"skip sample: {e2}")
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        msg = f"Encode done: {n_ok} samples collected into corpus"
        print(f"[train_bpe_3d] {msg}", flush=True)
        logger.info(msg)

        if cache_path is not None:
            _save_corpus_npz(cache_path, corpus, base_vocab_size, logger)

    if not corpus:
        raise SystemExit("[train_bpe_3d] Empty corpus; nothing to train.")

    # ------------------------------------------------------------------
    # Train BPE
    # ------------------------------------------------------------------
    tok = BPE3DTokenizer(base_vocab_size=base_vocab_size)
    sum_before = sum(int(r["tokens"].shape[0]) for r in corpus)

    msg = (
        f"Starting BPE training: {len(corpus)} samples, "
        f"{sum_before} total tokens, "
        f"base_vocab={base_vocab_size}, "
        f"num_merges={args.num_merges}, "
        f"min_freq={args.min_freq}, "
        f"workers={args.workers}"
    )
    print(f"[train_bpe_3d] {msg}", flush=True)
    logger.info(msg)

    tok.train(
        corpus,
        num_merges=args.num_merges,
        min_freq=args.min_freq,
        verbose=verbose,
        num_workers=args.workers,
        progress=progress,
        train_mode=getattr(args, "bpe_train_mode", None),
    )

    tok.save(args.out_merge_table)
    msg = f"Saved merge table → {args.out_merge_table}  vocab_size={tok.vocab_size}"
    print(f"[train_bpe_3d] {msg}", flush=True)
    logger.info(msg)

    # ------------------------------------------------------------------
    # Compression ratio
    # ------------------------------------------------------------------
    msg_comp = "Computing compression ratio …"
    print(f"[train_bpe_3d] {msg_comp}", flush=True)
    logger.info(msg_comp)

    sum_after = sum(_macro_token_count_encode(tok, rec) for rec in corpus)
    ratio = float(sum_before) / max(float(sum_after), 1.0)
    msg = (
        f"Compression: {sum_before} leaf-blocks → {sum_after} macro-tokens  "
        f"ratio={ratio:.4f}x  "
        f"reduction={100.0*(1-1/max(ratio,1e-9)):.1f}%"
    )
    print(f"[train_bpe_3d] {msg}", flush=True)
    logger.info(msg)

    elapsed = time.perf_counter() - t_total_start
    msg_done = f"train_bpe_3d finished  total_time={elapsed:.1f}s"
    print(f"[train_bpe_3d] {msg_done}", flush=True)
    logger.info(msg_done)
    if log_file:
        print(f"[train_bpe_3d] Full log written to: {log_file}", flush=True)


if __name__ == "__main__":
    main()
