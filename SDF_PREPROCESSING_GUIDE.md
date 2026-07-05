# SDF Preprocessing Guide

This guide explains how to use `sdf_voxelize.py` to preprocess mesh files into sparse SDF format for VQVAE training.

## Overview

The `sdf_voxelize.py` script converts GLB mesh files to sparse Signed Distance Field (SDF) representations at configurable resolutions. It supports two data formats:

1. **TRELLIS-500K format**: ObjaverseXL data with object-paths.json
2. **Custom labeled format**: Your own 3D models with metadata.csv

## Prerequisites

- CUDA-capable GPU (required for SDF computation)
- PyTorch with CUDA support
- Trimesh library
- TRELLIS utils installed

## Usage Examples

### 1. Process TRELLIS-500K Data

```bash
python dataset_toolkits/sdf_voxelize.py \
    --format trellis500k \
    --input_dir ./TRELLIS-500K/raw/hf-objaverse-v1 \
    --output_dir ./train_sdf_dataset \
    --resolutions 64,512 \
    --max_workers 4
```

**With aesthetic score filtering:**

```bash
python dataset_toolkits/sdf_voxelize.py \
    --format trellis500k \
    --input_dir ./TRELLIS-500K/raw/hf-objaverse-v1 \
    --output_dir ./train_sdf_dataset \
    --resolutions 512 \
    --filter_aesthetic_score 6.0 \
    --max_workers 4
```

### 2. Process Custom Labeled Data

```bash
python dataset_toolkits/sdf_voxelize.py \
    --format custom \
    --input_dir ./my_3d_dataset \
    --output_dir ./train_sdf_dataset \
    --resolutions 512 \
    --max_workers 4
```

## Command-Line Arguments

### Required Arguments

- `--format`: Dataset format type
  - `trellis500k`: TRELLIS-500K format with object-paths.json
  - `custom`: Custom format with metadata.csv
  
- `--input_dir`: Input directory containing mesh files and metadata

- `--output_dir`: Output directory for SDF files

### Optional Arguments

- `--resolutions`: Comma-separated list of resolutions (default: `512`)
  - Example: `64,512,1024` for multiple resolutions
  - Higher resolutions = more detail but slower and larger files
  
- `--threshold_factor`: UDF threshold factor for sparse extraction (default: `4.0`)
  - Controls how many points are kept near the surface
  - Higher values = more points, larger files
  
- `--max_workers`: Worker processes per GPU
  - Default: `1`
  - Increase to `2`-`4` when GPU utilization is low and VRAM is available
  - Total worker slots = selected GPUs (`--gpu_nums`) × `--max_workers`
  
- `--no_skip`: Reprocess all files even if they already exist
  - By default, already processed files are skipped
  
- `--filter_aesthetic_score`: Filter by aesthetic score (TRELLIS-500K only)
  - Only process objects with aesthetic_score >= this value

## Input Data Formats

### TRELLIS-500K Format

Expected directory structure:
```
TRELLIS-500K/
├── raw/
│   └── hf-objaverse-v1/
│       ├── glbs/
│       │   └── 000-023/
│       │       └── {sha256}.glb
│       └── object-paths.json
├── metadata.csv (optional, for captions)
└── ...
```

The `object-paths.json` file maps sha256 IDs to GLB file paths:
```json
{
  "8476c4170df24cf5bbe6967222d1a42d": "glbs/000-023/8476c4170df24cf5bbe6967222d1a42d.glb",
  ...
}
```

### Custom Labeled Format

Expected directory structure:
```
my_dataset/
├── metadata.csv
├── {sha256_1}.glb
├── {sha256_2}.glb
└── ...
```

The `metadata.csv` should have these columns:
- `sha256`: Unique identifier
- `glb_file`: GLB filename (can be just filename or relative path)
- `overall_label`: Overall description (optional)
- `materials_captions`: Part-level descriptions (optional)

## Output Format

The script generates:

### 1. SDF Files (NPZ format)

```
output_dir/
├── {sha256}_r64.npz
├── {sha256}_r512.npz
├── {sha256}_r1024.npz
└── ...
```

Each NPZ file contains:
- `sparse_sdf`: SDF values [N, 1] (float32)
- `sparse_index`: 3D coordinates [N, 3] (int32)
- `resolution`: Grid resolution (int)

### 2. Metadata CSV

Located at `{parent_of_output_dir}/metadata.csv`:
- Original metadata columns preserved
- Added columns:
  - `sdf_computed`: Boolean indicating success
  - `r{resolution}_num_points`: Number of points for each resolution
  - `r{resolution}_error`: Error message if processing failed (optional)

## Performance Tips

### GPU Memory Management

- **Multiple workers**: More workers = more GPU memory usage
  - Start with `--max_workers 2` and increase if you have enough VRAM
  - Each worker needs ~2-4GB VRAM depending on resolution
  
- **Resolution**: Higher resolution requires more memory
  - Resolution 64: Very fast, low memory
  - Resolution 512: Moderate (recommended)
  - Resolution 1024: Slow, high memory

### Processing Speed

Typical processing times (with RTX 3090):
- Resolution 64: ~0.5-1 second per mesh
- Resolution 512: ~2-5 seconds per mesh
- Resolution 1024: ~10-20 seconds per mesh

### Resuming Interrupted Processing

The script automatically skips already processed files (unless `--no_skip` is used). If processing is interrupted:

1. Simply re-run the same command
2. The script will detect existing NPZ files and skip them
3. Only unprocessed files will be processed

## Troubleshooting

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `--max_workers` (try 1 or 2)
- Lower `--resolutions` (use 256 or 512 instead of 1024)
- Process in smaller batches

### Missing Mesh Files

**Problem**: "mesh file not found" errors

**Solutions**:
- Check that GLB files exist at the specified paths
- For TRELLIS-500K: Some files may not be downloaded yet
- Files with errors will be skipped, metadata will show which ones failed

### Empty or Invalid Meshes

**Problem**: "empty mesh" or "invalid mesh format" errors

**Solutions**:
- These files will be skipped automatically
- Check the metadata CSV output to see which files failed
- Failed files will have `sdf_computed=False` and an error message

### Import Errors

**Problem**: `ImportError: No module named 'trellis'`

**Solutions**:
```bash
# Make sure you're in the project root directory
cd /path/to/Med-3D-LLM-main

# Install TRELLIS dependencies
pip install -r requirements.txt

# Compile CUDA extensions
cd third_party/voxelize
pip install -e . --no-build-isolation
cd ../..
```

## Example Workflow

### Complete TRELLIS-500K Processing Pipeline

```bash
# 1. Download TRELLIS-500K metadata
git clone https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K

# 2. Download ObjaverseXL metadata
python dataset_toolkits/build_metadata.py ObjaverseXL \
    --source sketchfab \
    --output_dir TRELLIS-500K

# 3. Download GLB files
python dataset_toolkits/download.py ObjaverseXL \
    --output_dir TRELLIS-500K \
    --rank 0 --world_size 1

# 4. Convert to SDF format (this script!)
#    默认 --gpu_nums -1：使用当前可见的全部 GPU 并行（数据分片，无重叠写 .npz），最后合并 metadata.csv。
#    --max_workers 是每张 GPU 的 worker 数；GPU 利用率低且显存空余时可从 2 逐步加到 4。
#    单卡或调试：加 --gpu_nums 1；指定前 N 张卡：--gpu_nums N。
python dataset_toolkits/sdf_voxelize.py \
    --format trellis500k \
    --input_dir ./TRELLIS-500K/ObjaverseXL/raw/hf-objaverse-v1 \
    --output_dir ./train_sdf_dataset_256 \
    --filter_aesthetic_score 4.0 \
    --resolutions 256 \
    --threshold_factor 4.0 \
    --max_workers 24
    

# 5. Check results
ls -lh train_sdf_dataset/*.npz | wc -l
head metadata.csv
```

## Integration with Training Pipeline

After preprocessing, the SDF files can be used with the VQVAE training pipeline:

1. **Location**: Place processed SDF files in `train_sdf_dataset/res512_thre0.1`
2. **Metadata**: Update your training config to point to the generated `metadata.csv`
3. **Dataset class**: Use the SDF dataset loader that reads NPZ files

See `CT_VQVAE_TRAINING_README.md` for more details on training setup.

## Qwen3-VL SFT data (LLaMA-Factory): SDF → VQ tokens → JSONL

加 `--enable_bpe`，并让 `--bpe_merge_table` 指向**不存在**的路径（或不传），脚本分三段执行：

1. **Phase-1（多卡 Encode）**：写原始 JSONL，同时落盘 `*.corpus.part{r}.npz` / `*.meta.part{r}.jsonl`；
2. **Phase-2（CPU，默认增量 BPE）**：合并所有 shard 训练 `BPE3DTokenizer` 并保存 `merge_table.json`。默认使用 **incremental** 训练核心（局部更新邻接对频次，适合百万级 cell）。若需旧版「每轮全量重扫 + 可选多进程」，加 `--bpe_train_mode legacy`；`--bpe_num_workers` 仅对 legacy 生效（`0`=auto：`min(os.cpu_count(),8)`）。
3. **Phase-3（CPU 单进程）**：复用同一份 cache 生成 `*_bpe.jsonl` 与 `mesh_tokens_comma_bpe.txt`（扩展词表 `num_embeddings + bpe_extra_vocab_size`，默认多出 8192 个宏 token `<mesh_8192>` …）。

**Phase-2 内存与进度**：训练开始前会打印主机 RAM 与静态峰值估算（`[BPE3D][memory]`）；构建 slot 图时 `BPE3D build graph` 进度条显示 `cells=已建/总量`、`rss=当前进程 RSS`、`~graph=按 cells 线性外推的图构建结束 RSS`、`~peak=含 merge 阶段余量`。以外推 `~peak` 为准判断是否 OOM；静态估算常偏低。`BPE_MEMORY_ABORT=1` 可按静态估算提前退出。关闭进度条：`BPE3D_GRAPH_PROGRESS=0`。

**Phase-2 merge 速度**：默认 `incremental` 为**单线程**（`--workers` 无效）。`BPE3D_DEFER_HEAP=1`（默认）在 merge 阶段只对**本轮变更的 pair key** 做 heap 更新（dirty flush），避免每轮对千万级 `pair_freq` 全量 `heapify`（旧实现约 40–50s/merge）。图构建结束仍会一次性 `heapify` 全部 key。调试回退：`BPE3D_HEAP_FLUSH=rebuild`（每轮全量 heapify）；`BPE3D_DEFER_HEAP=0`（边变更即 heappush）。仍慢时可减 `--num_merges`，或 `train_bpe_3d.py --bpe_train_mode legacy --workers 16`（多核、更高内存，30 核机器建议 16–20）。

```bash
BPE3D_ENCODE_VERIFY=1 python dataset_toolkits/build_qwen3vl_sft_3d_jsonl.py \
  --sdf_dir ./train_sdf_dataset_256 \
  --vae_config ./configs/vae/sdf_vqvae_stage2.json \
  --vae_ckpt ./outputs_pad_sdf/sdf_vqvae_256_0.5/ckpts/vqvae_step0041000.pt \
  --out_jsonl ./outputs_pad_sdf/qwen3_3d_sft_256_0.5.jsonl \
  --resolution 256 \
  --gpu_ids 0,1,2,3 \
  --batch_size 1 \
  --enable_bpe \
  --out_jsonl_bpe ./outputs_pad_sdf/qwen3_3d_sft_256_0.5_bpe.jsonl \
  --bpe_merge_table ./outputs_pad_sdf/merge_table.json \
  --bpe_extra_vocab_size 8192 \
  --bpe_num_merges 8192 \
  --bpe_min_freq 2
```

### 断点复用

Phase-1 结束后，`--out_jsonl` 同目录会留下 `*.corpus.part{r}.npz` + `*.meta.part{r}.jsonl`。**再次用完全相同的命令运行**，脚本会按下表自动决定从哪个阶段恢复：

| merge table 状态 | BPE JSONL 状态 | 自动行为 |
|---|---|---|
| **不存在**（auto-train 模式） | 不存在 | 跳过 Phase-1，重跑 **Phase-2（训 BPE）+ Phase-3** |
| **已存在**（load 模式） | **不存在** | 跳过 Phase-1 和 Phase-2，直接 **Phase-3**（用现有 merge table 改写） |
| 已存在 | 已存在 | 走 load 模式：重新 Encode + 同步写两份 JSONL |

- 日志会打印 `Resume: detected N complete Phase-1 shard(s) … skipping Phase-1` 或 `Resume Phase-3 only: …`
- shard 按 `part0, part1, …` 顺序扫描，**中间缺一个就视为不完整**，回退到正常 Phase-1。
- 想强制重编码：加 `--force_reencode`。
- Resume 阶段仅用 CPU，无需 GPU。

### 已有 merge table 且 BPE JSONL 也已存在时（完整 load 模式）

`--bpe_merge_table` 指向已存在文件 + BPE JSONL 也已存在 → 单阶段：重新 Encode 并同步写原始与 BPE 两份 JSONL（适合换数据集或重新处理）。

### 输出一览

- `*.jsonl`：ShareGPT `messages`（`role`/`content`），默认每对象两条（`3D→caption`、`caption→3D`）；加 `--multiturn` 改为单轮 4 条消息。
- `mesh_tokens_comma.txt`（或 `--token_list_out`）：逗号分隔的 `<mesh_start>, <mesh_end>, <mesh_empty>, <mesh_0>, …`，用于 `add_tokens` / `resize_vocab`。
- 启用 BPE 时额外产出 `*_bpe.jsonl` + `mesh_tokens_comma_bpe.txt` + `merge_table.json`。

把 JSONL 复制进 `LLaMA-Factory/data/`；数据集 key `qwen3vl_3d_sft` 已在 `LLaMA-Factory/data/dataset_info.json` 注册。

**Optional image + 3D：** 加多视角渲染（如 `dataset_toolkits/render.py`），在 `user` content 放 `<image>` 占位符，并按 `LLaMA-Factory/data/mllm_demo.json` 加 `images` 列；本脚本不输出图像。


<!-- Usage -->
## 💡 Usage

### 3D BPE Round-Trip Verification

The current `bpe_3d.py` format stores `rel_offset` in each merge entry so BPE macro tokens can be decoded back to the exact pre-BPE `(token, coord)` sparse set. Old `merge_table.json` files that do not contain `rel_offset` are intentionally rejected by `BPE3DTokenizer.load`; retrain the merge table with this code before building or verifying BPE JSONL.

Run local unit checks:

```sh
cd /yangjunjie/Med-3D-LLM-main
python -m py_compile bpe_3d.py test_bpe_3d.py dataset_toolkits/verify_bpe_jsonl_decode.py
python test_bpe_3d.py
```

Benchmark a small server subset before the full run. Use a fresh merge table path instead of overwriting the old incompatible one:

```sh
cd /yangjunjie/Med-3D-LLM-main
python dataset_toolkits/build_qwen3vl_sft_3d_jsonl.py \
  --sdf_dir ./train_sdf_dataset/res512_thre0.5 \
  --vae_config ./configs/vae/sdf_vqvae_stage2.json \
  --vae_ckpt ./pad_outputs/sdf_vqvae_stage2_512_0.5/ckpts/vqvae_step0000300.pt \
  --out_jsonl ./pad_outputs/qwen3vl_3d_sft_8cat_1k.jsonl \
  --gpu_ids 0,1 \
  --batch_size 1 \
  --max_samples 1000 \
  --enable_bpe \
  --out_jsonl_bpe ./pad_outputs/qwen3vl_3d_sft_8cat_1k_bpe.jsonl \
  --bpe_merge_table ./pad_outputs/merge_table_rel_offset_1k.json \
  --bpe_extra_vocab_size 8192 \
  --bpe_num_merges 8192 \
  --bpe_min_freq 2
```

Verify the small subset. The expected summary is `decode_mismatch=0`, `jsonl_orig_mismatch=0`, and `jsonl_bpe_mismatch=0`.

```sh
python dataset_toolkits/verify_bpe_jsonl_decode.py \
  --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat_1k.jsonl \
  --merge_table ./pad_outputs/merge_table_rel_offset_1k.json \
  --jsonl_orig ./pad_outputs/qwen3vl_3d_sft_8cat_1k.jsonl \
  --jsonl_bpe ./pad_outputs/qwen3vl_3d_sft_8cat_1k_bpe.jsonl \
  --task both
```

Full retrain and verification:

```sh
source /yangjunjie/anaconda3/etc/profile.d/conda.sh && conda activate trellis && cd /yangjunjie/Med-3D-LLM-main
ATTN_BACKEND=xformers python dataset_toolkits/build_qwen3vl_sft_3d_jsonl.py \
  --sdf_dir ./train_sdf_dataset/res512_thre0.5 \
  --vae_config ./configs/vae/sdf_vqvae_stage2.json \
  --vae_ckpt ./outputs_pad_sdf/sdf_vqvae_stage2_512_0.5/ckpts/vqvae_step0000200.pt \
  --out_jsonl ./outputs_pad_sdf/qwen3vl_3d_sft_8cat.jsonl \
  --out_jsonl_bpe ./outputs_pad_sdf/qwen3vl_3d_sft_8cat_bpe_with_cord.jsonl \
  --gpu_ids 0,1,2,3 \
  --batch_size 1 \
  --enable_bpe \
  --bpe_merge_table ./merge_table.json \
  --bpe_extra_vocab_size 8192 \
  --bpe_num_merges 8192 \
  --bpe_min_freq 2


python dataset_toolkits/verify_bpe_jsonl_decode.py \
  --corpus_prefix ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl \
  --merge_table ./merge_table.json \
  --jsonl_orig ./pad_outputs/qwen3vl_3d_sft_8cat.jsonl \
  --jsonl_bpe ./pad_outputs/qwen3vl_3d_sft_8cat_bpe_rel_offset.jsonl \
  --max_samples 400000 \
  --task both
```
