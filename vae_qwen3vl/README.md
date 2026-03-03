# VAE + Qwen2-VL / Qwen3-VL for 3D Understanding and Mesh Reconstruction

This package connects the trained 3D VQ-VAE (SparseSDFVQVAE) to Qwen2-VL or Qwen3-VL for 3D understanding and mesh reconstruction. The representation sent to the LLM is the **codebook vector combination** (VQ-VAE output): we encode to discrete indices, then `feats = codebook(indices)`—not the raw encoder output. The same `encoding_indices` can be passed to `vae.Decode(encoding_indices)` to reconstruct mesh.

---

## 重要注意事项（使用前必读）

| 类别 | 说明 |
|------|------|
| **环境变量** | 训练/推理前需设置 `export SPARSE_BACKEND=spconv`，否则稀疏张量后端可能报错。 |
| **Projector 结构** | `projector_layers=1` 为单层 Linear；`projector_layers>=2` 才是 MLP。配置中 `projector_layers: 3` 表示 MLP。 |
| **分布式训练** | 多卡 DDP 下通过 `getattr(model, "module", model)` 正确调用 `forward_with_3d`，无需手动处理。 |
| **训练日志** | 日志写入 `logs/`，按时间与 rank 分别记录；控制台过滤 `[DEBUG]`，仅 rank 0 回显，避免多卡输出混乱。 |
| **输出目录** | 训练结果保存到 `output_dir/ep{epochs}_lr{lr}_bs{bs}_n{samples}_lora{r}_{时间戳}/`，多次训练不覆盖。 |
| **eval 指定 run** | 评估时在 config 中设置 `output_run` 为实际训练子目录名（含时间戳，如 `ep10_lr5e-5_bs4_nall_lora16_20260216_143022`）。 |
| **生成效果** | 若大量输出 "red robot" 等重复内容，可能是 VQ 码本利用不足、训练不足或数据问题，可增加 epochs、调大 `max_samples`。 |
| **LoRA 评估** | 使用 LoRA 训练时，eval 需同时指定 `--projector_ckpt` 和 `--lora_dir`（或通过 `--config` 自动推断）。 |
| **Chat 模板** | 训练与 eval 均使用 Qwen chat 模板 + `add_generation_prompt`，确保模型在 assistant 位置正确生成。 |

---

## 数据目录说明

| 目录 | 用途 |
|------|------|
| **train_sdf_dataset** | 3D–文本对齐训练的数据源。含 `metadata.csv` 和 `{sha256}_r512.npz`（稀疏 SDF）。由 `dataset_toolkits/sdf_voxelize.py` 从 mesh（GLB）生成。当前约 **18,293** 个可训练样本（sdf_computed + r512 + captions + npz 存在）。 |
| **M3D_Seg_processed** | **M3D-Seg 医学 CT 分割数据集**的预处理输出。每个子目录（如 0000、0001）含 `dataset_config.json`、`metadata.csv`、`processed/`，保存的是 CT 图像、器官掩码、分辨率适配等中间结果。由 `dataset_toolkits/process_m3d_seg_format.py` 处理 M3D-Seg 原始数据得到。与 `train_sdf_dataset` 是**不同数据流水线**，不直接用于本模块的 3D–VL 对齐训练。 |

---

## Components

- **vae_latent_extractor**: `extract_3d_latent(batch, vae_model)` returns `(feats [N, embed_dim], coords [N, 4])` where feats are codebook vectors. `extract_3d_latent_and_indices(...)` returns `(feats, coords, encoding_indices)` so you can use `encoding_indices` for mesh reconstruction.
- **sequence_3d**: `prepare_3d_sequence` / `prepare_3d_sequence_batched`: sort, truncate/pad to `max_3d_tokens`, build `attention_mask`.
- **projector**: `Projector3D(latent_dim -> hidden_size)`. When `vae_model` is provided, `latent_dim` is inferred from the VAE codebook; optional `PositionEncoder3D` from coords.
- **model**: `Qwen3VLWith3DBranch`: loads Qwen2-VL or Qwen3-VL, adds projector; `forward_with_3d(...)` merges 3D tokens with text. When using `inputs_3d` + `vae_model`, the returned dict includes `encoding_indices_3d` for reconstruction. Use `get_3d_embeds_and_encoding_indices(inputs_3d)` to get embeddings and indices without a full forward.
- **train_finetune.py**: Fine-tune the projector (and optional LoRA) on 3D–text pairs.
- **run_smoke_test.py**: 端到端烟雾测试，验证整条流程可跑通（3D 输入 → forward_with_3d → encoding_indices_3d → Decode 重建）。

## 跑通流程：烟雾测试

在改动代码或环境后，可用烟雾测试确认模型结构与数据流正确、能跑起来。

**快速模式**（不下载 Qwen2-VL，不依赖 trellis 全量环境；用 mock VL + 假 VAE 或真实 VAE）：

```bash
# 项目根目录执行
python vae_qwen3vl/run_smoke_test.py --vae_config configs/vae/sdf_vqvae_stage1.json --quick
```

若本机未安装 trellis 或其依赖（如 easydict），脚本会自动使用「假 VAE」仅验证 3D 分支与 projector 的接口与形状，仍会打印 `Smoke test OK.`。

**完整模式**（加载真实 Qwen2-VL 与真实 VAE，需网络与显存）：

```bash
python vae_qwen3vl/run_smoke_test.py --vae_config configs/vae/sdf_vqvae_stage1.json
# 若有 VAE 权重
python vae_qwen3vl/run_smoke_test.py --vae_config configs/vae/sdf_vqvae_stage1.json --vae_ckpt path/to/vae.pt
```

通过后即可认为：forward_with_3d、encoding_indices_3d、Decode、get_3d_embeds_and_encoding_indices、extract_3d_latent_and_indices 等预定功能在当下代码下可跑通。

## Usage

### Inference (3D -> text)

```python
from vae_qwen3vl import Qwen3VLWith3DBranch, extract_3d_latent
import torch

# Load VAE (from your trained checkpoint)
# vae = ... trellis SparseSDFVQVAE ...

model = Qwen3VLWith3DBranch(
    model_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
    vae_model=vae,
    max_3d_tokens=2048,
    use_3d_pos=True,
)
model.eval()

# batch: dict with sparse_sdf, sparse_index, batch_idx (from your dataset)
outputs = model.forward_with_3d(
    input_ids=input_ids,
    attention_mask=attention_mask,
    inputs_3d=batch,
)
# Or pass precomputed feats_3d [N, latent_dim], coords_3d [N, 4]
# outputs = model.forward_with_3d(..., feats_3d=feats, coords_3d=coords)

# When using inputs_3d, outputs include encoding_indices_3d for mesh reconstruction:
# encoding_indices = outputs["encoding_indices_3d"]
# recon = model.vae_model.Decode(encoding_indices)  # sparse SDF
# meshes = model.vae_model.decode_mesh(recon, ...)  # optional: to mesh
```

### Mesh reconstruction from the same 3D input

Use the same encoding indices used for the LLM input to decode back to 3D:

```python
# Option 1: from forward_with_3d return value
outputs = model.forward_with_3d(input_ids=..., attention_mask=..., inputs_3d=batch)
encoding_indices = outputs["encoding_indices_3d"]
recon = model.vae_model.Decode(encoding_indices)  # SparseTensor of reconstructed SDF
# Optional: convert to mesh
# meshes = model.vae_model.decode_mesh(recon, voxel_resolution=512, mc_threshold=0.2)

# Option 2: get embeddings and indices without running the full LLM forward
embeds_3d, mask_3d, encoding_indices = model.get_3d_embeds_and_encoding_indices(batch, device="cuda")
recon = model.vae_model.Decode(encoding_indices)
```

### Fine-tuning

From project root:

```bash
# Dummy data (quick test)
python vae_qwen3vl/train_finetune.py --dummy_data --dummy_samples 50 --epochs 1 --batch_size 2 --output_dir ./out_3dvl

# With VAE and LoRA
python vae_qwen3vl/train_finetune.py \
  --vl_model Qwen/Qwen2-VL-2B-Instruct \
  --vae_config configs/vae/ct_vqvae_stage1.json \
  --vae_ckpt path/to/vae.pt \
  --use_lora --max_3d_tokens 2048 --output_dir ./out_3dvl
```

Replace the dummy dataset with your 3D–text dataset (e.g. 3D captioning or QA).

---

## 训练指令

在项目根目录执行。需先准备好 VAE 权重与数据，并修改 `configs/3d_align_train.yaml` 中的路径（`vl_model`、`vae_ckpt`、`data_dir`）。

### 多卡训练（推荐）

```bash
bash scripts/run_3d_align_train.sh
```

### 单卡调试

```bash
python scripts/run_3d_align_train.py --config configs/3d_align_train.yaml --debug
```

### 指定配置文件

```bash
bash scripts/run_3d_align_train.sh configs/3d_align_train.yaml
```

### 训练输出

- 保存目录：`output_dir/ep{epochs}_lr{lr}_bs{bs}_n{samples}_lora{r}_{时间戳}/`
- 日志：`logs/train_{时间戳}_rank{rank}.log`
- 权重：`projector_final.pt`、`lora_final/`（若启用 LoRA）
- 指标：`training_metrics.jsonl`（每步 step / epoch / loss / lr），用于画曲线

### 可视化训练进度

训练结束后（或中途）用同一 run 目录下的 `training_metrics.jsonl` 画 loss 与学习率曲线：

```bash
# 指定 run 目录（自动找 training_metrics.jsonl）
python vae_qwen3vl/plot_training.py --run_dir outputs_3d_align/ep10_lr5e-5_bs4_nall_lora16_20260219_023800

# 或直接指定指标文件
python vae_qwen3vl/plot_training.py --metrics outputs_3d_align/xxx/training_metrics.jsonl --out ./curve.png

# 对 loss 做 10% 窗口平滑
python vae_qwen3vl/plot_training.py --run_dir outputs_3d_align/xxx --smooth 0.1
```

图片会保存到 run 目录下的 `training_curve.png`（需安装 `matplotlib`）。

**收敛过程动态刷新（边训练边看图）**：在训练**进行中**另开一个终端，对同一 `run_dir` 加 `--live`，会弹窗并每隔几秒重读 `training_metrics.jsonl` 重绘曲线，实时看收敛；同时会更新 run 目录下的 `training_curve.png`。按 Ctrl+C 结束动态刷新。

```bash
# 终端 1：正常启动训练
python vae_qwen3vl/train_finetune.py ...

# 终端 2：指定与上面相同的 output_dir（即 run_dir），加 --live
python vae_qwen3vl/plot_training.py --run_dir outputs_3d_align/你的run目录 --live

# 可选：刷新间隔（秒）、平滑
python vae_qwen3vl/plot_training.py --run_dir outputs_3d_align/xxx --live --interval 3 --smooth 0.1
```

---

## 测试 / 评估指令

在项目根目录执行。评估前需将 `configs/3d_align_train.yaml` 中的 `output_run` 设为本次训练的实际子目录名（如 `ep10_lr5e-5_bs4_nall_lora16_20260216_143022`）。

### 使用配置文件（推荐）

```bash
python vae_qwen3vl/eval_3d_vl.py --config configs/3d_align_train.yaml
```

从 config 自动加载 VAE、projector、LoRA、data_dir 等，输出写入 `output_dir/output_run/eval_out/`。

### 指定输出目录

```bash
python vae_qwen3vl/eval_3d_vl.py --config configs/3d_align_train.yaml --output_dir ./eval_out
```

### 限制评估样本数

```bash
python vae_qwen3vl/eval_3d_vl.py --config configs/3d_align_train.yaml --max_eval_samples 50
```

### 保存 mesh 重建

```bash
python vae_qwen3vl/eval_3d_vl.py --config configs/3d_align_train.yaml --save_mesh
```

### 不使用 config，手动指定路径

```bash
python vae_qwen3vl/eval_3d_vl.py \
  --vae_config configs/vae/sdf_vqvae_stage2.json \
  --vae_ckpt outputs/sdf_vqvae_stage2_1/ckpts/vqvae_step0000459.pt \
  --projector_ckpt outputs_3d_align/ep10_lr5e-5_bs4_nall_lora16_xxx/projector_final.pt \
  --lora_dir outputs_3d_align/ep10_lr5e-5_bs4_nall_lora16_xxx/lora_final \
  --vl_model /path/to/model_qwen3vl_2B \
  --data_dir train_sdf_dataset \
  --projector_layers 3 \
  --output_dir ./eval_out
```

### 评估结果

- `eval_results.jsonl`：每行包含 `idx`、`generated`、`gt_caption`
- `generated.txt`：仅单样本时生成
- `recon_mesh_*.obj`：`--save_mesh` 时保存 mesh 重建

---

## 配置说明

主要参数在 `configs/3d_align_train.yaml` 中：

| 参数 | 说明 |
|------|------|
| `vl_model` | Qwen3-VL 模型路径 |
| `vae_ckpt` | VAE 权重路径 |
| `data_dir` | SDF 数据集目录（含 metadata.csv、*.npz） |
| `output_run` | 评估时使用的训练子目录名（带时间戳） |
| `epochs` | 训练轮数 |
| `max_samples` | 0=全量，>0 限制样本数 |
| `projector_layers` | Projector MLP 层数（须与训练一致） |

**完整说明**（数据格式、脚本参数、VAE 冻结、自定义数据集）见 [TRAIN_ALIGNMENT.md](TRAIN_ALIGNMENT.md)。

## Dependencies

- torch
- transformers (Qwen2-VL / Qwen3-VL)
- trellis (for VAE and sparse SDF data)
- Optional: peft (for LoRA)
