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
| **输出目录** | 训练结果按参数命名保存到 `output_dir/ep{epochs}_lr{lr}_bs{bs}_n{samples}_lora{r}/`，多次训练不会覆盖。 |
| **eval 指定 run** | 评估时在 config 中设置 `output_run`（如 `ep2_lr1e-4_bs2_n500_lora8`）以加载对应子目录的 LoRA/projector。 |
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

## 对齐训练与效果检验（Encoder 已训练好后）

Encoder（VQ-VAE）已训练好后，用本小节完成 **3D–文本对齐**（只训练 projector，可选 LoRA）并**检验效果**。  
**完整说明（数据格式、脚本参数、VAE 冻结、自定义数据集接入等）见 [TRAIN_ALIGNMENT.md](TRAIN_ALIGNMENT.md)。**

### 0. 推荐工作流（配置文件 + 脚本）

修改 `configs/3d_align_train.yaml` 后，通过脚本一键启动：

```bash
# 多卡训练
bash scripts/run_3d_align_train.sh

# 或单卡调试
python scripts/run_3d_align_train.py --config configs/3d_align_train.yaml --debug
```

配置中可设置 `data_dir`、`max_samples`、`output_run` 等；训练完成后，将 config 中 `output_run` 改为实际生成的子目录名（如 `ep2_lr1e-4_bs2_n500_lora8`），再执行 eval。

### 1. 对齐训练

- **数据**：每个样本需要「3D 表示」+「文本」（如描述/问答）。两种用法：
  - **方式 A**：用 VAE 先抽好码本特征，每个样本为 `(feats_3d [N, latent_dim], coords_3d [N, 4], input_ids, attention_mask, labels)`，DataLoader 的 `collate_fn` 用 `collate_3d_text(batch, latent_dim=model.projector.latent_dim)`。
  - **方式 B**：直接提供原始 3D 批次，每个样本为 `(inputs_3d: {sparse_sdf, sparse_index, batch_idx}, input_ids, attention_mask, labels)`；`forward_with_3d(..., inputs_3d=batch["inputs_3d"])` 会在内部用 VAE 得到码本 feats（与 Decode 同源）。
- **命令示例**（项目根目录下）：

```bash
# 必须：指定已训练好的 VAE 配置与权重；输出里会保存 projector 权重
python vae_qwen3vl/train_finetune.py \
  --vl_model Qwen/Qwen2-VL-2B-Instruct \
  --vae_config configs/vae/sdf_vqvae_stage1.json \
  --vae_ckpt path/to/your_trained_vae.pt \
  --output_dir ./outputs_3d_vl \
  --max_3d_tokens 2048 \
  --use_3d_pos \
  --epochs 3 --batch_size 4 --lr 1e-4
```

- 若要对 LLM 做轻量微调，可加 `--use_lora --lora_r 8`（需安装 peft）。
- 训练结束后，projector 权重在 `--output_dir` 下，例如 `projector_final.pt`。

### 2. 效果检验

- **3D → 文本生成**：用训练好的 VAE + projector 做推理，看模型能否根据 3D 生成合理描述。
- **Mesh 重建**：用同一 3D 的 `encoding_indices` 做 `vae.Decode`，确认能稳定重建 mesh，说明 3D 理解与重建共用同一套码本表示。

**推荐：用脚本 `eval_3d_vl.py` 一次做完「生成 + 可选 mesh 保存」：**

```bash
# 使用配置文件（与训练一致，自动加载 LoRA + projector + 真实数据）
python vae_qwen3vl/eval_3d_vl.py --config configs/3d_align_train.yaml --output_dir ./eval_out

# 仅 3D → 文本生成（无 data_path 时用随机 3D 做快速测试）
python vae_qwen3vl/eval_3d_vl.py \
  --vae_config configs/vae/sdf_vqvae_stage1.json \
  --vae_ckpt path/to/your_trained_vae.pt \
  --projector_ckpt path/to/projector_final.pt \
  --vl_model Qwen/Qwen2-VL-2B-Instruct \
  --output_dir ./eval_out

# 使用 LoRA 时需同时指定 lora_dir
python vae_qwen3vl/eval_3d_vl.py ... --lora_dir outputs_3d_align/ep2_lr1e-4_bs2_n500_lora8/lora_final

# 使用真实数据（与训练相同格式）
python vae_qwen3vl/eval_3d_vl.py ... --data_dir train_sdf_dataset

# 指定一个 3D 批次（.pt 里为 dict: sparse_sdf, sparse_index, batch_idx）
python vae_qwen3vl/eval_3d_vl.py ... --data_path path/to/one_batch.pt --prompt "Describe this 3D shape in one sentence:"

# 同时做 mesh 重建并保存到 output_dir
python vae_qwen3vl/eval_3d_vl.py ... --save_mesh --voxel_resolution 256
```

- 生成结果会写在 `{output_dir}/generated.txt`。
- 使用 `--save_mesh` 时，会用当前 3D 的 `encoding_indices` 走 `Decode` → `sparse2mesh`，并在 `output_dir` 下保存 `.obj`，用于肉眼或下游检查重建质量。

### 3. 自建 3D–文本数据集接入训练

**内置 SDF 数据集**：通过 config 设置 `data_dir: train_sdf_dataset` 和 `data_format: sdf_caption`，可使用 `SDF3DCaptionDataset`，要求目录内有 `metadata.csv`（含 sha256、captions、sdf_computed、r512_num_points 等列）及 `{sha256}_r512.npz`。可通过 `max_samples` 限制样本数（0 表示全量）。

若不用 `--dummy_data` 且非 sdf_caption 格式，需要实现自己的 `Dataset` 和 DataLoader，并替换 `train_finetune.py` 里相应分支：

- 每个 batch 要么包含 `feats_3d`、`coords_3d`（与 `model.projector.latent_dim` 一致），要么包含 `inputs_3d`（sparse_sdf, sparse_index, batch_idx）。
- 同时提供 `input_ids`、`attention_mask`、`labels`（与 Qwen 文本格式一致，labels 中非生成部分可为 -100）。

这样即可在 Encoder 已固定的前提下，只训练对齐层并系统检验 3D 理解与 mesh 重建效果。

## Dependencies

- torch
- transformers (Qwen2-VL / Qwen3-VL)
- trellis (for VAE and sparse SDF data)
- Optional: peft (for LoRA)
