# 3D–VL 对齐训练使用说明

本文档说明如何在 **VAE（Encoder）已训练好** 的前提下，完成 3D 与 Qwen2-VL/Qwen3-VL 的**对齐训练**：只训练 projector（及可选 LoRA），冻结 VAE 与（默认）VL 主体，使用 3D–文本配对数据。

---

## 一、前提条件

| 项目 | 说明 |
|------|------|
| **VAE 已训练** | 已有 SparseSDFVQVAE 的 checkpoint（`.pt`），以及对应配置文件（如 `configs/vae/sdf_vqvae_stage1.json`）。 |
| **VAE 配置与代码** | 项目内 trellis 可导入（或能通过 `load_vae_from_config` 加载 VAE）。配置中 `latent_channels`（如 16）即码本维度，与 projector 的 `latent_dim` 一致。 |
| **环境** | Python 3.8+，torch，transformers（Qwen2-VL），trellis；若使用 LoRA 需安装 peft。 |
| **数据** | 3D–文本配对数据，格式见下文。 |

---

## 二、数据格式与两种用法

对齐训练支持两种数据形式，二选一即可。

### 方式 A：预提取的码本特征（feats_3d + coords_3d）

每个样本在**训练前**用 VAE 跑一遍，得到码本特征与坐标，训练时直接读入，无需在 DataLoader 里再跑 VAE。

**每个样本需包含：**

| 键 | 形状 / 类型 | 说明 |
|----|-------------|------|
| `feats_3d` | `[N, latent_dim]`，float | 码本向量，`latent_dim` 与 VAE 一致（如 16）。必须由 `vae.Encode` → `vq.embeddings(indices)` 得到，不能是编码器原始输出。 |
| `coords_3d` | `[N, 4]`，long | 格点坐标，格式为 `(batch_idx, x, y, z)`，与 VAE 潜在空间分辨率一致（如 0~63）。 |
| `input_ids` | `[L]`，long | 文本的 token id（含 prompt + 回答），与 Qwen 词表一致。 |
| `attention_mask` | `[L]`，long/bool | 1 表示有效 token，0 表示 padding。 |
| `labels` | `[L]`，long | 用于计算 loss：**只对“需要模型生成”的位置保留 token id，其余填 -100**（不参与 loss）。通常 3D 对应部分与 prompt 部分在 labels 中均为 -100，仅回答部分为真实 id。 |

**如何得到 feats_3d / coords_3d：**

对单条 3D 数据（如稀疏 SDF）构造 batch dict：`sparse_sdf`、`sparse_index`、`batch_idx`，然后调用：

```python
from vae_qwen3vl import extract_3d_latent_and_indices

feats_3d, coords_3d, _ = extract_3d_latent_and_indices(batch_dict, vae_model, device="cuda")
# feats_3d: [N, latent_dim], coords_3d: [N, 4]
```

可离线对所有样本跑一遍，将 `feats_3d`、`coords_3d` 与对应文本一起存成数据集（如 `.pt`、`.npz` 或 Dataset 返回的 dict）。

**Batch 内变长处理：**  
同一 batch 内各样本的 `N` 可不同；collate 时需对 3D 做 padding 到同一长度（如 `max_n = max(N_i)`），`feats_3d` 用 0 填充，`coords_3d` 用 0 填充。训练脚本中的 `collate_3d_text(batch, latent_dim=...)` 即按此方式 pad；自定义 DataLoader 时需保证 `latent_dim` 与 `model.projector.latent_dim` 一致。

---

### 方式 B：原始 3D 批次（inputs_3d）

不预提取特征，每个 batch 直接提供原始 3D 输入，由模型内部用 VAE 现场 Encode 得到码本特征（与 Decode 同源）。

**每个样本需包含：**

| 键 | 形状 / 类型 | 说明 |
|----|-------------|------|
| `inputs_3d` | dict | 键：`sparse_sdf`、`sparse_index`、`batch_idx`（格式与 trellis SparseSDF 一致）；可选 `factor`。 |
| `input_ids` | `[L]`，long | 同上。 |
| `attention_mask` | `[L]` | 同上。 |
| `labels` | `[L]`，long | 同上。 |

**注意：**  
- `inputs_3d` 在 collate 时需按你的数据形式组织成 batch（例如同一 batch 内多样本的 `batch_idx` 区分开）。  
- 当前 `train_finetune.py` 在**非 dummy** 模式下若检测到 batch 中有 `inputs_3d` 且非 None，会走 `forward_with_3d(..., inputs_3d=batch["inputs_3d"])`；否则走 `feats_3d`/`coords_3d`。  
- 使用 `inputs_3d` 时**必须**在训练脚本中传入 `vae_config` 与 `vae_ckpt`，否则模型内部没有 VAE 无法编码。

---

## 三、文本与 labels 的构造建议

- **词表与 tokenizer**：与所选 VL 一致（如 Qwen2-VL 使用 HuggingFace 对应 tokenizer）。  
- **格式**：建议使用 Qwen 的 chat 模板，将“系统/用户/助手”与 3D 占位、问题、答案拼成一条序列，再转为 `input_ids`。  
- **labels**：  
  - 仅对**模型需要生成**的 token 保留真实 id；  
  - 3D 前缀、prompt、padding 等位置在 `labels` 中设为 **-100**，不参与 loss。  
- 这样等价于“给定 3D + prompt，只对回答部分做语言建模 loss”。

---

## 四、训练脚本：`train_finetune.py`

**路径：** 项目根目录下执行  
`python vae_qwen3vl/train_finetune.py <参数>`

### 4.1 必需 / 强烈建议参数（真实数据训练）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--vae_config` | str | None | VAE 配置 JSON 路径（如 `configs/vae/sdf_vqvae_stage1.json`）。**真实对齐训练时必须提供。** |
| `--vae_ckpt` | str | None | VAE 权重路径（`.pt`）。**真实对齐训练时必须提供。** |
| `--output_dir` | str | `./outputs_3d_vl` | 保存 projector 权重与 `train_args.json` 的目录。 |

### 4.2 VL 与 3D 相关

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--vl_model` | str | `Qwen/Qwen2-VL-2B-Instruct` | HuggingFace 模型名或本地路径。 |
| `--max_3d_tokens` | int | 2048 | 3D 序列最大 token 数（过长会截断）。显存紧张可适当减小。 |
| `--use_3d_pos` | flag | False | 是否在 projector 中使用 3D 位置编码。 |
| `--projector_layers` | int | 1 | Projector 的 MLP 层数（1 表示单层 Linear）。 |

### 4.3 训练超参

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--lr` | float | 1e-4 | 学习率（仅作用于可训练参数）。 |
| `--epochs` | int | 3 | 训练轮数。 |
| `--batch_size` | int | 2 | 每步 batch 大小。 |
| `--use_lora` | flag | False | 是否对 VL 使用 LoRA（需安装 peft）。 |
| `--lora_r` | int | 8 | LoRA rank。 |

### 4.4 测试用（dummy）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--dummy_data` | flag | False | 使用脚本内置 Dummy3DTextDataset，不读真实数据。 |
| `--dummy_samples` | int | 50 | dummy 时样本数。 |

**注意：** 未传 `--dummy_data` 时，脚本当前会报错 `Provide your 3D-text dataset`，需要你接入自己的 Dataset（见第五节）。

### 4.5 示例命令

```bash
# 使用真实 VAE + 真实数据（需自行实现 Dataset 并替换脚本中的 else 分支）
python vae_qwen3vl/train_finetune.py \
  --vae_config configs/vae/sdf_vqvae_stage1.json \
  --vae_ckpt path/to/your/vae.pt \
  --vl_model Qwen/Qwen2-VL-2B-Instruct \
  --output_dir ./outputs_3d_vl \
  --max_3d_tokens 2048 \
  --use_3d_pos \
  --epochs 3 \
  --batch_size 4 \
  --lr 1e-4

# 仅快速跑通（dummy，不写真实数据）
python vae_qwen3vl/train_finetune.py \
  --vae_config configs/vae/sdf_vqvae_stage1.json \
  --vae_ckpt path/to/vae.pt \
  --dummy_data --dummy_samples 100 \
  --epochs 1 --batch_size 2 \
  --output_dir ./out_dummy
```

---

## 五、VAE 是否已冻结？可训练参数有哪些？

**是的，VAE 冻结已实现。**

- **在 `model.py` 中**（约 77–79 行）：若传入了 `vae_model`，则  
  `for p in vae_model.parameters(): p.requires_grad = False`  
  即 VAE 全部参数不更新。

- **在 `train_finetune.py` 中**（约 145–151 行）：  
  - 默认只把 `projector` 的 `requires_grad` 设为 True，其余（VL、VAE）为 False。  
  - 若使用 `--use_lora`，会再对 VL 中名字含 `lora` 的参数设为 True。

因此：  
- **不开 LoRA**：仅 **projector** 可训练，VAE 与 VL 主体均冻结。  
- **开 LoRA**：**projector** + **VL 的 LoRA 参数** 可训练，VAE 仍冻结。

---

## 六、如何接入自己的 3D–文本数据集

当前脚本在**未使用 `--dummy_data`** 时会抛出“Provide your 3D-text dataset”。需要你：

1. **实现一个 `torch.utils.data.Dataset`**  
   - 每个 `__getitem__` 返回一个 dict，键至少包含：  
     - **方式 A**：`feats_3d`、`coords_3d`、`input_ids`、`attention_mask`、`labels`（格式见第二节）。  
     - **方式 B**：`inputs_3d`、`input_ids`、`attention_mask`、`labels`。  
   - 若用方式 A，`feats_3d` 的最后一维必须等于 VAE 的 `latent_channels`（如 16），即与 `model.projector.latent_dim` 一致。

2. **实现或复用 collate**  
   - **方式 A**：可用脚本中的 `collate_3d_text(batch, latent_dim=model.projector.latent_dim)`，或自写：对 `feats_3d`/`coords_3d` 按 batch 内最大 N 做 padding，其余键按常规 stack。  
   - **方式 B**：按你的 `inputs_3d` 结构做 batching（保证 `batch_idx` 等与 forward 约定一致）。

3. **在 `train_finetune.py` 的 `main()` 里替换数据分支**  
   - 将 `else: raise ValueError("Provide your 3D-text dataset. ...")` 改为：  
     - 实例化你的 Dataset。  
     - 用 `DataLoader(..., collate_fn=你的 collate)`；若用方式 A 且用脚本里的 collate，则 `collate_fn=lambda b: collate_3d_text(b, latent_dim=latent_dim)`，其中 `latent_dim` 来自 `model.projector.latent_dim`（脚本在构建 model 后会取 `latent_dim = model.projector.latent_dim`）。

4. **方式 B 时**  
   - 保证传入的 `vae_config`、`vae_ckpt` 有效，这样 `forward_with_3d(..., inputs_3d=batch["inputs_3d"])` 才会在内部用 VAE 编码得到码本特征。

按上述接入后，即可用“真实 3D + 真实文本”完成对齐训练。

---

## 七、训练输出与后续使用

- **输出目录（`--output_dir`）** 中会得到：  
  - `projector_final.pt`：最终 projector 权重（推荐用于推理）。  
  - `projector_epoch0.pt`、`projector_epoch1.pt`、…：各 epoch 的 projector 权重。  
  - `train_args.json`：本次训练使用的全部参数（便于复现）。

- **推理 / 评估时**：  
  - 加载同一 VAE（同一 config + ckpt）、同一 VL（同一 `vl_model`），再 `model.projector.load_state_dict(torch.load("projector_final.pt", map_location="cpu"), strict=True)` 即可。  
  - 评估脚本用法见主 README 中的「效果检验」与 `eval_3d_vl.py`。

---

## 八、完成训练需要兼顾的因素小结

| 因素 | 说明 |
|------|------|
| **数据格式** | 二选一：预提取 `feats_3d`/`coords_3d`，或原始 `inputs_3d`；文本为 `input_ids`/`attention_mask`/`labels`，labels 中非生成位置为 -100。 |
| **VAE 与 latent_dim** | 必须提供已训练好的 VAE（config + ckpt）；`latent_dim` 由 VAE 的码本维度决定，脚本会从 VAE 推断，无需手填。 |
| **训练脚本** | 使用 `vae_qwen3vl/train_finetune.py`；真实训练必须传 `--vae_config`、`--vae_ckpt`，并接入自己的 Dataset 或先用 `--dummy_data` 验证流程。 |
| **VAE 冻结** | 已实现：VAE 全部参数 `requires_grad=False`；默认只训 projector，可选 LoRA 再训 VL 部分 LoRA 参数。 |
| **显存与长度** | 通过 `--max_3d_tokens`、`--batch_size` 控制；3D 序列过长或 batch 过大会 OOM，可先减小再试。 |
| **文本与 loss** | 使用 Qwen tokenizer/chat 模板，labels 只对“回答”部分非 -100，保证 loss 只对生成部分计算。 |
| **保存与复现** | 使用 `output_dir` 下的 `projector_final.pt` 和 `train_args.json` 做推理与复现。 |

按上述准备数据、参数与数据集接入，即可完成从数据到训练再到保存的完整对齐训练流程。
