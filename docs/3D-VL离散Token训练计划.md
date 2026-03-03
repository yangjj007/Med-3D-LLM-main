# 3D-VL 离散 Token 训练计划（可执行命令）

本文档给出**离散 3D token 对齐**（8³ 空间池化 + 扩表、无 Projector）的完整训练流程与可直接复制的命令。对应执行计划见项目内计划书；配置说明见 `configs/3d_align_train.yaml`。

---

## 一、环境与前置

- **Python**：建议 3.8+，已安装 `torch`、`transformers`、`accelerate`、`peft`、`pyyaml` 等（见项目 `requirements`）。
- **稀疏后端**：训练前设置 `export SPARSE_BACKEND=spconv`（或由启动脚本设置）。
- **必须准备**：
  - **VAE 权重**：`vae_ckpt` 指向已训练好的 SparseSDFVQVAE 的 `.pt` 文件。
  - **VAE 配置**：`vae_config` 指向 `configs/vae/sdf_vqvae_stage2.json` 或等价 JSON。
  - **数据目录**：`data_dir` 下需有 SDF 的 `.npz` 及 `metadata.csv`（含 `captions` 列），格式与 `SDF3DCaptionDataset` 一致。
  - **VL 模型**：`vl_model` 为 Qwen2-VL / Qwen3-VL 的本地路径或 HuggingFace 名称。

以下命令均在**项目根目录**（即 `Med-3D-LLM-main/`）下执行。

---

## 二、配置中与离散训练相关的项

在 `configs/3d_align_train.yaml` 中（或通过命令行覆盖）：

| 配置项 | 含义 | 离散训练建议 |
|--------|------|--------------|
| `use_discrete_3d_tokens` | 是否使用离散 mesh token（8³ 池化、无 Projector） | **true** |
| `reconstruction_ratio` | 3D 重建任务占比（0~1），与 caption 混合 | 0.2~0.3 |
| `training_stage` | warmup = 只训 embed+lm_head；sft = LoRA + embed | 先 warmup 再 sft |
| `data_dir` | SDF+Caption 数据目录 | 必填（离散模式不支持 dummy_data） |
| `vae_config` / `vae_ckpt` | VAE 配置与权重 | 必填 |
| `use_variable_length_3d_tokens` | 变长 3D：保留 VAE 全部点 + Morton 排序 + 动态 pad | 可选，见下文「变长 3D」 |
| `max_safe_3d_length` | 软上限：仅当点数 > 此值才 FPS 截断（如 15000） | 变长时 15000 |
| `coord_max_3d` | Morton 坐标上界：当前 trellis VAE 输出 64³ latent → 64；若 VAE 输出 512³ 则 512 | 64 |
| `max_length_variable` | 变长时 tokenizer 最大长度（避免截断 8k~12k） | 32768 |
| `gradient_accumulation_steps` | 梯度累积步数（变长建议 batch_size=1 + 4~8） | 1 或 4~8 |
| `use_flash_attn_2` | Flash Attention 2（长序列省显存） | 变长建议 true |
| `use_gradient_checkpointing` | 梯度检查点（以时间换显存） | 变长建议 true |
| `use_deepspeed` | DeepSpeed ZeRO-2 分片（多卡时降低每卡显存） | 显存紧张时 true |

其余如 `vl_model`、`batch_size`、`epochs`、`lr`、`use_lora`、`lora_r` 等与原有 3D-VL 训练一致。

---

## 二点五、变长 3D Token（Variable-Length 3D Tokenization）

在离散路径基础上，若希望**保留 VAE 输出的全部点**（约 8k~12k，不做 8³ 固定池化），可采用变长 3D 序列化，以提升几何保真度。实现上已做四道保险：

1. **莫顿码排序 (Morton Code)**：按 (x,y,z) 的 Z-Order 排序，使序列相邻 token 在 3D 空间相邻，便于模型理解局部结构。
2. **动态 Padding + Attention Mask**：按 batch 内最长序列 padding，`attention_mask` 屏蔽 `<pad>`，不设全局截断。
3. **Flash Attention 2 + 显存策略**：`--use_flash_attn_2`、`--use_gradient_checkpointing`，并建议 `batch_size=1` + `gradient_accumulation_steps` 4~8。
4. **软性上限**：仅当点数 > `max_safe_3d_length`（默认 15000）时用 FPS 降采样到该长度，正常物体原样通过。

**说明**：VAE **输入**是 512³ 的 SDF；当前 trellis Encoder 有 3 次 stride-2 下采样，**输出** latent 是 **64³**（坐标 0..63），不是 8³ 也不是 512³。`coord_max_3d` 应对应编码器输出的坐标范围（当前=64）；若将来改用「无下采样、输出 512³ 坐标」的 VAE，再设为 512。

**推荐命令（变长 + 单卡）**：

```bash
export SPARSE_BACKEND=spconv
accelerate launch --num_processes 1 vae_qwen3vl/train_finetune.py \
  --config configs/3d_align_train.yaml \
  --use_discrete_3d_tokens \
  --use_variable_length_3d_tokens \
  --max_safe_3d_length 15000 \
  --max_length_variable 32768 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --use_flash_attn_2 \
  --use_gradient_checkpointing \
  --training_stage warmup \
  --lr 1e-4 \
  --epochs 2
```

若 YAML 中已设置上述项，可直接：

```bash
export SPARSE_BACKEND=spconv
python scripts/run_3d_align_train.py --config configs/3d_align_train_variable_length.yaml
```

（需自建 `3d_align_train_variable_length.yaml` 或在本机 YAML 中打开 `use_variable_length_3d_tokens: true` 等。）

**并行与降显存**：在 YAML 中设置 `use_deepspeed: true` 可启用 **DeepSpeed ZeRO-2**，在多卡时把优化器状态和梯度分片到各 GPU，降低每卡显存。变长配置 `configs/3d_align_train_variable_length.yaml` 已默认开启；普通配置需显存时也可在 `configs/3d_align_train.yaml` 里打开。需安装：`pip install deepspeed`。

---

## 三、离散 Token 两阶段训练

### 3.1 Stage 1：词汇扫盲预热（Modality Warm-up）

**目的**：让模型认识 8195 个 mesh token，只训练 `input_embeddings` 与 `lm_head`，其余冻结。

**建议**：1~2 epoch，lr 约 1e-4 或 2e-4，单卡即可。

**方式一：用 YAML 配置（推荐）**

1. 复制一份配置并改为 Stage 1 专用（或直接改原配置跑完 Stage 1 再改回 Stage 2）：

```bash
# 在项目根目录执行
cp configs/3d_align_train.yaml configs/3d_align_train_discrete_warmup.yaml
```

2. 编辑 `configs/3d_align_train_discrete_warmup.yaml`，确保包含且取值为：

```yaml
use_discrete_3d_tokens: true
reconstruction_ratio: 0.0        # Stage 1 可只用 caption
training_stage: "warmup"
lr: 1.0e-4
epochs: 2
batch_size: 2
# vl_model / vae_config / vae_ckpt / data_dir 按实际路径填写
```

3. 单卡执行：

```bash
export SPARSE_BACKEND=spconv
accelerate launch --num_processes 1 vae_qwen3vl/train_finetune.py \
  --config configs/3d_align_train_discrete_warmup.yaml
```

4. 多卡执行（与现有脚本一致，先生成 accelerate 配置再 launch）：

```bash
export SPARSE_BACKEND=spconv
python scripts/run_3d_align_train.py --config configs/3d_align_train_discrete_warmup.yaml
```

（若需单卡调试可加 `--debug`：`python scripts/run_3d_align_train.py --config ... --debug`）

**方式二：命令行覆盖（不改 YAML）**

在已有 YAML 基础上用命令行打开离散 + warmup：

```bash
export SPARSE_BACKEND=spconv
accelerate launch --num_processes 1 vae_qwen3vl/train_finetune.py \
  --config configs/3d_align_train.yaml \
  --use_discrete_3d_tokens \
  --training_stage warmup \
  --reconstruction_ratio 0 \
  --lr 1e-4 \
  --epochs 2
```

**输出**：  
- 日志与指标在 `outputs_3d_align/<run_dir>/` 下，其中 `<run_dir>` 为自动生成的子目录（含 epochs、lr、timestamp 等）。  
- 离散模式下会保存 **tokenizer**（如 `tokenizer_final`、`tokenizer_epoch{N}`），**不保存** projector。  
- 若使用 LoRA，Stage 1 仍可挂 LoRA 但仅训练 embed；若希望 Stage 1 纯 embed，可暂时将 YAML 中 `use_lora` 设为 false。

---

### 3.2 Stage 2：LoRA SFT（混合 Caption + 3D 重建）

**目的**：在 Stage 1 基础上，用 LoRA 微调注意力等层，并混合 3D→caption 与 3D→3D 重建任务，促使模型关注前文 3D token。

**建议**：3~5 epoch，lr 约 2e-5 或 5e-5，`reconstruction_ratio` 约 0.2~0.3。

**加载 Stage 1 结果**：  
Stage 2 需在**已扩表且做过 warmup 的模型**上继续。当前实现下，Stage 2 仍从**原始 VL 基座**重新加载并再次执行 `add_mesh_tokens` + `resize_token_embeddings_and_init_mesh`，然后加载 **Stage 1 保存的 LoRA**（若有）和** tokenizer**。若 Stage 1 只训了 embed、未保存 LoRA，则 Stage 2 从基座 + 扩表 + 初始化后的 embed 开始，再挂 LoRA 训练。  
（若你希望 Stage 2 从 Stage 1 的 checkpoint 继续，需自行在训练脚本中增加“加载上一阶段 ckpt”的逻辑；此处给出的是“独立跑 Stage 2”的用法。）

**方式一：用 YAML 配置（推荐）**

1. 使用或复制一份配置，设为 SFT 阶段：

```yaml
use_discrete_3d_tokens: true
reconstruction_ratio: 0.3
training_stage: "sft"
lr: 5.0e-5
epochs: 5
use_lora: true
lora_r: 16
# 其他同前
```

2. 单卡：

```bash
export SPARSE_BACKEND=spconv
accelerate launch --num_processes 1 vae_qwen3vl/train_finetune.py \
  --config configs/3d_align_train.yaml
```

3. 多卡：

```bash
export SPARSE_BACKEND=spconv
python scripts/run_3d_align_train.py --config configs/3d_align_train.yaml
```

**方式二：命令行覆盖**

```bash
export SPARSE_BACKEND=spconv
accelerate launch --num_processes 1 vae_qwen3vl/train_finetune.py \
  --config configs/3d_align_train.yaml \
  --use_discrete_3d_tokens \
  --training_stage sft \
  --reconstruction_ratio 0.3 \
  --lr 5e-5 \
  --epochs 5
```

**输出**：  
- 同上，结果在 `outputs_3d_align/<run_dir>/`。  
- 会保存 `tokenizer_final`、`lora_final`（若 `use_lora: true`），无 projector。

---

## 四、连续特征路径（Projector）训练（对照）

若不使用离散 token，仍用原有「连续特征 + Projector」方案，则保持：

- `use_discrete_3d_tokens: false`（或不加该参数）
- 不设 `training_stage` 或为 `sft`，按需设 `max_3d_tokens`、`truncate_mode` 等。

**单卡示例**：

```bash
export SPARSE_BACKEND=spconv
accelerate launch --num_processes 1 vae_qwen3vl/train_finetune.py \
  --config configs/3d_align_train.yaml
```

**多卡示例**：

```bash
export SPARSE_BACKEND=spconv
python scripts/run_3d_align_train.py --config configs/3d_align_train.yaml
```

---

## 五、评估（离散 run）

离散训练得到的 run 目录下会有 `tokenizer_final`、`lora_final`（若用了 LoRA），**没有** `projector_final.pt`。评估脚本会通过「存在 `tokenizer_final`」自动识别为离散 run，并加载该 tokenizer、做 resize/init、走离散生成路径。

**命令**（将 `output_run` 换成你本次训练产生的子目录名）：

```bash
export SPARSE_BACKEND=spconv
python vae_qwen3vl/eval_3d_vl.py \
  --config configs/3d_align_train.yaml \
  --output_run <你的run目录名> \
  --data_dir train_sdf_dataset/res512_thre0.5 \
  --max_eval_samples 20
```

可选：`--save_mesh` 会顺带用 VAE 解码并保存 mesh。

---

## 六、命令速查

| 场景 | 命令 |
|------|------|
| 离散 Stage 1（单卡） | `accelerate launch --num_processes 1 vae_qwen3vl/train_finetune.py --config configs/3d_align_train.yaml --use_discrete_3d_tokens --training_stage warmup --reconstruction_ratio 0 --lr 1e-4 --epochs 2` |
| 离散 Stage 2（单卡） | `accelerate launch --num_processes 1 vae_qwen3vl/train_finetune.py --config configs/3d_align_train.yaml --use_discrete_3d_tokens --training_stage sft --reconstruction_ratio 0.3 --lr 5e-5 --epochs 5` |
| 离散多卡 | `python scripts/run_3d_align_train.py --config configs/3d_align_train.yaml`（YAML 中已设 use_discrete_3d_tokens: true、training_stage、reconstruction_ratio） |
| 评估离散 run | `python vae_qwen3vl/eval_3d_vl.py --config configs/3d_align_train.yaml --output_run <run_dir> --data_dir <data_dir>` |

以上命令均需在项目根目录执行，并保证 `vl_model`、`vae_config`、`vae_ckpt`、`data_dir` 在配置或环境中正确指向现有路径。
