# VQVAE 训练配置文件参数详解

本文档详细介绍了 VQVAE 训练配置文件中各个参数的含义、作用以及如何调整。

## 目录

- [配置文件结构](#配置文件结构)
- [模型参数 (models.vqvae.args)](#模型参数-modelsvqvaeargs)
- [数据集参数 (dataset.args)](#数据集参数-datasetargs)
- [训练器参数 (trainer.args)](#训练器参数-trainerargs)
- [优化器参数](#优化器参数)
- [学习率调度器参数](#学习率调度器参数)
- [梯度裁剪参数](#梯度裁剪参数)
- [参数调整建议](#参数调整建议)
- [常见场景配置](#常见场景配置)

---

## 配置文件结构

配置文件采用 JSON 格式，包含三个主要部分：

```json
{
    "models": { ... },      // 模型架构配置
    "dataset": { ... },     // 数据集配置
    "trainer": { ... }      // 训练器配置
}
```

---

## 模型参数 (models.vqvae.args)

### 基础架构参数（一般不变，需要跟预训练权重一致）

#### `resolution` (int)
- **含义**: 模型内部处理的分辨率（潜在空间分辨率）
- **默认值**: `64`
- **取值范围**: `32, 64, 128` 等
- **说明**: 
  - 这是 encoder/decoder 内部处理的分辨率，不是输入数据的分辨率
  - 较小的值可以降低显存占用，但可能影响细节重建
  - 通常设置为 `64`，对应输入分辨率 `512`（下采样 8 倍）
- **调整建议**:
  - 显存充足：保持 `64`
  - 显存不足：可尝试 `32`，但会降低重建质量

#### `model_channels` (int)
- **含义**: Encoder 和 Decoder 的基础通道数
- **默认值**: `512`
- **取值范围**: `128, 256, 512, 1024` 等
- **说明**:
  - 控制模型的容量和表达能力
  - 更大的值意味着更强的模型能力，但显存占用也更大
  - 通道数会随着网络层数变化（如 decoder 中会逐步减半）
- **显存影响**: 
  - 显存占用与 `model_channels²` 成正比
  - `512` → `256` 可减少约 75% 的显存占用
- **调整建议**:
  - 标准配置：`512`
  - 显存不足：`256` 或 `384`
  - 追求更高质量：`768` 或 `1024`（需要更多显存）

#### `latent_channels` (int)
- **含义**: 潜在空间的通道维度（embedding 维度）
- **默认值**: `16`
- **取值范围**: `8, 16, 32, 64` 等
- **说明**:
  - 控制 codebook 的 embedding 维度
  - 更大的值可以存储更多信息，但会增加 codebook 大小
- **显存影响**:
  - Codebook 大小 = `num_embeddings × latent_channels × 4 bytes` (FP32)
  - `8192 × 16 × 4 = 512 KB`（较小）
- **调整建议**:
  - 标准配置：`16`
  - 需要更高压缩比：`8`
  - 需要更丰富表示：`32`

#### `num_blocks` (int)
- **含义**: Encoder 和 Decoder 中 Transformer 块的层数
- **默认值**: `8`
- **取值范围**: `4, 6, 8, 12, 16` 等
- **说明**:
  - 控制模型的深度
  - 更多层数可以学习更复杂的特征，但训练更慢、显存更大
- **显存影响**:
  - 显存占用与 `num_blocks` 成正比
  - 激活值显存 = `batch_size × num_points × model_channels × num_blocks`
- **调整建议**:
  - 标准配置：`8`
  - 显存不足：`4` 或 `6`
  - 追求更高质量：`12` 或 `16`

### Codebook 参数

#### `num_embeddings` (int)
- **含义**: Codebook 中 code 的数量（词汇表大小）
- **默认值**: `8192`
- **取值范围**: `1024, 2048, 4096, 8192, 16384` 等
- **说明**:
  - 控制离散表示的丰富程度
  - 更大的 codebook 可以表示更多样化的特征，但训练更困难
- **显存影响**:
  - Codebook 显存 = `num_embeddings × latent_channels × 4 bytes`
  - `8192 × 16 × 4 = 512 KB`（很小）
- **调整建议**:
  - 标准配置：`8192`
  - 简单数据：`4096`
  - 复杂数据：`16384`

### Transformer 参数

#### `num_heads` (int | null)
- **含义**: 多头注意力的头数
- **默认值**: `null`（自动计算）
- **取值范围**: `null` 或 `4, 8, 16` 等
- **说明**:
  - `null` 时根据 `num_head_channels` 自动计算：`num_heads = model_channels / num_head_channels`
  - 手动设置时，必须满足 `model_channels % num_heads == 0`
- **计算方式**: `num_heads = model_channels / num_head_channels = 512 / 64 = 8`
- **调整建议**: 通常保持 `null`，让系统自动计算

#### `num_head_channels` (int)
- **含义**: 每个注意力头的通道数
- **默认值**: `64`
- **取值范围**: `32, 64, 128` 等
- **说明**:
  - 与 `num_heads` 配合使用：`num_heads = model_channels / num_head_channels`
  - 影响注意力的表达能力
- **调整建议**: 通常保持 `64`

#### `mlp_ratio` (float)
- **含义**: MLP（前馈网络）的扩展比例
- **默认值**: `4.0`
- **取值范围**: `2.0, 4.0, 8.0` 等
- **说明**:
  - MLP 隐藏层维度 = `model_channels × mlp_ratio`
  - 更大的值增加模型容量，但显存占用也更大
- **显存影响**:
  - MLP 显存与 `mlp_ratio` 成正比
- **调整建议**:
  - 标准配置：`4.0`
  - 显存不足：`2.0`
  - 追求更高质量：`8.0`

#### `attn_mode` (str)
- **含义**: 注意力机制的模式
- **默认值**: `"swin"`
- **可选值**: `"full"`, `"shift_window"`, `"shift_sequence"`, `"shift_order"`, `"swin"`
- **说明**:
  - `"swin"`: Swin Transformer 风格的窗口注意力（推荐，效率高）
  - `"full"`: 全局注意力（显存占用大，但表达能力最强）
  - 其他：各种局部注意力变体
- **显存影响**:
  - `"full"` 显存占用最大（O(N²)）
  - `"swin"` 显存占用最小（O(N)）
- **调整建议**: 通常保持 `"swin"`

#### `window_size` (int)
- **含义**: Swin Transformer 的窗口大小（仅当 `attn_mode="swin"` 时有效）
- **默认值**: `8`
- **取值范围**: `4, 8, 16` 等
- **说明**:
  - 控制局部注意力的范围
  - 更大的窗口可以捕获更长距离的依赖，但显存占用更大
- **显存影响**:
  - 显存占用与 `window_size²` 成正比
- **调整建议**:
  - 标准配置：`8`
  - 显存不足：`4`
  - 需要更长依赖：`16`

#### `pe_mode` (str)
- **含义**: 位置编码模式
- **默认值**: `"ape"`（绝对位置编码）
- **可选值**: `"ape"`, `"rope"`（旋转位置编码）
- **说明**:
  - `"ape"`: 学习的位置编码，需要额外参数
  - `"rope"`: 无需学习的位置编码，更节省参数
- **调整建议**: 通常保持 `"ape"`

### 训练优化参数

#### `use_fp16` (bool)
- **含义**: 是否在模型内部使用 FP16（半精度）
- **默认值**: `false`
- **说明**:
  - `true`: 模型参数和计算使用 FP16，可节省约 50% 显存
  - `false`: 使用 FP32（全精度），更稳定但显存占用更大
- **注意**: 与 `trainer.args.fp16_mode` 不同，这里是模型内部的精度
- **调整建议**:
  - 显存充足：`false`（更稳定）
  - 显存不足：`true`（节省显存）

#### `use_checkpoint` (bool)
- **含义**: 是否启用梯度检查点（Gradient Checkpointing）
- **默认值**: `false`
- **说明**:
  - `true`: 以时间换显存，减少激活值显存占用约 50-70%
  - `false`: 正常训练，显存占用更大但速度更快
- **显存影响**:
  - 可显著减少激活值显存，但训练速度会降低约 20-30%
- **调整建议**:
  - **Stage 2 训练强烈推荐**: `true`（decoder 显存占用大）
  - Stage 1 训练：`false`（只训练 codebook，显存占用小）

#### `chunk_size` (int)
- **含义**: Decoder 分块训练的大小（仅在 Stage 2 有效）
- **默认值**: `1`（不分块）
- **取值范围**: `1, 2, 4, 8` 等
- **说明**:
  - `1`: 处理整个 tensor，显存占用最大
  - `>1`: 将空间分成 `chunk_size³` 个块，每次只处理一个块
  - 例如 `chunk_size=4` 会分成 64 个块
- **显存影响**:
  - 显存占用与 `chunk_size³` 成反比
  - `chunk_size=4` 可减少约 64 倍的显存占用
- **调整建议**:
  - **Stage 2 训练 OOM 时强烈推荐**: `4` 或 `8`
  - Stage 1 训练：`1`（不需要）

#### `qk_rms_norm` (bool)
- **含义**: 是否在注意力中使用 RMSNorm（而非 LayerNorm）
- **默认值**: `false`
- **说明**: 实验性功能，通常保持 `false`
- **调整建议**: 保持 `false`

#### `representation_config` (dict | null)
- **含义**: 表示学习相关配置
- **默认值**: `null`
- **说明**: 高级配置，通常保持 `null`
- **调整建议**: 保持 `null`

---

## 数据集参数 (dataset.args)

### 基础参数

#### `name` (str)
- **含义**: 数据集类名
- **默认值**: `"CTWindowSparseSDF"`
- **可选值**: `"CTWindowSparseSDF"`, `"SparseSDF"` 等
- **说明**: 指定使用的数据集类
- **调整建议**: 使用 CT 数据时保持 `"CTWindowSparseSDF"`

#### `resolution` (int)
- **含义**: 输入数据的空间分辨率
- **默认值**: `512`
- **取值范围**: `256, 512, 1024` 等
- **说明**:
  - 这是**输入数据**的分辨率，不是模型内部分辨率
  - 必须与预处理后的数据分辨率一致
  - 更大的分辨率包含更多细节，但显存占用更大
- **显存影响**:
  - 输入数据显存与 `resolution³` 成正比
  - `512³` → `256³` 可减少 8 倍显存
- **调整建议**:
  - 标准配置：`512`
  - 显存不足：`256`
  - 追求更高细节：`1024`（需要大量显存）

#### `window_type` (str)
- **含义**: CT 窗口类型
- **默认值**: `"lung"`
- **可选值**: `"lung"`, `"bone"`, `"soft_tissue"`, `"brain"`
- **说明**:
  - 不同窗口类型对应不同的 HU 值范围
  - `lung`: 窗宽 1500，窗位 -600（适合肺组织）
  - `bone`: 窗宽 1500，窗位 300（适合骨骼）
  - `soft_tissue`: 窗宽 400，窗位 50（适合软组织）
  - `brain`: 窗宽 80，窗位 35（适合脑组织）
- **调整建议**: 根据训练目标选择合适的窗口类型

#### `min_points` (int)
- **含义**: 稀疏点数量的最小值阈值
- **默认值**: `100`
- **说明**:
  - 少于 `min_points` 的样本会被过滤掉
  - 用于过滤过于稀疏的数据
- **调整建议**: 通常保持 `100`

#### `max_points` (int)
- **含义**: 稀疏点数量的最大值阈值
- **默认值**: `100000`（Stage 1），`500000`（Stage 2）
- **说明**:
  - 超过 `max_points` 的样本会被下采样
  - **重要**: 这个值直接影响显存占用
  - Decoder 的显存占用与点数成指数关系（因为 upsample）
- **显存影响**:
  - 显存占用与 `max_points` 成正比
  - 但 decoder upsample 后点数会 ×8³ = 512 倍增长
  - 例如：`max_points=100000` → upsample 后可能达到 5000 万点
- **调整建议**:
  - **Stage 1**: `100000`（只训练 codebook，显存占用小）
  - **Stage 2**: `50000-100000`（联合训练，显存占用大）
  - OOM 时：逐步减小到 `30000, 50000` 等

#### `cache_data` (bool)
- **含义**: 是否将数据缓存到内存
- **默认值**: `false`
- **说明**:
  - `true`: 预加载所有数据到内存，加快训练速度但占用 CPU 内存
  - `false`: 按需加载，节省内存但可能稍慢
- **调整建议**:
  - 数据集较小（<100 样本）：`true`
  - 数据集较大：`false`

#### `precompute_sparse` (bool)
- **含义**: 是否预计算稀疏索引
- **默认值**: `true`
- **说明**:
  - `true`: 预处理时计算稀疏索引，训练时直接使用
  - `false`: 训练时实时计算，可能稍慢
- **调整建议**: 通常保持 `true`

---

## 训练器参数 (trainer.args)

### 训练阶段参数

#### `training_stage` (int)
- **含义**: 训练阶段
- **可选值**: `1`, `2`
- **说明**:
  - `1`: Stage 1 - 冻结 Encoder/Decoder，只训练 Codebook
  - `2`: Stage 2 - 联合训练 Encoder/Decoder/Codebook
- **调整建议**: 
  - 第一阶段：`1`
  - 第二阶段：`2`

#### `pretrained_vae_path` (str | null)
- **含义**: 预训练 VAE 权重路径
- **默认值**: `null` 或路径字符串
- **说明**:
  - Stage 1: 如果有 Direct3D 预训练权重，设置路径；否则 `null`
  - Stage 2: 设置为 Stage 1 的输出 checkpoint 路径
- **示例**:
  - Stage 1: `"./vae_weights/sparse_vae_512.pth"` 或 `null`
  - Stage 2: `"./outputs/ct_vqvae_lung_stage1/ckpts/vqvae_step0000002.pth"`
- **调整建议**: 根据训练阶段正确设置

### 训练循环参数

#### `max_steps` (int)
- **含义**: 最大训练步数
- **默认值**: `100000`（Stage 1），`200000`（Stage 2）
- **说明**: 训练会在达到 `max_steps` 时停止
- **调整建议**:
  - Stage 1: `100000`
  - Stage 2: `200000` 或更多

#### `max_epoch` (int | null)
- **含义**: 最大训练轮数
- **默认值**: `null` 或整数
- **说明**:
  - `null`: 只使用 `max_steps` 控制训练
  - 整数: 训练会在达到 `max_epoch` 时停止
  - 实际停止条件：`min(max_steps, max_epoch × steps_per_epoch)`
- **调整建议**: 通常保持 `null`，使用 `max_steps` 控制

#### `batch_size_per_gpu` (int)
- **含义**: 每个 GPU 的批次大小
- **默认值**: `2`（Stage 1），`1`（Stage 2）
- **说明**:
  - 总 batch size = `batch_size_per_gpu × num_gpus`
  - 直接影响显存占用和训练速度
- **显存影响**:
  - 显存占用与 `batch_size_per_gpu` 成正比
  - 减小 batch size 是解决 OOM 最直接的方法
- **调整建议**:
  - Stage 1: `2-4`（只训练 codebook，显存占用小）
  - Stage 2: `1-2`（联合训练，显存占用大）
  - OOM 时：逐步减小到 `1` 或更小

#### `batch_split` (int)
- **含义**: 批次分割数（梯度累积）
- **默认值**: `2`
- **说明**:
  - 将 `batch_size_per_gpu` 分成 `batch_split` 个小批次
  - 每个小批次独立计算梯度，最后累加
  - 等效于更大的 batch size，但显存占用不变
- **示例**:
  - `batch_size_per_gpu=4`, `batch_split=2`: 每次处理 2 个样本，累积 2 次梯度
  - 等效于 `batch_size_per_gpu=4`，但显存占用只有一半
- **调整建议**:
  - 显存不足时：增大 `batch_split`（如 `4` 或 `8`）
  - 显存充足时：`1` 或 `2`

#### `num_workers` (int)
- **含义**: 数据加载的进程数
- **默认值**: `30`
- **说明**:
  - 控制数据预加载的并行度
  - 更多进程可以加快数据加载，但占用更多 CPU 内存
- **调整建议**:
  - CPU 核心多：`20-30`
  - CPU 核心少：`4-8`
  - 内存不足：`1-4`

### 损失函数参数

#### `loss_type` (str)
- **含义**: 重建损失类型
- **默认值**: `"l1"`（Stage 1），`"l1_l2"`（Stage 2）
- **可选值**: `"mse"`, `"l1"`, `"l1_l2"`
- **说明**:
  - `"mse"`: 均方误差（L2 损失）
  - `"l1"`: L1 损失（更鲁棒）
  - `"l1_l2"`: L1 + L2 组合损失（`0.5 × L1 + 0.5 × L2`）
- **调整建议**:
  - Stage 1: `"l1"`（只训练 codebook）
  - Stage 2: `"l1_l2"`（联合训练，更好的细节重建）

#### `lambda_vq` (float)
- **含义**: VQ 损失的权重
- **默认值**: `1.0`
- **说明**:
  - 控制 codebook 学习的强度
  - 更大的值鼓励使用 codebook，但可能影响重建质量
- **调整建议**: 通常保持 `1.0`

#### `lambda_commitment` (float)
- **含义**: Commitment 损失的权重
- **默认值**: `0.25`
- **说明**:
  - 控制 encoder 输出与 codebook 的绑定强度
  - 防止 encoder 输出远离 codebook
- **调整建议**: 通常保持 `0.25`

### 混合精度训练参数

#### `fp16_mode` (str)
- **含义**: FP16 训练模式
- **默认值**: `"inflat_all"`
- **可选值**: `null`, `"amp"`, `"inflat_all"`
- **说明**:
  - `null`: 不使用 FP16，全部使用 FP32
  - `"amp"`: PyTorch 自动混合精度（Automatic Mixed Precision）
  - `"inflat_all"`: 为所有参数维护 FP32 master 副本（推荐）
- **显存影响**:
  - `"inflat_all"`: 模型参数 FP16，但优化器状态仍为 FP32
  - 可节省约 50% 模型参数显存
- **调整建议**: 通常保持 `"inflat_all"`

#### `fp16_scale_growth` (float)
- **含义**: FP16 梯度缩放的增长因子（仅 `"inflat_all"` 模式）
- **默认值**: `0.001`
- **说明**:
  - 控制梯度缩放的动态调整
  - 如果梯度溢出，会减小 scale；如果稳定，会增大 scale
- **调整建议**: 通常保持 `0.001`

### EMA 参数

#### `ema_rate` (list[float])
- **含义**: 指数移动平均的衰减率
- **默认值**: `[0.9999]`
- **说明**:
  - 维护模型参数的指数移动平均，用于推理
  - 更大的值更新更慢，更稳定
  - 可以设置多个 EMA 率（如 `[0.999, 0.9999]`）
- **调整建议**: 通常保持 `[0.9999]`

### 日志和保存参数

#### `i_print` (int)
- **含义**: 打印日志的间隔（步数）
- **默认值**: `1000`
- **说明**: 每 `i_print` 步打印一次训练信息到控制台
- **调整建议**: 
  - 调试时：`10-100`
  - 正常训练：`100-1000`

#### `i_log` (int)
- **含义**: 记录日志的间隔（步数）
- **默认值**: `500`
- **说明**: 每 `i_log` 步记录一次日志到 TensorBoard
- **调整建议**: 通常设置为 `i_print` 的一半或相等

#### `i_sample` (int)
- **含义**: 生成样本的间隔（步数）
- **默认值**: `10000`
- **说明**: 每 `i_sample` 步生成一次可视化样本
- **调整建议**: 根据训练长度调整，通常 `5000-20000`

#### `i_save` (int)
- **含义**: 保存 checkpoint 的间隔（步数）
- **默认值**: `10000`
- **说明**: 每 `i_save` 步保存一次模型 checkpoint
- **调整建议**:
  - 调试时：`1-10`（频繁保存）
  - 正常训练：`5000-10000`

#### `i_ddpcheck` (int)
- **含义**: DDP 一致性检查的间隔（步数）
- **默认值**: `10000`
- **说明**: 每 `i_ddpcheck` 步检查一次多 GPU 训练的一致性
- **调整建议**: 通常保持 `10000`，调试时可设为 `10-100`

#### `disable_snapshot` (bool)
- **含义**: 是否禁用快照生成
- **默认值**: `false`
- **说明**:
  - `true`: 不生成可视化样本，节省时间和显存
  - `false`: 正常生成样本
- **调整建议**: 
  - 调试时：`false`
  - 快速训练：`true`

---

## 优化器参数

### `optimizer.name` (str)
- **含义**: 优化器类型
- **默认值**: `"AdamW"`
- **可选值**: `"AdamW"`, `"Adam"`, `"SGD"` 等
- **调整建议**: 通常使用 `"AdamW"`

### `optimizer.args.lr` (float)
- **含义**: 学习率
- **默认值**: `5e-3`（Stage 1），`1e-4`（Stage 2）
- **说明**:
  - Stage 1: 只训练 codebook，可以使用较大学习率（`5e-3`）
  - Stage 2: 联合训练，使用较小学习率（`1e-4` 或 `5e-3`）
- **调整建议**:
  - Stage 1: `5e-3` 或 `1e-3`
  - Stage 2: `1e-4` 到 `5e-3`（根据收敛情况调整）

### `optimizer.args.weight_decay` (float)
- **含义**: 权重衰减（L2 正则化）
- **默认值**: `0.0`
- **说明**: 防止过拟合，通常 VQVAE 不需要
- **调整建议**: 通常保持 `0.0`

---

## 学习率调度器参数

### `lr_scheduler.name` (str)
- **含义**: 学习率调度器类型
- **默认值**: `"CosineAnnealingLR"`（Stage 1），`"ExponentialLR"`（Stage 2）
- **可选值**: `"CosineAnnealingLR"`, `"ExponentialLR"`, `"StepLR"`, `null` 等
- **说明**:
  - `null`: 不使用学习率调度
  - `"CosineAnnealingLR"`: 余弦退火（推荐 Stage 1）
  - `"ExponentialLR"`: 指数衰减（推荐 Stage 2）
- **调整建议**: 根据训练阶段选择合适的调度器

### `lr_scheduler.args.T_max` (int) - CosineAnnealingLR
- **含义**: 余弦退火的最大步数
- **默认值**: `100000`
- **说明**: 学习率会在 `T_max` 步内从初始值降到 `eta_min`
- **调整建议**: 设置为 `max_steps`

### `lr_scheduler.args.eta_min` (float) - CosineAnnealingLR
- **含义**: 最小学习率
- **默认值**: `5e-5`
- **说明**: 学习率衰减的下限
- **调整建议**: 通常设置为初始学习率的 `1/100`

### `lr_scheduler.args.gamma` (float) - ExponentialLR
- **含义**: 指数衰减因子
- **默认值**: `0.999`
- **说明**: 每步学习率乘以 `gamma`
- **调整建议**: 
  - 快速衰减：`0.99`
  - 慢速衰减：`0.999` 或 `0.9999`

---

## 梯度裁剪参数

### `grad_clip.name` (str)
- **含义**: 梯度裁剪方法
- **默认值**: `"AdaptiveGradClipper"`
- **可选值**: `"AdaptiveGradClipper"`, 或直接使用 `float` 值
- **说明**:
  - `float`: 固定阈值梯度裁剪
  - `"AdaptiveGradClipper"`: 自适应梯度裁剪（推荐）

### `grad_clip.args.max_norm` (float) - AdaptiveGradClipper
- **含义**: 梯度范数的最大阈值
- **默认值**: `1.0`
- **说明**: 超过此值的梯度会被裁剪
- **调整建议**: 通常保持 `1.0`

### `grad_clip.args.clip_percentile` (int) - AdaptiveGradClipper
- **含义**: 裁剪的百分位数
- **默认值**: `95`
- **说明**: 只裁剪超过 95% 分位数的梯度
- **调整建议**: 通常保持 `95`

---

## 参数调整建议

### 解决 OOM（显存不足）问题

1. **减小 batch size**
   ```json
   "batch_size_per_gpu": 1
   ```

2. **启用分块训练**（Stage 2 强烈推荐）
   ```json
   "chunk_size": 4
   ```

3. **启用梯度检查点**
   ```json
   "use_checkpoint": true
   ```

4. **减小 max_points**
   ```json
   "max_points": 50000
   ```

5. **增大 batch_split**
   ```json
   "batch_split": 4
   ```

6. **减小模型容量**
   ```json
   "model_channels": 256,
   "num_blocks": 4
   ```

### 提高训练速度

1. **增大 batch_size_per_gpu**（如果显存允许）
2. **减小 num_workers**（如果 CPU 是瓶颈）
3. **禁用快照生成**
   ```json
   "disable_snapshot": true
   ```
4. **禁用梯度检查点**（如果显存充足）
   ```json
   "use_checkpoint": false
   ```

### 提高重建质量

1. **增大模型容量**
   ```json
   "model_channels": 768,
   "num_blocks": 12
   ```

2. **增大 codebook 大小**
   ```json
   "num_embeddings": 16384
   ```

3. **使用组合损失**（Stage 2）
   ```json
   "loss_type": "l1_l2"
   ```

4. **提高输入分辨率**（如果数据支持）
   ```json
   "resolution": 1024
   ```

---

## 常见场景配置

### 场景 1: Stage 1 训练（只训练 Codebook）

```json
{
    "models": {
        "vqvae": {
            "args": {
                "resolution": 64,
                "model_channels": 512,
                "latent_channels": 16,
                "num_blocks": 8,
                "use_checkpoint": false,
                "chunk_size": 1
            }
        }
    },
    "trainer": {
        "args": {
            "training_stage": 1,
            "batch_size_per_gpu": 2,
            "batch_split": 2,
            "loss_type": "l1",
            "lr": 5e-3,
            "lr_scheduler": {
                "name": "CosineAnnealingLR",
                "args": {
                    "T_max": 100000,
                    "eta_min": 5e-5
                }
            }
        }
    }
}
```

### 场景 2: Stage 2 训练（联合训练，显存充足）

```json
{
    "models": {
        "vqvae": {
            "args": {
                "resolution": 64,
                "model_channels": 512,
                "latent_channels": 16,
                "num_blocks": 8,
                "use_checkpoint": true,
                "chunk_size": 1
            }
        }
    },
    "trainer": {
        "args": {
            "training_stage": 2,
            "batch_size_per_gpu": 2,
            "batch_split": 2,
            "loss_type": "l1_l2",
            "lr": 1e-4,
            "lr_scheduler": {
                "name": "ExponentialLR",
                "args": {
                    "gamma": 0.999
                }
            }
        }
    }
}
```

### 场景 3: Stage 2 训练（显存不足，OOM）

```json
{
    "models": {
        "vqvae": {
            "args": {
                "resolution": 64,
                "model_channels": 256,
                "latent_channels": 16,
                "num_blocks": 4,
                "use_checkpoint": true,
                "chunk_size": 4
            }
        }
    },
    "dataset": {
        "args": {
            "max_points": 50000
        }
    },
    "trainer": {
        "args": {
            "training_stage": 2,
            "batch_size_per_gpu": 1,
            "batch_split": 4,
            "loss_type": "l1_l2"
        }
    }
}
```

### 场景 4: 高分辨率训练（1024³）

```json
{
    "models": {
        "vqvae": {
            "args": {
                "resolution": 128,
                "model_channels": 512,
                "use_checkpoint": true,
                "chunk_size": 8
            }
        }
    },
    "dataset": {
        "args": {
            "resolution": 1024,
            "max_points": 200000
        }
    },
    "trainer": {
        "args": {
            "batch_size_per_gpu": 1,
            "batch_split": 8
        }
    }
}
```

---

## 总结

### 关键参数优先级

1. **显存相关**（OOM 时优先调整）:
   - `chunk_size`（Stage 2）
   - `batch_size_per_gpu`
   - `max_points`
   - `use_checkpoint`
   - `model_channels`, `num_blocks`

2. **训练质量**:
   - `model_channels`, `num_blocks`
   - `num_embeddings`
   - `loss_type`
   - `lr`, `lr_scheduler`

3. **训练速度**:
   - `batch_size_per_gpu`
   - `num_workers`
   - `use_checkpoint`（禁用可提速）
   - `disable_snapshot`

### 推荐配置流程

1. **Stage 1**: 使用标准配置，只训练 codebook，显存占用小
2. **Stage 2**: 
   - 先尝试标准配置
   - 如果 OOM，按顺序尝试：
     1. 设置 `chunk_size: 4`
     2. 减小 `batch_size_per_gpu` 到 `1`
     3. 减小 `max_points` 到 `50000`
     4. 减小 `model_channels` 到 `256`
     5. 增大 `batch_split` 到 `4` 或 `8`

---

## 参考

- [TRELLIS 文档](https://github.com/microsoft/TRELLIS)
- [Direct3D-S2 文档](https://github.com/DreamTechAI/Direct3D-S2)
- PyTorch 文档: [混合精度训练](https://pytorch.org/docs/stable/amp.html)
- PyTorch 文档: [梯度检查点](https://pytorch.org/docs/stable/checkpoint.html)

