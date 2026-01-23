# CT VQVAE 训练流程使用指南

## 概述

本指南说明如何使用CT窗口数据训练VQVAE模型，实现两阶段训练流程：
1. **阶段1**：冻结预训练VAE的Encoder和Decoder，只训练VQ码本
2. **阶段2**：联合训练Encoder、Decoder和码本，提升重建质量

## 目录结构

```
TRELLIS-main/
├── trellis/
│   ├── datasets/
│   │   ├── ct_window_sparse.py         # CT窗口稀疏数据集 (新增)
│   │   └── __init__.py                 # 数据集注册 (已更新)
│   └── trainers/vae/
│       └── sparse_sdf_vqvae.py         # VQVAE训练器 (已更新)
├── configs/vae/
│   ├── ct_vqvae_stage1.json            # 阶段1配置 (新增)
│   └── ct_vqvae_stage2.json            # 阶段2配置 (新增)
├── tests/
│   └── test_ct_dataset.py              # 数据集测试脚本 (新增)
└── train.py                            # 训练主脚本
```

## 前置要求

### 1. 数据准备

确保已经完成CT数据预处理，输出目录结构如下：

```
processed_dataset/
├── dataset_name_1/
│   ├── metadata.csv
│   ├── dataset_config.json
│   └── processed/
│       ├── case_001/
│       │   ├── ct_normalized_512.npy
│       │   └── windows/
│       │       ├── lung_w1500_l-600.npy
│       │       ├── bone_w1500_l300.npy
│       │       ├── soft_tissue_w400_l50.npy
│       │       └── brain_w80_l35.npy
│       ├── case_002/
│       └── ...
└── dataset_name_2/
    └── ...
```

运行预处理：
```bash
bash scripts/prepare_ct_recursive.sh \
    /path/to/raw_ct_data \
    /path/to/processed_dataset \
    ./organ_mapping.json \
    8
```

### 2. 预训练VAE权重（可选）

如果有Direct3D-S2的预训练VAE权重，可以使用它来初始化：
- 将权重文件放在合适的位置（如 `./pretrained_weights/direct3d_vae.pth`）
- 在配置文件中设置 `pretrained_vae_path`

如果没有预训练权重，将 `pretrained_vae_path` 设为 `null`，从头训练。

## 快速开始

### 步骤1：配置训练参数

#### 阶段1配置 (`configs/vae/ct_vqvae_stage1.json`)

关键参数：
- `training_stage`: 1 (冻结VAE)
- `pretrained_vae_path`: 预训练权重路径（如果有）
- `window_type`: "lung" | "bone" | "soft_tissue" | "brain"
- `batch_size_per_gpu`: 根据显存调整（建议2-4）
- `learning_rate`: 1e-3（只训练码本）
- `loss_type`: "l1"
- `max_steps`: 100000

#### 阶段2配置 (`configs/vae/ct_vqvae_stage2.json`)

关键参数：
- `training_stage`: 2 (联合训练)
- `pretrained_vae_path`: null（从阶段1加载）
- `learning_rate`: 5e-3（联合训练使用更高学习率）
- `lr_scheduler`: ExponentialLR（学习率衰减）
- `loss_type`: "l1_l2"（组合损失）
- `max_steps`: 200000

### 步骤2：阶段1训练 - 冻结VAE，训练码本
<!-- 
为了解决int64_t(N) * int64_t(C) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max assert faild. your data exceed int32 range. 报错，加入环境变量 SPCONV_ALGO='native'、 -->

如果需要Debug，精确定位错误位置：export CUDA_LAUNCH_BLOCKING=1

```bash
export ATTN_BACKEND=xformers
python train.py \
    --config configs/vae/ct_vqvae_stage1.json \
    --output_dir outputs/ct_vqvae_stage1 \
    --data_dir ./processed_dataset \
    --num_gpus 4 > stage1_train.log 2>&1
```

### 步骤3：阶段2训练 - 联合微调

```bash
export ATTN_BACKEND=xformers
python train.py \
    --config configs/vae/ct_vqvae_stage2.json \
    --output_dir outputs/ct_vqvae_stage2 \
    --data_dir ./processed_dataset \
    --num_gpus 4 > stage2_train.log 2>&1
```

可选参数，或者config中配置阶段一训练权重：
```bash
    --load_dir outputs/ct_vqvae_lung_stage1 \
    --ckpt latest \
```

训练日志示例：
```
Loading checkpoint from step 100000...

================================================================================
[Stage 2] Joint training: Encoder + Decoder + Codebook
================================================================================
Total parameters: 125,829,120
Trainable parameters: 125,829,120
================================================================================

Step 1000: loss=0.156 recon=0.145 vq=0.008 commitment=0.003
Step 2000: loss=0.132 recon=0.124 vq=0.006 commitment=0.002
...
```

## 高级用法

### 多窗口训练

训练不同窗口类型的模型：

```bash
# 肺窗
python train.py --config configs/vae/ct_vqvae_stage1.json \
    --output_dir outputs/ct_vqvae_lung_stage1 \
    --data_dir /path/to/processed_dataset

# 骨窗
python train.py --config configs/vae/ct_vqvae_stage1.json \
    --output_dir outputs/ct_vqvae_bone_stage1 \
    --data_dir /path/to/processed_dataset
```

修改配置文件中的 `window_type` 参数。

### 多GPU训练

```bash
python train.py \
    --config configs/vae/ct_vqvae_stage1.json \
    --output_dir outputs/ct_vqvae_stage1 \
    --data_dir /path/to/processed_dataset \
    --num_gpus 4 \
    --num_nodes 1
```

### 从中断的训练恢复

```bash
python train.py \
    --config configs/vae/ct_vqvae_stage1.json \
    --output_dir outputs/ct_vqvae_stage1 \
    --load_dir outputs/ct_vqvae_stage1 \
    --ckpt latest \
    --data_dir /path/to/processed_dataset
```

### 调整超参数

常见调整：

**增加批次大小**（如果显存充足）：
```json
"batch_size_per_gpu": 4,
"batch_split": 1
```

**减少显存使用**：
```json
"batch_size_per_gpu": 1,
"batch_split": 4,  // 梯度累积
"use_checkpoint": true  // 启用梯度检查点
```

**调整点云采样**：
```json
"dataset": {
    "args": {
        "min_points": 100,      // 最小点数
        "max_points": 300000    // 最大点数（减少以节省显存）
    }
}
```

## 监控训练

### TensorBoard

```bash
tensorboard --logdir outputs/ct_vqvae_stage1/tb_logs
```

### 检查点文件

```
outputs/ct_vqvae_stage1/
├── ckpts/
│   ├── vqvae_step0010000.pt          # 模型权重
│   ├── vqvae_ema0.9999_step0010000.pt  # EMA权重
│   └── misc_step0010000.pt            # 优化器状态
├── tb_logs/                           # TensorBoard日志
├── samples/                           # 可视化样本
├── vqvae_model_summary.txt           # 模型参数摘要
├── config.json                        # 训练配置
└── command.txt                        # 训练命令
```

### 验证参数冻结

检查 `vqvae_model_summary.txt`：
```
Parameters:
================================================================================
Name                                                                    Shape                           Type            Grad
Encoder.input_layer1.weight                                            [512, 1]                        torch.float32   False
Encoder.downsample.0.weight                                             [512, 512]                      torch.float32   False
...
vq.embeddings.weight                                                    [8192, 32]                      torch.float32   True
```

阶段1中，Encoder和Decoder的 `Grad` 应为 `False`，VQ的 `Grad` 应为 `True`。

## 常见问题

### Q: 数据集为空或找不到数据？

**A**: 检查以下几点：
1. 数据目录路径是否正确
2. 确认预处理已完成，存在 `processed/` 目录
3. 窗口文件是否存在（如 `lung_w1500_l-600.npy`）
4. 运行测试脚本诊断问题

### Q: 内存不足 (OOM)?

**A**: 尝试以下方法：
1. 减少 `batch_size_per_gpu`
2. 增加 `batch_split`（梯度累积）
3. 减少 `max_points`（点云采样上限）
4. 启用 `use_checkpoint=true`

### Q: 训练速度慢？

**A**: 优化建议：
1. 增加 `num_workers`（数据加载并行）
2. 使用更快的存储（SSD）
3. 减少 `i_sample` 和 `i_save` 频率
4. 使用多GPU训练

### Q: 损失不下降？

**A**: 检查：
1. 学习率是否合适（阶段1: 1e-3, 阶段2: 5e-3）
2. 是否正确加载预训练权重
3. 数据质量（运行测试脚本检查）
4. 尝试调整 `lambda_vq` 和 `lambda_commitment`

### Q: 如何使用训练好的模型？

**A**: 加载模型权重：
```python
import torch
from trellis.models import Direct3DS2_VQVAE

# 加载模型
vqvae = Direct3DS2_VQVAE(
    resolution=64,
    model_channels=512,
    latent_channels=32,
    num_embeddings=8192
)

# 加载checkpoint
checkpoint = torch.load('outputs/ct_vqvae_stage2/ckpts/vqvae_step0200000.pt')
vqvae.load_state_dict(checkpoint)
vqvae.eval().cuda()

# 推理
with torch.no_grad():
    encoding_indices = vqvae.Encode(sparse_input)
    reconstruction = vqvae.Decode(encoding_indices)
```

## 实现细节

### 数据流程

```
CT预处理输出 (512³密集体素)
    ↓
CTWindowSparseSDF.get_instance()
    ↓
找到非零体素 (稀疏化)
    ↓
SparseTensor (coords + features)
    ↓
SparseSDF_VQVAETrainer
    ↓
Direct3DS2_VQVAE
    ↓
训练/推理
```

### 关键特性

1. **自动递归数据发现**：自动扫描多层目录结构
2. **动态稀疏转换**：训练时将密集体素转为稀疏格式
3. **两阶段训练**：通过 `training_stage` 参数控制
4. **多窗口支持**：支持4种标准CT窗口类型
5. **灵活的批处理**：自定义collate_fn处理可变大小点云

## 参考

- **数据预处理**: `QUICKSTART_CT_PREPROCESSING.md`
- **VQVAE集成**: `VQVAE_INTEGRATION_README.md`
- **训练框架**: `train.py`
- **模型定义**: `trellis/models/autoencoders/ss_vqvae.py`

## 贡献者

实现基于：
- Direct3D-S2 VQVAE架构
- TRELLIS训练框架
- CT数据预处理流程

## 更新日志

- **2024-12**: 初始实现
  - CT窗口稀疏数据集
  - 两阶段VQVAE训练器
  - 配置文件和测试脚本

