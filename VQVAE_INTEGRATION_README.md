# Direct3D Sparse SDF VQVAE Integration

本文档说明如何使用整合到 Trellis 框架中的 Direct3D 稀疏 VQVAE 进行训练。

## 快速开始

### 1. 环境准备

确保已安装 Trellis 的所有依赖。另外需要编译 UDF CUDA 扩展（用于 mesh 到稀疏 SDF 转换）：

```bash
cd Direct3D-S2-main/third_party/voxelize
python setup.py install
```

### 2. 数据准备

准备医学数据集（mesh 文件或体素数组）：

```bash
bash TRELLIS-main/scripts/prepare_medical_dataset.sh /path/to/medical/data ./data/medical
```

这将：
- 构建元数据
- 计算多分辨率稀疏 SDF (64³, 512³, 1024³)
- 保存为 `.npz` 格式

### 3. 开始训练

训练 512³ 分辨率的 VQVAE：

```bash
bash TRELLIS-main/scripts/train_vqvae.sh 512 ./data/medical ./outputs/vqvae_512
```

支持的分辨率：64, 512, 1024

### 4. 测试

运行单元测试：

```bash
bash TRELLIS-main/scripts/test_data_preprocessing.sh
```

测试模型前向传播：

```bash
cd TRELLIS-main
python scripts/test_model_forward.py
```

## 项目结构

```
TRELLIS-main/
├── trellis/
│   ├── utils/
│   │   └── mesh_utils.py              # Mesh 处理工具
│   ├── datasets/
│   │   └── sparse_sdf.py              # 稀疏 SDF 数据集
│   ├── trainers/vae/
│   │   └── sparse_sdf_vqvae.py        # VQVAE 训练器
│   └── models/autoencoders/
│       └── ss_vqvae.py                # VQVAE 模型（已存在）
├── configs/vae/
│   ├── sparse_sdf_vqvae_64.json       # 64³ 配置
│   ├── sparse_sdf_vqvae_512.json      # 512³ 配置
│   └── sparse_sdf_vqvae_1024.json     # 1024³ 配置
├── dataset_toolkits/
│   ├── compute_sparse_sdf.py          # 稀疏 SDF 预处理
│   └── datasets/
│       └── MedicalData.py             # 医学数据集工具
└── scripts/
    ├── prepare_medical_dataset.sh     # 数据准备
    ├── train_vqvae.sh                 # 训练脚本
    ├── test_data_preprocessing.sh     # 测试脚本
    └── test_model_forward.py          # 模型测试
```

## 关键特性

1. **多分辨率支持**: 64³, 512³, 1024³
2. **预训练 VAE 加载**: 可从 Direct3D 预训练权重初始化
3. **分布式训练**: 内置 DDP 支持
4. **混合精度**: FP16 训练支持
5. **自动检查点**: 自动保存和恢复

## 配置说明

主要配置参数（在 `configs/vae/*.json` 中）：

- `resolution`: 训练分辨率
- `num_embeddings`: Codebook 大小（默认: 8192）
- `latent_channels`: 潜在空间维度（默认: 32）
- `lambda_vq`: VQ 损失权重（默认: 1.0）
- `lambda_commitment`: Commitment 损失权重（默认: 0.25）
- `pretrained_vae_path`: 预训练 VAE 路径

## 训练监控

训练日志保存在输出目录：
- `{output_dir}/ckpts/`: 检查点
- `{output_dir}/tb_logs/`: TensorBoard 日志
- `{output_dir}/samples/`: 可视化样本

启动 TensorBoard：

```bash
tensorboard --logdir ./outputs/vqvae_512/tb_logs
```

## 常见问题

**Q: UDF 扩展编译失败？**
A: 确保 CUDA 版本匹配，检查 nvcc 是否在 PATH 中。

**Q: 内存不足？**
A: 减小 `batch_size_per_gpu` 或增加 `batch_split`。

**Q: 训练慢？**
A: 启用 `use_checkpoint=true` 进行梯度检查点，节省显存但会降低速度。

## 参考

- Direct3D-S2: [github.com/DreamTechAI/Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)
- TRELLIS: [github.com/microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)

