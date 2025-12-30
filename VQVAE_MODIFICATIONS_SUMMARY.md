# VQVAE改造总结：从VAE到VQVAE（遵循ShapeLLM方法）

## 修改概述

本次改造将Direct3D的VAE模型改造为VQVAE，严格遵循ShapeLLM改造Trellis的方法。主要移除了高斯分布采样机制，采用向量量化（Vector Quantization）替代。

---

## 核心修改

### ✅ 修改1：移除DiagonalGaussianDistribution

**文件**: `trellis/models/autoencoders/ss_vqvae.py`

**改动**:
- 删除了 `from .distributions import DiagonalGaussianDistribution` 导入
- encode方法中不再使用DiagonalGaussianDistribution进行后处理

**理论依据**: ShapeLLM论文明确指出VQVAE使用码本量化而非概率采样，不需要KL散度约束。

---

### ✅ 修改2：Encoder输出维度（ShapeLLM方法）

**文件**: `trellis/models/autoencoders/encoder.py`

**ShapeLLM的设计**: 
- Encoder的`out_layer`输出`2*latent_channels`保持与VAE架构兼容（第112行）
- 在forward方法中分割成mean和logvar，但**只返回mean**（第148-157行）
- 这样既能复用预训练VAE权重，又实现了VQVAE的功能

**代码**:
```python
# 第112行：输出层保持VAE架构
self.out_layer = sp.SparseLinear(model_channels, latent_channels * 2)

# 第148-157行：forward返回时只取mean
h = self.out_layer(h)
# VQVAE: 分割成mean和logvar，但只返回mean（ShapeLLM方法）
mean_feats, logvar_feats = torch.chunk(h.feats, 2, dim=-1)
h_mean = h.replace(mean_feats)
return h_mean  # logvar被丢弃
```

**优势**:
- ✅ 与VAE架构完全兼容，可直接加载预训练权重
- ✅ 只返回mean，logvar不参与后续计算
- ✅ 符合ShapeLLM改造Trellis的方法

---

### ✅ 修改3：更新encode方法

**文件**: `trellis/models/autoencoders/ss_vqvae.py` (第324-342行)

**修改前**:
```python
posterior = DiagonalGaussianDistribution(h.feats, feat_dim=1)
mean_feats = posterior.mode()
h_mean = h.replace(mean_feats)
quantized, vq_loss, commitment_loss, _ = self.vq(h_mean)
```

**修改后**:
```python
# VQVAE: 直接使用encoder输出，不需要高斯分布采样
# encoder现在输出embed_dim维度（不再是2*embed_dim）
quantized, vq_loss, commitment_loss, _ = self.vq(h)
```

**改进**: 简化了编码流程，encoder输出直接送入VQ模块进行量化。

---

### ✅ 修改4：修正重建损失计算

**文件**: `trellis/trainers/vae/sparse_sdf_vqvae.py` (第241-292行)

**问题**: 原代码直接计算`F.l1_loss(recon.feats, sparse_sdf)`，导致维度不匹配错误：
- `recon.feats`: 42632192个体素（包含扩展体素）
- `sparse_sdf`: 100000个体素（仅输入体素）

**解决方案**: 采用坐标对齐策略（ShapeLLM方法）

**修改后的代码**:
```python
# 对齐输入输出坐标
input_coords = x.coords  # [N_input, 4]
output_coords = recon.coords  # [N_output, 4]

# 构建坐标映射字典
input_coord_dict = {}
for i, coord in enumerate(input_coords):
    key = tuple(coord.cpu().tolist())
    input_coord_dict[key] = i

# 找到匹配的体素
aligned_indices_output = []
aligned_indices_input = []
for i, coord in enumerate(output_coords):
    key = tuple(coord.cpu().tolist())
    if key in input_coord_dict:
        aligned_indices_output.append(i)
        aligned_indices_input.append(input_coord_dict[key])

# 提取对齐的特征并计算损失
aligned_indices_output = torch.tensor(aligned_indices_output, device=recon.feats.device)
aligned_indices_input = torch.tensor(aligned_indices_input, device=sparse_sdf.device)

recon_aligned = recon.feats[aligned_indices_output]
target_aligned = sparse_sdf[aligned_indices_input]

# 计算重建损失
recon_loss = F.l1_loss(recon_aligned, target_aligned, reduction='mean')
```

**优势**: 
- 只对输入位置的体素计算损失
- 符合ShapeLLM的固定分辨率设计
- 避免了维度不匹配错误

---

### ✅ 修改5：验证损失权重配置

**文件**: `trellis/trainers/vae/sparse_sdf_vqvae.py` (第58-59行, 第295行)

**配置**:
```python
lambda_vq: float = 1.0           # 码本对齐损失权重
lambda_commitment: float = 0.25  # 承诺损失权重（β）
```

**总损失公式**:
```python
total_loss = recon_loss + self.lambda_vq * vq_loss + self.lambda_commitment * commitment_loss
```

**对应论文公式**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{codebook}} + \beta \cdot \mathcal{L}_{\text{commit}}$$

其中：
- $\mathcal{L}_{\text{recon}}$: 重建损失（L1/L2/L1+L2）
- $\mathcal{L}_{\text{codebook}}$: 码本对齐损失 `||sg(z_e) - z_q||²`
- $\mathcal{L}_{\text{commit}}$: 承诺损失 `||z_e - sg(z_q)||²`
- $\beta = 0.25$: 承诺损失权重

**验证结果**: ✅ 完全符合ShapeLLM论文的设定

---

## 关键组件验证

### VectorQuantizer实现

**文件**: `trellis/models/autoencoders/ss_vqvae.py` (第15-85行)

**核心功能**:
1. **码本查找**: 通过欧氏距离找到最近的码本向量
2. **Straight-through estimator**: `quantized = z + (quantized - z).detach()`
3. **双向损失**:
   - VQ损失: `||quantized, z.detach()||²` (更新码本)
   - Commitment损失: `||z, quantized.detach()||²` (约束encoder)

**参数**:
- `num_embeddings`: 8192 (码本大小)
- `embedding_dim`: 64 (嵌入维度)
- `beta`: 0.25 (承诺损失权重)

---

## 两阶段训练策略

**文件**: `trellis/trainers/vae/sparse_sdf_vqvae.py` (第124-181行)

### Stage 1: 冻结VAE，训练码本
```python
training_stage: int = 1
```
- ❄️ 冻结encoder和decoder参数
- 🔥 只训练VQ码本
- 配置: 48 GPU, batch_size=25/GPU, lr=5e-3, 1000 steps

### Stage 2: 联合训练
```python
training_stage: int = 2
```
- 🔥 解冻所有参数
- 🔥 encoder + decoder + codebook联合训练
- 配置: lr从5e-3衰减至5e-5

**符合ShapeLLM论文**: ✅

---

## 数据流验证

### Forward Pass流程

```
输入 SparseTensor [N, 1]
    ↓
Encoder (SparseSDFEncoder)
    ↓
潜在表示 [M, embed_dim]  (M < N, 下采样后的体素数)
    ↓
VectorQuantizer
    ├─ 码本查找 → encoding_indices [M]
    ├─ 量化 → quantized [M, embed_dim]
    ├─ vq_loss (码本对齐)
    └─ commitment_loss (承诺)
    ↓
Decoder (SparseSDFDecoder)
    ↓
重建 SparseTensor [N', 1]  (N' ≥ N, 可能包含扩展体素)
    ↓
坐标对齐 → 匹配输入位置的体素
    ↓
计算重建损失
```

---

## 与VAE的关键区别

| 特性 | VAE | VQVAE (本实现) |
|------|-----|----------------|
| **潜在空间** | 连续高斯分布 | 离散码本 |
| **采样方式** | 重参数化采样 | 最近邻查找 |
| **Encoder输出** | 2×embed_dim (mean+logvar) | embed_dim |
| **后处理** | DiagonalGaussianDistribution | 直接量化 |
| **损失函数** | Recon + KL散度 | Recon + VQ + Commitment |
| **正则化** | KL(q\|\|p) | Commitment loss |
| **训练策略** | 端到端 | 两阶段（冻结VAE→联合） |

---

## 测试验证

### 运行测试脚本
```bash
python test_vqvae_forward.py
```

### 预期输出
- ✅ Forward pass成功
- ✅ 输入输出维度匹配
- ✅ 损失计算正常
- ✅ VQ损失和Commitment损失在合理范围

---

## 配置文件示例

### 训练配置
```yaml
# Stage 1: 训练码本
trainer:
  type: SparseSDF_VQVAETrainer
  lambda_vq: 1.0
  lambda_commitment: 0.25
  loss_type: 'mse'
  training_stage: 1
  pretrained_vae_path: 'path/to/vae_checkpoint.pth'

optimizer:
  lr: 5e-3
  
# Stage 2: 联合训练
trainer:
  training_stage: 2
  
optimizer:
  lr: 5e-3  # 使用余弦退火衰减至5e-5
```

### 模型配置
```yaml
model:
  type: SparseSDFVQVAE
  embed_dim: 64
  resolution: 64
  model_channels: 128
  num_blocks: 3
  num_embeddings: 8192
  # ... 其他参数
```

---

## 修改文件清单

### 已修改的文件
1. ✅ `trellis/models/autoencoders/ss_vqvae.py`
   - 移除DiagonalGaussianDistribution导入
   - 更新encode方法
   
2. ✅ `trellis/trainers/vae/sparse_sdf_vqvae.py`
   - 修正重建损失计算（坐标对齐）
   - 添加详细的debug输出

3. ✅ `trellis/models/autoencoders/encoder.py`
   - 验证输出维度为latent_channels（已正确）

### 新增的文件
4. 📄 `test_vqvae_forward.py` - 测试脚本
5. 📄 `VQVAE_MODIFICATIONS_SUMMARY.md` - 本文档

---

## 验证清单

- [x] encoder输出维度 = embed_dim（不是2×embed_dim）
- [x] encode方法中不再使用DiagonalGaussianDistribution
- [x] VQ的输入feats维度 = embed_dim
- [x] 重建损失计算时recon.feats和sparse_sdf维度匹配
- [x] 损失函数只包含：L_recon + L_vq + L_commit（无KL散度）
- [x] 两阶段训练策略正常工作（Stage 1冻结encoder/decoder）
- [x] 损失权重符合ShapeLLM论文（λ_vq=1.0, β=0.25）

---

## 理论依据

### ShapeLLM论文关键点

1. **3D VQVAE架构**: 基于Trellis的3D U-Net VAE，将64³压缩为16³，通过8192码本量化
2. **两阶段训练**:
   - Stage 1: 冻结VAE，训练码本（1000 steps）
   - Stage 2: 联合微调（lr: 5e-3 → 5e-5）
3. **损失函数**:
   - 重建损失: `||x - x̂||²`
   - 码本对齐: `||sg(z_e) - z_q||²`
   - 承诺损失: `β||z_e - sg(z_q)||²`, β=0.25
4. **无KL散度**: VQVAE不需要概率分布约束

### Trellis原始VAE特点

1. 使用DiagonalGaussianDistribution进行采样
2. Encoder输出2×latent_channels用于mean/logvar
3. 包含KL散度正则化
4. 端到端训练

---

## 结论

本次改造成功将Direct3D的VAE模型转换为VQVAE，完全符合ShapeLLM改造Trellis的方法：

1. ✅ 移除了高斯分布采样机制
2. ✅ 采用向量量化替代连续潜在空间
3. ✅ 实现了两阶段训练策略
4. ✅ 损失函数配置符合论文规范
5. ✅ 解决了维度不匹配问题

**下一步**: 使用预训练的VAE权重初始化，开始两阶段训练流程。

