# K-means 码本周期性重新初始化功能使用说明

## 功能概述

已成功为 VQ-VAE 训练流程添加了周期性 K-means 聚类码本重新初始化功能，参考 VQFR 的实现。

## 主要特性

1. **水塘采样器**：持续收集编码器输出的特征向量（固定容量 16384）
2. **K-means 重估计**：周期性地用 K-means 聚类中心整体重置码本
3. **可配置开关**：通过 `use_kmeans_reinit` 参数控制是否启用
4. **双模式支持**：梯度模式和 EMA 模式都支持 K-means
5. **分布式训练支持**：使用 `torch.distributed` 支持多 GPU 训练

## 配置方法

### 1. 启用 K-means 重估计

在模型配置文件中添加以下参数（如 `configs/vae/sparse_sdf_vqvae_1024_kmeans.json`）：

```json
{
    "models": {
        "vqvae": {
            "name": "SparseSDFVQVAE",
            "args": {
                "num_embeddings": 8192,
                "use_ema_update": false,
                "use_kmeans_reinit": true,      // 启用 K-means
                "kmeans_interval": 2000,         // 每 2000 步重估计一次
                "reservoir_size": 16384          // 水塘采样器容量
            }
        }
    }
}
```

### 2. 禁用 K-means（默认行为）

```json
{
    "models": {
        "vqvae": {
            "args": {
                "use_kmeans_reinit": false  // 或不设置此参数
            }
        }
    }
}
```

## 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_kmeans_reinit` | bool | False | 是否启用 K-means 重新初始化 |
| `kmeans_interval` | int | 2000 | K-means 重估计周期（步数） |
| `reservoir_size` | int | 16384 | 水塘采样器容量（应远大于码本大小） |

## 与现有更新模式的兼容性

### 梯度模式 + K-means
```json
"use_ema_update": false,
"use_kmeans_reinit": true
```
- 码本通过梯度反向传播更新
- 周期性 K-means 重置码本

### EMA 模式 + K-means
```json
"use_ema_update": true,
"use_kmeans_reinit": true
```
- 码本通过 EMA 统计更新
- 周期性 K-means 重置码本
- K-means 重置时会清空 EMA 统计量

### 仅梯度/EMA（无 K-means）
```json
"use_kmeans_reinit": false
```
- 保持原有行为，不进行 K-means 重估计

## 训练日志示例

启用 K-means 后，训练日志会显示：

```
[K-means 重估计] 已启用，间隔=2000步，水塘容量=16384
...
[K-means 重估计] 开始，使用 16384 个样本重建 8192 个码本向量...
[K-means 重估计] 完成！码本已更新
```

如果样本不足：
```
[K-means 重估计] 跳过：样本数不足 (5000 < 8192)
```

## 验证方法

### 1. 检查日志输出
运行训练时，在步数为 `kmeans_interval` 的倍数时应该看到 K-means 重估计日志。

### 2. 监控码本变化
K-means 重估计后，码本权重会发生显著变化，可以通过监控码本统计量（min/max/mean/std）来验证。

### 3. 测试示例

使用提供的配置文件运行训练：

```bash
python train.py \
    --config configs/vae/sparse_sdf_vqvae_1024_kmeans.json \
    --output_dir ./outputs/vqvae_kmeans_test \
    --data_dir ./data \
    --num_gpus 1
```

预期在以下步数看到 K-means 重估计：
- 步数 2000
- 步数 4000
- 步数 6000
- ...

## 实现细节

### 水塘采样算法
- 前 N 个样本直接填充缓冲区
- 之后的样本以概率 N/i 替换现有样本（i 为已见样本数）
- 保证每个样本被采样的概率相等

### K-means 整体重置策略（参考 VQFR）
- **不统计使用频率**（训练时）
- **不筛选低频条目**
- **直接重建整个码本**：用 K-means 聚类中心替换所有码本向量
- **周期性无条件触发**

### 分布式训练支持
- 使用 `torch.distributed.all_gather` 收集所有 GPU 的特征
- K-means 聚类在所有进程上同步执行
- 码本更新后自动同步到所有 GPU

## 预期效果

1. **提高码本利用率**：避免死亡向量，接近 100% 码本利用率
2. **加速收敛**：周期性重置确保码本始终覆盖当前特征分布
3. **更好的特征覆盖**：K-means 确保码本向量均匀分布在特征空间

## 注意事项

1. **依赖库**：确保安装 `scikit-learn` 库（用于 K-means 聚类）
   ```bash
   pip install scikit-learn
   ```

2. **首次重估计**：需要水塘采样器收集足够样本（至少 `num_embeddings` 个）

3. **计算开销**：K-means 聚类可能耗时 1-2 秒，但相对于训练周期可忽略

4. **内存使用**：水塘采样器会占用额外内存（`reservoir_size × embedding_dim × 4` 字节）

5. **推荐设置**：
   - `reservoir_size` 应至少为 `num_embeddings` 的 2 倍
   - `kmeans_interval` 建议设为 1000-5000 步
   - 训练初期（前 50K 步）启用 K-means 效果最佳

## 文件修改清单

以下文件已被修改以支持 K-means 功能：

1. **trellis/models/autoencoders/ss_vqvae.py**
   - 添加 `ReservoirSampler` 水塘采样器类
   - 修改 `SparseVectorQuantizer` 添加 K-means 支持
   - 修改 `SparseSDFVQVAE` 传递 `current_step`

2. **trellis/trainers/vae/sparse_sdf_vqvae.py**
   - 修改 `training_losses()` 传递 `self.step` 到模型

3. **configs/vae/sparse_sdf_vqvae_1024_kmeans.json** （新增）
   - 提供启用 K-means 的配置示例

## 故障排查

### 问题：K-means 从未触发
- **原因**：`current_step` 未正确传递，或 `use_kmeans_reinit=False`
- **解决**：检查配置文件和训练日志

### 问题：K-means 总是跳过（样本不足）
- **原因**：水塘采样器样本收集不足
- **解决**：增加 `reservoir_size` 或减小 `num_embeddings`

### 问题：训练速度明显变慢
- **原因**：`kmeans_interval` 设置过小
- **解决**：增加重估计间隔（如 5000）

## 参考资料

- VQFR 论文实现：`VQFR码本K-means初始化实现总结.md`
- 原始 VQFR 代码：`vqfr/archs/quantizer_arch.py`

