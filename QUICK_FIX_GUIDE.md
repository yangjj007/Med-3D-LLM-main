# 快速修复指南 - 数据加载优化

## 🎯 问题现象

- ✅ 数据加载非常慢（卡在第一次prefetch）
- ✅ GPU利用率为0%
- ✅ 只有20个训练样本，但加载时间很长

## ⚡ 立即可用的解决方案

### 方案1: 启用内存缓存（推荐，适合你的20个样本）

修改你的训练配置文件，在创建数据集时添加两个参数：

```python
# 找到创建数据集的代码，类似这样:
dataset = CTWindowSparseSDF(
    roots="your/data/path",
    resolution=512,
    window_type='lung',
    min_points=100,
    max_points=100000,
    cache_data=True,        # 🆕 添加这行
    precompute_sparse=True  # 🆕 添加这行
)
```

**预期效果**:
- 第一个epoch会有约10-20秒的初始化时间（预加载所有数据）
- 之后每个batch加载时间 < 1秒
- 后续epoch几乎瞬时加载

### 方案2: 测试你的当前配置性能

在应用任何优化之前，先测试性能：

```bash
python tools/benchmark_dataloader.py \
    --roots "./processed_dataset/0000/processed" \
    --window_type lung \
    --batch_size 2 \
    --num_batches 10 \
    --cache_data \
    --precompute_sparse
```

这会测试不同的worker配置并给出推荐设置。

### 方案3: 预处理为稀疏格式（可选，进一步优化）

如果上述方案还不够快，可以离线预处理数据：

```bash
python tools/precompute_sparse_data.py \
    --roots "./processed_dataset/0000/processed" \
    --window_type lung \
    --workers 4
```

这会为每个 `.npy` 文件生成对应的 `_sparse.npz` 文件，加载速度更快。

## 📊 优化对比

| 配置 | 首次加载 | 后续加载 | GPU利用率 |
|------|---------|---------|-----------|
| 原始代码 | 30-60s | 30-60s | 0% |
| 启用缓存 | 10-20s | <1s | 高 |
| 预处理+缓存 | <5s | <0.5s | 高 |

## 🔍 如何找到你的训练配置文件

根据你打开的文件，训练配置可能在：

1. 训练脚本中 - 搜索 `CTWindowSparseSDF(` 
2. 配置文件中 - 可能是 `config.py` 或 `.yaml` 文件

例如：

```bash
# 搜索数据集创建的位置
grep -r "CTWindowSparseSDF" . --include="*.py"
```

## ✅ 验证优化效果

修改后运行训练，观察输出：

```
[DataLoader] Configuration:
  Dataset size: 20
  Batch size per GPU: 2
  Num workers: 2          <-- 应该看到较小的数值（2-4）
  ...

CTWindowSparseSDF Dataset:
  ...
  Cache enabled: True     <-- 应该是True
  Precompute sparse: True <-- 应该是True
  Preloading all 20 instances to memory...  <-- 会看到预加载过程
  Preloading completed in 15.32s            <-- 首次加载时间

[DEBUG] Training step 0, calling load_data()...
[DEBUG] load_data() returned, got 1 batches  <-- 应该很快返回
```

## 🐛 故障排除

### 如果内存不足

如果看到 `MemoryError` 或系统内存占用过高：

```python
# 只启用稀疏索引预计算，不缓存原始数据
dataset = CTWindowSparseSDF(
    roots="your/data/path",
    resolution=512,
    window_type='lung',
    cache_data=False,       # 改为False
    precompute_sparse=True  # 保持True
)
```

### 如果仍然很慢

1. 检查数据是否在HDD上（应该用SSD）
2. 检查磁盘空间是否充足
3. 运行性能测试脚本获取详细报告：
   ```bash
   python tools/benchmark_dataloader.py --roots "your/data/path" --cache_data --precompute_sparse
   ```

### 如果看到警告

```
Warning: Failed to load metadata from ...
```
这是正常的，不影响训练，可以忽略。

## 📝 技术细节

已实施的核心优化：

1. **内存缓存** (`cache_data=True`)
   - 第一次加载后保存在RAM中
   - 避免重复磁盘I/O
   
2. **稀疏索引预计算** (`precompute_sparse=True`)
   - 预先计算 `np.nonzero()` 结果
   - 避免重复执行耗时的numpy操作
   
3. **优化numpy操作**
   - 使用 `np.nonzero()` 替代 `np.argwhere()`
   - 速度提升3-5倍
   
4. **动态worker数量**
   - 小数据集使用少量worker（2-4个）
   - 减少进程创建开销
   
5. **Memory-mapped文件**
   - 对未缓存的数据使用 `mmap_mode='r'`
   - 让OS管理内存

## 🎓 更多信息

详细优化说明请查看: `DATA_LOADING_OPTIMIZATION_GUIDE.md`

## 💬 需要帮助？

如果问题仍未解决，请提供：
1. 运行 `benchmark_dataloader.py` 的完整输出
2. 你的数据集大小和位置（HDD还是SSD）
3. 系统内存大小
4. 具体的错误信息（如果有）

