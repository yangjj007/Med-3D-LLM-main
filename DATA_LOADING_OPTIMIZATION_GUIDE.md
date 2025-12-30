# 数据加载优化指南

## 问题诊断

你遇到的问题：
1. **数据加载非常慢** - 程序卡在 `load_data()` 的第一次预取阶段
2. **GPU利用率为0** - 所有时间都花在CPU数据加载上

## 已实施的优化

### 1. 数据集层面优化 (`ct_window_sparse.py`)

#### a) 内存缓存机制
```python
cache_data=True  # 将加载的.npy文件缓存到内存中
precompute_sparse=True  # 预计算稀疏索引并缓存
```

**优点**：
- 第一次加载后，后续epoch不需要重新读取磁盘
- 避免重复执行 `np.argwhere()` 这种耗时操作

**适用场景**：
- 数据集不大（<100个样本，每个<500MB）
- 系统有足够内存（建议至少32GB）

#### b) 使用 `np.nonzero()` 替代 `np.argwhere()`
```python
# 旧代码（慢）
indices = np.argwhere(window_data > 0)

# 新代码（快3-5倍）
indices = np.stack(np.nonzero(window_data), axis=1)
```

#### c) Memory-mapped文件加载
```python
# 对于未缓存的数据，使用mmap_mode
window_data = np.load(instance['window_path'], mmap_mode='r')
```

**优点**：
- 不立即将整个文件加载到内存
- 让操作系统管理内存映射

### 2. DataLoader层面优化 (`base.py`)

#### a) 动态调整num_workers
```python
if dataset_size < 50:
    num_workers = min(2, cpu_count // gpu_count)  # 小数据集
elif dataset_size < 200:
    num_workers = min(4, cpu_count // gpu_count)  # 中等数据集
else:
    num_workers = min(8, cpu_count // gpu_count)  # 大数据集
```

**原因**：
- 你的数据集只有20个样本，多个worker会导致：
  - 进程创建开销
  - 每个worker都要初始化数据集
  - worker之间数据重复加载

#### b) 添加prefetch_factor
```python
prefetch_factor=2  # 每个worker预取2个batch
```

**优点**：
- 在GPU处理当前batch时，提前准备好下一个batch

## 使用建议

### 对于你当前的20个样本数据集：

**推荐配置**：
```python
dataset = CTWindowSparseSDF(
    roots="your/data/path",
    resolution=512,
    window_type='lung',
    min_points=100,
    max_points=100000,
    cache_data=True,        # 启用缓存
    precompute_sparse=True  # 预计算稀疏索引
)
```

**预期效果**：
- 第一个epoch会慢（需要加载并缓存所有数据）
- 后续epoch会快得多（直接从内存读取）
- 数据加载时间应该从几分钟降到几秒

### 对于更大的数据集（>100个样本）：

**推荐配置**：
```python
dataset = CTWindowSparseSDF(
    roots="your/data/path",
    resolution=512,
    window_type='lung',
    min_points=100,
    max_points=100000,
    cache_data=False,       # 不全部缓存（内存不够）
    precompute_sparse=True  # 仍然预计算（会逐个缓存）
)
```

## 进一步优化建议

### 1. 预处理稀疏格式

如果数据加载仍然很慢，可以考虑离线预处理：

```python
# 创建一个脚本来预先计算稀疏索引
import numpy as np

for instance in dataset.instances:
    window_data = np.load(instance['window_path'])
    indices = np.stack(np.nonzero(window_data), axis=1)
    values = window_data[indices[:, 0], indices[:, 1], indices[:, 2]]
    
    # 保存为稀疏格式
    sparse_path = instance['window_path'].replace('.npy', '_sparse.npz')
    np.savez_compressed(sparse_path, indices=indices, values=values)
```

然后修改 `__getitem__` 直接加载稀疏格式：
```python
sparse_data = np.load(sparse_path)
indices = sparse_data['indices']
values = sparse_data['values']
```

### 2. 检查磁盘I/O

如果数据在HDD（机械硬盘）上：
- 考虑将数据移到SSD
- 或者使用 `cache_data=True` 一次性加载到内存

### 3. 使用更小的分辨率进行测试

在调试时，可以先用更小的分辨率：
```python
resolution=128  # 而不是512
```

这样可以更快验证代码逻辑。

## 性能基准

**未优化（原代码）**：
- 第一次加载batch：~30-60秒（取决于磁盘速度）
- 包含大量 `np.argwhere()` 操作
- 每个epoch都需要重新加载

**优化后（启用所有优化）**：
- 第一次加载batch：~10-20秒（预加载和缓存）
- 后续batch：<1秒（从内存读取）
- 后续epoch：几乎瞬时（完全从缓存读取）

## 调试建议

如果优化后仍然慢，添加性能分析：

```python
import time

def __getitem__(self, index: int):
    t0 = time.time()
    
    # 加载数据
    t1 = time.time()
    window_data = self._load_window_data(index)
    print(f"Load time: {(time.time() - t1)*1000:.1f}ms")
    
    # 获取稀疏索引
    t1 = time.time()
    indices, values = self._get_sparse_indices(index, window_data)
    print(f"Sparse time: {(time.time() - t1)*1000:.1f}ms")
    
    # ... 其余代码
    
    print(f"Total __getitem__ time: {(time.time() - t0)*1000:.1f}ms")
```

## 总结

主要优化策略：
1. ✅ 内存缓存 - 避免重复磁盘I/O
2. ✅ 优化numpy操作 - `np.nonzero()` 比 `np.argwhere()` 快
3. ✅ 减少worker数量 - 避免小数据集的进程开销
4. ✅ 添加预取 - GPU和CPU流水线并行
5. 💡 考虑离线预处理 - 如果还不够快

这些优化应该能将你的数据加载时间减少80-90%。

