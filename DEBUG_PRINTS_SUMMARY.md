# 调试信息添加总结

## 问题描述
训练过程中出现 RuntimeError：
```
RuntimeError: The size of tensor a (6640567) must match the size of tensor b (59944) at non-singleton dimension 0
```

错误发生在稀疏张量加法操作中，具体位置：
- `trellis/modules/sparse/basic.py` 第318行的 `__elemwise__` 方法
- `trellis/models/autoencoders/encoder.py` 第52行的 skip connection 加法

## 添加的调试打印位置

### 1. `trellis/modules/sparse/basic.py` - SparseTensor.__elemwise__()
**位置**: 第309-343行

**添加的调试信息**:
- 操作开始时的分隔线标记
- self 和 other 的类型
- self.feats.shape, self.coords.shape, self.shape
- 如果 other 是 torch.Tensor，打印 broadcast 前后的形状
- 如果 other 是 SparseTensor，打印其 feats 和 coords 的形状
- 执行操作前，打印两个操作数的准确大小

**目的**: 准确定位哪个张量的大小不匹配，以及为什么不匹配

### 2. `trellis/models/autoencoders/encoder.py` - SparseDownBlock3d._forward()
**位置**: 第42-75行

**添加的调试信息**:
- 输入 x 的 feats 和 coords 形状
- act_layers 后的形状
- down(h) 和 down(x) 后的形状（下采样操作）
- out_layers 后的形状
- **关键**: skip_connection(x) 的输出形状
- 加法操作前，打印两个张量的大小对比
- 打印两个张量的唯一坐标数量

**目的**: 追踪 skip connection 分支和主分支在下采样后的体素数量变化

### 3. `trellis/modules/sparse/spatial.py` - SparseDownsample.forward()
**位置**: 第22-61行

**添加的调试信息**:
- 输入的 feats 和 coords 形状
- 下采样因子
- 原始体素数和下采样后的唯一体素数
- idx 的形状（映射关系）
- 输出的 feats 和 coords 形状

**目的**: 理解下采样如何改变体素数量，这是导致大小不匹配的根本原因

### 4. `trellis/modules/sparse/conv/conv_spconv.py` - SparseConv3d.forward()
**位置**: 第23-76行

**添加的调试信息**:
- 输入 coords 和 feats 的 dtype 及形状
- 坐标的最小值和最大值
- 检查 NaN 和 Inf
- 大特征值警告
- 卷积的 stride 和 padding 参数
- 输出的 feats 和 coords 形状

**目的**: 追踪卷积操作（用于 skip_connection）如何影响稀疏张量

### 5. `trellis/modules/sparse/conv/conv_torchsparse.py` - SparseConv3d.forward()
**位置**: 第13-25行

**添加的调试信息**:
- 输入 feats 和 coords 形状
- Conv stride
- 输出 feats 和 coords 形状

**目的**: 为 torchsparse 后端提供相同的调试能力

## 调试输出格式

所有调试信息都使用清晰的前缀：
- `[DEBUG ...]`: 一般调试信息
- `[WARNING ...]`: 警告信息
- `[ERROR ...]`: 错误信息
- `[CRITICAL ...]`: 严重问题

使用分隔线标记重要操作块：
- `{'='*80}`: SparseTensor 元素级操作
- `{'#'*80}`: DownBlock 前向传播
- `{'>'*80}`: SparseDownsample 操作

## 问题定位策略

通过这些调试信息，您可以：

1. **识别大小不匹配的来源**: 查看 `__elemwise__` 中的打印，确认是哪两个张量大小不匹配

2. **追踪下采样效果**: 查看 `SparseDownsample` 的输出，了解每次下采样如何改变体素数量

3. **对比两个分支**: 在 `DownBlock` 中，对比：
   - 主分支: x → act_layers → down → out_layers → h
   - skip分支: x → down → skip_connection → skip_result
   
   查看它们的最终体素数量（feats.shape[0]）是否一致

4. **检查坐标一致性**: 通过打印唯一坐标数量，判断两个分支的空间位置是否对齐

## 可能的根本原因

基于错误信息和代码结构，问题可能是：

1. **下采样不一致**: 两个分支的下采样操作可能产生了不同数量的体素点
2. **坐标不对齐**: 虽然两个分支都经过了 down(x)，但后续操作可能改变了体素的空间分布
3. **out_layers 影响**: out_layers 中的卷积操作可能改变了体素数量
4. **医疗CT数据特殊性**: 大的体素数差异（6640567 vs 59944）暗示可能是医疗数据的稀疏性导致的

## 使用说明

1. 重新运行训练脚本
2. 查看终端输出，按照上述标记找到关键信息
3. 重点关注最后一个 `DownBlock` 操作中，加法前的两个张量形状
4. 将完整的调试输出发送给我，以便进一步分析

## 文件修改清单

- ✅ `trellis/modules/sparse/basic.py`
- ✅ `trellis/models/autoencoders/encoder.py`
- ✅ `trellis/modules/sparse/spatial.py`
- ✅ `trellis/modules/sparse/conv/conv_spconv.py`
- ✅ `trellis/modules/sparse/conv/conv_torchsparse.py`

