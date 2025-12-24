# CT数据3D可视化工具

## 概述

为 `bash scripts/prepare_ct_recursive.sh` 数据处理流程生成的CT数据集提供完整的3D交互式可视化解决方案。

### ✨ 主要特性

- 🎮 **3D交互式可视化** - 可拖动、旋转、缩放的3D视图
- 📊 **多种视图模式** - 切片、体渲染、表面渲染
- 🪟 **窗口对比** - 肺窗、骨窗、软组织窗等多窗口对比
- 🫀 **器官分割** - 器官的3D表面渲染和可视化
- 🌐 **独立HTML** - 生成独立的HTML文件，无需服务器
- 📱 **响应式设计** - 美观的用户界面，支持各种屏幕尺寸

## 快速开始

### 1. 安装依赖

```bash
pip install plotly scikit-image kaleido numpy
```

### 2. 可视化单个病例

```bash
bash scripts/visualize_ct.sh /processed_dataset/processed/0000
```

### 3. 查看结果

可视化文件将保存在 `/processed_dataset/processed/0000/visualization/`

在浏览器中打开 `index.html` 查看所有可视化结果。

## 使用方法

### 方法1：使用Bash脚本（推荐）

#### 可视化单个病例

```bash
bash scripts/visualize_ct.sh <数据集路径> [输出目录]
```

**示例：**
```bash
# 使用默认输出目录（病例目录下的visualization文件夹）
bash scripts/visualize_ct.sh /processed_dataset/processed/0000

# 指定自定义输出目录
bash scripts/visualize_ct.sh /processed_dataset/processed/0000 /custom/output/path
```

#### 批量可视化多个病例

```bash
bash scripts/visualize_ct_batch.sh <processed目录> [最大数量]
```

**示例：**
```bash
# 可视化所有病例
bash scripts/visualize_ct_batch.sh /processed_dataset/processed

# 只可视化前5个病例
bash scripts/visualize_ct_batch.sh /processed_dataset/processed 5
```

### 方法2：使用Python脚本

#### 可视化单个病例

```bash
python dataset_toolkits/visualize_ct_dataset.py \
    --dataset_path /processed_dataset/processed/0000
```

#### 指定输出目录

```bash
python dataset_toolkits/visualize_ct_dataset.py \
    --dataset_path /processed_dataset/processed/0000 \
    --output_dir /custom/output/path
```

### 方法3：在Python代码中使用

```python
from dataset_toolkits.visualize_ct_dataset import visualize_ct_dataset

# 可视化单个病例
visualize_ct_dataset('/processed_dataset/processed/0000')

# 指定输出目录
visualize_ct_dataset(
    dataset_path='/processed_dataset/processed/0000',
    output_dir='/custom/output/path'
)
```

## 输入数据格式

工具需要以下数据结构（由 `prepare_ct_recursive.sh` 生成）：

```
/processed_dataset/processed/0000/
├── ct_normalized_512.npy          # 必需：标准化的CT数据
├── windows/                        # 可选：不同窗口的数据
│   ├── lung_w1500_l-600.npy
│   ├── bone_w1500_l300.npy
│   ├── soft_tissue_w400_l40.npy
│   └── brain_w80_l40.npy
├── organs/                         # 可选：器官分割数据
│   ├── liver/
│   │   └── liver_binary_512.npy
│   ├── lung/
│   │   └── lung_binary_512.npy
│   └── ...
└── masks/                          # 可选：分割掩码
    └── segmentation_masks.npz
```

### 必需文件

- `ct_normalized_512.npy` 或 `ct_normalized_1024.npy` - 标准化的CT数据

### 可选文件

- `windows/*.npy` - 窗口数据（肺窗、骨窗等）
- `organs/*/binary*.npy` - 器官二值掩码
- `masks/*.npz` - 分割掩码

## 输出文件说明

可视化工具生成以下HTML文件：

### 1. 索引页面
- **文件名：** `index.html`
- **说明：** 总览页面，包含所有可视化的链接和说明
- **特点：** 美观的响应式设计，易于导航

### 2. 总览仪表板
- **文件名：** `01_overview_dashboard.html`
- **内容：**
  - 三个正交切片（横断面、冠状面、矢状面）
  - CT值分布直方图
  - 数据统计信息
  - 窗口对比

### 3. CT三切片视图
- **文件名：** `02_ct_slices.html`
- **内容：** 矢状面、冠状面、横断面的详细切片视图

### 4. CT 3D体渲染
- **文件名：** `03_ct_3d_volume.html`
- **内容：** CT数据的交互式3D体渲染
- **交互：** 可拖动旋转、滚轮缩放、双击重置

### 5. 窗口对比
- **文件名：** `04_windows_comparison.html`
- **内容：** 多个窗口的并排对比视图

### 6. 器官3D渲染
- **文件名：** `05_organ_<器官名>_3d.html`
- **内容：** 每个器官的3D表面渲染
- **示例：** `05_organ_liver_3d.html`, `05_organ_lung_3d.html`

## 可视化示例

### 示例1：基本可视化

```bash
# 准备数据
bash scripts/prepare_ct_recursive.sh \
    ./med_datasets \
    ./processed_datasets \
    ./organ_mapping.json \
    8

# 可视化第一个病例
bash scripts/visualize_ct.sh ./processed_datasets/processed/0000
```

### 示例2：批量可视化

```bash
# 可视化所有病例
bash scripts/visualize_ct_batch.sh ./processed_datasets/processed

# 生成的文件：
# - ./processed_datasets/processed/0000/visualization/index.html
# - ./processed_datasets/processed/0001/visualization/index.html
# - ...
# - ./processed_datasets/processed/visualization_summary.html (总索引)
```

### 示例3：编程方式使用

```python
# 查看 examples/visualize_ct_example.py 了解更多示例

# 运行示例
python examples/visualize_ct_example.py \
    --dataset_root ./processed_datasets \
    --case_id 0000 \
    --example all
```

## 交互操作说明

在生成的HTML文件中，您可以进行以下操作：

### 3D视图交互

- **旋转：** 鼠标左键拖动
- **平移：** Shift + 鼠标左键拖动
- **缩放：** 鼠标滚轮
- **重置视图：** 双击
- **保存图片：** 点击工具栏的相机图标

### 工具栏功能

- 📸 **拍照** - 保存当前视图为PNG图片
- 🔍 **缩放** - 缩放工具
- ↔️ **平移** - 平移工具
- 🔲 **框选缩放** - 框选区域放大
- 🏠 **重置** - 重置视图
- ⚙️ **设置** - 更多选项

## 高级功能

### 自定义切片位置

在Python代码中使用：

```python
from dataset_toolkits.visualize_ct_dataset import create_slices_plot, load_npy_data
import numpy as np

# 加载数据
ct_volume = load_npy_data('path/to/ct_normalized_512.npy')

# 创建自定义切片位置的可视化
custom_slices = [128, 256, 384]  # X, Y, Z切片位置
fig = create_slices_plot(ct_volume, "自定义切片", custom_slices)
fig.write_html('custom_slices.html')
```

### 自定义3D渲染参数

```python
from dataset_toolkits.visualize_ct_dataset import create_3d_volume_plot

# 创建高透明度的3D渲染
fig = create_3d_volume_plot(
    volume=ct_volume,
    title="高透明度渲染",
    opacity=0.3,  # 增加透明度
    colorscale='Viridis'  # 改变颜色映射
)
fig.write_html('custom_3d.html')
```

### 器官表面渲染

```python
from dataset_toolkits.visualize_ct_dataset import create_organ_surface_plot

# 加载器官掩码
organ_mask = load_npy_data('path/to/liver_binary_512.npy')

# 创建器官表面渲染
fig = create_organ_surface_plot(
    mask=organ_mask,
    organ_name='Liver',
    color='red',
    opacity=0.7
)
fig.write_html('liver_surface.html')
```

## 性能优化

### 降采样

为了提高渲染性能，工具会自动对3D数据进行降采样（每2个体素取1个）。这不会影响视觉效果，但会显著提高加载速度。

### 浏览器性能

- **推荐浏览器：** Chrome, Firefox, Edge (最新版本)
- **硬件加速：** 确保浏览器启用了硬件加速
- **内存：** 建议至少8GB RAM用于大型数据集

### 大数据集处理

对于1024³分辨率的数据：

```python
# 手动降采样以提高性能
ct_volume_small = ct_volume[::4, ::4, ::4]
```

## 故障排除

### 问题1：找不到模块

```bash
错误: ModuleNotFoundError: No module named 'plotly'
```

**解决方案：**
```bash
pip install plotly scikit-image kaleido numpy
```

### 问题2：无法找到CT数据文件

```bash
错误: 未找到CT数据文件
```

**解决方案：**
确保数据路径正确，且包含以下文件之一：
- `ct_normalized_512.npy`
- `ct_normalized_1024.npy`

### 问题3：器官表面渲染失败

```bash
警告: 无法为liver生成表面渲染
```

**可能原因：**
- 器官掩码为空
- 掩码数据格式不正确（应该是0/1二值数据）

**解决方案：**
检查器官掩码文件，确保包含有效的二值数据。

### 问题4：浏览器无法打开HTML文件

**解决方案：**
1. 检查文件路径是否正确
2. 尝试不同的浏览器
3. 确保浏览器没有阻止本地文件访问

### 问题5：内存不足

```bash
错误: MemoryError
```

**解决方案：**
1. 减少并行处理数量
2. 使用较小分辨率的数据
3. 增加系统内存或交换空间

## 技术细节

### 使用的库

- **Plotly** - 3D可视化和交互式图表
- **NumPy** - 数组处理
- **scikit-image** - Marching Cubes算法（表面提取）
- **Kaleido** - 静态图片导出（可选）

### 数据流程

```
输入数据 (.npy/.npz)
    ↓
加载和验证
    ↓
降采样（如需要）
    ↓
生成Plotly图表
    ↓
导出HTML文件
    ↓
创建索引页面
```

### 文件大小

- 单个HTML文件：通常 2-10 MB
- 完整可视化集：通常 20-50 MB/病例
- 批量可视化：取决于病例数量

## 示例工作流程

### 完整的数据处理和可视化流程

```bash
# 1. 预处理CT数据
bash scripts/prepare_ct_recursive.sh \
    ./raw_datasets \
    ./processed_datasets \
    ./organ_mapping.json \
    8

# 2. 批量可视化
bash scripts/visualize_ct_batch.sh ./processed_datasets/processed 10

# 3. 打开总索引查看所有可视化
# file:///path/to/processed_datasets/processed/visualization_summary.html
```

## 常见问题 (FAQ)

### Q: 可以在没有互联网的情况下使用吗？

**A:** 可以！生成的HTML文件是完全独立的，包含所有必要的JavaScript库（通过CDN加载，但会缓存）。首次打开需要互联网，之后可以离线使用。

### Q: 可以自定义颜色和样式吗？

**A:** 可以！您可以在Python代码中使用不同的参数，或者直接编辑生成的HTML文件。

### Q: 支持哪些数据格式？

**A:** 目前支持：
- `.npy` - NumPy数组
- `.npz` - 压缩的NumPy数组

原始的`.nii.gz`文件需要先通过预处理脚本转换。

### Q: 可以导出图片吗？

**A:** 可以！点击Plotly工具栏的相机图标可以保存当前视图为PNG图片。

### Q: 如何处理非常大的数据集？

**A:** 
1. 使用批量脚本的最大数量参数限制处理数量
2. 分批处理
3. 考虑使用更低的分辨率（512³而不是1024³）

## 贡献和反馈

如果您发现问题或有改进建议，请：
1. 检查本文档的故障排除部分
2. 查看 `examples/visualize_ct_example.py` 的示例代码
3. 提交Issue或Pull Request

## 许可证

本工具遵循项目的主许可证。

## 更新日志

### v1.0.0 (2024)
- ✨ 初始版本
- 🎮 3D交互式可视化
- 📊 多种视图模式
- 🪟 窗口对比功能
- 🫀 器官分割可视化
- 📱 响应式设计
- 🔄 批量处理支持

---

**祝您使用愉快！** 🎉

