# CT数据预处理 - 快速开始指南

## 5分钟快速上手

### 步骤1：安装依赖

```bash
pip install numpy scipy pandas tqdm monai nibabel
```

### 步骤2：准备数据

系统支持两种数据格式：

#### 格式1：NIfTI格式

```
your_data/
├── imagesTr/
│   ├── case_001_0000.nii.gz
│   ├── case_002_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── case_001.nii.gz
    ├── case_002.nii.gz
    └── ...
```

#### 格式2：M3D-Seg格式

```
dataset_0000/
├── 0000.json
├── 1/
│   ├── image.npy
│   └── mask_(1, 512, 512, 96).npz
├── 2/
│   ├── image.npy
│   └── mask_*.npz
└── ...
```

#### 递归处理多个数据集

如果您有包含多个数据集的大文件夹：

```
all_datasets/
├── dataset_A/
│   ├── imagesTr/
│   └── labelsTr/
├── dataset_B/
│   ├── imagesTr/
│   └── labelsTr/
├── m3d_0000/
│   ├── 0000.json
│   ├── 1/
│   └── 2/
└── ...
```

使用递归处理命令（见步骤3B）。

### 步骤3：运行预处理

#### 方法A：处理单个数据集

**NIfTI格式：**
```bash
bash scripts/prepare_medical_ct_dataset.sh \
    ./your_data \
    ./output_ct \
    ./dataset_toolkits/ct_preprocessing/organ_mapping_example.json \
    4
```

**M3D-Seg格式：**
```bash
python dataset_toolkits/process_m3d_seg_format.py \
    --data_root ./dataset_0000 \
    --output_dir ./output_ct \
    --num_workers 4
```

#### 方法B：递归处理多个数据集（推荐）

一次性处理包含多个数据集的大文件夹：

```bash
bash scripts/prepare_ct_recursive.sh \
    ./med_dataset \
    ./processed_dataset \
    ./organ_mapping.json \
    8
```

系统会自动：
- 🔍 递归扫描所有子文件夹
- 📊 自动识别数据格式（NIfTI或M3D-Seg）
- ⚡ 并行处理所有数据集
- 📋 生成统一格式的输出

预处理会自动完成：
- ✅ 分辨率适配（自动选择512³或1024³）
- ✅ 保存原始CT（HU值）
- ✅ 4种窗口二值化（肺、骨、软组织、脑，直接在原始HU值上二值化）
- ✅ 器官特定窗口提取
- ✅ 生成元数据

### 步骤4：使用数据

```python
import numpy as np

# 加载原始CT（HU值）
ct = np.load('output_ct/processed/case_001/ct_original_512.npy')
print(f"CT形状: {ct.shape}")  # (512, 512, 512)
print(f"HU值范围: [{ct.min():.2f}, {ct.max():.2f}]")

# 加载肺窗（二值化后的结果）
lung_window = np.load('output_ct/processed/case_001/windows/lung_w1500_l-600.npy')
print(f"肺窗形状: {lung_window.shape}")  # (512, 512, 512)
```

## 命令行参数

### 基本用法

```bash
python dataset_toolkits/process_medical_ct.py \
    --data_root /path/to/nifti_data \
    --output_dir ./output
```

### 完整参数

```bash
python dataset_toolkits/process_medical_ct.py \
    --data_root /path/to/nifti_data \
    --output_dir ./output \
    --organ_labels ./organ_mapping.json \
    --default_resolution 512 \
    --num_workers 8 \
    --max_cases 10
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_root` | NIfTI数据目录 | 必需 |
| `--output_dir` | 输出目录 | 必需 |
| `--organ_labels` | 器官映射JSON | None |
| `--default_resolution` | 目标分辨率 | 512 |
| `--num_workers` | 并行进程数 | 4 |
| `--max_cases` | 最大处理数（测试用） | None |

## 器官映射配置

创建 `organ_mapping.json`：

```json
{
  "dataset_name": "MyDataset",
  "modality": "CT",
  "organ_labels": {
    "1": {"name": "liver", "window": "soft_tissue"},
    "2": {"name": "lung", "window": "lung"},
    "3": {"name": "bone", "window": "bone"}
  },
  "default_resolution": 512
}
```

## 输出结构

```
output_dir/
├── metadata.csv              # 元数据
├── dataset_config.json       # 配置
└── processed/
    └── case_001/
        ├── ct_original_512.npy    # 原始CT（HU值，已适配分辨率）
        ├── windows/               # 窗口二值化结果（基于原始HU值）
        │   ├── lung_w1500_l-600.npy
        │   ├── bone_w1500_l300.npy
        │   └── ...
        ├── organs/
        │   ├── liver/
        │   └── ...
        └── masks/
            └── segmentation_masks.npz
```

## 数据加载

### 方法1：直接加载

```python
import numpy as np
# 加载原始CT（HU值）
ct = np.load('output_ct/processed/case_001/ct_original_512.npy')
```

### 方法2：使用示例脚本

```bash
python examples/ct_data_loader_example.py ./output_ct case_001
```

### 方法3：使用PyTorch Dataset

```python
from examples.ct_data_loader_example import CTDataset
from torch.utils.data import DataLoader

dataset = CTDataset('./output_ct')
loader = DataLoader(dataset, batch_size=2)

for batch in loader:
    print(batch['ct'].shape)  # torch.Size([2, 1, 512, 512, 512])
```

## 数据可视化 🎮

### 一键可视化

**可视化单个病例：**
```bash
bash scripts/visualize_ct.sh ./processed_dataset/0000/processed/1
```

**批量可视化：**
```bash
bash scripts/visualize_ct_batch.sh ./processed_dataset/0000/processed
```

### 生成的可视化

系统会自动生成：
- ✅ **3D交互式视图** - 可拖动、旋转、缩放
- ✅ **多切片视图** - 矢状面、冠状面、横断面
- ✅ **窗口对比** - 肺窗、骨窗、软组织窗等
- ✅ **器官3D渲染** - 每个器官的表面可视化
- ✅ **数据统计** - 分布直方图和统计信息

### 查看可视化

可视化文件保存在 `<病例目录>/visualization/index.html`

```bash
# 在浏览器中打开
file:///path/to/output_ct/processed/0000/visualization/index.html
```

### 详细文档

更多功能和使用方法，请查看：[CT_VISUALIZATION_README.md](CT_VISUALIZATION_README.md)

## 常见问题

### Q: 如何处理没有标签的数据？
**A:** 直接运行即可，脚本会自动跳过器官处理：
```bash
python dataset_toolkits/process_medical_ct.py \
    --data_root ./data \
    --output_dir ./output
    # 不指定--organ_labels
```

### Q: 如何加快处理速度？
**A:** 增加并行进程数：
```bash
--num_workers 16  # 使用16个进程
```

### Q: 内存不足怎么办？
**A:** 减少并行进程数：
```bash
--num_workers 1  # 串行处理
```

### Q: 如何测试流程？
**A:** 限制处理数量：
```bash
--max_cases 5  # 只处理前5个病例
```

### Q: 如何修改窗口设置？
**A:** 编辑 `dataset_toolkits/ct_preprocessing/config.py` 或使用Python API：
```python
from dataset_toolkits.ct_preprocessing import add_custom_window

add_custom_window(
    window_name='my_window',
    window_width=500,
    window_level=100
)
```


