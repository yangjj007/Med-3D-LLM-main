# CT数据预处理模块 - 使用文档

## 概述

本模块提供3D医学CT数据的完整预处理流程，包括分辨率适配、窗宽/窗位二值化、器官特定窗口处理和语义分割标签处理。预处理后的数据可用于TRELLIS Sparse SDF模型训练或其他3D医学图像分析任务。

### 核心功能

1. **分辨率适配**：将不规则的3D CT数组适配到标准分辨率（512³或1024³）
   - 只支持向上兼容，不支持向下压缩
   - 不足的维度用空气HU值（-1000）填充

2. **窗宽/窗位二值化**：根据预定义的窗口设置对CT进行二值化
   - 肺窗：窗宽1500 HU，窗位-600 HU
   - 骨窗：窗宽1500 HU，窗位300 HU
   - 软组织窗：窗宽400 HU，窗位50 HU
   - 脑窗：窗宽80 HU，窗位35 HU

3. **器官特定处理**：结合分割掩码，提取每个器官在对应窗口下的数据

4. **数据管理**：统一的文件命名规范和目录结构

## 安装依赖

```bash
# 基础依赖
pip install numpy scipy pandas tqdm

# NIfTI文件处理（必需）
pip install monai nibabel

# 可选：用于Sparse SDF生成（需要CUDA）
pip install torch trimesh

# 编译voxelize库（必需）
pip install ./third_party/voxelize/
```

## 快速开始

### 1. 准备数据

系统支持多种数据格式和目录结构：

#### 格式1：NIfTI格式（医学影像标准格式）

```
your_data_root/
├── imagesTr/
│   ├── case_001_0000.nii.gz
│   ├── case_002_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── case_001.nii.gz
    ├── case_002.nii.gz
    └── ...
```

#### 格式2：M3D-Seg格式（NPY数组格式）

```
dataset_0000/
├── 0000.json          # 数据集配置和标签信息
├── 1/
│   ├── image.npy      # CT图像数组
│   └── mask_(1, 512, 512, 96).npz  # 分割掩码（稀疏格式）
├── 2/
│   ├── image.npy
│   └── mask_*.npz
└── ...
```

#### 格式3：包含多个数据集的大文件夹

```
all_datasets/
├── dataset_A/          # NIfTI格式
│   ├── imagesTr/
│   └── labelsTr/
├── dataset_B/          # NIfTI格式
│   ├── imagesTr/
│   └── labelsTr/
├── m3d_0000/          # M3D-Seg格式
│   ├── 0000.json
│   ├── 1/
│   └── 2/
├── m3d_0001/          # M3D-Seg格式
│   ├── 0001.json
│   └── ...
└── ...
```

系统会自动递归扫描并识别所有数据集！

### 2. 创建器官映射配置

创建一个JSON文件（例如`organ_labels.json`）：

```json
{
  "dataset_name": "MyDataset",
  "modality": "CT",
  "organ_labels": {
    "1": {"name": "liver", "window": "soft_tissue"},
    "2": {"name": "right_kidney", "window": "soft_tissue"},
    "3": {"name": "left_kidney", "window": "soft_tissue"},
    "4": {"name": "spleen", "window": "soft_tissue"}
  },
  "default_resolution": 512
}
```

参考示例：`dataset_toolkits/ct_preprocessing/organ_mapping_example.json`

### 3. 运行预处理

#### 方法1：递归处理多个数据集（推荐）

适用于包含多个数据集的大文件夹，自动识别格式：

```bash
bash scripts/prepare_ct_recursive.sh \
    ./med_dataset \
    ./processed_dataset \
    ./organ_labels.json \
    8 \
    5
```

参数说明：
- 第1个参数：根目录（包含多个数据集）
- 第2个参数：输出基础目录
- 第3个参数：器官标签映射JSON（可选，用于NIfTI格式）
- 第4个参数：并行进程数（可选，默认4）
- 第5个参数：最大递归深度（可选，默认5）

特点：
- 🔍 自动递归扫描所有子文件夹
- 📊 自动识别NIfTI和M3D-Seg格式
- ⚡ 并行处理多个数据集
- 📋 统一输出格式
- 📝 生成总结报告

#### 方法2：处理单个NIfTI数据集

```bash
bash scripts/prepare_medical_ct_dataset.sh \
    /path/to/nifti_data \
    ./data/processed_ct \
    ./organ_labels.json \
    8
```

#### 方法3：处理单个M3D-Seg数据集

```bash
python dataset_toolkits/process_m3d_seg_format.py \
    --data_root /path/to/m3d_dataset \
    --output_dir ./data/processed_ct \
    --num_workers 8
```

### 4. 预计算SDF


### 5. 输出结果

处理完成后，输出目录结构如下：

```
processed_ct/
├── metadata.csv                    # 元数据（包含所有样本信息）
├── dataset_config.json             # 数据集全局配置
└── processed/
    ├── case_001/
    │   ├── ct_original_512.npy    # 原始CT（适配到512³）
    │   ├── ct_normalized_512.npy  # 标准化后的CT
    │   ├── windows/               # 全局窗口二值化结果
    │   │   ├── lung_w1500_l-600.npy
    │   │   ├── bone_w1500_l300.npy
    │   │   ├── soft_tissue_w400_l50.npy
    │   │   └── brain_w80_l35.npy
    │   ├── organs/                # 器官特定窗口结果
    │   │   ├── liver/
    │   │   │   └── soft_tissue_w400_l50.npy
    │   │   ├── lung/
    │   │   │   └── lung_w1500_l-600.npy
    │   │   └── ...
    │   ├── masks/                 # 原始分割掩码
    │   │   └── segmentation_masks.npz
    │   └── info.json              # 样本元信息
    └── case_002/
        └── ...
```

## 数据加载示例

### 加载预处理后的数据

```python
import numpy as np
import pandas as pd
from scipy import sparse
import json
import os

# 1. 加载元数据
metadata_path = './data/processed_ct/metadata.csv'
metadata = pd.read_csv(metadata_path)
print(f"总病例数: {len(metadata)}")

# 2. 选择一个病例
case_id = metadata.iloc[0]['case_id']
print(f"加载病例: {case_id}")

# 3. 加载标准化的CT数据
ct_path = f'./data/processed_ct/processed/{case_id}/ct_normalized_512.npy'
ct_array = np.load(ct_path)
print(f"CT形状: {ct_array.shape}")

# 4. 加载窗口数据
lung_window = np.load(f'./data/processed_ct/processed/{case_id}/windows/lung_w1500_l-600.npy')
bone_window = np.load(f'./data/processed_ct/processed/{case_id}/windows/bone_w1500_l300.npy')
print(f"肺窗形状: {lung_window.shape}")
print(f"骨窗形状: {bone_window.shape}")

# 5. 加载分割掩码
mask_path = f'./data/processed_ct/processed/{case_id}/masks/segmentation_masks.npz'
seg_sparse = sparse.load_npz(mask_path)
seg_shape = (512, 512, 512)  # 从info.json获取
seg_array = seg_sparse.toarray().reshape(seg_shape)
print(f"分割形状: {seg_array.shape}")
print(f"唯一标签: {np.unique(seg_array)}")

# 6. 加载器官特定数据
liver_path = f'./data/processed_ct/processed/{case_id}/organs/liver/soft_tissue_w400_l50.npy'
if os.path.exists(liver_path):
    liver_window = np.load(liver_path)
    print(f"肝脏窗口体素数: {np.sum(liver_window)}")

# 7. 加载元信息
info_path = f'./data/processed_ct/processed/{case_id}/info.json'
with open(info_path, 'r') as f:
    info = json.load(f)
print(f"原始形状: {info['original_shape']}")
print(f"处理时间: {info['processing_time_sec']}秒")
```

### 批量数据加载器

```python
class CTDataLoader:
    """CT数据加载器"""
    
    def __init__(self, processed_dir, resolution=512):
        self.processed_dir = processed_dir
        self.resolution = resolution
        
        # 加载元数据
        metadata_path = os.path.join(processed_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path)
        
    def __len__(self):
        return len(self.metadata)
    
    def load_case(self, idx):
        """加载单个病例"""
        row = self.metadata.iloc[idx]
        case_id = row['case_id']
        
        case_dir = os.path.join(self.processed_dir, 'processed', case_id)
        
        # 加载CT
        ct_path = os.path.join(case_dir, f'ct_normalized_{self.resolution}.npy')
        ct = np.load(ct_path)
        
        # 加载分割（如果有）
        seg = None
        if row.get('has_segmentation', False):
            mask_path = os.path.join(case_dir, 'masks', 'segmentation_masks.npz')
            if os.path.exists(mask_path):
                seg_sparse = sparse.load_npz(mask_path)
                seg_shape = tuple(map(int, row['adapted_shape'].split(',')))
                seg = seg_sparse.toarray().reshape(seg_shape)
        
        return {
            'case_id': case_id,
            'ct': ct,
            'segmentation': seg,
            'info': row.to_dict()
        }
    
    def load_window(self, case_id, window_name):
        """加载特定窗口"""
        window_path = os.path.join(
            self.processed_dir, 'processed', case_id, 
            'windows', f'{window_name}.npy'
        )
        return np.load(window_path)

# 使用示例
loader = CTDataLoader('./data/processed_ct')
print(f"数据集大小: {len(loader)}")

# 加载第一个病例
data = loader.load_case(0)
print(f"病例ID: {data['case_id']}")
print(f"CT形状: {data['ct'].shape}")
```

### 用于训练的数据迭代器

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CTDataset(Dataset):
    """PyTorch Dataset for CT data"""
    
    def __init__(self, processed_dir, resolution=512, load_windows=True):
        self.processed_dir = processed_dir
        self.resolution = resolution
        self.load_windows = load_windows
        
        # 加载元数据
        metadata_path = os.path.join(processed_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        case_id = row['case_id']
        case_dir = os.path.join(self.processed_dir, 'processed', case_id)
        
        # 加载CT
        ct_path = os.path.join(case_dir, f'ct_normalized_{self.resolution}.npy')
        ct = np.load(ct_path)
        ct_tensor = torch.from_numpy(ct).float()
        
        # 添加通道维度 (1, H, W, D)
        if ct_tensor.dim() == 3:
            ct_tensor = ct_tensor.unsqueeze(0)
        
        data = {'ct': ct_tensor, 'case_id': case_id}
        
        # 可选：加载窗口
        if self.load_windows:
            windows_dir = os.path.join(case_dir, 'windows')
            if os.path.exists(windows_dir):
                lung_window = np.load(os.path.join(windows_dir, 'lung_w1500_l-600.npy'))
                data['lung_window'] = torch.from_numpy(lung_window).float()
        
        return data

# 使用示例
dataset = CTDataset('./data/processed_ct')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

for batch in dataloader:
    print(f"Batch CT shape: {batch['ct'].shape}")
    print(f"Case IDs: {batch['case_id']}")
    break
```

## 高级用法

### 自定义窗口设置

```python
from dataset_toolkits.ct_preprocessing import add_custom_window

# 添加自定义窗口
add_custom_window(
    window_name='custom_liver',
    window_width=350,
    window_level=60,
    organ_types=['liver'],
    description='肝脏专用窗口'
)
```

### 单个文件处理（不使用主脚本）

```python
from dataset_toolkits.ct_preprocessing import (
    adapt_resolution,
    process_all_windows,
    normalize_ct
)
import numpy as np

# 加载你的CT数据（假设已加载为numpy数组）
ct_array = np.load('your_ct.npy')
print(f"原始形状: {ct_array.shape}")

# 1. 分辨率适配
ct_adapted = adapt_resolution(ct_array, target_resolution=512)
print(f"适配后形状: {ct_adapted.shape}")

# 2. 标准化
ct_normalized = normalize_ct(ct_adapted, method='foreground')

# 3. 窗口处理
windows = process_all_windows(ct_normalized, binarize=True)
for window_name, binary_array in windows.items():
    print(f"{window_name}: {np.sum(binary_array)} 正值体素")

# 4. 保存结果
np.save('ct_normalized_512.npy', ct_normalized)
np.save('lung_window.npy', windows['lung'])
```

## 参数说明

### process_medical_ct.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | str | 必需 | NIfTI数据根目录 |
| `--output_dir` | str | 必需 | 输出目录 |
| `--organ_labels` | str | None | 器官标签映射JSON文件 |
| `--default_resolution` | int | 512 | 默认目标分辨率（512或1024） |
| `--num_workers` | int | 4 | 并行进程数 |
| `--max_cases` | int | None | 最大处理病例数（用于测试） |

### 分辨率适配规则

| 输入最大维度 | 目标分辨率 | 示例 |
|------------|----------|------|
| ≤ 512 | 512³ | (512,512,100) → (512,512,512) |
| 512 < d ≤ 1024 | 1024³ | (600,600,200) → (1024,1024,1024) |
| > 1024 | 错误 | 不支持 |

### 窗口配置

| 窗口名称 | 窗宽(HU) | 窗位(HU) | HU范围 | 适用器官 |
|---------|---------|---------|--------|---------|
| lung | 1500 | -600 | [-1350, 150] | 肺、支气管 |
| bone | 1500 | 300 | [-450, 1050] | 骨骼、椎骨 |
| soft_tissue | 400 | 50 | [-150, 250] | 肝、肾、脾 |
| brain | 80 | 35 | [-5, 75] | 脑组织 |

## 文件命名规范

### CT文件
- 原始适配CT：`ct_original_{resolution}.npy`
- 标准化CT：`ct_normalized_{resolution}.npy`

### 窗口文件
- 格式：`{window_name}_w{width}_l{level}.npy`
- 示例：`lung_w1500_l-600.npy`、`bone_w1500_l300.npy`

### 器官文件
- 路径：`organs/{organ_name}/{window_name}_w{width}_l{level}.npy`
- 示例：`organs/liver/soft_tissue_w400_l50.npy`

### 掩码文件
- 稀疏格式：`masks/segmentation_masks.npz`

## 测试

运行单元测试验证所有模块：

```bash
python tests/test_ct_preprocessing.py
```

测试内容包括：
- 配置模块
- 分辨率适配器
- 窗口处理器
- 器官提取器
- 集成测试
- 文件操作

## 性能优化

### 并行处理

使用多进程加速处理：

```bash
# 使用8个进程
python dataset_toolkits/process_medical_ct.py \
    --data_root /path/to/data \
    --output_dir ./output \
    --num_workers 8
```

### 内存优化

- 使用uint8存储二值化数组（节省75%空间）
- 使用稀疏矩阵存储分割掩码
- 使用np.savez_compressed压缩存储

### 存储空间估算

以512³分辨率为例，单个病例约占：
- 原始CT（float32）：512 MB
- 标准化CT（float32）：512 MB
- 全局窗口（4个，uint8）：512 MB
- 器官窗口：取决于器官数量
- 总计：约1.5-2 GB/病例

## 常见问题

### Q1: 如何处理没有分割标签的数据？

A: 脚本会自动检测。如果没有标签文件，将只处理CT数据和全局窗口，跳过器官特定处理。

```bash
# 无标签数据也可以正常处理
python dataset_toolkits/process_medical_ct.py \
    --data_root /path/to/data \
    --output_dir ./output
    # 不需要指定--organ_labels
```

### Q2: 如何修改窗宽/窗位设置？

A: 有两种方法：

1. 修改配置文件 `dataset_toolkits/ct_preprocessing/config.py`
2. 在代码中动态添加：

```python
from dataset_toolkits.ct_preprocessing import add_custom_window

add_custom_window(
    window_name='my_window',
    window_width=500,
    window_level=100
)
```

### Q3: 内存不足怎么办？

A: 
1. 减少并行进程数：`--num_workers 1`
2. 分批处理：`--max_cases 10`
3. 使用更小的分辨率（修改DEFAULT_RESOLUTION）

### Q4: 处理速度慢怎么办？

A:
1. 增加并行进程数：`--num_workers 16`
2. 使用SSD存储
3. 关闭不需要的窗口处理
4. 不保存中间结果（修改save_intermediate参数）

### Q5: 如何为不同器官使用不同窗口？

A: 在器官标签映射JSON中指定：

```json
{
  "organ_labels": {
    "1": {"name": "lung", "window": "lung"},
    "2": {"name": "liver", "window": "soft_tissue"},
    "3": {"name": "bone", "window": "bone"}
  }
}
```

### Q6: 如何验证处理结果是否正确？

A: 使用可视化工具检查：

```python
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
ct = np.load('processed_ct/processed/case_001/ct_normalized_512.npy')
lung_window = np.load('processed_ct/processed/case_001/windows/lung_w1500_l-600.npy')

# 可视化中间切片
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(ct[:, :, 256], cmap='gray')
axes[0].set_title('原始CT')
axes[1].imshow(lung_window[:, :, 256], cmap='gray')
axes[1].set_title('肺窗')
plt.show()
```

## 与TRELLIS集成

预处理后的数据可以直接用于TRELLIS训练：

```bash
# 1. 预处理CT数据
bash scripts/prepare_medical_ct_dataset.sh \
    /path/to/nifti_data \
    ./data/processed_ct

# 2. 生成Sparse SDF（需要CUDA）
python dataset_toolkits/compute_sparse_sdf.py \
    --output_dir ./data/processed_ct \
    --resolutions 512 \
    --input_type voxel \
    --max_workers 8

# 3. 训练TRELLIS
python train.py \
    --config configs/vae/sparse_sdf_vqvae_512.json \
    --output_dir ./outputs/ct_vqvae \
    --data_dir ./data/processed_ct
```

## 数据读取API参考

完整的数据加载API在 `dataset_toolkits/datasets/MedicalCT.py` 中：

```python
from dataset_toolkits.datasets.MedicalCT import get_data_loader

# 获取数据加载器
metadata = pd.read_csv('processed_ct/metadata.csv')
loader = get_data_loader(metadata, 'processed_ct', resolution=512)

# 加载数据
sha256 = metadata.iloc[0]['sha256']
data = loader(sha256)

print(data.keys())  # ['ct', 'case_id', 'segmentation', 'windows']
```

## 贡献与支持

如有问题或建议，请查看项目文档或提交Issue。

## 许可证

本模块遵循项目整体许可证。

