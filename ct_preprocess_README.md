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
    5 \
    --compute_sdf \
    --replace_npy
```

参数说明：
- 第1个参数：根目录（包含多个数据集）
- 第2个参数：输出基础目录
- 第3个参数：器官标签映射JSON（可选，用于NIfTI格式）
- 第4个参数：并行进程数（可选，默认4）
- 第5个参数：最大递归深度（可选，默认5）
- 第6个参数：是否预计算SDF
- 第7个参数：是否用SDF替代体素网格

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

#### 方法A：在预处理时同时计算SDF（推荐）

使用 `--compute_sdf` 参数，在预处理时直接生成 SDF 文件：

```bash
bash scripts/prepare_ct_recursive.sh \
    ./med_dataset \
    ./processed_dataset \
    ./organ_labels.json \
    8 \
    5 \
    --compute_sdf \
    --replace_npy
```

**✨ 新特性（已修复）：**
- ✅ 同时生成全局窗口（`windows/`）和器官窗口（`organs/`）的 SDF 文件
- ✅ 支持 `--replace_npy` 参数，用 `.npz` 文件替换 `.npy` 文件以节省空间

#### 方法B：后处理转换为SDF表示

如果已经完成预处理，可以使用独立脚本转换：

```bash
python scripts/precompute_ct_window_sdf.py \
    --data_root ./processed_dataset/0000 \
    --resolution 512 \
    --max_workers 4
```

**参数说明：**
- `--data_root`: 数据根目录（包含processed子目录）
- `--window_type`: 窗口类型（lung, bone, soft_tissue, brain, all），**默认all（处理所有类型）**
- `--resolution`: 目标分辨率（默认512）
- `--threshold_factor`: UDF阈值因子（默认4.0）
- `--max_workers`: 并行处理的worker数量（默认4）
- `--force_recompute`: 强制重新计算已存在的SDF文件
- `--replace_npy`: 用npz文件替换原npy文件

**输出：**
- 将 `windows/*.npy` 文件转换为 `windows/*.npz` 文件
- 将 `organs/*/​*.npy` 文件转换为 `organs/*/​*.npz` 文件
- 生成处理日志CSV文件

#### 测试SDF加载（可选）

```bash
python scripts/test_sdf_loading.py \
    --data_root ./processed_dataset/0000 \
    --window_type lung \
    --num_samples 5
```

### 5. 输出结果

处理完成后，输出目录结构如下：

```
processed_ct/
├── metadata.csv                    # 元数据（包含所有样本信息）
├── dataset_config.json             # 数据集全局配置
└── processed/
    ├── case_001/
    │   ├── ct_original_512.npy    # 原始CT（适配到512³）
    │   ├── windows/               # 全局窗口二值化和sdf化结果
    │   │   ├── lung_w1500_l-600.npy
    │   │   ├── bone_w1500_l300.npy
    │   │   ├── soft_tissue_w400_l50.npy
    │   │   └── brain_w80_l35.npy
    │   │   ├── lung_w1500_l-600.npz
    │   │   ├── bone_w1500_l300.npz
    │   │   ├── soft_tissue_w400_l50.npz
    │   │   └── brain_w80_l35.npz
    │   ├── organs/                # 器官特定窗口结果（使用--window_type all自动处理）
    │   │   ├── liver/
    │   │   │   ├── soft_tissue_w400_l50.npy
    │   │   │   └── soft_tissue_w400_l50.npz
    │   │   ├── lung/
    │   │   │   ├── lung_w1500_l-600.npy
    │   │   │   └── lung_w1500_l-600.npz
    │   │   └── ...
    │   ├── masks/                 # 原始分割掩码
    │   │   └── segmentation_masks.npz
    │   └── info.json              # 样本元信息
    └── case_002/
        └── ...
```


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

