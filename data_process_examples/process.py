import os
import numpy as np
import multiprocessing
from scipy import sparse
from sklearn.model_selection import train_test_split
import json

from monai.transforms import (
    Compose,
)
from monai.transforms import LoadImaged, Orientationd, EnsureChannelFirstd

# ==================== 配置参数 ====================
# 3D-IRCADB数据集的类别标签
# 根据3D-IRCADB官方定义，该数据集包含以下器官
CATEGORY = [
    'liver',           # 1: 肝脏
    'right kidney',    # 2: 右肾
    'left kidney',     # 3: 左肾
    'spleen'          # 4: 脾脏
]

# 路径配置
IMAGE_DIR = './3Dircad/imagesTr'
LABEL_DIR = './3Dircad/labelsTr'
DATASET_CODE = '0009'  # 3D-IRCADB在M3D_Seg中的编号
SAVE_ROOT = './'
TEST_RATIO = 0.2  # 测试集比例

# ==================================================

# 获取图像和标签列表
image_list_all = [item for item in sorted(os.listdir(IMAGE_DIR))]
label_list_all = [item for item in sorted(os.listdir(LABEL_DIR))]

# 匹配图像和标签文件
# 图像文件名格式: 1_0000.nii.gz
# 标签文件名格式: 1.nii.gz
matched_pairs = []
for img_file in image_list_all:
    # 提取图像文件的ID (如 1_0000.nii.gz -> 1)
    img_id = img_file.split('_')[0]
    # 查找对应的标签文件
    label_file = img_id + '.nii.gz'
    if label_file in label_list_all:
        matched_pairs.append((img_file, label_file, img_id))

print(f'数据集大小: {len(matched_pairs)}')

# 构建数据路径列表
data_path_list_all = []
for idx, (img_file, label_file, name) in enumerate(matched_pairs):
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_path = os.path.join(LABEL_DIR, label_file)
    info = (idx, name, img_path, label_path)
    data_path_list_all.append(info)

# MONAI图像加载器
img_loader = Compose([
    LoadImaged(keys=['image', 'label']),
    EnsureChannelFirstd(keys=['image', 'label']),
    Orientationd(keys=['image', 'label'], axcodes="RAS"),
])

# 创建保存路径
save_path = os.path.join(SAVE_ROOT, DATASET_CODE)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 检查已存在的文件
exist_folders = os.listdir(save_path) if os.path.exists(save_path) else []
print(f'已存在的文件夹: {exist_folders}')


def normalize(ct_narray):
    """标准化CT图像"""
    ct_voxel_ndarray = ct_narray.flatten()
    
    # 对所有数据计算阈值
    thred = np.mean(ct_voxel_ndarray)
    voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
    
    # 对前景数据进行标准化
    upper_bound = np.percentile(voxel_filtered, 99.95)
    lower_bound = np.percentile(voxel_filtered, 0.05)
    mean = np.mean(voxel_filtered)
    std = np.std(voxel_filtered)
    
    # 释放临时变量
    del ct_voxel_ndarray, voxel_filtered
    
    # 变换（直接在原数组上操作）
    ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
    ct_narray = (ct_narray - mean) / max(std, 1e-8)
    return ct_narray


def run(info):
    """处理单个数据样本"""
    idx, file_name, case_path, label_path = info
    
    # 创建样本保存文件夹
    sample_save_path = os.path.join(save_path, file_name)
    if os.path.exists(sample_save_path):
        print(f'{file_name} 已存在，跳过')
        return
    
    os.makedirs(sample_save_path)
    print(f'处理 {idx} --- {file_name}')
    
    # 加载图像和标签
    item_load = {
        'image': case_path,
        'label': label_path,
    }
    item_load = img_loader(item_load)
    # 将MetaTensor转换为numpy数组
    ct_voxel_ndarray = np.array(item_load['image'])
    gt_voxel_ndarray = np.array(item_load['label'])
    
    ct_shape = ct_voxel_ndarray.shape
    print(f'  CT形状: {ct_shape}')
    
    # 处理标签：将多类别标签转换为多通道二值mask
    gt_voxel_ndarray = gt_voxel_ndarray.squeeze()
    gt_shape = gt_voxel_ndarray.shape  # 获取squeeze后的标签形状
    print(f'  标签形状: {gt_shape}')
    
    present_categories = np.unique(gt_voxel_ndarray)
    print(f'  存在的类别: {present_categories}')
    
    gt_masks = []
    for cls_idx in range(len(CATEGORY)):
        cls = cls_idx + 1
        if cls not in present_categories:
            gt_voxel_ndarray_category = np.zeros(gt_shape)  # 使用gt_shape而不是ct_shape
            gt_masks.append(gt_voxel_ndarray_category)
            print(f'  样本 {idx} ==> 零类别 {CATEGORY[cls_idx]}')
        else:
            gt_voxel_ndarray_category = gt_voxel_ndarray.copy()
            gt_voxel_ndarray_category[gt_voxel_ndarray != cls] = 0
            gt_voxel_ndarray_category[gt_voxel_ndarray == cls] = 1
            gt_masks.append(gt_voxel_ndarray_category)
            print(f'  样本 {idx} ==> 包含 {CATEGORY[cls_idx]}')
    
    gt_voxel_ndarray = np.stack(gt_masks, axis=0)
    
    assert gt_voxel_ndarray.shape[0] == len(CATEGORY), f'标签通道数不匹配: {gt_voxel_ndarray.shape[0]}'
    # 比较时要考虑CT图像可能有通道维度
    ct_spatial_shape = ct_voxel_ndarray.shape[1:] if ct_voxel_ndarray.shape[0] == 1 else ct_voxel_ndarray.shape
    assert gt_voxel_ndarray.shape[1:] == ct_spatial_shape, f'图像和标签形状不匹配: {gt_voxel_ndarray.shape[1:]} vs {ct_spatial_shape}'
    
    # 标准化图像
    ct_voxel_ndarray = normalize(ct_voxel_ndarray)
    print(f'  {idx} 标准化完成')
    
    # 保存图像
    image_save_file = os.path.join(sample_save_path, 'image.npy')
    np.save(image_save_file, ct_voxel_ndarray)
    
    # 保存标签（稀疏矩阵格式）
    mask_shape = gt_voxel_ndarray.shape
    allmatrix_sp = sparse.csr_matrix(gt_voxel_ndarray.reshape(mask_shape[0], -1))
    mask_save_file = os.path.join(sample_save_path, f'mask_{mask_shape}.npz')
    sparse.save_npz(mask_save_file, allmatrix_sp)
    
    print(f'  {file_name} CT和GT形状: {ct_voxel_ndarray.shape}, {gt_voxel_ndarray.shape}')
    print(f'  {file_name} 保存完成！\n')


def generate_dataset_json(root_dir, output_file, test_ratio=0.2):
    """生成数据集划分JSON文件"""
    sample_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    
    # 收集所有图像和标签路径
    data_pairs = []
    for folder in sample_folders:
        folder_path = os.path.join(root_dir, folder)
        image_file = os.path.join(DATASET_CODE, folder, 'image.npy')
        
        # 查找mask文件
        mask_files = [f for f in os.listdir(folder_path) if f.startswith('mask_') and f.endswith('.npz')]
        if mask_files:
            mask_file = os.path.join(DATASET_CODE, folder, mask_files[0])
            data_pairs.append((image_file, mask_file))
    
    print(f'总共找到 {len(data_pairs)} 个样本')
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(data_pairs, test_size=test_ratio, random_state=42)
    
    # 构建标签字典
    labels = {'0': 'background'}
    for idx, label_name in enumerate(CATEGORY):
        labels[str(idx + 1)] = label_name
    
    # 构建数据集字典
    dataset = {
        'name': f'{DATASET_CODE} Dataset',
        'description': '3D-IRCADB Dataset',
        'tensorImageSize': '4D',
        'modality': {
            '0': 'CT',
        },
        'labels': labels,
        'numTrain': len(train_data),
        'numTest': len(test_data),
        'train': [{'image': ct_path, 'label': gt_path} for ct_path, gt_path in train_data],
        'test': [{'image': ct_path, 'label': gt_path} for ct_path, gt_path in test_data]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    print(f'JSON文件已保存到: {output_file}')
    print(f'训练集样本数: {len(train_data)}')
    print(f'测试集样本数: {len(test_data)}')


if __name__ == "__main__":
    print('=' * 60)
    print('开始处理3D-IRCADB数据集')
    print('=' * 60)
    
    # 多进程处理数据
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(run, data_path_list_all)
    print('\n数据处理完成！')
    
    # 生成JSON文件
    print('=' * 60)
    print('生成数据集划分JSON文件')
    print('=' * 60)
    generate_dataset_json(
        root_dir=save_path,
        output_file=os.path.join(save_path, f'{DATASET_CODE}.json'),
        test_ratio=TEST_RATIO
    )
    print('\n全部完成！')

