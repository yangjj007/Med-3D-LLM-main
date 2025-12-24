"""
Medical CT数据集处理器

扩展MedicalData.py，添加CT特定的处理功能。
支持从NIfTI文件构建元数据，并集成到dataset_toolkits工作流。
"""

import os
import glob
import hashlib
import json
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def get_file_hash(file_path: str) -> str:
    """
    计算文件的SHA256哈希值
    
    Args:
        file_path: 文件路径
    
    Returns:
        SHA256哈希字符串
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()


def scan_nifti_files(data_root: str) -> List[Dict[str, str]]:
    """
    扫描NIfTI文件并配对图像和标签
    
    假设目录结构：
    data_root/
        imagesTr/
            case_001_0000.nii.gz
            case_002_0000.nii.gz
        labelsTr/
            case_001.nii.gz
            case_002.nii.gz
    
    Args:
        data_root: 数据根目录
    
    Returns:
        文件信息列表
    """
    image_dir = os.path.join(data_root, 'imagesTr')
    label_dir = os.path.join(data_root, 'labelsTr')
    
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    
    files = []
    
    # 查找所有图像文件
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
    
    for image_file in image_files:
        # 提取case ID
        basename = os.path.basename(image_file)
        case_id = basename.split('_')[0]
        
        # 生成SHA256（基于相对路径）
        rel_path = os.path.relpath(image_file, data_root)
        sha256 = hashlib.sha256(rel_path.encode()).hexdigest()
        
        # 查找对应的标签文件
        label_file = os.path.join(label_dir, f'{case_id}.nii.gz')
        has_label = os.path.exists(label_file)
        
        files.append({
            'sha256': sha256,
            'case_id': case_id,
            'image_path': image_file,
            'label_path': label_file if has_label else None,
            'has_label': has_label,
            'relative_image_path': rel_path,
            'relative_label_path': os.path.relpath(label_file, data_root) if has_label else None,
        })
    
    return files


def build_metadata_from_nifti(data_root: str) -> pd.DataFrame:
    """
    从NIfTI文件构建元数据
    
    Args:
        data_root: 数据根目录
    
    Returns:
        元数据DataFrame
    """
    print(f"扫描NIfTI文件: {data_root}")
    files = scan_nifti_files(data_root)
    print(f"  发现 {len(files)} 个CT病例")
    
    # 创建元数据DataFrame
    metadata = pd.DataFrame(files)
    
    # 添加默认列（与其他数据集保持一致）
    metadata['rendered'] = False
    metadata['voxelized'] = False
    metadata['sparse_sdf_computed'] = False
    metadata['aesthetic_score'] = 5.0  # CT数据默认质量分数
    metadata['file_type'] = 'nifti'
    
    return metadata


def build_metadata_from_processed(processed_dir: str) -> pd.DataFrame:
    """
    从已处理的CT数据构建元数据
    
    Args:
        processed_dir: 处理后的数据目录（包含processed子目录和metadata.csv）
    
    Returns:
        元数据DataFrame
    """
    metadata_path = os.path.join(processed_dir, 'metadata.csv')
    
    if not os.path.exists(metadata_path):
        raise ValueError(f"Metadata file not found: {metadata_path}")
    
    print(f"加载已处理数据的元数据: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    print(f"  {len(metadata)} 个已处理病例")
    
    # 添加必要的列
    if 'sha256' not in metadata.columns:
        # 基于case_id生成SHA256
        metadata['sha256'] = metadata['case_id'].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()
        )
    
    if 'rendered' not in metadata.columns:
        metadata['rendered'] = False
    
    if 'voxelized' not in metadata.columns:
        metadata['voxelized'] = True  # 已处理的数据认为是已体素化的
    
    if 'sparse_sdf_computed' not in metadata.columns:
        metadata['sparse_sdf_computed'] = False
    
    return metadata


def add_ct_specific_args(parser):
    """
    添加CT特定的命令行参数
    
    Args:
        parser: argparse.ArgumentParser对象
    """
    parser.add_argument(
        '--data_root',
        type=str,
        help='NIfTI数据根目录（原始数据）或处理后的数据目录'
    )
    parser.add_argument(
        '--organ_labels',
        type=str,
        default=None,
        help='器官标签映射JSON文件路径'
    )
    parser.add_argument(
        '--default_resolution',
        type=int,
        default=512,
        choices=[512, 1024],
        help='默认目标分辨率'
    )
    parser.add_argument(
        '--data_type',
        type=str,
        default='nifti',
        choices=['nifti', 'processed'],
        help='数据类型: nifti（原始NIfTI文件）或processed（已处理的NPY文件）'
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default=None,
        help='已处理数据的目录（如果data_type=processed）'
    )


def build_metadata(opt) -> pd.DataFrame:
    """
    构建CT数据集元数据
    
    Args:
        opt: 命令行参数
    
    Returns:
        元数据DataFrame
    """
    data_type = getattr(opt, 'data_type', 'nifti')
    
    if data_type == 'nifti':
        # 从原始NIfTI文件构建
        if not hasattr(opt, 'data_root') or opt.data_root is None:
            raise ValueError("--data_root is required for nifti data type")
        
        metadata = build_metadata_from_nifti(opt.data_root)
        
    elif data_type == 'processed':
        # 从已处理的数据构建
        processed_dir = getattr(opt, 'processed_dir', opt.output_dir)
        metadata = build_metadata_from_processed(processed_dir)
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return metadata


def get_data_loader(metadata: pd.DataFrame, 
                   processed_dir: str,
                   resolution: int = 512) -> Callable:
    """
    获取数据加载函数
    
    Args:
        metadata: 元数据DataFrame
        processed_dir: 处理后的数据目录
        resolution: 目标分辨率
    
    Returns:
        数据加载函数
    """
    def load_data(sha256: str) -> Dict[str, Any]:
        """
        加载单个样本的数据
        
        Args:
            sha256: 样本的SHA256标识
        
        Returns:
            数据字典
        """
        import numpy as np
        from scipy import sparse
        
        # 查找样本
        row = metadata[metadata['sha256'] == sha256]
        if len(row) == 0:
            raise ValueError(f"Sample not found: {sha256}")
        
        row = row.iloc[0]
        case_id = row['case_id']
        
        # 加载CT数据
        ct_path = os.path.join(processed_dir, 'processed', case_id, f'ct_normalized_{resolution}.npy')
        if not os.path.exists(ct_path):
            ct_path = os.path.join(processed_dir, 'processed', case_id, f'ct_original_{resolution}.npy')
        
        ct_array = np.load(ct_path)
        
        data = {
            'ct': ct_array,
            'case_id': case_id,
            'sha256': sha256,
        }
        
        # 加载分割掩码（如果有）
        if row.get('has_segmentation', False):
            mask_path = os.path.join(processed_dir, 'processed', case_id, 'masks', 'segmentation_masks.npz')
            if os.path.exists(mask_path):
                seg_sparse = sparse.load_npz(mask_path)
                seg_shape = tuple(map(int, row['adapted_shape'].split(',')))
                seg_array = seg_sparse.toarray().reshape(seg_shape)
                data['segmentation'] = seg_array
        
        # 加载窗口数据（可选）
        windows_dir = os.path.join(processed_dir, 'processed', case_id, 'windows')
        if os.path.exists(windows_dir):
            window_files = glob.glob(os.path.join(windows_dir, '*.npy'))
            data['windows'] = {}
            for window_file in window_files:
                window_name = os.path.basename(window_file).replace('.npy', '')
                data['windows'][window_name] = np.load(window_file)
        
        return data
    
    return load_data


if __name__ == '__main__':
    """
    示例用法:
    
    # 从原始NIfTI文件构建元数据
    python dataset_toolkits/datasets/MedicalCT.py \
        --data_root /path/to/nifti_data \
        --output_dir ./data/ct_metadata \
        --data_type nifti
    
    # 从已处理的数据构建元数据
    python dataset_toolkits/datasets/MedicalCT.py \
        --processed_dir ./data/processed_ct \
        --output_dir ./data/ct_metadata \
        --data_type processed
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Build CT dataset metadata')
    add_ct_specific_args(parser)
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for metadata'
    )
    
    opt = parser.parse_args()
    
    # 构建元数据
    metadata = build_metadata(opt)
    
    # 保存元数据
    os.makedirs(opt.output_dir, exist_ok=True)
    output_path = os.path.join(opt.output_dir, 'metadata.csv')
    metadata.to_csv(output_path, index=False)
    
    print(f"\n元数据已保存: {output_path}")
    print(f"总样本数: {len(metadata)}")
    
    if 'has_label' in metadata.columns:
        has_label_count = metadata['has_label'].sum()
        print(f"有标签样本数: {has_label_count}")
    
    print("\n元数据列:")
    for col in metadata.columns:
        print(f"  - {col}")

