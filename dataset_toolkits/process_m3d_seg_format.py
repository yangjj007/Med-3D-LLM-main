"""
M3D-Seg格式数据预处理脚本

处理已经是NPY格式的M3D-Seg数据，应用相同的预处理流程：
- 分辨率适配
- CT标准化
- 窗口二值化
- 器官特定窗口处理

输入格式:
dataset_folder/
├── 0000.json
├── 1/
│   ├── image.npy
│   └── mask_(1, 512, 512, 96).npz
├── 2/
│   ├── image.npy
│   └── mask_(...).npz
└── ...
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob
from scipy import sparse
import ast

# 导入预处理模块
from ct_preprocessing import (
    DEFAULT_RESOLUTION,
    determine_target_resolution,
    adapt_resolution,
    process_all_windows,
    process_all_organs,
    compute_organ_statistics,
    get_window_filename,
    validate_segmentation,
    get_organs_present
)


def normalize_ct(ct_array: np.ndarray) -> np.ndarray:
    """标准化CT图像（与process_medical_ct.py保持一致）"""
    ct_voxel_ndarray = ct_array.flatten()
    
    thred = np.mean(ct_voxel_ndarray)
    voxel_filtered = ct_voxel_ndarray[ct_voxel_ndarray > thred]
    
    if len(voxel_filtered) == 0:
        voxel_filtered = ct_voxel_ndarray
    
    upper_bound = np.percentile(voxel_filtered, 99.95)
    lower_bound = np.percentile(voxel_filtered, 0.05)
    mean = np.mean(voxel_filtered)
    std = np.std(voxel_filtered)
    
    del ct_voxel_ndarray, voxel_filtered
    
    ct_normalized = np.clip(ct_array, lower_bound, upper_bound)
    ct_normalized = (ct_normalized - mean) / max(std, 1e-8)
    
    return ct_normalized


def load_m3d_seg_case(case_dir: str) -> tuple:
    """
    加载M3D-Seg格式的单个病例
    
    Args:
        case_dir: 病例目录（如 dataset/1/）
    
    Returns:
        (ct_array, seg_array, mask_shape)
    """
    # 加载image.npy
    image_path = os.path.join(case_dir, 'image.npy')
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    ct_array = np.load(image_path)
    
    # 加载mask文件
    mask_files = glob.glob(os.path.join(case_dir, 'mask_*.npz'))
    if not mask_files:
        return ct_array, None, None
    
    mask_file = mask_files[0]
    
    # 从文件名解析形状
    mask_filename = os.path.basename(mask_file)
    shape_str = mask_filename.split('.')[-2].split('_')[-1]
    mask_shape = ast.literal_eval(shape_str)
    
    # 加载稀疏矩阵
    seg_sparse = sparse.load_npz(mask_file)
    seg_array = seg_sparse.toarray().reshape(mask_shape)
    
    return ct_array, seg_array, mask_shape


def load_dataset_json(json_path: str) -> Dict:
    """加载数据集JSON配置"""
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    return dataset_info


def build_organ_mapping_from_json(dataset_json: Dict) -> Dict:
    """
    从M3D-Seg的JSON构建器官映射
    
    Args:
        dataset_json: M3D-Seg数据集JSON
    
    Returns:
        器官映射配置
    """
    labels = dataset_json.get('labels', {})
    
    organ_mapping = {
        'dataset_name': dataset_json.get('name', 'Unknown'),
        'modality': 'CT',
        'organ_labels': {}
    }
    
    # 转换标签格式
    for label_id, label_name in labels.items():
        if label_id == '0':  # 跳过背景
            continue
        
        # 简单的器官到窗口的映射
        window = 'soft_tissue'  # 默认
        label_lower = label_name.lower()
        
        if 'lung' in label_lower or 'bronchus' in label_lower:
            window = 'lung'
        elif 'bone' in label_lower or 'vertebra' in label_lower or 'rib' in label_lower:
            window = 'bone'
        elif 'brain' in label_lower:
            window = 'brain'
        
        organ_mapping['organ_labels'][label_id] = {
            'name': label_name.replace(' ', '_').lower(),
            'window': window
        }
    
    return organ_mapping


def process_m3d_seg_case(case_info: Dict,
                         output_dir: str,
                         organ_mapping: Optional[Dict] = None,
                         default_resolution: int = DEFAULT_RESOLUTION,
                         compute_sdf: bool = False,
                         sdf_resolution: int = 512,
                         sdf_threshold_factor: float = 4.0,
                         replace_npy: bool = False) -> Dict:
    """
    处理M3D-Seg格式的单个病例
    
    Args:
        case_info: 病例信息
        output_dir: 输出目录
        organ_mapping: 器官映射
        default_resolution: 目标分辨率
        compute_sdf: 是否计算SDF
        sdf_resolution: SDF分辨率
        sdf_threshold_factor: SDF阈值因子
        replace_npy: 是否用NPZ替换NPY文件
    
    Returns:
        处理结果信息
    """
    case_id = case_info['case_id']
    case_dir = case_info['case_dir']
    
    print(f"\n处理病例: {case_id}")
    start_time = time.time()
    
    # 创建输出目录
    case_output_dir = os.path.join(output_dir, 'processed', case_id)
    os.makedirs(case_output_dir, exist_ok=True)
    os.makedirs(os.path.join(case_output_dir, 'windows'), exist_ok=True)
    
    # 步骤1: 加载数据
    print(f"  1. 加载M3D-Seg数据...")
    ct_array, seg_array, mask_shape = load_m3d_seg_case(case_dir)
    
    # 处理CT数组维度
    if ct_array.ndim == 4:
        if ct_array.shape[0] == 1:
            ct_array = ct_array[0]
        else:
            print(f"     警告: CT有多个通道 {ct_array.shape}，只使用第一个通道")
            ct_array = ct_array[0]
    
    original_shape = ct_array.shape
    print(f"     原始形状: {original_shape}")
    print(f"     HU值范围: [{np.min(ct_array):.2f}, {np.max(ct_array):.2f}]")
    
    # 步骤2: 分辨率适配
    print(f"  2. 分辨率适配...")
    target_resolution = determine_target_resolution(original_shape, default_resolution)
    print(f"     目标分辨率: {target_resolution}³")
    
    ct_adapted = adapt_resolution(ct_array, target_resolution)
    adapted_shape = ct_adapted.shape
    print(f"     适配后形状: {adapted_shape}")
    
    # 如果有分割标签，也进行适配
    seg_adapted = None
    if seg_array is not None:
        # 处理分割数组维度
        if seg_array.ndim == 4:
            seg_array = seg_array.squeeze()
        
        seg_adapted = adapt_resolution(seg_array, target_resolution, fill_value=0, mode='constant')
        print(f"     分割标签已适配")
    
    # 保存原始适配后的CT
    ct_original_path = os.path.join(case_output_dir, f'ct_original_{target_resolution}.npy')
    np.save(ct_original_path, ct_adapted)
    print(f"     保存原始CT: ct_original_{target_resolution}.npy")
    
    # 步骤3: 全局窗口处理（直接在原始CT上进行二值化）
    print(f"  3. 全局窗口处理（基于原始HU值）...")
    if compute_sdf:
        print(f"     - 同时计算SDF (分辨率={sdf_resolution}, 阈值因子={sdf_threshold_factor})")
    
    global_windows = process_all_windows(
        ct_adapted, 
        binarize=True,
        compute_sdf=compute_sdf,
        sdf_resolution=sdf_resolution,
        sdf_threshold_factor=sdf_threshold_factor
    )
    
    # 保存窗口结果
    from ct_preprocessing.window_processor import save_window_results
    windows_dir = os.path.join(case_output_dir, 'windows')
    saved_paths = save_window_results(global_windows, windows_dir, replace_npy=replace_npy)
    
    for window_name, result in global_windows.items():
        if isinstance(result, dict) and 'binary' in result:
            binary_array = result['binary']
            sdf_points = len(result['sdf']['sparse_index']) if 'sdf' in result else 0
            positive_ratio = np.sum(binary_array) / binary_array.size
            print(f"     {window_name}: {positive_ratio:.2%} 正值, SDF点数: {sdf_points}")
        else:
            binary_array = result
            positive_ratio = np.sum(binary_array) / binary_array.size
            print(f"     {window_name}: {positive_ratio:.2%} 正值")
    
    # 步骤4: 器官特定窗口处理
    organs_info = []
    if seg_adapted is not None and organ_mapping is not None:
        print(f"  4. 器官特定窗口处理...")
        
        # 验证分割
        is_valid, message = validate_segmentation(seg_adapted, ct_adapted)
        if not is_valid:
            print(f"     警告: {message}")
        else:
            print(f"     {message}")
        
        # 获取存在的器官
        organs_present = get_organs_present(seg_adapted, organ_mapping)
        print(f"     发现 {len(organs_present)} 个器官")
        
        # 处理所有器官
        organ_results = process_all_organs(
            ct_adapted,
            seg_adapted,
            organ_mapping,
            save_global_windows=False,
            compute_sdf=compute_sdf,
            sdf_resolution=sdf_resolution,
            sdf_threshold_factor=sdf_threshold_factor
        )
        
        # 保存器官数据
        for organ_name, organ_data in organ_results['organs'].items():
            organ_dir = os.path.join(case_output_dir, 'organs', organ_name)
            os.makedirs(organ_dir, exist_ok=True)
            
            for window_filename, window_result in organ_data.items():
                if window_filename in ['mask', 'label', 'window_used']:
                    continue
                
                # 检查是否包含SDF结果
                if isinstance(window_result, dict) and 'sdf' in window_result:
                    # 保存二值化结果
                    npy_path = os.path.join(organ_dir, window_filename + '.npy')
                    np.save(npy_path, window_result['binary'])
                    
                    # 保存SDF结果
                    from ct_preprocessing.sdf_processor import save_sdf_result
                    npz_path = os.path.join(organ_dir, window_filename + '.npz')
                    save_sdf_result(
                        window_result['sdf'],
                        npz_path,
                        replace_source=replace_npy,
                        source_path=npy_path if replace_npy else None
                    )
                else:
                    # 只有二值化结果
                    window_path = os.path.join(organ_dir, window_filename + '.npy')
                    np.save(window_path, window_result)
            
            # 统计
            organ_label = organ_data['label']
            organ_stats = compute_organ_statistics(ct_adapted, seg_adapted, organ_label)
            organs_info.append({
                'name': organ_name,
                'label': organ_label,
                'window': organ_data['window_used'],
                'voxel_count': organ_stats['voxel_count'],
                'hu_mean': organ_stats['hu_mean'],
                'hu_std': organ_stats['hu_std']
            })
            
            print(f"       {organ_name}: {organ_stats['voxel_count']} 体素")
        
        # 保存分割掩码
        masks_dir = os.path.join(case_output_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        mask_shape_save = seg_adapted.shape
        seg_flat = seg_adapted.reshape(-1)
        seg_sparse = sparse.csr_matrix(seg_flat)
        mask_path = os.path.join(masks_dir, 'segmentation_masks.npz')
        sparse.save_npz(mask_path, seg_sparse)
        print(f"     保存分割掩码")
    else:
        print(f"  4. 跳过器官处理")
    
    # 生成元信息
    processing_time = time.time() - start_time
    
    total_size = 0
    for root, dirs, files in os.walk(case_output_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    
    file_size_mb = total_size / (1024 * 1024)
    
    info = {
        'case_id': case_id,
        'original_shape': list(original_shape),
        'adapted_shape': list(adapted_shape),
        'resolution': target_resolution,
        'has_segmentation': seg_array is not None,
        'organs_present': organs_info,
        'windows_processed': list(global_windows.keys()),
        'file_size_mb': round(file_size_mb, 2),
        'processing_time_sec': round(processing_time, 2),
        'ct_path': f'processed/{case_id}/ct_original_{target_resolution}.npy',
        'masks_path': f'processed/{case_id}/masks/segmentation_masks.npz' if seg_array is not None else None,
        'source_format': 'm3d_seg'
    }
    
    info_path = os.path.join(case_output_dir, 'info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"  完成！耗时: {processing_time:.2f}秒")
    
    return info


def scan_m3d_seg_dataset(dataset_root: str) -> tuple:
    """
    扫描M3D-Seg格式的数据集
    
    Args:
        dataset_root: 数据集根目录
    
    Returns:
        (case_list, dataset_json)
    """
    # 查找JSON文件
    json_files = glob.glob(os.path.join(dataset_root, '*.json'))
    
    dataset_json = None
    if json_files:
        json_path = json_files[0]
        dataset_json = load_dataset_json(json_path)
        print(f"  发现数据集配置: {json_path}")
    
    # 扫描子文件夹
    case_list = []
    for item in os.listdir(dataset_root):
        item_path = os.path.join(dataset_root, item)
        if os.path.isdir(item_path):
            # 检查是否包含image.npy
            image_path = os.path.join(item_path, 'image.npy')
            if os.path.exists(image_path):
                case_list.append({
                    'case_id': item,
                    'case_dir': item_path
                })
    
    return case_list, dataset_json


def process_m3d_seg_dataset(dataset_root: str,
                            output_dir: str,
                            default_resolution: int = DEFAULT_RESOLUTION,
                            num_workers: int = 1,
                            compute_sdf: bool = False,
                            sdf_resolution: int = 512,
                            sdf_threshold_factor: float = 4.0,
                            replace_npy: bool = False) -> pd.DataFrame:
    """
    处理完整的M3D-Seg数据集
    
    Args:
        dataset_root: M3D-Seg数据集根目录
        output_dir: 输出目录
        default_resolution: 默认分辨率
        num_workers: 并行进程数
        compute_sdf: 是否计算SDF
        sdf_resolution: SDF分辨率
        sdf_threshold_factor: SDF阈值因子
        replace_npy: 是否用NPZ替换NPY文件
    
    Returns:
        元数据DataFrame
    """
    print("=" * 70)
    print("M3D-Seg格式数据预处理")
    print("=" * 70)
    
    # 检查SDF依赖
    if compute_sdf:
        from ct_preprocessing import check_cuda_available, check_trellis_available
        if not check_cuda_available():
            print("\n⚠️  警告: CUDA不可用，SDF计算需要GPU支持")
            compute_sdf = False
        # elif not check_trellis_available():
        #     print("\n⚠️  警告: TRELLIS不可用，跳过SDF计算")
        #     compute_sdf = False
        else:
            print(f"\n✓ SDF计算已启用 (分辨率={sdf_resolution}, 替换NPY={replace_npy})")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
    
    # 扫描数据集
    print(f"\n扫描M3D-Seg数据集: {dataset_root}")
    case_list, dataset_json = scan_m3d_seg_dataset(dataset_root)
    print(f"  发现 {len(case_list)} 个病例")
    
    # 构建器官映射
    organ_mapping = None
    if dataset_json:
        organ_mapping = build_organ_mapping_from_json(dataset_json)
        print(f"  数据集: {organ_mapping['dataset_name']}")
        print(f"  器官数: {len(organ_mapping['organ_labels'])}")
    
    # 处理所有病例
    print(f"\n开始处理（并行进程数: {num_workers}）...")
    print("=" * 70)
    
    metadata_list = []
    
    if num_workers == 1:
        for case_info in case_list:
            try:
                info = process_m3d_seg_case(
                    case_info,
                    output_dir,
                    organ_mapping,
                    default_resolution,
                    compute_sdf,
                    sdf_resolution,
                    sdf_threshold_factor,
                    replace_npy
                )
                metadata_list.append(info)
            except Exception as e:
                print(f"  错误: {case_info['case_id']}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for case_info in case_list:
                future = executor.submit(
                    process_m3d_seg_case,
                    case_info,
                    output_dir,
                    organ_mapping,
                    default_resolution,
                    compute_sdf,
                    sdf_resolution,
                    sdf_threshold_factor,
                    replace_npy
                )
                futures.append((future, case_info['case_id']))
            
            for future, case_id in tqdm(futures, desc="处理进度"):
                try:
                    info = future.result()
                    metadata_list.append(info)
                except Exception as e:
                    print(f"\n  错误: {case_id}: {e}")
    
    # 生成元数据
    print("\n" + "=" * 70)
    print("生成元数据...")
    
    metadata_df = pd.DataFrame(metadata_list)
    
    # 转换字段
    if 'organs_present' in metadata_df.columns:
        metadata_df['organs_present'] = metadata_df['organs_present'].apply(json.dumps)
    if 'windows_processed' in metadata_df.columns:
        metadata_df['windows_processed'] = metadata_df['windows_processed'].apply(json.dumps)
    if 'original_shape' in metadata_df.columns:
        metadata_df['original_shape'] = metadata_df['original_shape'].apply(
            lambda x: ','.join(map(str, x))
        )
    if 'adapted_shape' in metadata_df.columns:
        metadata_df['adapted_shape'] = metadata_df['adapted_shape'].apply(
            lambda x: ','.join(map(str, x))
        )
    
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"  元数据已保存: {metadata_path}")
    
    # 数据集配置
    dataset_config = {
        'dataset_name': organ_mapping['dataset_name'] if organ_mapping else 'Unknown',
        'modality': 'CT',
        'source_format': 'm3d_seg',
        'num_cases': len(metadata_df),
        'default_resolution': default_resolution,
        'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    config_path = os.path.join(output_dir, 'dataset_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)
    
    print("\n处理完成!")
    print(f"  总病例数: {len(metadata_df)}")
    print(f"  输出目录: {output_dir}")
    print("=" * 70)
    
    return metadata_df


def main():
    parser = argparse.ArgumentParser(description='M3D-Seg格式数据预处理')
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='M3D-Seg数据集根目录（包含子文件夹和JSON文件）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--default_resolution', type=int, default=DEFAULT_RESOLUTION,
                       help='默认目标分辨率')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='并行进程数')
    parser.add_argument('--compute_sdf', action='store_true',
                       help='计算窗口数据的SDF表示（需要CUDA和TRELLIS）')
    parser.add_argument('--sdf_resolution', type=int, default=512,
                       help='SDF目标分辨率（默认: 512）')
    parser.add_argument('--sdf_threshold_factor', type=float, default=4.0,
                       help='SDF阈值因子（默认: 4.0）')
    parser.add_argument('--replace_npy', action='store_true',
                       help='用NPZ文件替换原NPY文件')
    
    args = parser.parse_args()
    
    metadata_df = process_m3d_seg_dataset(
        dataset_root=args.data_root,
        output_dir=args.output_dir,
        default_resolution=args.default_resolution,
        num_workers=args.num_workers,
        compute_sdf=args.compute_sdf,
        sdf_resolution=args.sdf_resolution,
        sdf_threshold_factor=args.sdf_threshold_factor,
        replace_npy=args.replace_npy
    )
    
    print("\n全部完成！")


if __name__ == '__main__':
    main()

