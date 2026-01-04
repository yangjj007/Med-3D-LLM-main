"""
CT数据预处理主脚本

集成所有预处理模块，实现完整的CT数据处理流程：
1. 加载NIfTI格式的CT和分割数据
2. 分辨率适配（向上兼容到512³或1024³）
3. 全局窗口二值化（直接在原始CT上进行）
4. 器官特定窗口处理
5. 保存所有结果
6. 生成元数据
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob
from scipy import sparse
import multiprocessing

# 设置multiprocessing启动方法为spawn，避免CUDA fork问题
# 在多进程中使用CUDA时必须使用spawn模式
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass

# 添加MONAI支持
try:
    from monai.transforms import Compose, LoadImaged, Orientationd, EnsureChannelFirstd
    MONAI_AVAILABLE = True
except ImportError:
    print("Warning: MONAI not available. Install would change torch version, please be careful: pip install monai nibabel")
    MONAI_AVAILABLE = False

# 导入预处理模块
from ct_preprocessing import (
    DEFAULT_RESOLUTION,
    SUPPORTED_RESOLUTIONS,
    determine_target_resolution,
    adapt_resolution,
    process_all_windows,
    process_all_organs,
    load_organ_mapping,
    get_organs_present,
    validate_segmentation,
    compute_organ_statistics,
    get_window_filename,
    save_window_results,
    check_cuda_available,
    check_trellis_available
)


def normalize_ct(ct_array: np.ndarray, 
                method: str = 'foreground') -> np.ndarray:
    """
    标准化CT图像
    
    Args:
        ct_array: 输入CT数组
        method: 标准化方法
            - 'foreground': 基于前景的标准化（参考data_process_examples）
            - 'simple': 简单的z-score标准化
    
    Returns:
        标准化后的CT数组
    """
    if method == 'foreground':
        # 参考data_process_examples/process.py的normalize函数
        ct_voxel_ndarray = ct_array.flatten()
        
        # 对所有数据计算阈值
        thred = np.mean(ct_voxel_ndarray)
        voxel_filtered = ct_voxel_ndarray[ct_voxel_ndarray > thred]
        
        if len(voxel_filtered) == 0:
            print("  警告: 没有前景体素，使用全局标准化")
            voxel_filtered = ct_voxel_ndarray
        
        # 对前景数据进行标准化
        upper_bound = np.percentile(voxel_filtered, 99.95)
        lower_bound = np.percentile(voxel_filtered, 0.05)
        mean = np.mean(voxel_filtered)
        std = np.std(voxel_filtered)
        
        # 释放临时变量
        del ct_voxel_ndarray, voxel_filtered
        
        # 变换
        ct_normalized = np.clip(ct_array, lower_bound, upper_bound)
        ct_normalized = (ct_normalized - mean) / max(std, 1e-8)
        
    elif method == 'simple':
        mean = np.mean(ct_array)
        std = np.std(ct_array)
        ct_normalized = (ct_array - mean) / max(std, 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return ct_normalized


def load_nifti_data(image_path: str, 
                   label_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    加载NIfTI格式的数据
    
    Args:
        image_path: CT图像路径
        label_path: 分割标签路径（可选）
    
    Returns:
        (ct_array, segmentation_array)
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for loading NIfTI files")
    
    # 构建数据字典
    data_dict = {'image': image_path}
    keys = ['image']
    
    if label_path is not None:
        data_dict['label'] = label_path
        keys.append('label')
    
    # MONAI加载器
    loader = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
    ])
    
    # 加载数据
    loaded = loader(data_dict)
    ct_array = np.array(loaded['image'])
    
    segmentation_array = None
    if label_path is not None:
        segmentation_array = np.array(loaded['label'])
    
    return ct_array, segmentation_array


def process_single_case(case_info: Dict,
                       output_dir: str,
                       organ_mapping: Optional[Dict] = None,
                       default_resolution: int = DEFAULT_RESOLUTION,
                       save_intermediate: bool = True,
                       compute_sdf: bool = False,
                       sdf_resolution: int = 512,
                       sdf_threshold_factor: float = 4.0,
                       replace_npy: bool = False) -> Dict:
    """
    处理单个CT病例
    
    Args:
        case_info: 病例信息字典
            {
                'case_id': 'case_001',
                'image_path': 'path/to/ct.nii.gz',
                'label_path': 'path/to/seg.nii.gz',  # 可选
            }
        output_dir: 输出根目录
        organ_mapping: 器官标签映射配置
        default_resolution: 默认目标分辨率
        save_intermediate: 是否保存中间结果
        compute_sdf: 是否计算SDF
        sdf_resolution: SDF分辨率
        sdf_threshold_factor: SDF阈值因子
        replace_npy: 是否用NPZ替换NPY文件
    
    Returns:
        处理结果元信息字典
    """
    case_id = case_info['case_id']
    image_path = case_info['image_path']
    label_path = case_info.get('label_path')
    
    print(f"\n处理病例: {case_id}")
    start_time = time.time()
    
    # 创建输出目录
    case_output_dir = os.path.join(output_dir, 'processed', case_id)
    os.makedirs(case_output_dir, exist_ok=True)
    os.makedirs(os.path.join(case_output_dir, 'windows'), exist_ok=True)
    
    # 步骤1: 加载NIfTI数据
    print(f"  1. 加载数据...")
    ct_array, seg_array = load_nifti_data(image_path, label_path)
    
    # 处理通道维度
    if ct_array.ndim == 4 and ct_array.shape[0] == 1:
        ct_array = ct_array[0]  # 移除通道维度
    
    if seg_array is not None and seg_array.ndim == 4:
        seg_array = seg_array.squeeze()
    
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
        seg_adapted = adapt_resolution(seg_array, target_resolution, fill_value=0, mode='constant')
        print(f"     分割标签也已适配")
    
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
        
        # 验证分割数据
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
        
        # 保存器官特定数据
        for organ_name, organ_data in organ_results['organs'].items():
            organ_dir = os.path.join(case_output_dir, 'organs', organ_name)
            os.makedirs(organ_dir, exist_ok=True)
            
            # 保存窗口结果
            for window_filename, window_result in organ_data.items():
                if window_filename == 'mask' or window_filename in ['label', 'window_used']:
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
            
            # 计算统计信息
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
        
        # 保存分割掩码（稀疏格式）
        masks_dir = os.path.join(case_output_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        # 保存为稀疏矩阵格式
        mask_shape = seg_adapted.shape
        seg_flat = seg_adapted.reshape(-1)
        seg_sparse = sparse.csr_matrix(seg_flat)
        mask_path = os.path.join(masks_dir, 'segmentation_masks.npz')
        sparse.save_npz(mask_path, seg_sparse)
        print(f"     保存分割掩码: masks/segmentation_masks.npz")
    
    else:
        print(f"  4. 跳过器官处理（无分割标签或器官映射）")
    
    # 步骤5: 生成元信息
    processing_time = time.time() - start_time
    
    # 计算文件大小
    total_size = 0
    for root, dirs, files in os.walk(case_output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    file_size_mb = total_size / (1024 * 1024)
    
    # 生成元信息
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
    }
    
    # 保存元信息
    info_path = os.path.join(case_output_dir, 'info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"  完成！耗时: {processing_time:.2f}秒, 大小: {file_size_mb:.2f}MB")
    
    return info


def scan_nifti_directory(data_root: str) -> list:
    """
    扫描目录，查找所有NIfTI文件并配对
    
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
        病例信息列表
    """
    image_dir = os.path.join(data_root, 'imagesTr')
    label_dir = os.path.join(data_root, 'labelsTr')
    
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    
    # 查找所有图像文件
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
    
    case_list = []
    for image_file in image_files:
        # 提取case ID（去除_0000等后缀）
        basename = os.path.basename(image_file)
        case_id = basename.split('_')[0]
        
        # 查找对应的标签文件
        label_file = os.path.join(label_dir, f'{case_id}.nii.gz')
        
        case_info = {
            'case_id': case_id,
            'image_path': image_file,
            'label_path': label_file if os.path.exists(label_file) else None
        }
        
        case_list.append(case_info)
    
    return case_list


def process_dataset(data_root: str,
                   output_dir: str,
                   organ_mapping_file: Optional[str] = None,
                   default_resolution: int = DEFAULT_RESOLUTION,
                   num_workers: int = 1,
                   max_cases: Optional[int] = None,
                   compute_sdf: bool = False,
                   sdf_resolution: int = 512,
                   sdf_threshold_factor: float = 4.0,
                   replace_npy: bool = False) -> pd.DataFrame:
    """
    处理整个数据集
    
    Args:
        data_root: 数据根目录
        output_dir: 输出目录
        organ_mapping_file: 器官映射JSON文件
        default_resolution: 默认分辨率
        num_workers: 并行工作进程数
        max_cases: 最大处理病例数（用于测试）
        compute_sdf: 是否计算SDF
        sdf_resolution: SDF分辨率
        sdf_threshold_factor: SDF阈值因子
        replace_npy: 是否用NPZ替换NPY文件
    
    Returns:
        元数据DataFrame
    """
    print("=" * 70)
    print("CT数据预处理")
    print("=" * 70)
    
    # 检查SDF依赖
    if compute_sdf:
        if not check_cuda_available():
            print("\n⚠️  警告: CUDA不可用，SDF计算需要GPU支持")
            compute_sdf = False
        # elif not check_trellis_available():
        #     print("\n⚠️  警告: TRELLIS不可用，跳过SDF计算")
        #     compute_sdf = False
        else:
            print(f"\n✓ SDF计算已启用 (分辨率={sdf_resolution}, 替换NPY={replace_npy})")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
    
    # 加载器官映射
    organ_mapping = None
    if organ_mapping_file and os.path.exists(organ_mapping_file):
        print(f"\n加载器官映射: {organ_mapping_file}")
        organ_mapping = load_organ_mapping(organ_mapping_file)
        print(f"  数据集: {organ_mapping.get('dataset_name', 'Unknown')}")
        print(f"  器官数: {len(organ_mapping.get('organ_labels', {}))}")
    
    # 扫描数据目录
    print(f"\n扫描数据目录: {data_root}")
    case_list = scan_nifti_directory(data_root)
    print(f"  发现 {len(case_list)} 个病例")
    
    if max_cases is not None:
        case_list = case_list[:max_cases]
        print(f"  限制处理前 {max_cases} 个病例")
    
    # 处理所有病例
    print(f"\n开始处理（并行进程数: {num_workers}）...")
    print("=" * 70)
    
    metadata_list = []
    
    if num_workers == 1:
        # 串行处理
        for case_info in case_list:
            try:
                info = process_single_case(
                    case_info,
                    output_dir,
                    organ_mapping,
                    default_resolution,
                    save_intermediate=True,
                    compute_sdf=compute_sdf,
                    sdf_resolution=sdf_resolution,
                    sdf_threshold_factor=sdf_threshold_factor,
                    replace_npy=replace_npy
                )
                metadata_list.append(info)
            except Exception as e:
                print(f"  错误处理 {case_info['case_id']}: {e}")
    else:
        # 并行处理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for case_info in case_list:
                future = executor.submit(
                    process_single_case,
                    case_info,
                    output_dir,
                    organ_mapping,
                    default_resolution,
                    True,  # save_intermediate
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
                    print(f"\n  错误处理 {case_id}: {e}")
    
    # 生成元数据CSV
    print("\n" + "=" * 70)
    print("生成元数据...")
    
    metadata_df = pd.DataFrame(metadata_list)
    
    # 转换列表字段为JSON字符串
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
    
    # 保存元数据
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"  元数据已保存: {metadata_path}")
    
    # 生成数据集配置
    dataset_config = {
        'dataset_name': organ_mapping.get('dataset_name', 'Unknown') if organ_mapping else 'Unknown',
        'modality': 'CT',
        'num_cases': len(metadata_df),
        'default_resolution': default_resolution,
        'has_segmentation': metadata_df['has_segmentation'].sum() if 'has_segmentation' in metadata_df.columns else 0,
        'total_size_mb': metadata_df['file_size_mb'].sum() if 'file_size_mb' in metadata_df.columns else 0,
        'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    config_path = os.path.join(output_dir, 'dataset_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)
    print(f"  配置已保存: {config_path}")
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("处理完成统计:")
    print(f"  总病例数: {len(metadata_df)}")
    print(f"  有分割标签: {dataset_config['has_segmentation']}")
    print(f"  总大小: {dataset_config['total_size_mb']:.2f} MB")
    
    if 'resolution' in metadata_df.columns:
        print(f"  分辨率分布:")
        for res, count in metadata_df['resolution'].value_counts().items():
            print(f"    {res}³: {count} 个病例")
    
    if 'processing_time_sec' in metadata_df.columns:
        total_time = metadata_df['processing_time_sec'].sum()
        avg_time = metadata_df['processing_time_sec'].mean()
        print(f"  总耗时: {total_time:.2f} 秒")
        print(f"  平均耗时: {avg_time:.2f} 秒/病例")
    
    print("=" * 70)
    
    return metadata_df


def main():
    parser = argparse.ArgumentParser(description='CT数据预处理')
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据根目录（包含imagesTr和labelsTr子目录）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--organ_labels', type=str, default=None,
                       help='器官标签映射JSON文件路径')
    parser.add_argument('--default_resolution', type=int, default=DEFAULT_RESOLUTION,
                       choices=SUPPORTED_RESOLUTIONS,
                       help=f'默认目标分辨率（默认: {DEFAULT_RESOLUTION}）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='并行工作进程数（默认: 4）')
    parser.add_argument('--max_cases', type=int, default=None,
                       help='最大处理病例数（用于测试，默认: 处理全部）')
    parser.add_argument('--compute_sdf', action='store_true',
                       help='计算窗口数据的SDF表示（需要CUDA和TRELLIS）')
    parser.add_argument('--sdf_resolution', type=int, default=512,
                       help='SDF目标分辨率（默认: 512）')
    parser.add_argument('--sdf_threshold_factor', type=float, default=4.0,
                       help='SDF阈值因子（默认: 4.0）')
    parser.add_argument('--replace_npy', action='store_true',
                       help='用NPZ文件替换原NPY文件')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not MONAI_AVAILABLE:
        print("错误: 需要安装MONAI")
        print("请运行: pip install monai nibabel, but it would change torch version, please be careful")
        sys.exit(1)
    
    # 处理数据集
    metadata_df = process_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        organ_mapping_file=args.organ_labels,
        default_resolution=args.default_resolution,
        num_workers=args.num_workers,
        max_cases=args.max_cases,
        compute_sdf=args.compute_sdf,
        sdf_resolution=args.sdf_resolution,
        sdf_threshold_factor=args.sdf_threshold_factor,
        replace_npy=args.replace_npy
    )
    
    print("\n全部完成！")


if __name__ == '__main__':
    main()

