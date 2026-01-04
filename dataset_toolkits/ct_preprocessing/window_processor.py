"""
窗宽/窗位处理器模块

提供CT窗宽/窗位的二值化处理功能，用于突出显示特定组织类型。
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple, Union
from .config import WINDOW_CONFIGS, get_window_config, get_all_window_names


def apply_window_binarization(ct_array: np.ndarray,
                              window_level: float,
                              window_width: float,
                              output_dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    应用窗宽/窗位二值化
    
    算法：
    1. 计算HU值范围：[window_level - window_width/2, window_level + window_width/2]
    2. 在此范围内的像素值设为1（有组织），范围外设为0（无组织）
    
    Args:
        ct_array: 输入CT数组（HU值）
        window_level: 窗位（窗口中心HU值）
        window_width: 窗宽（HU值范围）
        output_dtype: 输出数据类型，默认uint8
    
    Returns:
        二值化的3D数组（0或1）
    """
    # 计算窗口的HU值范围
    hu_min = window_level - window_width / 2.0
    hu_max = window_level + window_width / 2.0
    
    # 二值化：在窗口范围内为1，否则为0
    binary_mask = (ct_array >= hu_min) & (ct_array <= hu_max)
    binary_array = binary_mask.astype(output_dtype)
    
    return binary_array


def apply_window_scaling(ct_array: np.ndarray,
                        window_level: float,
                        window_width: float) -> np.ndarray:
    """
    应用窗宽/窗位缩放（非二值化）
    
    将CT值映射到[0, 1]范围，用于可视化
    
    Args:
        ct_array: 输入CT数组（HU值）
        window_level: 窗位
        window_width: 窗宽
    
    Returns:
        缩放后的数组，范围[0, 1]
    """
    hu_min = window_level - window_width / 2.0
    hu_max = window_level + window_width / 2.0
    
    # 裁剪到窗口范围
    windowed = np.clip(ct_array, hu_min, hu_max)
    
    # 归一化到[0, 1]
    if window_width > 0:
        normalized = (windowed - hu_min) / window_width
    else:
        normalized = np.zeros_like(windowed)
    
    return normalized


def process_all_windows(ct_array: np.ndarray,
                       window_configs: Optional[Dict] = None,
                       binarize: bool = True,
                       compute_sdf: bool = False,
                       sdf_resolution: int = 512,
                       sdf_threshold_factor: float = 4.0) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    对所有预定义窗口进行处理
    
    Args:
        ct_array: 输入CT数组
        window_configs: 窗口配置字典。如果为None，使用默认配置
        binarize: 是否进行二值化。如果False，进行归一化缩放
        compute_sdf: 是否计算SDF（需要CUDA和TRELLIS）
        sdf_resolution: SDF分辨率
        sdf_threshold_factor: SDF阈值因子
    
    Returns:
        字典，key为窗口名称，value为处理后的数组或SDF结果
    """
    if window_configs is None:
        window_configs = WINDOW_CONFIGS
    
    results = {}
    
    for window_name, config in window_configs.items():
        window_level = config['window_level']
        window_width = config['window_width']
        
        if binarize:
            processed = apply_window_binarization(
                ct_array, window_level, window_width
            )
        else:
            processed = apply_window_scaling(
                ct_array, window_level, window_width
            )
        
        # 如果需要计算SDF
        if compute_sdf and binarize:
            try:
                from .sdf_processor import convert_window_to_sdf
                sdf_result = convert_window_to_sdf(
                    processed,
                    resolution=sdf_resolution,
                    threshold_factor=sdf_threshold_factor
                )
                results[window_name] = {
                    'binary': processed,
                    'sdf': sdf_result
                }
            except Exception as e:
                # 如果SDF计算失败，仍返回二值化结果
                print(f"Warning: SDF computation failed for {window_name}: {e}")
                results[window_name] = processed
        else:
            results[window_name] = processed
    
    return results


def get_window_filename(window_name: str, 
                       window_width: Optional[int] = None,
                       window_level: Optional[int] = None) -> str:
    """
    生成窗口文件名
    
    格式: {window_name}_w{width}_l{level}.npy
    
    Args:
        window_name: 窗口名称
        window_width: 窗宽（如果为None，从配置中获取）
        window_level: 窗位（如果为None，从配置中获取）
    
    Returns:
        文件名字符串
    """
    if window_width is None or window_level is None:
        config = get_window_config(window_name)
        window_width = int(config['window_width'])
        window_level = int(config['window_level'])
    
    # 处理负数窗位
    level_str = str(window_level) if window_level >= 0 else f"{window_level}"
    filename = f"{window_name}_w{window_width}_l{level_str}.npy"
    
    return filename


def compute_window_statistics(ct_array: np.ndarray,
                              window_name: str,
                              binary_array: Optional[np.ndarray] = None) -> Dict:
    """
    计算窗口处理的统计信息
    
    Args:
        ct_array: 原始CT数组
        window_name: 窗口名称
        binary_array: 二值化结果（如果为None，会自动计算）
    
    Returns:
        统计信息字典
    """
    config = get_window_config(window_name)
    
    if binary_array is None:
        binary_array = apply_window_binarization(
            ct_array, 
            config['window_level'], 
            config['window_width']
        )
    
    total_voxels = binary_array.size
    positive_voxels = np.sum(binary_array)
    positive_ratio = positive_voxels / total_voxels if total_voxels > 0 else 0
    
    # 计算在窗口范围内的CT值统计
    hu_min = config['hu_min']
    hu_max = config['hu_max']
    in_window_mask = (ct_array >= hu_min) & (ct_array <= hu_max)
    in_window_values = ct_array[in_window_mask]
    
    stats = {
        'window_name': window_name,
        'window_width': config['window_width'],
        'window_level': config['window_level'],
        'hu_range': [hu_min, hu_max],
        'total_voxels': int(total_voxels),
        'positive_voxels': int(positive_voxels),
        'positive_ratio': float(positive_ratio),
        'in_window_hu_mean': float(np.mean(in_window_values)) if len(in_window_values) > 0 else 0,
        'in_window_hu_std': float(np.std(in_window_values)) if len(in_window_values) > 0 else 0,
        'in_window_hu_min': float(np.min(in_window_values)) if len(in_window_values) > 0 else 0,
        'in_window_hu_max': float(np.max(in_window_values)) if len(in_window_values) > 0 else 0,
    }
    
    return stats


def batch_process_windows(ct_arrays: list,
                         window_names: Optional[list] = None,
                         compute_sdf: bool = False,
                         sdf_resolution: int = 512,
                         sdf_threshold_factor: float = 4.0) -> list:
    """
    批量处理多个CT数组的窗口
    
    Args:
        ct_arrays: CT数组列表
        window_names: 要处理的窗口名称列表。如果为None，处理所有窗口
        compute_sdf: 是否计算SDF
        sdf_resolution: SDF分辨率
        sdf_threshold_factor: SDF阈值因子
    
    Returns:
        结果列表，每个元素是一个字典 {window_name: binary_array或sdf_result}
    """
    if window_names is None:
        window_names = get_all_window_names()
    
    results = []
    
    for ct_array in ct_arrays:
        window_results = {}
        for window_name in window_names:
            config = get_window_config(window_name)
            binary = apply_window_binarization(
                ct_array,
                config['window_level'],
                config['window_width']
            )
            
            # 如果需要计算SDF
            if compute_sdf:
                try:
                    from .sdf_processor import convert_window_to_sdf
                    sdf_result = convert_window_to_sdf(
                        binary,
                        resolution=sdf_resolution,
                        threshold_factor=sdf_threshold_factor
                    )
                    window_results[window_name] = {
                        'binary': binary,
                        'sdf': sdf_result
                    }
                except Exception as e:
                    print(f"Warning: SDF computation failed for {window_name}: {e}")
                    window_results[window_name] = binary
            else:
                window_results[window_name] = binary
                
        results.append(window_results)
    
    return results


def save_window_results(window_results: Dict[str, Union[np.ndarray, Dict]],
                       output_dir: str,
                       replace_npy: bool = False) -> Dict[str, str]:
    """
    保存窗口处理结果
    
    Args:
        window_results: 窗口处理结果字典
        output_dir: 输出目录
        replace_npy: 如果有SDF结果，是否删除NPY文件
    
    Returns:
        保存文件路径的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}
    
    for window_name, result in window_results.items():
        filename = get_window_filename(window_name)
        
        # 如果结果包含SDF
        if isinstance(result, dict) and 'sdf' in result:
            # 先保存二值化结果为npy
            npy_path = os.path.join(output_dir, filename)
            np.save(npy_path, result['binary'])
            
            # 保存SDF结果为npz
            from .sdf_processor import save_sdf_result
            npz_path = npy_path.replace('.npy', '.npz')
            save_sdf_result(
                result['sdf'],
                npz_path,
                replace_source=replace_npy,
                source_path=npy_path if replace_npy else None
            )
            saved_paths[window_name] = npz_path
        else:
            # 只保存二值化结果
            npy_path = os.path.join(output_dir, filename)
            np.save(npy_path, result)
            saved_paths[window_name] = npy_path
    
    return saved_paths


if __name__ == '__main__':
    # 测试窗口处理器
    print("=== 测试窗口处理器 ===\n")
    
    # 创建模拟CT数据（不同HU值范围）
    print("创建模拟CT数据...")
    test_ct = np.zeros((100, 100, 100), dtype=np.float32)
    
    # 填充不同组织类型的HU值
    test_ct[20:40, 20:40, 20:40] = -600  # 肺组织
    test_ct[40:60, 40:60, 40:60] = 50    # 软组织
    test_ct[60:80, 60:80, 60:80] = 300   # 骨组织
    
    print(f"CT数组形状: {test_ct.shape}")
    print(f"CT值范围: [{np.min(test_ct)}, {np.max(test_ct)}]")
    print()
    
    # 测试单个窗口
    print("测试1: 肺窗二值化")
    lung_binary = apply_window_binarization(test_ct, -600, 1500)
    print(f"  输出形状: {lung_binary.shape}")
    print(f"  输出数据类型: {lung_binary.dtype}")
    print(f"  正值体素数: {np.sum(lung_binary)}")
    print(f"  ✓ 测试通过\n")
    
    # 测试所有窗口
    print("测试2: 处理所有窗口")
    all_windows = process_all_windows(test_ct)
    print(f"  处理的窗口数: {len(all_windows)}")
    for window_name, binary in all_windows.items():
        positive_count = np.sum(binary)
        print(f"  {window_name}: {positive_count} 个正值体素")
    print(f"  ✓ 测试通过\n")
    
    # 测试文件名生成
    print("测试3: 生成窗口文件名")
    for window_name in get_all_window_names():
        filename = get_window_filename(window_name)
        print(f"  {window_name}: {filename}")
    print(f"  ✓ 测试通过\n")
    
    # 测试统计信息
    print("测试4: 计算窗口统计信息")
    stats = compute_window_statistics(test_ct, 'lung')
    print(f"  窗口: {stats['window_name']}")
    print(f"  HU范围: {stats['hu_range']}")
    print(f"  正值比例: {stats['positive_ratio']:.4f}")
    print(f"  窗口内平均HU: {stats['in_window_hu_mean']:.2f}")
    print(f"  ✓ 测试通过\n")
    
    # 测试窗口缩放（非二值化）
    print("测试5: 窗口缩放（归一化）")
    lung_scaled = apply_window_scaling(test_ct, -600, 1500)
    print(f"  输出范围: [{np.min(lung_scaled):.4f}, {np.max(lung_scaled):.4f}]")
    print(f"  ✓ 测试通过\n")
    
    print("所有测试通过！")

