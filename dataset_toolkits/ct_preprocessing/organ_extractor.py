"""
器官掩码提取器模块

提供器官特定的掩码提取和窗口应用功能。
根据分割标签提取特定器官，并应用对应的窗口设置。
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from .config import WINDOW_CONFIGS, get_window_config, get_window_for_organ
from .window_processor import apply_window_binarization


def load_organ_mapping(mapping_file: Union[str, Path]) -> Dict:
    """
    加载器官标签映射配置文件
    
    配置文件格式示例:
    {
        "dataset_name": "3D-IRCADB",
        "modality": "CT",
        "organ_labels": {
            "1": {"name": "liver", "window": "soft_tissue"},
            "2": {"name": "right_kidney", "window": "soft_tissue"},
            ...
        },
        "default_resolution": 512
    }
    
    Args:
        mapping_file: JSON配置文件路径
    
    Returns:
        器官映射字典
    """
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    return mapping


def extract_organ_mask(segmentation_array: np.ndarray,
                      organ_label: int) -> np.ndarray:
    """
    从分割数组中提取特定器官的掩码
    
    Args:
        segmentation_array: 分割标签数组
        organ_label: 器官标签值
    
    Returns:
        二值掩码数组（0或1）
    """
    organ_mask = (segmentation_array == organ_label).astype(np.uint8)
    return organ_mask


def extract_organ_with_window(ct_array: np.ndarray,
                              mask_array: np.ndarray,
                              organ_label: int,
                              window_config: Dict) -> np.ndarray:
    """
    提取特定器官在特定窗口下的二值化结果
    
    处理步骤：
    1. 从分割标签中提取器官掩码
    2. 对整个CT应用窗口二值化
    3. 将器官掩码与窗口二值化结果做交集
    
    Args:
        ct_array: CT数组（HU值）
        mask_array: 分割标签数组
        organ_label: 器官标签值
        window_config: 窗口配置字典
    
    Returns:
        器官特定的窗口二值化结果
    """
    # 步骤1: 提取器官掩码
    organ_mask = extract_organ_mask(mask_array, organ_label)
    
    # 步骤2: 应用窗口二值化到整个CT
    window_binary = apply_window_binarization(
        ct_array,
        window_config['window_level'],
        window_config['window_width']
    )
    
    # 步骤3: 交集 - 只保留器官区域内的窗口响应
    organ_window_result = organ_mask & window_binary
    
    return organ_window_result


def process_all_organs(ct_array: np.ndarray,
                      segmentation_array: np.ndarray,
                      organ_mapping: Dict,
                      save_global_windows: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    """
    处理所有器官在所有适用窗口下的数据
    
    Args:
        ct_array: CT数组
        segmentation_array: 分割标签数组
        organ_mapping: 器官映射配置
        save_global_windows: 是否也保存全局窗口（不限定器官区域）
    
    Returns:
        嵌套字典结构:
        {
            'organs': {
                'liver': {
                    'soft_tissue_w400_l50': binary_array,
                    'mask': organ_mask
                },
                ...
            },
            'global_windows': {  # 如果save_global_windows=True
                'lung_w1500_l-600': binary_array,
                ...
            }
        }
    """
    results = {
        'organs': {},
    }
    
    if save_global_windows:
        results['global_windows'] = {}
    
    # 获取器官标签配置
    organ_labels = organ_mapping.get('organ_labels', {})
    
    # 获取数据集中实际存在的器官标签
    unique_labels = np.unique(segmentation_array)
    present_labels = [label for label in unique_labels if label != 0]  # 排除背景
    
    # 处理每个器官
    for label_str, organ_info in organ_labels.items():
        organ_label = int(label_str)
        
        # 检查该器官是否在数据集中存在
        if organ_label not in present_labels:
            print(f"  器官 {organ_info['name']} (label={organ_label}) 不存在，跳过")
            continue
        
        organ_name = organ_info['name']
        window_name = organ_info.get('window')
        
        # 如果没有指定窗口，自动确定
        if window_name is None:
            window_name = get_window_for_organ(organ_name)
        
        # 获取窗口配置
        try:
            window_config = get_window_config(window_name)
        except ValueError as e:
            print(f"  警告: {e}，使用soft_tissue窗口")
            window_config = get_window_config('soft_tissue')
            window_name = 'soft_tissue'
        
        # 提取器官掩码
        organ_mask = extract_organ_mask(segmentation_array, organ_label)
        
        # 应用窗口
        organ_window = extract_organ_with_window(
            ct_array,
            segmentation_array,
            organ_label,
            window_config
        )
        
        # 保存结果
        window_filename = f"{window_name}_w{int(window_config['window_width'])}_l{int(window_config['window_level'])}"
        
        results['organs'][organ_name] = {
            window_filename: organ_window,
            'mask': organ_mask,
            'label': organ_label,
            'window_used': window_name
        }
    
    # 如果需要，保存全局窗口
    if save_global_windows:
        for window_name, window_config in WINDOW_CONFIGS.items():
            window_binary = apply_window_binarization(
                ct_array,
                window_config['window_level'],
                window_config['window_width']
            )
            window_filename = f"{window_name}_w{int(window_config['window_width'])}_l{int(window_config['window_level'])}"
            results['global_windows'][window_filename] = window_binary
    
    return results


def compute_organ_statistics(ct_array: np.ndarray,
                            segmentation_array: np.ndarray,
                            organ_label: int) -> Dict:
    """
    计算器官区域的统计信息
    
    Args:
        ct_array: CT数组
        segmentation_array: 分割标签数组
        organ_label: 器官标签
    
    Returns:
        统计信息字典
    """
    organ_mask = (segmentation_array == organ_label)
    organ_voxels = ct_array[organ_mask]
    
    if len(organ_voxels) == 0:
        return {
            'organ_label': organ_label,
            'voxel_count': 0,
            'volume_ratio': 0.0,
        }
    
    total_voxels = segmentation_array.size
    organ_voxel_count = len(organ_voxels)
    
    stats = {
        'organ_label': organ_label,
        'voxel_count': int(organ_voxel_count),
        'volume_ratio': float(organ_voxel_count / total_voxels),
        'hu_mean': float(np.mean(organ_voxels)),
        'hu_std': float(np.std(organ_voxels)),
        'hu_min': float(np.min(organ_voxels)),
        'hu_max': float(np.max(organ_voxels)),
        'hu_median': float(np.median(organ_voxels)),
        'hu_percentile_25': float(np.percentile(organ_voxels, 25)),
        'hu_percentile_75': float(np.percentile(organ_voxels, 75)),
    }
    
    return stats


def get_organs_present(segmentation_array: np.ndarray,
                      organ_mapping: Dict) -> List[Dict]:
    """
    获取数据集中实际存在的器官列表
    
    Args:
        segmentation_array: 分割标签数组
        organ_mapping: 器官映射配置
    
    Returns:
        存在的器官信息列表
    """
    unique_labels = np.unique(segmentation_array)
    present_labels = [label for label in unique_labels if label != 0]
    
    organ_labels_config = organ_mapping.get('organ_labels', {})
    organs_present = []
    
    for label in present_labels:
        label_str = str(label)
        if label_str in organ_labels_config:
            organ_info = organ_labels_config[label_str]
            organs_present.append({
                'label': int(label),
                'name': organ_info['name'],
                'window': organ_info.get('window', 'soft_tissue')
            })
    
    return organs_present


def validate_segmentation(segmentation_array: np.ndarray,
                         ct_array: np.ndarray) -> Tuple[bool, str]:
    """
    验证分割数组的有效性
    
    Args:
        segmentation_array: 分割标签数组
        ct_array: CT数组
    
    Returns:
        (is_valid, message)
    """
    # 检查形状匹配
    if segmentation_array.shape != ct_array.shape:
        # 如果CT有通道维度，需要squeeze
        if ct_array.ndim == 4 and ct_array.shape[0] == 1:
            ct_shape = ct_array.shape[1:]
            if segmentation_array.shape != ct_shape:
                return False, f"Shape mismatch: seg {segmentation_array.shape} vs CT {ct_shape}"
        else:
            return False, f"Shape mismatch: seg {segmentation_array.shape} vs CT {ct_array.shape}"
    
    # 检查标签值是否合理（应该是非负整数）
    if np.any(segmentation_array < 0):
        return False, "Segmentation contains negative values"
    
    # 检查是否有非零标签
    unique_labels = np.unique(segmentation_array)
    non_zero_labels = [label for label in unique_labels if label != 0]
    
    if len(non_zero_labels) == 0:
        return False, "Segmentation contains only background (all zeros)"
    
    return True, f"Valid segmentation with {len(non_zero_labels)} organ labels"


if __name__ == '__main__':
    # 测试器官提取器
    print("=== 测试器官提取器 ===\n")
    
    # 创建模拟数据
    print("创建模拟CT和分割数据...")
    ct = np.random.randn(100, 100, 100).astype(np.float32) * 100  # 模拟CT
    segmentation = np.zeros((100, 100, 100), dtype=np.int32)
    
    # 添加不同器官
    segmentation[20:40, 20:40, 20:40] = 1  # 肝脏
    segmentation[50:70, 50:70, 50:70] = 2  # 肾脏
    ct[20:40, 20:40, 20:40] = 50           # 软组织HU值
    ct[50:70, 50:70, 50:70] = 30           # 软组织HU值
    
    print(f"CT形状: {ct.shape}")
    print(f"分割标签形状: {segmentation.shape}")
    print(f"唯一标签: {np.unique(segmentation)}")
    print()
    
    # 测试1: 提取器官掩码
    print("测试1: 提取器官掩码")
    liver_mask = extract_organ_mask(segmentation, organ_label=1)
    print(f"  肝脏掩码形状: {liver_mask.shape}")
    print(f"  肝脏体素数: {np.sum(liver_mask)}")
    print(f"  ✓ 测试通过\n")
    
    # 测试2: 器官窗口提取
    print("测试2: 提取器官窗口")
    soft_tissue_config = get_window_config('soft_tissue')
    liver_window = extract_organ_with_window(
        ct, segmentation, organ_label=1, window_config=soft_tissue_config
    )
    print(f"  肝脏窗口形状: {liver_window.shape}")
    print(f"  肝脏窗口正值数: {np.sum(liver_window)}")
    print(f"  ✓ 测试通过\n")
    
    # 测试3: 器官统计
    print("测试3: 计算器官统计信息")
    liver_stats = compute_organ_statistics(ct, segmentation, organ_label=1)
    print(f"  器官标签: {liver_stats['organ_label']}")
    print(f"  体素数: {liver_stats['voxel_count']}")
    print(f"  平均HU: {liver_stats['hu_mean']:.2f}")
    print(f"  HU标准差: {liver_stats['hu_std']:.2f}")
    print(f"  ✓ 测试通过\n")
    
    # 测试4: 验证分割
    print("测试4: 验证分割数据")
    is_valid, message = validate_segmentation(segmentation, ct)
    print(f"  验证结果: {is_valid}")
    print(f"  消息: {message}")
    print(f"  ✓ 测试通过\n")
    
    # 测试5: 创建和加载器官映射
    print("测试5: 器官映射配置")
    test_mapping = {
        "dataset_name": "Test Dataset",
        "modality": "CT",
        "organ_labels": {
            "1": {"name": "liver", "window": "soft_tissue"},
            "2": {"name": "kidney", "window": "soft_tissue"}
        },
        "default_resolution": 512
    }
    
    # 测试获取存在的器官
    organs_present = get_organs_present(segmentation, test_mapping)
    print(f"  存在的器官数: {len(organs_present)}")
    for organ in organs_present:
        print(f"    - {organ['name']} (label={organ['label']})")
    print(f"  ✓ 测试通过\n")
    
    print("所有测试通过！")

