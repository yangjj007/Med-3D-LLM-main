"""
SDF处理器模块

将二值化窗口数据转换为稀疏SDF格式，用于TRELLIS模型训练
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple
import torch

# 添加项目根目录到Python路径
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def convert_window_to_sdf(window_data: np.ndarray,
                          resolution: int = 512,
                          threshold_factor: float = 4.0) -> Dict:
    """
    将二值化窗口数据转换为稀疏SDF格式
    
    Args:
        window_data: 二值化窗口数据 (0/1数组)
        resolution: 目标分辨率
        threshold_factor: UDF阈值因子
    
    Returns:
        包含sparse_sdf, sparse_index, resolution的字典
    """
    try:
        from trellis.utils.mesh_utils import dense_voxel_to_sparse_sdf
    except ImportError:
        import traceback
        traceback.print_exc()
        raise ImportError(
            "TRELLIS not available. Cannot convert to SDF. "
            "Please install TRELLIS or skip SDF conversion."
        )
    
    # 检查数据是否太稀疏
    if window_data.sum() < 100:
        raise ValueError(f"Window data too sparse (< 100 voxels)")
    
    # 转换为SDF
    sdf_result = dense_voxel_to_sparse_sdf(
        window_data,
        resolution=resolution,
        threshold_factor=threshold_factor,
        marching_cubes_level=0.5
    )
    
    return sdf_result


def save_sdf_result(sdf_result: Dict, 
                    output_path: str,
                    replace_source: bool = False,
                    source_path: Optional[str] = None) -> None:
    """
    保存SDF结果到NPZ文件
    
    Args:
        sdf_result: SDF转换结果
        output_path: 输出NPZ文件路径
        replace_source: 是否删除源文件
        source_path: 源文件路径（如果replace_source=True需要提供）
    """
    # 保存为压缩的npz文件
    np.savez_compressed(
        output_path,
        sparse_sdf=sdf_result['sparse_sdf'],
        sparse_index=sdf_result['sparse_index'],
        resolution=np.array(sdf_result['resolution'])
    )
    
    # 如果指定替换源文件
    if replace_source and source_path and os.path.exists(source_path):
        os.remove(source_path)


def load_sdf_result(npz_path: str) -> Dict:
    """
    从NPZ文件加载SDF结果
    
    Args:
        npz_path: NPZ文件路径
    
    Returns:
        包含sparse_sdf, sparse_index, resolution的字典
    """
    data = np.load(npz_path)
    return {
        'sparse_sdf': data['sparse_sdf'],
        'sparse_index': data['sparse_index'],
        'resolution': int(data['resolution'])
    }


def process_window_to_sdf(window_path: str,
                          resolution: int = 512,
                          threshold_factor: float = 4.0,
                          replace_npy: bool = False) -> Tuple[str, Dict]:
    """
    处理单个窗口文件，转换为SDF并保存
    
    Args:
        window_path: 窗口NPY文件路径
        resolution: 目标分辨率
        threshold_factor: UDF阈值因子
        replace_npy: 是否用NPZ替换原NPY文件
    
    Returns:
        (output_path, stats) - 输出路径和统计信息
    """
    # 加载窗口数据
    window_data = np.load(window_path)
    
    # 转换为SDF
    sdf_result = convert_window_to_sdf(
        window_data, 
        resolution=resolution,
        threshold_factor=threshold_factor
    )
    
    # 确定输出路径
    output_path = window_path.replace('.npy', '.npz')
    
    # 保存结果
    save_sdf_result(
        sdf_result, 
        output_path,
        replace_source=replace_npy,
        source_path=window_path
    )
    
    # 统计信息
    stats = {
        'num_points': len(sdf_result['sparse_index']),
        'resolution': sdf_result['resolution'],
        'output_path': output_path
    }
    
    return output_path, stats


def batch_process_windows_to_sdf(window_dict: Dict[str, np.ndarray],
                                 output_dir: str,
                                 resolution: int = 512,
                                 threshold_factor: float = 4.0,
                                 replace_npy: bool = False) -> Dict[str, Dict]:
    """
    批量处理窗口数据，转换为SDF
    
    Args:
        window_dict: 窗口名称到数据的字典
        output_dir: 输出目录
        resolution: 目标分辨率
        threshold_factor: UDF阈值因子
        replace_npy: 是否用NPZ替换原NPY文件
    
    Returns:
        窗口名称到处理结果的字典
    """
    results = {}
    
    for window_name, window_data in window_dict.items():
        try:
            # 转换为SDF
            sdf_result = convert_window_to_sdf(
                window_data,
                resolution=resolution,
                threshold_factor=threshold_factor
            )
            
            results[window_name] = {
                'success': True,
                'sdf_result': sdf_result,
                'num_points': len(sdf_result['sparse_index'])
            }
            
        except Exception as e:
            results[window_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results


def check_cuda_available() -> bool:
    """
    检查CUDA是否可用
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_trellis_available() -> bool:
    """
    检查TRELLIS是否可用
    
    Returns:
        True if TRELLIS is available, False otherwise
    """
    try:
        from trellis.utils.mesh_utils import dense_voxel_to_sparse_sdf
        return True
    except ImportError:
        print("Trellis mesh util import error: TRELLIS not available. ")
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 测试SDF处理器
    print("=== 测试SDF处理器 ===\n")
    
    # 检查依赖
    print("检查依赖:")
    print(f"  CUDA可用: {check_cuda_available()}")
    print(f"  TRELLIS可用: {check_trellis_available()}")
    print()
    
    if not check_trellis_available():
        print("警告: TRELLIS不可用，跳过测试")
    else:
        # 创建模拟窗口数据
        print("创建模拟窗口数据...")
        test_window = np.zeros((64, 64, 64), dtype=np.uint8)
        test_window[16:48, 16:48, 16:48] = 1  # 中心立方体
        print(f"  形状: {test_window.shape}")
        print(f"  正值体素: {np.sum(test_window)}")
        print()
        
        # 测试转换
        print("测试SDF转换...")
        try:
            sdf_result = convert_window_to_sdf(test_window, resolution=64)
            print(f"  ✓ 转换成功")
            print(f"  稀疏点数: {len(sdf_result['sparse_index'])}")
            print(f"  分辨率: {sdf_result['resolution']}")
        except Exception as e:
            print(f"  ✗ 转换失败: {e}")
        
        print("\n所有测试完成！")

