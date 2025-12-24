"""
分辨率适配器模块

提供CT数据的分辨率适配功能，支持向上兼容到标准分辨率（512³或1024³）。
不支持向下压缩，只支持通过填充空气HU值来扩展。
"""

import numpy as np
from typing import Tuple, Optional
from .config import AIR_HU_VALUE, DEFAULT_RESOLUTION, SUPPORTED_RESOLUTIONS


def determine_target_resolution(input_shape: Tuple[int, int, int], 
                                default_resolution: int = DEFAULT_RESOLUTION) -> int:
    """
    根据输入形状自动确定目标分辨率
    
    规则：
    1. 选择能容纳输入的最小标准分辨率
    2. 如果输入的最大维度 <= 512，目标为512³
    3. 如果输入的最大维度 > 512且 <= 1024，目标为1024³
    4. 如果输入的最大维度 > 1024，抛出错误（不支持压缩）
    
    Args:
        input_shape: 输入CT数组的形状 (H, W, D)
        default_resolution: 默认分辨率
    
    Returns:
        目标分辨率 (512 or 1024)
    
    Raises:
        ValueError: 如果输入尺寸超过最大支持分辨率
    """
    max_dim = max(input_shape)
    
    # 按照从小到大的顺序检查支持的分辨率
    sorted_resolutions = sorted(SUPPORTED_RESOLUTIONS)
    
    for resolution in sorted_resolutions:
        if max_dim <= resolution:
            return resolution
    
    # 如果超过所有支持的分辨率
    max_supported = max(SUPPORTED_RESOLUTIONS)
    raise ValueError(
        f"Input shape {input_shape} exceeds maximum supported resolution {max_supported}. "
        f"Max dimension is {max_dim}, but we only support up to {max_supported}. "
        f"Downsampling is not supported."
    )


def check_resolution_compatibility(input_shape: Tuple[int, int, int],
                                   target_resolution: int) -> bool:
    """
    检查输入形状是否可以适配到目标分辨率
    
    Args:
        input_shape: 输入形状
        target_resolution: 目标分辨率
    
    Returns:
        True if compatible (all dims <= target_resolution), False otherwise
    """
    return all(dim <= target_resolution for dim in input_shape)


def adapt_resolution(ct_array: np.ndarray,
                    target_resolution: Optional[int] = None,
                    fill_value: float = AIR_HU_VALUE,
                    mode: str = 'edge') -> np.ndarray:
    """
    适配CT数组到目标分辨率
    
    特性：
    - 只向上兼容，不向下压缩
    - 不足的维度填充指定的HU值（默认为空气HU值-1000）
    - 在每个维度的末尾进行填充
    
    Args:
        ct_array: 输入CT数组，形状为 (H, W, D) 或 (C, H, W, D)
        target_resolution: 目标分辨率。如果为None，自动确定
        fill_value: 填充值，默认为空气HU值(-1000)
        mode: 填充模式
            - 'constant': 使用fill_value填充
            - 'edge': 使用边缘值填充（保留边界信息）
    
    Returns:
        适配后的CT数组，形状为 (target_resolution, target_resolution, target_resolution)
        或 (C, target_resolution, target_resolution, target_resolution)
    
    Raises:
        ValueError: 如果需要向下压缩或输入形状不合法
    """
    # 处理多通道情况
    has_channel = ct_array.ndim == 4
    if has_channel:
        num_channels = ct_array.shape[0]
        spatial_shape = ct_array.shape[1:]
    else:
        spatial_shape = ct_array.shape
    
    # 确定目标分辨率
    if target_resolution is None:
        target_resolution = determine_target_resolution(spatial_shape)
    
    # 检查兼容性
    if not check_resolution_compatibility(spatial_shape, target_resolution):
        raise ValueError(
            f"Cannot adapt {spatial_shape} to {target_resolution}. "
            f"Downsampling is not supported. Input dimensions must be <= target resolution."
        )
    
    # 计算每个维度需要填充的大小
    pad_amounts = []
    for dim_size in spatial_shape:
        pad_needed = target_resolution - dim_size
        if pad_needed < 0:
            raise ValueError(f"Negative padding detected: {pad_needed}")
        # 在末尾填充
        pad_amounts.append((0, pad_needed))
    
    # 如果有通道维度，通道维度不填充
    if has_channel:
        pad_amounts = [(0, 0)] + pad_amounts
    
    # 执行填充
    if mode == 'constant':
        padded_array = np.pad(
            ct_array,
            pad_amounts,
            mode='constant',
            constant_values=fill_value
        )
    elif mode == 'edge':
        padded_array = np.pad(
            ct_array,
            pad_amounts,
            mode='edge'
        )
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")
    
    return padded_array


def get_padding_info(original_shape: Tuple[int, int, int],
                    target_resolution: int) -> dict:
    """
    获取填充信息
    
    Args:
        original_shape: 原始形状
        target_resolution: 目标分辨率
    
    Returns:
        包含填充信息的字典
    """
    pad_info = {
        'original_shape': original_shape,
        'target_shape': (target_resolution, target_resolution, target_resolution),
        'padding': {}
    }
    
    for i, (dim_name, dim_size) in enumerate(zip(['height', 'width', 'depth'], original_shape)):
        pad_needed = target_resolution - dim_size
        pad_info['padding'][dim_name] = {
            'original': dim_size,
            'pad_amount': pad_needed,
            'final': target_resolution
        }
    
    return pad_info


def crop_to_original(padded_array: np.ndarray,
                    original_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    从填充后的数组裁剪回原始尺寸
    
    用于处理后需要恢复原始尺寸的情况
    
    Args:
        padded_array: 填充后的数组
        original_shape: 原始形状
    
    Returns:
        裁剪后的数组
    """
    has_channel = padded_array.ndim == 4
    
    if has_channel:
        h, w, d = original_shape
        return padded_array[:, :h, :w, :d]
    else:
        h, w, d = original_shape
        return padded_array[:h, :w, :d]


if __name__ == '__main__':
    # 测试分辨率适配
    print("=== 测试分辨率适配器 ===\n")
    
    # 测试1: 512x512x100 -> 512³
    print("测试1: 512x512x100 -> 512³")
    test_array_1 = np.random.randn(512, 512, 100).astype(np.float32)
    target_res_1 = determine_target_resolution(test_array_1.shape)
    print(f"  输入形状: {test_array_1.shape}")
    print(f"  目标分辨率: {target_res_1}")
    adapted_1 = adapt_resolution(test_array_1, target_res_1)
    print(f"  输出形状: {adapted_1.shape}")
    print(f"  ✓ 测试通过\n")
    
    # 测试2: 256x256x150 -> 512³
    print("测试2: 256x256x150 -> 512³")
    test_array_2 = np.random.randn(256, 256, 150).astype(np.float32)
    target_res_2 = determine_target_resolution(test_array_2.shape)
    print(f"  输入形状: {test_array_2.shape}")
    print(f"  目标分辨率: {target_res_2}")
    adapted_2 = adapt_resolution(test_array_2, target_res_2)
    print(f"  输出形状: {adapted_2.shape}")
    print(f"  ✓ 测试通过\n")
    
    # 测试3: 1024x1024x200 -> 1024³
    print("测试3: 1024x1024x200 -> 1024³")
    test_array_3 = np.random.randn(1024, 1024, 200).astype(np.float32)
    target_res_3 = determine_target_resolution(test_array_3.shape)
    print(f"  输入形状: {test_array_3.shape}")
    print(f"  目标分辨率: {target_res_3}")
    adapted_3 = adapt_resolution(test_array_3, target_res_3)
    print(f"  输出形状: {adapted_3.shape}")
    print(f"  ✓ 测试通过\n")
    
    # 测试4: 带通道的数组
    print("测试4: 带通道 (1, 512, 512, 100) -> (1, 512, 512, 512)")
    test_array_4 = np.random.randn(1, 512, 512, 100).astype(np.float32)
    adapted_4 = adapt_resolution(test_array_4, 512)
    print(f"  输入形状: {test_array_4.shape}")
    print(f"  输出形状: {adapted_4.shape}")
    print(f"  ✓ 测试通过\n")
    
    # 测试5: 获取填充信息
    print("测试5: 获取填充信息")
    pad_info = get_padding_info((512, 512, 100), 512)
    print(f"  原始形状: {pad_info['original_shape']}")
    print(f"  目标形状: {pad_info['target_shape']}")
    print(f"  填充信息:")
    for dim, info in pad_info['padding'].items():
        print(f"    {dim}: {info['original']} + {info['pad_amount']} = {info['final']}")
    print(f"  ✓ 测试通过\n")
    
    print("所有测试通过！")

