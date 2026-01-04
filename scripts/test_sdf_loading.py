"""
测试SDF数据加载是否正确
验证SDF值范围应该是连续的距离值，而不是全1.0
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trellis.datasets import CTWindowSparseSDF


def test_sdf_loading(data_root, window_type='lung', num_samples=5):
    """
    测试SDF数据加载
    
    Args:
        data_root: 数据根目录
        window_type: 窗口类型
        num_samples: 测试样本数量
    """
    print(f"\n{'='*80}")
    print(f"测试SDF数据加载")
    print(f"{'='*80}")
    print(f"数据根目录: {data_root}")
    print(f"窗口类型: {window_type}")
    print(f"测试样本数: {num_samples}")
    print(f"{'='*80}\n")
    
    # 创建数据集
    dataset = CTWindowSparseSDF(
        roots=data_root,
        resolution=512,
        window_type=window_type,
        min_points=100,
        max_points=100000,
    )
    
    print(f"数据集大小: {len(dataset)}\n")
    
    # 测试多个样本
    all_sdf_values = []
    
    for i in range(min(num_samples, len(dataset))):
        print(f"--- 样本 {i+1}/{num_samples} ---")
        
        sample = dataset[i]
        sparse_sdf = sample['sparse_sdf']
        sparse_index = sample['sparse_index']
        
        print(f"sparse_sdf.shape: {sparse_sdf.shape}")
        print(f"sparse_index.shape: {sparse_index.shape}")
        print(f"sparse_sdf统计:")
        print(f"  min: {sparse_sdf.min().item():.6f}")
        print(f"  max: {sparse_sdf.max().item():.6f}")
        print(f"  mean: {sparse_sdf.mean().item():.6f}")
        print(f"  std: {sparse_sdf.std().item():.6f}")
        print(f"  unique values: {len(torch.unique(sparse_sdf))}")
        
        all_sdf_values.append(sparse_sdf.numpy())
        print()
    
    # 总体统计
    all_sdf_values = np.concatenate(all_sdf_values)
    
    print(f"{'='*80}")
    print(f"总体统计（{num_samples}个样本）")
    print(f"{'='*80}")
    print(f"总点数: {len(all_sdf_values):,}")
    print(f"SDF值范围: [{all_sdf_values.min():.6f}, {all_sdf_values.max():.6f}]")
    print(f"SDF平均值: {all_sdf_values.mean():.6f}")
    print(f"SDF标准差: {all_sdf_values.std():.6f}")
    print(f"唯一值数量: {len(np.unique(all_sdf_values))}")
    
    # 验证
    print(f"\n{'='*80}")
    print(f"验证结果")
    print(f"{'='*80}")
    
    is_all_ones = np.allclose(all_sdf_values, 1.0)
    is_binary = np.all((all_sdf_values == 0) | (all_sdf_values == 1))
    is_continuous = len(np.unique(all_sdf_values)) > 10
    is_in_expected_range = (all_sdf_values.min() >= 0) and (all_sdf_values.max() <= 0.01)
    
    if is_all_ones:
        print("❌ 失败: SDF值全为1.0（问题未解决）")
        return False
    elif is_binary:
        print("❌ 失败: SDF值仍然是二值的（0或1）")
        return False
    elif not is_continuous:
        print("⚠️  警告: SDF值不够连续")
        return False
    elif not is_in_expected_range:
        print(f"⚠️  警告: SDF值不在预期范围 [0, ~0.008] 内")
        print(f"  实际范围: [{all_sdf_values.min():.6f}, {all_sdf_values.max():.6f}]")
        return False
    else:
        print("✅ 成功: SDF值正确！")
        print("  - 值是连续的距离值")
        print(f"  - 范围在 [0, ~0.008] 内")
        print("  - 不是二值数据")
        return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录')
    parser.add_argument('--window_type', type=str, default='lung',
                        help='窗口类型')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='测试样本数')
    
    args = parser.parse_args()
    
    success = test_sdf_loading(
        args.data_root,
        args.window_type,
        args.num_samples
    )
    
    sys.exit(0 if success else 1)

