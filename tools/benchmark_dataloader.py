#!/usr/bin/env python3
"""
数据加载性能测试脚本

测试不同配置下的数据加载性能，帮助找到最优配置。

使用方法：
    python tools/benchmark_dataloader.py \
        --roots /path/to/processed/data \
        --window_type lung \
        --batch_size 2 \
        --num_batches 10
"""

import os
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trellis.datasets.ct_window_sparse import CTWindowSparseSDF


def benchmark_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    prefetch_factor: int = 2,
    pin_memory: bool = True
):
    """
    测试DataLoader性能
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        num_workers: worker数量
        num_batches: 测试批次数量
        prefetch_factor: 预取因子
        pin_memory: 是否使用pin_memory
    
    Returns:
        性能统计字典
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        shuffle=True
    )
    
    times = []
    
    # 预热
    print("  预热中...", end='', flush=True)
    iterator = iter(dataloader)
    try:
        for _ in range(min(2, num_batches)):
            _ = next(iterator)
    except StopIteration:
        pass
    print(" 完成")
    
    # 实际测试
    print(f"  测试 {num_batches} 个批次...", end='', flush=True)
    iterator = iter(dataloader)
    
    for i in range(num_batches):
        try:
            t0 = time.time()
            batch = next(iterator)
            t1 = time.time()
            
            # 移动到GPU（如果可用）
            if torch.cuda.is_available():
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
                torch.cuda.synchronize()
            
            t2 = time.time()
            
            times.append({
                'load_time': (t1 - t0) * 1000,  # ms
                'transfer_time': (t2 - t1) * 1000,  # ms
                'total_time': (t2 - t0) * 1000,  # ms
            })
        except StopIteration:
            print(f" (只有 {i} 个批次)")
            break
    
    print(" 完成")
    
    if not times:
        return None
    
    # 计算统计
    avg_load = sum(t['load_time'] for t in times) / len(times)
    avg_transfer = sum(t['transfer_time'] for t in times) / len(times)
    avg_total = sum(t['total_time'] for t in times) / len(times)
    
    min_load = min(t['load_time'] for t in times)
    max_load = max(t['load_time'] for t in times)
    
    return {
        'num_batches': len(times),
        'avg_load_ms': avg_load,
        'avg_transfer_ms': avg_transfer,
        'avg_total_ms': avg_total,
        'min_load_ms': min_load,
        'max_load_ms': max_load,
        'std_load_ms': (sum((t['load_time'] - avg_load) ** 2 for t in times) / len(times)) ** 0.5,
        'throughput_batches_per_sec': 1000.0 / avg_total if avg_total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='数据加载性能测试',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--roots',
        type=str,
        required=True,
        help='数据根目录，多个目录用逗号分隔'
    )
    parser.add_argument(
        '--window_type',
        type=str,
        default='lung',
        help='窗口类型'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=512,
        help='分辨率'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='批次大小'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=10,
        help='测试批次数量'
    )
    parser.add_argument(
        '--test_workers',
        type=str,
        default='0,2,4,8',
        help='要测试的worker数量，用逗号分隔'
    )
    parser.add_argument(
        '--cache_data',
        action='store_true',
        help='启用数据缓存'
    )
    parser.add_argument(
        '--precompute_sparse',
        action='store_true',
        help='启用稀疏索引预计算'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("数据加载性能测试")
    print("=" * 80)
    print(f"数据根目录: {args.roots}")
    print(f"窗口类型: {args.window_type}")
    print(f"分辨率: {args.resolution}")
    print(f"批次大小: {args.batch_size}")
    print(f"测试批次数: {args.num_batches}")
    print(f"缓存数据: {args.cache_data}")
    print(f"预计算稀疏: {args.precompute_sparse}")
    print(f"GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print()
    
    # 创建数据集
    print("初始化数据集...")
    t0 = time.time()
    dataset = CTWindowSparseSDF(
        roots=args.roots,
        resolution=args.resolution,
        window_type=args.window_type,
        min_points=100,
        max_points=100000,
        cache_data=args.cache_data,
        precompute_sparse=args.precompute_sparse,
    )
    init_time = time.time() - t0
    print(f"数据集初始化完成，耗时: {init_time:.2f}s")
    print()
    
    # 检查数据集是否为空
    if len(dataset) == 0:
        print("=" * 80)
        print("错误: 数据集为空！")
        print("=" * 80)
        print("\n可能的原因:")
        print("1. 路径不正确")
        print(f"   当前路径: {args.roots}")
        print(f"   期望结构: {{root}}/{{case_id}}/windows/{dataset.window_filename}")
        print("\n2. 没有找到窗口文件")
        print(f"   窗口文件名: {dataset.window_filename}")
        print("\n建议:")
        print("- 检查路径是否正确")
        print("- 确保数据已经预处理完成")
        print("- 使用绝对路径而不是相对路径")
        print("- 或者尝试父目录，例如: ./processed_dataset/0000")
        print("\n退出。")
        return
    
    # 解析要测试的worker数量
    test_workers = [int(w) for w in args.test_workers.split(',')]
    
    # 测试不同的worker配置
    results = []
    
    print("=" * 80)
    print("开始性能测试")
    print("=" * 80)
    
    for num_workers in test_workers:
        print(f"\n测试配置: num_workers={num_workers}")
        print("-" * 40)
        
        stats = benchmark_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            num_batches=args.num_batches,
            prefetch_factor=2,
            pin_memory=True
        )
        
        if stats:
            results.append({
                'num_workers': num_workers,
                **stats
            })
            
            print(f"  平均加载时间: {stats['avg_load_ms']:.1f} ms")
            print(f"  平均传输时间: {stats['avg_transfer_ms']:.1f} ms")
            print(f"  平均总时间: {stats['avg_total_ms']:.1f} ms")
            print(f"  加载时间范围: [{stats['min_load_ms']:.1f}, {stats['max_load_ms']:.1f}] ms")
            print(f"  加载时间标准差: {stats['std_load_ms']:.1f} ms")
            print(f"  吞吐量: {stats['throughput_batches_per_sec']:.2f} batches/s")
    
    # 显示总结
    print("\n" + "=" * 80)
    print("性能测试总结")
    print("=" * 80)
    
    if results:
        # 找到最佳配置
        best_result = min(results, key=lambda r: r['avg_total_ms'])
        
        print("\n所有配置对比:")
        print(f"{'Workers':<10} {'平均加载(ms)':<15} {'平均传输(ms)':<15} {'平均总时间(ms)':<15} {'吞吐量(b/s)':<15}")
        print("-" * 80)
        
        for r in results:
            marker = " ⭐" if r['num_workers'] == best_result['num_workers'] else ""
            print(f"{r['num_workers']:<10} {r['avg_load_ms']:<15.1f} {r['avg_transfer_ms']:<15.1f} "
                  f"{r['avg_total_ms']:<15.1f} {r['throughput_batches_per_sec']:<15.2f}{marker}")
        
        print(f"\n推荐配置: num_workers={best_result['num_workers']}")
        print(f"  预期性能: {best_result['avg_total_ms']:.1f} ms/batch")
        print(f"  预期吞吐: {best_result['throughput_batches_per_sec']:.2f} batches/s")
        
        # 给出优化建议
        print("\n优化建议:")
        if best_result['avg_load_ms'] > 1000:
            print("  ⚠️  数据加载时间较长 (>1s)，建议:")
            print("      - 启用 cache_data=True（如果内存足够）")
            print("      - 启用 precompute_sparse=True")
            print("      - 考虑使用预处理脚本: python tools/precompute_sparse_data.py")
        
        if best_result['avg_transfer_ms'] > 100:
            print("  ⚠️  GPU传输时间较长 (>100ms)，建议:")
            print("      - 检查是否使用了 pin_memory=True")
            print("      - 考虑减少batch_size")
        
        if best_result['std_load_ms'] > best_result['avg_load_ms'] * 0.5:
            print("  ⚠️  加载时间波动较大，建议:")
            print("      - 检查磁盘I/O（考虑使用SSD）")
            print("      - 增加num_workers以提供更多缓冲")
    
    print("\n完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

