#!/usr/bin/env python3
"""
预计算稀疏数据脚本

将CT窗口数据从密集格式(.npy)转换为稀疏格式(.npz)，
这样可以大幅加快数据加载速度。

使用方法：
    python tools/precompute_sparse_data.py \
        --roots /path/to/processed/data \
        --window_type lung \
        --resolution 512

输出：
    为每个窗口文件生成对应的 *_sparse.npz 文件
"""

import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../dataset_toolkits'))

from ct_preprocessing.config import WINDOW_CONFIGS
from ct_preprocessing.window_processor import get_window_filename


def precompute_sparse_for_file(window_path: str, force: bool = False) -> dict:
    """
    为单个窗口文件预计算稀疏表示
    
    Args:
        window_path: 窗口数据文件路径
        force: 是否强制重新计算（即使稀疏文件已存在）
    
    Returns:
        统计信息字典
    """
    # 检查输出文件
    sparse_path = window_path.replace('.npy', '_sparse.npz')
    
    if os.path.exists(sparse_path) and not force:
        return {
            'status': 'skipped',
            'reason': 'already_exists',
            'sparse_path': sparse_path
        }
    
    try:
        # 加载窗口数据
        t0 = time.time()
        window_data = np.load(window_path)
        load_time = time.time() - t0
        
        # 转换为稀疏格式
        t1 = time.time()
        indices = np.stack(np.nonzero(window_data), axis=1)
        
        if len(indices) == 0:
            # 空数据
            return {
                'status': 'skipped',
                'reason': 'empty_data',
                'window_path': window_path
            }
        
        values = window_data[indices[:, 0], indices[:, 1], indices[:, 2]]
        sparse_time = time.time() - t1
        
        # 保存稀疏格式
        t2 = time.time()
        np.savez_compressed(
            sparse_path,
            indices=indices.astype(np.int16),  # 使用int16节省空间（最大32768）
            values=values.astype(np.float16)   # 使用float16节省空间
        )
        save_time = time.time() - t2
        
        # 统计信息
        original_size = os.path.getsize(window_path)
        sparse_size = os.path.getsize(sparse_path)
        compression_ratio = sparse_size / original_size
        
        return {
            'status': 'success',
            'window_path': window_path,
            'sparse_path': sparse_path,
            'num_points': len(indices),
            'original_size_mb': original_size / 1024 / 1024,
            'sparse_size_mb': sparse_size / 1024 / 1024,
            'compression_ratio': compression_ratio,
            'load_time_ms': load_time * 1000,
            'sparse_time_ms': sparse_time * 1000,
            'save_time_ms': save_time * 1000,
            'total_time_ms': (time.time() - t0) * 1000
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'window_path': window_path,
            'error': str(e)
        }


def discover_window_files(roots: list, window_filename: str) -> list:
    """
    发现所有窗口文件
    
    Args:
        roots: 根目录列表
        window_filename: 窗口文件名
    
    Returns:
        窗口文件路径列表
    """
    window_files = []
    
    for root in roots:
        root = os.path.expanduser(root)
        if not os.path.exists(root):
            print(f"警告: 根目录不存在: {root}")
            continue
        
        # 查找所有窗口文件
        pattern = os.path.join(root, '**/windows', window_filename)
        files = glob.glob(pattern, recursive=True)
        window_files.extend(files)
    
    return window_files


def main():
    parser = argparse.ArgumentParser(
        description='预计算CT窗口数据的稀疏表示',
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
        choices=list(WINDOW_CONFIGS.keys()),
        help='窗口类型'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=512,
        help='分辨率（用于显示，不影响计算）'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新计算（覆盖已存在的稀疏文件）'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='并行worker数量'
    )
    
    args = parser.parse_args()
    
    # 解析根目录
    roots = [r.strip() for r in args.roots.split(',')]
    
    # 获取窗口文件名
    window_filename = get_window_filename(args.window_type)
    
    print("=" * 80)
    print("预计算稀疏数据")
    print("=" * 80)
    print(f"根目录: {roots}")
    print(f"窗口类型: {args.window_type}")
    print(f"窗口文件名: {window_filename}")
    print(f"分辨率: {args.resolution}")
    print(f"强制重新计算: {args.force}")
    print(f"并行workers: {args.workers}")
    print()
    
    # 发现所有窗口文件
    print("正在搜索窗口文件...")
    window_files = discover_window_files(roots, window_filename)
    print(f"找到 {len(window_files)} 个窗口文件")
    print()
    
    if len(window_files) == 0:
        print("未找到窗口文件，退出")
        return
    
    # 处理每个文件
    results = []
    
    if args.workers > 1:
        # 多进程处理
        from multiprocessing import Pool
        from functools import partial
        
        # 使用partial而不是lambda，因为lambda无法pickle
        worker_func = partial(precompute_sparse_for_file, force=args.force)
        
        with Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap(worker_func, window_files),
                total=len(window_files),
                desc="处理中"
            ))
    else:
        # 单进程处理
        for window_file in tqdm(window_files, desc="处理中"):
            result = precompute_sparse_for_file(window_file, args.force)
            results.append(result)
    
    # 统计结果
    print("\n" + "=" * 80)
    print("处理结果统计")
    print("=" * 80)
    
    success_results = [r for r in results if r['status'] == 'success']
    skipped_results = [r for r in results if r['status'] == 'skipped']
    error_results = [r for r in results if r['status'] == 'error']
    
    print(f"成功: {len(success_results)}")
    print(f"跳过: {len(skipped_results)}")
    print(f"错误: {len(error_results)}")
    
    if success_results:
        print("\n成功处理的文件统计:")
        total_points = sum(r['num_points'] for r in success_results)
        total_original_size = sum(r['original_size_mb'] for r in success_results)
        total_sparse_size = sum(r['sparse_size_mb'] for r in success_results)
        avg_compression = np.mean([r['compression_ratio'] for r in success_results])
        avg_time = np.mean([r['total_time_ms'] for r in success_results])
        
        print(f"  总点数: {total_points:,}")
        print(f"  平均点数: {total_points / len(success_results):,.0f}")
        print(f"  原始总大小: {total_original_size:.2f} MB")
        print(f"  稀疏总大小: {total_sparse_size:.2f} MB")
        print(f"  平均压缩率: {avg_compression:.2%}")
        print(f"  节省空间: {total_original_size - total_sparse_size:.2f} MB")
        print(f"  平均处理时间: {avg_time:.1f} ms/file")
    
    if skipped_results:
        print(f"\n跳过的文件:")
        for r in skipped_results[:5]:  # 只显示前5个
            print(f"  - {r.get('window_path', 'unknown')}: {r['reason']}")
        if len(skipped_results) > 5:
            print(f"  ... 还有 {len(skipped_results) - 5} 个")
    
    if error_results:
        print(f"\n错误的文件:")
        for r in error_results:
            print(f"  - {r['window_path']}: {r['error']}")
    
    print("\n完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

