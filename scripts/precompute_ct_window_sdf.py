"""
预计算CT窗口数据的SDF表示
将二值化窗口数据转换为稀疏SDF格式
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置multiprocessing启动方法为spawn，避免CUDA fork问题
# 必须在主进程开始时设置
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# 仅导入配置，dense_voxel_to_sparse_sdf在子进程中导入避免CUDA初始化
from dataset_toolkits.ct_preprocessing.config import WINDOW_CONFIGS
from dataset_toolkits.ct_preprocessing.window_processor import get_window_filename


def process_single_case(case_dir, window_type, resolution=512, threshold_factor=4.0, organ_subdir=None):
    """
    处理单个case的窗口数据
    
    Args:
        case_dir: case目录路径
        window_type: 窗口类型
        resolution: 目标分辨率
        threshold_factor: UDF阈值因子
        organ_subdir: 器官子目录（如'liver', 'lung'等），None表示处理全局窗口
    
    Returns:
        处理结果字典
    """
    
    try:
        # 延迟导入，避免在主进程中初始化CUDA
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from trellis.utils.mesh_utils import dense_voxel_to_sparse_sdf
        from dataset_toolkits.ct_preprocessing.window_processor import get_window_filename
        
        # 获取窗口文件名
        window_filename = get_window_filename(window_type)
        
        case_id = os.path.basename(case_dir)
        
        # 根据是否指定器官子目录，确定窗口文件路径
        if organ_subdir:
            window_path = os.path.join(case_dir, 'organs', organ_subdir, window_filename)
            location = f'organs/{organ_subdir}'
        else:
            window_path = os.path.join(case_dir, 'windows', window_filename)
            location = 'windows'
        
        # 检查文件是否存在
        if not os.path.exists(window_path):
            return {
                'case_id': case_id,
                'location': location,
                'success': False,
                'error': f'Window file not found: {window_path}'
            }
        
        # 加载二值化窗口数据
        import numpy as np
        window_data = np.load(window_path)
        
        # 检查数据是否为空
        if window_data.sum() < 100:
            return {
                'case_id': case_id,
                'location': location,
                'success': False,
                'error': f'Window data too sparse (< 100 voxels)'
            }
        
        # 转换为SDF
        sdf_result = dense_voxel_to_sparse_sdf(
            window_data,
            resolution=resolution,
            threshold_factor=threshold_factor,
            marching_cubes_level=0.5
        )
        
        # 保存为npz文件（直接替换原文件，改扩展名为.npz）
        output_path = window_path.replace('.npy', '.npz')
        np.savez_compressed(
            output_path,
            sparse_sdf=sdf_result['sparse_sdf'],
            sparse_index=sdf_result['sparse_index'],
            resolution=np.array(sdf_result['resolution'])
        )
        
        return {
            'case_id': case_id,
            'location': location,
            'success': True,
            'num_points': len(sdf_result['sparse_index']),
            'output_path': output_path
        }
        
    except Exception as e:
        import traceback
        return {
            'case_id': case_id,
            'location': organ_subdir if organ_subdir else 'windows',
            'success': False,
            'error': f'{type(e).__name__}: {str(e)}',
            'traceback': traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(description='预计算CT窗口数据的SDF表示')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录（包含processed目录）')
    parser.add_argument('--window_type', type=str, default='all',
                        help='窗口类型（lung, bone, soft_tissue, brain, all）')
    parser.add_argument('--resolution', type=int, default=512,
                        help='目标分辨率')
    parser.add_argument('--threshold_factor', type=float, default=4.0,
                        help='UDF阈值因子')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='并行处理的worker数量')
    parser.add_argument('--force_recompute', action='store_true',
                        help='强制重新计算已存在的SDF文件')
    
    args = parser.parse_args()
    
    # 检查window_type是否有效
    valid_types = list(WINDOW_CONFIGS.keys()) + ['all']
    if args.window_type not in valid_types:
        print(f"❌ 错误: 无效的window_type '{args.window_type}'")
        print(f"可选值: {', '.join(valid_types)}")
        return
    
    # 如果是'all'，处理所有窗口类型（包括organs）
    if args.window_type == 'all':
        window_types = list(WINDOW_CONFIGS.keys())
        print(f"\n{'='*80}")
        print(f"将处理所有窗口类型: {', '.join(window_types)}")
        print(f"包含organs目录下的器官特定窗口")
        print(f"{'='*80}\n")
        
        for window_type in window_types:
            print(f"\n{'='*80}")
            print(f"开始处理窗口类型: {window_type}")
            print(f"{'='*80}\n")
            process_window_type(args.data_root, window_type, args.resolution, 
                              args.threshold_factor, args.max_workers, args.force_recompute,
                              include_organs=True)  # all模式下包含organs
        return
    
    # 处理单个窗口类型（不包含organs）
    process_window_type(args.data_root, args.window_type, args.resolution,
                       args.threshold_factor, args.max_workers, args.force_recompute,
                       include_organs=False)


def process_window_type(data_root, window_type, resolution, threshold_factor, max_workers, force_recompute, include_organs=False):
    """处理单个窗口类型的SDF预计算
    
    Args:
        data_root: 数据根目录
        window_type: 窗口类型
        resolution: 分辨率
        threshold_factor: 阈值因子
        max_workers: worker数量
        force_recompute: 是否强制重新计算
        include_organs: 是否包含organs目录下的窗口
    """
    
    print(f"\n{'='*80}")
    print(f"CT窗口SDF预计算")
    print(f"{'='*80}")
    print(f"数据根目录: {data_root}")
    print(f"窗口类型: {window_type}")
    print(f"分辨率: {resolution}")
    print(f"阈值因子: {threshold_factor}")
    print(f"并行Worker: {max_workers}")
    print(f"包含器官窗口: {'是' if include_organs else '否'}")
    print(f"{'='*80}\n")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ 错误: CUDA不可用！此脚本需要GPU支持。")
        return
    
    # 查找所有processed目录
    processed_dir = os.path.join(data_root, 'processed')
    if not os.path.exists(processed_dir):
        print(f"❌ 错误: processed目录不存在: {processed_dir}")
        return
    
    # 查找所有case目录
    case_dirs = [
        os.path.join(processed_dir, d)
        for d in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, d))
    ]
    
    print(f"找到 {len(case_dirs)} 个case目录")
    
    # 构建任务列表 (case_dir, organ_subdir)
    # organ_subdir=None 表示全局窗口，否则为器官特定窗口
    tasks = []
    window_filename = get_window_filename(window_type)
    
    for case_dir in case_dirs:
        # 1. 添加全局窗口任务
        global_window_path = os.path.join(case_dir, 'windows', window_filename)
        global_npz_path = global_window_path.replace('.npy', '.npz')
        
        if os.path.exists(global_window_path):
            if force_recompute or not os.path.exists(global_npz_path):
                tasks.append((case_dir, None))  # None表示全局窗口
        
        # 2. 添加器官特定窗口任务（如果启用）
        if include_organs:
            organs_dir = os.path.join(case_dir, 'organs')
            if os.path.exists(organs_dir):
                # 遍历所有器官子目录
                for organ_name in os.listdir(organs_dir):
                    organ_path = os.path.join(organs_dir, organ_name)
                    if os.path.isdir(organ_path):
                        organ_window_path = os.path.join(organ_path, window_filename)
                        organ_npz_path = organ_window_path.replace('.npy', '.npz')
                        
                        if os.path.exists(organ_window_path):
                            if force_recompute or not os.path.exists(organ_npz_path):
                                tasks.append((case_dir, organ_name))
    
    print(f"共收集到 {len(tasks)} 个待处理任务")
    if include_organs:
        global_tasks = sum(1 for _, organ in tasks if organ is None)
        organ_tasks = sum(1 for _, organ in tasks if organ is not None)
        print(f"  - 全局窗口: {global_tasks}")
        print(f"  - 器官窗口: {organ_tasks}")
    print()
    
    if len(tasks) == 0:
        print("✅ 所有任务已处理完成！")
        return
    
    # 并行处理
    results = []
    
    if max_workers == 1:
        # 单进程模式（方便调试）
        for case_dir, organ_subdir in tqdm(tasks, desc='处理进度'):
            result = process_single_case(
                case_dir, 
                window_type, 
                resolution, 
                threshold_factor,
                organ_subdir
            )
            results.append(result)
    else:
        # 多进程模式
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_case,
                    case_dir,
                    window_type,
                    resolution,
                    threshold_factor,
                    organ_subdir
                ): (case_dir, organ_subdir)
                for case_dir, organ_subdir in tasks
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc='处理进度'):
                results.append(future.result())
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count
    
    print(f"\n{'='*80}")
    print(f"处理完成")
    print(f"{'='*80}")
    print(f"总计: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    
    if success_count > 0:
        avg_points = np.mean([r['num_points'] for r in results if r['success']])
        print(f"平均点数: {avg_points:.0f}")
    
    # 打印失败的case
    if failed_count > 0:
        print(f"\n失败的case:")
        for r in results:
            if not r['success']:
                location_info = f" [{r['location']}]" if 'location' in r else ""
                print(f"  - {r['case_id']}{location_info}: {r['error']}")
    
    print(f"{'='*80}\n")
    
    # 保存结果日志
    results_df = pd.DataFrame(results)
    log_path = os.path.join(
        data_root, 
        f'sdf_precompute_{window_type}_log.csv'
    )
    results_df.to_csv(log_path, index=False)
    print(f"结果日志已保存: {log_path}")


if __name__ == '__main__':
    main()

