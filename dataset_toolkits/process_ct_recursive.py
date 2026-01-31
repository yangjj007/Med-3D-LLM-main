"""
递归CT数据预处理脚本

支持递归扫描大文件夹，自动识别数据格式并处理：
1. NIfTI格式（imagesTr/labelsTr结构）
2. M3D-Seg格式（子文件夹包含image.npy和mask文件）

自动检测数据集类型并调用相应的处理脚本，统一输出格式。
"""

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import time
import multiprocessing

# 设置multiprocessing启动方法为spawn，避免CUDA fork问题
# 在多进程中使用CUDA时必须使用spawn模式
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass

# 导入处理模块
from process_medical_ct import process_dataset as process_nifti_dataset, scan_nifti_directory
from process_m3d_seg_format import process_m3d_seg_dataset, scan_m3d_seg_dataset


def detect_dataset_type(dataset_path: str) -> str:
    """
    检测数据集类型
    
    Args:
        dataset_path: 数据集路径
    
    Returns:
        'nifti', 'm3d_seg', 或 'unknown'
    """
    # 检查NIfTI格式（imagesTr/labelsTr结构）
    images_tr = os.path.join(dataset_path, 'imagesTr')
    if os.path.exists(images_tr) and os.path.isdir(images_tr):
        # 检查是否有.nii.gz文件
        import glob
        nifti_files = glob.glob(os.path.join(images_tr, '*.nii.gz'))
        if nifti_files:
            return 'nifti'
    
    # 检查M3D-Seg格式（子文件夹包含image.npy）
    has_image_npy = False
    has_json = False
    
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        
        # 检查JSON文件
        if item.endswith('.json'):
            has_json = True
        
        # 检查子文件夹是否包含image.npy
        if os.path.isdir(item_path):
            image_path = os.path.join(item_path, 'image.npy')
            if os.path.exists(image_path):
                has_image_npy = True
    
    if has_image_npy:
        return 'm3d_seg'
    
    return 'unknown'


def find_datasets_recursive(root_dir: str, max_depth: int = 5) -> List[Dict[str, str]]:
    """
    递归查找所有数据集
    
    Args:
        root_dir: 根目录
        max_depth: 最大递归深度
    
    Returns:
        数据集列表，每个元素包含路径和类型
    """
    datasets = []
    
    def scan_dir(current_path: str, depth: int):
        if depth > max_depth:
            return
        
        # 检测当前目录是否是数据集
        dataset_type = detect_dataset_type(current_path)
        
        if dataset_type != 'unknown':
            datasets.append({
                'path': current_path,
                'type': dataset_type,
                'name': os.path.basename(current_path)
            })
            # 找到数据集后，不再继续深入
            return
        
        # 继续递归扫描子目录
        try:
            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                if os.path.isdir(item_path):
                    # 跳过一些明显不是数据集的目录
                    if item.startswith('.') or item in ['__pycache__', 'node_modules']:
                        continue
                    scan_dir(item_path, depth + 1)
        except PermissionError:
            print(f"  警告: 无权限访问 {current_path}")
    
    print(f"开始递归扫描: {root_dir}")
    scan_dir(root_dir, 0)
    
    return datasets


def process_single_dataset(dataset_info: Dict[str, str],
                          output_base_dir: str,
                          default_resolution: int,
                          num_workers: int,
                          organ_mapping_file: str = None,
                          compute_sdf: bool = False,
                          sdf_resolution: int = 512,
                          sdf_threshold_factor: float = 4.0,
                          replace_npy: bool = False,
                          use_mask: bool = False,
                          skip_existing: bool = True) -> Dict:
    """
    处理单个数据集
    
    Args:
        dataset_info: 数据集信息
        output_base_dir: 输出基础目录
        default_resolution: 默认分辨率
        num_workers: 并行进程数
        organ_mapping_file: 器官映射文件（NIfTI格式需要）
        compute_sdf: 是否计算SDF
        sdf_resolution: SDF分辨率
        sdf_threshold_factor: SDF阈值因子
        replace_npy: 是否替换NPY文件
        use_mask: 是否使用掩码模式（跳过窗位窗宽处理）
        skip_existing: 是否跳过已处理的病例（断点续传）
    
    Returns:
        处理结果信息
    """
    dataset_path = dataset_info['path']
    dataset_type = dataset_info['type']
    dataset_name = dataset_info['name']
    
    print("\n" + "=" * 80)
    print(f"处理数据集: {dataset_name}")
    print(f"  路径: {dataset_path}")
    print(f"  类型: {dataset_type}")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = os.path.join(output_base_dir, dataset_name)
    
    start_time = time.time()
    
    try:
        if dataset_type == 'nifti':
            # 处理NIfTI格式
            print("使用NIfTI处理流程...")
            metadata = process_nifti_dataset(
                data_root=dataset_path,
                output_dir=output_dir,
                organ_mapping_file=organ_mapping_file,
                default_resolution=default_resolution,
                num_workers=num_workers,
                compute_sdf=compute_sdf,
                sdf_resolution=sdf_resolution,
                sdf_threshold_factor=sdf_threshold_factor,
                replace_npy=replace_npy,
                use_mask=use_mask,
                skip_existing=skip_existing
            )
            
        elif dataset_type == 'm3d_seg':
            # 处理M3D-Seg格式
            print("使用M3D-Seg处理流程...")
            metadata = process_m3d_seg_dataset(
                dataset_root=dataset_path,
                output_dir=output_dir,
                default_resolution=default_resolution,
                num_workers=num_workers,
                compute_sdf=compute_sdf,
                sdf_resolution=sdf_resolution,
                sdf_threshold_factor=sdf_threshold_factor,
                replace_npy=replace_npy,
                use_mask=use_mask,
                skip_existing=skip_existing
            )
        
        else:
            raise ValueError(f"未知数据集类型: {dataset_type}")
        
        processing_time = time.time() - start_time
        
        result = {
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'dataset_type': dataset_type,
            'output_dir': output_dir,
            'num_cases': len(metadata),
            'processing_time_sec': round(processing_time, 2),
            'status': 'success'
        }
        
        print(f"\n✓ 数据集处理成功: {dataset_name}")
        print(f"  病例数: {len(metadata)}")
        print(f"  耗时: {processing_time:.2f}秒")
        print(f"  输出: {output_dir}")
        
    except Exception as e:
        print(f"\n✗ 数据集处理失败: {dataset_name}")
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        
        result = {
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'dataset_type': dataset_type,
            'output_dir': output_dir,
            'num_cases': 0,
            'processing_time_sec': 0,
            'status': 'failed',
            'error': str(e)
        }
    
    return result


def process_recursive(root_dir: str,
                     output_base_dir: str,
                     default_resolution: int = 512,
                     num_workers: int = 4,
                     organ_mapping_file: str = None,
                     max_depth: int = 5,
                     compute_sdf: bool = False,
                     sdf_resolution: int = 512,
                     sdf_threshold_factor: float = 4.0,
                     replace_npy: bool = False,
                     use_mask: bool = False,
                     skip_existing: bool = True):
    """
    递归处理所有数据集
    
    Args:
        root_dir: 根目录
        output_base_dir: 输出基础目录
        default_resolution: 默认分辨率
        num_workers: 并行进程数
        organ_mapping_file: 器官映射文件
        max_depth: 最大递归深度
        compute_sdf: 是否计算SDF
        sdf_resolution: SDF分辨率
        sdf_threshold_factor: SDF阈值因子
        replace_npy: 是否替换NPY文件
        use_mask: 是否使用掩码模式（跳过窗位窗宽处理）
        skip_existing: 是否跳过已处理的病例（断点续传）
    """
    print("=" * 80)
    print("CT数据递归预处理")
    print("=" * 80)
    print(f"根目录: {root_dir}")
    print(f"输出目录: {output_base_dir}")
    print(f"最大深度: {max_depth}")
    print(f"并行进程: {num_workers}")
    print("=" * 80)
    
    # 查找所有数据集
    datasets = find_datasets_recursive(root_dir, max_depth)
    
    if not datasets:
        print("\n未找到任何数据集！")
        print("请检查：")
        print("  1. 路径是否正确")
        print("  2. 数据集格式是否符合要求")
        print("     - NIfTI格式: 包含imagesTr/labelsTr目录")
        print("     - M3D-Seg格式: 子文件夹包含image.npy和mask文件")
        return
    
    print(f"\n发现 {len(datasets)} 个数据集:")
    for i, ds in enumerate(datasets, 1):
        print(f"  {i}. {ds['name']} ({ds['type']})")
        print(f"     路径: {ds['path']}")
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 处理每个数据集
    results = []
    total_start_time = time.time()
    
    for i, dataset_info in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] 处理数据集...")
        
        result = process_single_dataset(
            dataset_info=dataset_info,
            output_base_dir=output_base_dir,
            default_resolution=default_resolution,
            num_workers=num_workers,
            organ_mapping_file=organ_mapping_file,
            compute_sdf=compute_sdf,
            sdf_resolution=sdf_resolution,
            sdf_threshold_factor=sdf_threshold_factor,
            replace_npy=replace_npy,
            use_mask=use_mask,
            skip_existing=skip_existing
        )
        
        results.append(result)
    
    total_time = time.time() - total_start_time
    
    # 生成总结报告
    print("\n" + "=" * 80)
    print("处理完成总结")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    total_cases = sum(r['num_cases'] for r in results)
    
    print(f"\n总体统计:")
    print(f"  数据集总数: {len(results)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  总病例数: {total_cases}")
    print(f"  总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    
    if success_count > 0:
        print(f"\n成功的数据集:")
        for r in results:
            if r['status'] == 'success':
                print(f"  ✓ {r['dataset_name']}: {r['num_cases']} 病例, {r['processing_time_sec']:.2f}秒")
                print(f"    输出: {r['output_dir']}")
    
    if failed_count > 0:
        print(f"\n失败的数据集:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  ✗ {r['dataset_name']}: {r.get('error', 'Unknown error')}")
    
    # 保存总结报告
    summary_path = os.path.join(output_base_dir, 'processing_summary.json')
    summary = {
        'root_dir': root_dir,
        'output_dir': output_base_dir,
        'total_datasets': len(results),
        'success_count': success_count,
        'failed_count': failed_count,
        'total_cases': total_cases,
        'total_time_sec': round(total_time, 2),
        'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n总结报告已保存: {summary_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='递归CT数据预处理 - 自动检测和处理多个数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 递归处理包含多个数据集的大文件夹:
   python dataset_toolkits/process_ct_recursive.py \\
       --root_dir /path/to/datasets_folder \\
       --output_dir ./processed_all

2. 指定器官映射（用于NIfTI格式）:
   python dataset_toolkits/process_ct_recursive.py \\
       --root_dir /path/to/datasets \\
       --output_dir ./processed \\
       --organ_labels ./organ_mapping.json

3. 控制递归深度和并行数:
   python dataset_toolkits/process_ct_recursive.py \\
       --root_dir /path/to/datasets \\
       --output_dir ./processed \\
       --max_depth 3 \\
       --num_workers 8

支持的数据格式:
- NIfTI格式: 包含imagesTr/labelsTr目录
- M3D-Seg格式: 子文件夹包含image.npy和mask_*.npz文件
        """
    )
    
    parser.add_argument('--root_dir', type=str, required=True,
                       help='根目录（包含多个数据集）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出基础目录')
    parser.add_argument('--organ_labels', type=str, default=None,
                       help='器官标签映射JSON文件（用于NIfTI格式）')
    parser.add_argument('--default_resolution', type=int, default=512,
                       choices=[512, 1024],
                       help='默认目标分辨率（默认: 512）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='并行进程数（默认: 4）')
    parser.add_argument('--max_depth', type=int, default=5,
                       help='最大递归深度（默认: 5）')
    parser.add_argument('--compute_sdf', action='store_true',
                       help='计算窗口数据的SDF表示（需要CUDA和TRELLIS）')
    parser.add_argument('--sdf_resolution', type=int, default=512,
                       help='SDF目标分辨率（默认: 512）')
    parser.add_argument('--sdf_threshold_factor', type=float, default=4.0,
                       help='SDF阈值因子（默认: 4.0）')
    parser.add_argument('--replace_npy', action='store_true',
                       help='用NPZ文件替换原NPY文件')
    parser.add_argument('--use_mask', action='store_true',
                       help='直接使用分割掩码生成二值化体素网格，跳过窗位窗宽处理')
    parser.add_argument('--no_skip', action='store_true',
                       help='不跳过已处理的病例，强制重新处理所有病例')
    
    args = parser.parse_args()
    
    # 检查根目录
    if not os.path.exists(args.root_dir):
        print(f"错误: 根目录不存在: {args.root_dir}")
        sys.exit(1)
    
    # 执行递归处理
    process_recursive(
        root_dir=args.root_dir,
        output_base_dir=args.output_dir,
        default_resolution=args.default_resolution,
        num_workers=args.num_workers,
        organ_mapping_file=args.organ_labels,
        max_depth=args.max_depth,
        compute_sdf=args.compute_sdf,
        sdf_resolution=args.sdf_resolution,
        sdf_threshold_factor=args.sdf_threshold_factor,
        replace_npy=args.replace_npy,
        use_mask=args.use_mask,
        skip_existing=not args.no_skip
    )
    
    print("\n全部完成！")


if __name__ == '__main__':
    main()

