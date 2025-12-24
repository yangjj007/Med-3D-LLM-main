"""
CT数据可视化示例

演示如何使用可视化工具处理预处理后的CT数据
"""

import os
import sys
import argparse

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_toolkits.visualize_ct_dataset import visualize_ct_dataset


def example_single_case(dataset_root: str, case_id: str = "0000"):
    """
    示例1: 可视化单个病例
    
    Args:
        dataset_root: 数据集根目录
        case_id: 病例ID
    """
    print("\n" + "="*80)
    print("示例1: 可视化单个病例")
    print("="*80)
    
    case_path = os.path.join(dataset_root, "processed", case_id)
    
    if not os.path.exists(case_path):
        print(f"错误: 病例路径不存在: {case_path}")
        return
    
    # 可视化病例
    visualize_ct_dataset(case_path)
    
    print("\n✓ 示例1完成")


def example_custom_output(dataset_root: str, case_id: str = "0000", output_dir: str = None):
    """
    示例2: 指定自定义输出目录
    
    Args:
        dataset_root: 数据集根目录
        case_id: 病例ID
        output_dir: 自定义输出目录
    """
    print("\n" + "="*80)
    print("示例2: 指定自定义输出目录")
    print("="*80)
    
    case_path = os.path.join(dataset_root, "processed", case_id)
    
    if not os.path.exists(case_path):
        print(f"错误: 病例路径不存在: {case_path}")
        return
    
    if output_dir is None:
        output_dir = "./custom_visualization_output"
    
    # 可视化并指定输出目录
    visualize_ct_dataset(case_path, output_dir)
    
    print("\n✓ 示例2完成")


def example_batch_visualization(dataset_root: str, max_cases: int = 3):
    """
    示例3: 批量可视化多个病例
    
    Args:
        dataset_root: 数据集根目录
        max_cases: 最多处理的病例数
    """
    print("\n" + "="*80)
    print("示例3: 批量可视化多个病例")
    print("="*80)
    
    processed_dir = os.path.join(dataset_root, "processed")
    
    if not os.path.exists(processed_dir):
        print(f"错误: processed目录不存在: {processed_dir}")
        return
    
    # 查找所有病例
    cases = []
    for item in sorted(os.listdir(processed_dir)):
        item_path = os.path.join(processed_dir, item)
        if os.path.isdir(item_path):
            # 检查是否有CT数据
            ct_files = [
                os.path.join(item_path, 'ct_normalized_512.npy'),
                os.path.join(item_path, 'ct_normalized_1024.npy')
            ]
            if any(os.path.exists(f) for f in ct_files):
                cases.append(item_path)
    
    if not cases:
        print("未找到任何有效病例")
        return
    
    print(f"发现 {len(cases)} 个病例")
    print(f"将处理前 {min(max_cases, len(cases))} 个病例")
    
    # 处理每个病例
    success_count = 0
    failed_count = 0
    
    for i, case_path in enumerate(cases[:max_cases], 1):
        case_name = os.path.basename(case_path)
        print(f"\n[{i}/{min(max_cases, len(cases))}] 处理: {case_name}")
        
        try:
            visualize_ct_dataset(case_path)
            success_count += 1
            print(f"✓ {case_name} 完成")
        except Exception as e:
            failed_count += 1
            print(f"✗ {case_name} 失败: {e}")
    
    print("\n" + "-"*80)
    print(f"批量处理完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print("="*80)
    
    print("\n✓ 示例3完成")


def example_programmatic_visualization(dataset_root: str, case_id: str = "0000"):
    """
    示例4: 编程方式创建自定义可视化
    
    Args:
        dataset_root: 数据集根目录
        case_id: 病例ID
    """
    print("\n" + "="*80)
    print("示例4: 编程方式创建自定义可视化")
    print("="*80)
    
    import numpy as np
    from dataset_toolkits.visualize_ct_dataset import (
        create_slices_plot,
        create_3d_volume_plot,
        load_npy_data
    )
    
    case_path = os.path.join(dataset_root, "processed", case_id)
    
    if not os.path.exists(case_path):
        print(f"错误: 病例路径不存在: {case_path}")
        return
    
    # 加载CT数据
    ct_file = os.path.join(case_path, "ct_normalized_512.npy")
    if not os.path.exists(ct_file):
        ct_file = os.path.join(case_path, "ct_normalized_1024.npy")
    
    if not os.path.exists(ct_file):
        print("错误: 未找到CT数据文件")
        return
    
    print(f"加载CT数据: {ct_file}")
    ct_volume = load_npy_data(ct_file)
    
    if ct_volume is None:
        print("错误: 无法加载CT数据")
        return
    
    # 创建自定义输出目录
    custom_output = os.path.join(case_path, "custom_visualization")
    os.makedirs(custom_output, exist_ok=True)
    
    # 创建自定义切片位置的可视化
    print("创建自定义切片可视化...")
    custom_slices = [
        ct_volume.shape[0] // 4,  # 1/4位置
        ct_volume.shape[1] // 2,  # 中间位置
        ct_volume.shape[2] * 3 // 4  # 3/4位置
    ]
    
    slices_fig = create_slices_plot(ct_volume, f"自定义切片 - {case_id}", custom_slices)
    slices_path = os.path.join(custom_output, "custom_slices.html")
    slices_fig.write_html(slices_path)
    print(f"✓ 保存: {slices_path}")
    
    # 创建不同透明度的3D可视化
    print("创建自定义3D可视化...")
    volume_fig = create_3d_volume_plot(
        ct_volume, 
        f"自定义3D渲染 - {case_id}",
        opacity=0.2,
        colorscale='Viridis'
    )
    volume_path = os.path.join(custom_output, "custom_3d.html")
    volume_fig.write_html(volume_path)
    print(f"✓ 保存: {volume_path}")
    
    print(f"\n自定义可视化保存在: {custom_output}")
    print("\n✓ 示例4完成")


def main():
    parser = argparse.ArgumentParser(
        description='CT数据可视化示例',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例说明:

1. 可视化单个病例:
   python examples/visualize_ct_example.py --dataset_root /path/to/dataset --example 1

2. 指定自定义输出目录:
   python examples/visualize_ct_example.py --dataset_root /path/to/dataset --example 2

3. 批量可视化:
   python examples/visualize_ct_example.py --dataset_root /path/to/dataset --example 3

4. 编程方式自定义可视化:
   python examples/visualize_ct_example.py --dataset_root /path/to/dataset --example 4

5. 运行所有示例:
   python examples/visualize_ct_example.py --dataset_root /path/to/dataset --example all
        """
    )
    
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--case_id', type=str, default='0000',
                       help='病例ID (默认: 0000)')
    parser.add_argument('--example', type=str, default='1',
                       choices=['1', '2', '3', '4', 'all'],
                       help='要运行的示例 (1-4 或 all)')
    parser.add_argument('--max_cases', type=int, default=3,
                       help='批量处理的最大病例数 (默认: 3)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='自定义输出目录 (用于示例2)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CT数据可视化示例")
    print("="*80)
    print(f"数据集根目录: {args.dataset_root}")
    print(f"病例ID: {args.case_id}")
    print("="*80)
    
    # 检查数据集根目录
    if not os.path.exists(args.dataset_root):
        print(f"错误: 数据集根目录不存在: {args.dataset_root}")
        return
    
    # 运行指定的示例
    if args.example == 'all' or args.example == '1':
        example_single_case(args.dataset_root, args.case_id)
    
    if args.example == 'all' or args.example == '2':
        example_custom_output(args.dataset_root, args.case_id, args.output_dir)
    
    if args.example == 'all' or args.example == '3':
        example_batch_visualization(args.dataset_root, args.max_cases)
    
    if args.example == 'all' or args.example == '4':
        example_programmatic_visualization(args.dataset_root, args.case_id)
    
    print("\n" + "="*80)
    print("所有示例完成！")
    print("="*80)


if __name__ == '__main__':
    main()

