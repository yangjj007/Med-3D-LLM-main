#!/usr/bin/env python
"""
M3D-Seg数据验证脚本

用于检查M3D-Seg格式数据的完整性和有效性
可以快速发现数据文件的问题
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
from scipy import sparse
import re
import ast


def check_json_config(dataset_path):
    """检查JSON配置文件"""
    print("\n" + "="*60)
    print("检查JSON配置文件")
    print("="*60)
    
    json_files = glob.glob(os.path.join(dataset_path, "*.json"))
    
    if not json_files:
        print("❌ 未找到JSON配置文件")
        return False
    
    for json_file in json_files:
        print(f"\n文件: {os.path.basename(json_file)}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"  ✓ 格式正确")
            print(f"  数据集名称: {config.get('name', 'N/A')}")
            
            if 'labels' in config:
                print(f"  器官标签数: {len(config['labels'])}")
                for label_id, label_name in config['labels'].items():
                    print(f"    {label_id}: {label_name}")
            else:
                print("  ⚠ 未找到labels字段")
            
            if 'train' in config:
                print(f"  训练样本数: {len(config['train'])}")
            if 'test' in config:
                print(f"  测试样本数: {len(config['test'])}")
                
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON格式错误: {e}")
            return False
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            return False
    
    return True


def check_case_data(case_path, case_name):
    """检查单个病例的数据"""
    print(f"\n{'='*60}")
    print(f"病例: {case_name}")
    print(f"路径: {case_path}")
    print('='*60)
    
    issues = []
    
    # 1. 检查image.npy
    print("\n1. 检查 image.npy")
    image_path = os.path.join(case_path, 'image.npy')
    
    if not os.path.exists(image_path):
        print("  ❌ 文件不存在")
        issues.append(f"{case_name}: image.npy 不存在")
    else:
        try:
            img = np.load(image_path)
            print(f"  ✓ 加载成功")
            print(f"  形状: {img.shape}")
            print(f"  数据类型: {img.dtype}")
            print(f"  值范围: [{img.min():.2f}, {img.max():.2f}]")
            print(f"  文件大小: {os.path.getsize(image_path) / 1024 / 1024:.2f} MB")
            
            # 检查异常值
            if np.any(np.isnan(img)):
                print("  ⚠ 警告: 包含NaN值")
                issues.append(f"{case_name}: image.npy 包含NaN")
            if np.any(np.isinf(img)):
                print("  ⚠ 警告: 包含Inf值")
                issues.append(f"{case_name}: image.npy 包含Inf")
                
            # 检查维度
            if img.ndim not in [3, 4]:
                print(f"  ⚠ 警告: 维度异常 (期望3或4维，实际{img.ndim}维)")
                issues.append(f"{case_name}: image.npy 维度异常")
                
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            issues.append(f"{case_name}: image.npy 加载失败 - {e}")
    
    # 2. 检查mask文件
    print("\n2. 检查 mask 文件")
    mask_files = glob.glob(os.path.join(case_path, 'mask_*.npz'))
    
    if not mask_files:
        print("  ❌ 未找到mask文件")
        issues.append(f"{case_name}: mask文件不存在")
    else:
        for mask_file in mask_files:
            mask_name = os.path.basename(mask_file)
            print(f"\n  文件: {mask_name}")
            
            try:
                # 从文件名解析形状
                shape_match = re.search(r'\(([\d,\s]+)\)', mask_name)
                if shape_match:
                    shape_str = shape_match.group(0)
                    expected_shape = ast.literal_eval(shape_str)
                    print(f"    文件名中的形状: {expected_shape}")
                else:
                    print("    ⚠ 无法从文件名解析形状")
                    expected_shape = None
                
                # 加载稀疏矩阵
                seg_sparse = sparse.load_npz(mask_file)
                print(f"    ✓ 加载成功")
                print(f"    稀疏矩阵形状: {seg_sparse.shape}")
                print(f"    非零元素数: {seg_sparse.nnz}")
                print(f"    稀疏度: {(1 - seg_sparse.nnz / np.prod(seg_sparse.shape)) * 100:.2f}%")
                print(f"    文件大小: {os.path.getsize(mask_file) / 1024 / 1024:.2f} MB")
                
                # 尝试转换为密集数组
                if expected_shape:
                    try:
                        seg_dense = seg_sparse.toarray().reshape(expected_shape)
                        print(f"    ✓ 成功重塑为: {seg_dense.shape}")
                        
                        # 检查标签值
                        unique_labels = np.unique(seg_dense)
                        print(f"    唯一标签值: {unique_labels}")
                        
                        if len(unique_labels) == 1 and unique_labels[0] == 0:
                            print("    ⚠ 警告: 只有背景标签(0)，没有前景")
                            issues.append(f"{case_name}: mask只有背景")
                        
                        # 检查每个器官的体素数
                        for label in unique_labels:
                            if label == 0:
                                continue
                            count = np.sum(seg_dense == label)
                            percentage = count / seg_dense.size * 100
                            print(f"    标签 {label}: {count} 个体素 ({percentage:.2f}%)")
                            
                    except Exception as e:
                        print(f"    ⚠ 重塑失败: {e}")
                        issues.append(f"{case_name}: mask重塑失败 - {e}")
                        
            except Exception as e:
                print(f"    ❌ 加载失败: {e}")
                issues.append(f"{case_name}: {mask_name} 加载失败 - {e}")
    
    return issues


def main():
    parser = argparse.ArgumentParser(
        description='M3D-Seg数据验证工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 验证单个数据集:
   python scripts/verify_m3d_data.py --dataset M3D_Seg/0008/0008

2. 验证并只显示错误:
   python scripts/verify_m3d_data.py --dataset M3D_Seg/0008/0008 --errors-only

3. 验证前3个病例:
   python scripts/verify_m3d_data.py --dataset M3D_Seg/0008/0008 --max-cases 3

4. 对比两个数据集:
   python scripts/verify_m3d_data.py --dataset M3D_Seg/0001/0001
   python scripts/verify_m3d_data.py --dataset M3D_Seg/0008/0008
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集路径')
    parser.add_argument('--max-cases', type=int, default=None,
                       help='最多检查的病例数（默认：全部）')
    parser.add_argument('--errors-only', action='store_true',
                       help='只显示有错误的病例')
    
    args = parser.parse_args()
    
    dataset_path = args.dataset
    
    print("="*60)
    print("M3D-Seg数据验证工具")
    print("="*60)
    print(f"数据集路径: {dataset_path}")
    print()
    
    # 检查目录是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 目录不存在: {dataset_path}")
        sys.exit(1)
    
    # 1. 检查JSON配置
    check_json_config(dataset_path)
    
    # 2. 查找所有病例
    print("\n" + "="*60)
    print("扫描病例")
    print("="*60)
    
    case_dirs = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    case_dirs.sort()
    
    print(f"找到 {len(case_dirs)} 个子目录")
    
    # 过滤出包含image.npy的目录
    valid_cases = []
    for case_dir in case_dirs:
        case_path = os.path.join(dataset_path, case_dir)
        if os.path.exists(os.path.join(case_path, 'image.npy')):
            valid_cases.append(case_dir)
    
    print(f"其中 {len(valid_cases)} 个包含 image.npy")
    
    if not valid_cases:
        print("\n❌ 未找到有效的病例数据")
        sys.exit(1)
    
    # 限制检查数量
    if args.max_cases:
        valid_cases = valid_cases[:args.max_cases]
        print(f"将检查前 {len(valid_cases)} 个病例")
    
    # 3. 检查每个病例
    all_issues = []
    
    for i, case_name in enumerate(valid_cases, 1):
        case_path = os.path.join(dataset_path, case_name)
        
        if not args.errors_only:
            print(f"\n[{i}/{len(valid_cases)}]")
        
        issues = check_case_data(case_path, case_name)
        if issues:
            all_issues.extend(issues)
            if args.errors_only:
                print(f"\n[{i}/{len(valid_cases)}] {case_name} - 发现问题:")
                for issue in issues:
                    print(f"  ❌ {issue}")
    
    # 4. 生成总结报告
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    print(f"检查的病例数: {len(valid_cases)}")
    print(f"发现的问题数: {len(all_issues)}")
    
    if all_issues:
        print("\n所有问题列表:")
        for issue in all_issues:
            print(f"  ❌ {issue}")
        print("\n⚠ 验证未通过，请修复上述问题")
        sys.exit(1)
    else:
        print("\n✓ 所有检查通过，数据完整!")
        sys.exit(0)


if __name__ == '__main__':
    main()

