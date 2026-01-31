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


def classify_mask_type(seg_array, unique_labels):
    """
    判断掩码类型
    
    返回字典包含:
    - type: 类型名称
    - description: 详细说明
    - warning: 警告信息（如果有）
    - ndim: 维度
    - num_channels: 通道数
    """
    shape = seg_array.shape
    ndim = seg_array.ndim
    dtype = seg_array.dtype
    
    result = {
        'type': 'Unknown',
        'description': '',
        'warning': None,
        'ndim': ndim,
        'num_channels': 1 if ndim == 3 else shape[0],
        'spatial_dims': shape[-3:] if ndim == 4 else shape
    }
    
    # 1. 三维数组（单通道）
    if ndim == 3:
        if dtype == bool or set(unique_labels).issubset({0, 1, False, True}):
            result['type'] = '单通道二值掩码'
            result['description'] = '3D二值掩码，值为0/1或False/True'
        elif len(unique_labels) > 2:
            result['type'] = '单通道多标签格式'
            result['description'] = f'3D标签数组，包含{len(unique_labels)}个不同标签值（含背景）'
        else:
            result['type'] = '单通道掩码'
            result['description'] = '3D掩码，标签值较少'
    
    # 2. 四维数组（多通道或带batch维度）
    elif ndim == 4:
        first_dim = shape[0]
        
        # 2.1 第一维度为1（batch=1或单通道）
        if first_dim == 1:
            if dtype == bool or set(unique_labels).issubset({0, 1, False, True}):
                result['type'] = '单通道二值掩码（4D）'
                result['description'] = '形状为(1,H,W,D)的二值掩码，第一维度可squeeze'
                result['warning'] = '建议squeeze掉第一维度'
            else:
                result['type'] = '单通道多标签格式（4D）'
                result['description'] = f'形状为(1,H,W,D)的标签数组，包含{len(unique_labels)}个标签'
                result['warning'] = '建议squeeze掉第一维度'
        
        # 2.2 第一维度 > 1（多通道）
        else:
            # 检查是否为布尔类型
            if dtype == bool or set(unique_labels).issubset({False, True}):
                result['type'] = '多通道布尔掩码'
                result['description'] = f'{first_dim}个通道，每个通道是一个布尔掩码（一个器官）'
                result['warning'] = '需要转换为单通道标签格式才能处理'
                result['num_channels'] = first_dim
            
            # 检查是否为one-hot编码
            elif set(unique_labels).issubset({0, 1}):
                # 检查每个空间位置是否只有一个通道为1
                sum_along_channel = seg_array.sum(axis=0)
                max_overlap = sum_along_channel.max()
                min_overlap = sum_along_channel.min()
                
                if max_overlap <= 1.1:
                    result['type'] = 'One-hot编码'
                    result['description'] = f'{first_dim}个通道，每个空间位置只有一个通道为1（互斥）'
                    result['warning'] = '需要使用argmax转换为单通道标签格式'
                else:
                    result['type'] = '多通道二值掩码（可重叠）'
                    result['description'] = f'{first_dim}个通道，通道间可能重叠（max_overlap={max_overlap}）'
                    result['warning'] = '器官可能有重叠，需要确认处理策略'
            
            # 其他多通道格式
            else:
                result['type'] = '多通道多标签格式'
                result['description'] = f'{first_dim}个通道，每个通道包含多个标签值'
                result['warning'] = '格式复杂，可能需要特殊处理'
    
    else:
        result['type'] = '未知维度格式'
        result['description'] = f'{ndim}维数组，不是标准的3D或4D格式'
        result['warning'] = '维度异常，无法处理'
    
    return result


def check_compatibility(mask_type):
    """
    检查与预处理脚本的兼容性
    """
    type_name = mask_type['type']
    
    compatibility_map = {
        '单通道二值掩码': '✓ 完全兼容',
        '单通道多标签格式': '✓ 完全兼容',
        '单通道二值掩码（4D）': '✓ 兼容（会自动squeeze）',
        '单通道多标签格式（4D）': '✓ 兼容（会自动squeeze）',
        '多通道布尔掩码': '✓ 兼容（修复后，会自动转换）',
        'One-hot编码': '✓ 兼容（会使用argmax转换）',
        '多通道二值掩码（可重叠）': '⚠ 可能兼容（需确认重叠处理方式）',
        '多通道多标签格式': '❌ 可能不兼容（格式复杂）',
        '未知维度格式': '❌ 不兼容',
    }
    
    return compatibility_map.get(type_name, '❓ 未知兼容性')


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
    mask_type_result = None  # 记录掩码类型
    
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
                        
                        # ========== 判断掩码类型 ==========
                        mask_type = classify_mask_type(seg_dense, unique_labels)
                        mask_type_result = mask_type  # 保存以便返回
                        
                        print(f"    >>> 掩码类型: {mask_type['type']}")
                        print(f"    >>> 说明: {mask_type['description']}")
                        if mask_type['warning']:
                            print(f"    ⚠ 警告: {mask_type['warning']}")
                        
                        # 检查兼容性
                        compatibility = check_compatibility(mask_type)
                        print(f"    >>> 预处理兼容性: {compatibility}")
                        # ===================================
                        
                        if len(unique_labels) == 1 and unique_labels[0] == 0:
                            print("    ⚠ 警告: 只有背景标签(0)，没有前景")
                            issues.append(f"{case_name}: mask只有背景")
                        
                        # 检查每个器官的体素数
                        for label in unique_labels:
                            if label == 0 or label == False:
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
    
    return issues, mask_type_result


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
    mask_types_summary = {}  # 统计掩码类型
    
    for i, case_name in enumerate(valid_cases, 1):
        case_path = os.path.join(dataset_path, case_name)
        
        if not args.errors_only:
            print(f"\n[{i}/{len(valid_cases)}]")
        
        issues, mask_type = check_case_data(case_path, case_name)
        
        # 统计掩码类型
        if mask_type:
            type_name = mask_type['type']
            if type_name not in mask_types_summary:
                mask_types_summary[type_name] = 0
            mask_types_summary[type_name] += 1
        
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
    
    # 显示掩码类型统计
    if mask_types_summary:
        print("\n掩码类型分布:")
        for mask_type, count in sorted(mask_types_summary.items(), key=lambda x: -x[1]):
            percentage = count / len(valid_cases) * 100
            print(f"  • {mask_type}: {count} 个病例 ({percentage:.1f}%)")
            
            # 显示兼容性
            compatibility = check_compatibility({'type': mask_type})
            print(f"    兼容性: {compatibility}")
    
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

