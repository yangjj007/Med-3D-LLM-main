"""
CT数据加载示例

演示如何加载和使用预处理后的CT数据。
"""

import numpy as np
import pandas as pd
from scipy import sparse
import json
import os
import sys
from pathlib import Path

def load_single_case(processed_dir, case_id, resolution=512):
    """
    加载单个病例的所有数据
    
    Args:
        processed_dir: 处理后的数据目录
        case_id: 病例ID
        resolution: 分辨率
    
    Returns:
        包含所有数据的字典
    """
    case_dir = os.path.join(processed_dir, 'processed', case_id)
    
    if not os.path.exists(case_dir):
        raise ValueError(f"Case directory not found: {case_dir}")
    
    data = {'case_id': case_id}
    
    # 1. 加载CT数据
    ct_normalized_path = os.path.join(case_dir, f'ct_normalized_{resolution}.npy')
    ct_original_path = os.path.join(case_dir, f'ct_original_{resolution}.npy')
    
    if os.path.exists(ct_normalized_path):
        data['ct_normalized'] = np.load(ct_normalized_path)
        print(f"✓ 加载标准化CT: {data['ct_normalized'].shape}")
    
    if os.path.exists(ct_original_path):
        data['ct_original'] = np.load(ct_original_path)
        print(f"✓ 加载原始CT: {data['ct_original'].shape}")
    
    # 2. 加载全局窗口
    windows_dir = os.path.join(case_dir, 'windows')
    if os.path.exists(windows_dir):
        data['windows'] = {}
        for window_file in os.listdir(windows_dir):
            if window_file.endswith('.npy'):
                window_name = window_file.replace('.npy', '')
                window_path = os.path.join(windows_dir, window_file)
                data['windows'][window_name] = np.load(window_path)
        print(f"✓ 加载 {len(data['windows'])} 个窗口")
    
    # 3. 加载器官数据
    organs_dir = os.path.join(case_dir, 'organs')
    if os.path.exists(organs_dir):
        data['organs'] = {}
        for organ_name in os.listdir(organs_dir):
            organ_dir = os.path.join(organs_dir, organ_name)
            if os.path.isdir(organ_dir):
                data['organs'][organ_name] = {}
                for organ_file in os.listdir(organ_dir):
                    if organ_file.endswith('.npy'):
                        organ_window_name = organ_file.replace('.npy', '')
                        organ_path = os.path.join(organ_dir, organ_file)
                        data['organs'][organ_name][organ_window_name] = np.load(organ_path)
        print(f"✓ 加载 {len(data['organs'])} 个器官")
    
    # 4. 加载分割掩码
    mask_path = os.path.join(case_dir, 'masks', 'segmentation_masks.npz')
    if os.path.exists(mask_path):
        seg_sparse = sparse.load_npz(mask_path)
        # 从info.json获取形状
        info_path = os.path.join(case_dir, 'info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            seg_shape = tuple(info['adapted_shape'])
        else:
            seg_shape = (resolution, resolution, resolution)
        
        data['segmentation'] = seg_sparse.toarray().reshape(seg_shape)
        print(f"✓ 加载分割掩码: {data['segmentation'].shape}")
    
    # 5. 加载元信息
    info_path = os.path.join(case_dir, 'info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            data['info'] = json.load(f)
        print(f"✓ 加载元信息")
    
    return data


def visualize_case(data, slice_idx=None):
    """
    可视化病例数据（需要matplotlib）
    
    Args:
        data: 病例数据字典
        slice_idx: 切片索引（如果为None，使用中间切片）
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要安装matplotlib: pip install matplotlib")
        return
    
    ct = data.get('ct_normalized', data.get('ct_original'))
    if ct is None:
        print("没有CT数据可视化")
        return
    
    if slice_idx is None:
        slice_idx = ct.shape[2] // 2
    
    # 确定子图数量
    num_plots = 1  # CT
    if 'windows' in data:
        num_plots += len(data['windows'])
    if 'segmentation' in data:
        num_plots += 1
    
    # 计算布局
    cols = min(4, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows * cols > 1 else [axes]
    
    plot_idx = 0
    
    # 绘制CT
    axes[plot_idx].imshow(ct[:, :, slice_idx], cmap='gray')
    axes[plot_idx].set_title(f'CT (切片 {slice_idx})')
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # 绘制窗口
    if 'windows' in data:
        for window_name, window_data in data['windows'].items():
            if plot_idx < len(axes):
                axes[plot_idx].imshow(window_data[:, :, slice_idx], cmap='gray')
                axes[plot_idx].set_title(window_name)
                axes[plot_idx].axis('off')
                plot_idx += 1
    
    # 绘制分割
    if 'segmentation' in data:
        if plot_idx < len(axes):
            seg_slice = data['segmentation'][:, :, slice_idx]
            axes[plot_idx].imshow(seg_slice, cmap='tab20')
            axes[plot_idx].set_title('分割标签')
            axes[plot_idx].axis('off')
            plot_idx += 1
    
    # 隐藏多余的子图
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"病例: {data['case_id']}", fontsize=16, y=1.02)
    plt.show()


def batch_loader(processed_dir, batch_size=4, resolution=512):
    """
    批量数据加载器（生成器）
    
    Args:
        processed_dir: 处理后的数据目录
        batch_size: 批量大小
        resolution: 分辨率
    
    Yields:
        批量数据
    """
    # 加载元数据
    metadata_path = os.path.join(processed_dir, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)
    
    batch_ct = []
    batch_ids = []
    
    for idx, row in metadata.iterrows():
        case_id = row['case_id']
        
        # 加载CT
        ct_path = os.path.join(
            processed_dir, 'processed', case_id, 
            f'ct_normalized_{resolution}.npy'
        )
        
        if os.path.exists(ct_path):
            ct = np.load(ct_path)
            batch_ct.append(ct)
            batch_ids.append(case_id)
        
        # 达到批量大小时返回
        if len(batch_ct) == batch_size:
            yield {
                'ct': np.stack(batch_ct),
                'case_ids': batch_ids
            }
            batch_ct = []
            batch_ids = []
    
    # 返回剩余数据
    if len(batch_ct) > 0:
        yield {
            'ct': np.stack(batch_ct),
            'case_ids': batch_ids
        }


def print_dataset_statistics(processed_dir):
    """
    打印数据集统计信息
    
    Args:
        processed_dir: 处理后的数据目录
    """
    print("=" * 60)
    print("数据集统计")
    print("=" * 60)
    
    # 加载元数据
    metadata_path = os.path.join(processed_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print("元数据文件不存在")
        return
    
    metadata = pd.read_csv(metadata_path)
    
    print(f"\n总病例数: {len(metadata)}")
    
    if 'has_segmentation' in metadata.columns:
        has_seg = metadata['has_segmentation'].sum()
        print(f"有分割标签: {has_seg} ({has_seg/len(metadata)*100:.1f}%)")
    
    if 'resolution' in metadata.columns:
        print(f"\n分辨率分布:")
        for res, count in metadata['resolution'].value_counts().sort_index().items():
            print(f"  {res}³: {count} 病例")
    
    if 'file_size_mb' in metadata.columns:
        total_size = metadata['file_size_mb'].sum()
        avg_size = metadata['file_size_mb'].mean()
        print(f"\n存储统计:")
        print(f"  总大小: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
        print(f"  平均大小: {avg_size:.2f} MB/病例")
    
    if 'processing_time_sec' in metadata.columns:
        total_time = metadata['processing_time_sec'].sum()
        avg_time = metadata['processing_time_sec'].mean()
        print(f"\n处理时间:")
        print(f"  总时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        print(f"  平均时间: {avg_time:.2f} 秒/病例")
    
    # 加载数据集配置
    config_path = os.path.join(processed_dir, 'dataset_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\n数据集配置:")
        print(f"  名称: {config.get('dataset_name', 'Unknown')}")
        print(f"  模态: {config.get('modality', 'Unknown')}")
        print(f"  处理日期: {config.get('processing_date', 'Unknown')}")
    
    print("=" * 60)


def main():
    """主函数 - 演示数据加载"""
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python ct_data_loader_example.py <processed_dir> [case_id]")
        print("\n示例:")
        print("  python ct_data_loader_example.py ./data/processed_ct")
        print("  python ct_data_loader_example.py ./data/processed_ct case_001")
        sys.exit(1)
    
    processed_dir = sys.argv[1]
    
    if not os.path.exists(processed_dir):
        print(f"错误: 目录不存在: {processed_dir}")
        sys.exit(1)
    
    # 打印数据集统计
    print_dataset_statistics(processed_dir)
    print()
    
    # 加载元数据
    metadata_path = os.path.join(processed_dir, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)
    
    # 确定要加载的病例
    if len(sys.argv) >= 3:
        case_id = sys.argv[2]
    else:
        # 加载第一个病例
        case_id = metadata.iloc[0]['case_id']
    
    print(f"加载病例: {case_id}")
    print("=" * 60)
    
    # 加载数据
    data = load_single_case(processed_dir, case_id)
    
    # 打印数据信息
    print("\n数据内容:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            if value.size > 0:
                print(f"         range=[{np.min(value):.4f}, {np.max(value):.4f}]")
        elif isinstance(value, dict):
            print(f"  {key}: {len(value)} 项")
            for sub_key in value.keys():
                print(f"    - {sub_key}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    # 可视化（可选）
    print("\n要可视化数据吗？(y/n): ", end='')
    try:
        response = input().strip().lower()
        if response == 'y':
            visualize_case(data)
    except:
        pass
    
    # 批量加载示例
    print("\n批量加载示例:")
    print("=" * 60)
    for i, batch in enumerate(batch_loader(processed_dir, batch_size=2)):
        print(f"批次 {i+1}:")
        print(f"  形状: {batch['ct'].shape}")
        print(f"  病例: {batch['case_ids']}")
        if i >= 2:  # 只展示前3个批次
            break


if __name__ == '__main__':
    main()

