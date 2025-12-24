"""
CT数据集3D可视化工具

为预处理后的CT数据生成交互式3D HTML可视化，支持：
- 3D体渲染
- 切片查看
- 器官分割可视化
- 窗口数据对比
- 可拖动、缩放的交互界面
"""

import os
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_npy_data(file_path: str) -> Optional[np.ndarray]:
    """加载.npy文件"""
    try:
        if os.path.exists(file_path):
            data = np.load(file_path)
            print(f"  ✓ 加载: {os.path.basename(file_path)} - 形状: {data.shape}")
            return data
        else:
            print(f"  ✗ 文件不存在: {file_path}")
            return None
    except Exception as e:
        print(f"  ✗ 加载失败 {file_path}: {e}")
        return None


def load_npz_masks(file_path: str) -> Optional[Dict[str, np.ndarray]]:
    """加载.npz分割掩码"""
    try:
        if os.path.exists(file_path):
            data = np.load(file_path)
            masks = {key: data[key] for key in data.files}
            print(f"  ✓ 加载掩码: {len(masks)} 个标签")
            return masks
        else:
            print(f"  ✗ 掩码文件不存在: {file_path}")
            return None
    except Exception as e:
        print(f"  ✗ 加载掩码失败: {e}")
        return None


def create_3d_volume_plot(volume: np.ndarray, title: str, opacity: float = 0.1, 
                          colorscale: str = 'Gray') -> go.Figure:
    """
    创建3D体渲染图
    
    Args:
        volume: 3D numpy数组
        title: 图表标题
        opacity: 透明度
        colorscale: 颜色映射
    """
    # 降采样以提高性能（每2个体素取1个）
    volume_ds = volume[::2, ::2, ::2]
    
    # 创建体渲染
    X, Y, Z = np.mgrid[0:volume_ds.shape[0], 
                        0:volume_ds.shape[1], 
                        0:volume_ds.shape[2]]
    
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume_ds.flatten(),
        isomin=volume_ds.min(),
        isomax=volume_ds.max(),
        opacity=opacity,
        surface_count=15,
        colorscale=colorscale,
        showscale=True,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    
    return fig


def create_slices_plot(volume: np.ndarray, title: str, 
                       slice_indices: Optional[List[int]] = None) -> go.Figure:
    """
    创建三个正交切片的可视化
    
    Args:
        volume: 3D numpy数组
        title: 图表标题
        slice_indices: [x, y, z] 切片索引，None则取中间切片
    """
    if slice_indices is None:
        slice_indices = [s // 2 for s in volume.shape]
    
    x_slice, y_slice, z_slice = slice_indices
    
    # 创建子图
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(f'矢状面 (X={x_slice})', 
                       f'冠状面 (Y={y_slice})', 
                       f'横断面 (Z={z_slice})'),
        horizontal_spacing=0.05,
    )
    
    # 矢状面 (YZ平面)
    fig.add_trace(
        go.Heatmap(z=volume[x_slice, :, :].T, colorscale='Gray', showscale=False),
        row=1, col=1
    )
    
    # 冠状面 (XZ平面)
    fig.add_trace(
        go.Heatmap(z=volume[:, y_slice, :].T, colorscale='Gray', showscale=False),
        row=1, col=2
    )
    
    # 横断面 (XY平面)
    fig.add_trace(
        go.Heatmap(z=volume[:, :, z_slice].T, colorscale='Gray', showscale=True),
        row=1, col=3
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=400,
        showlegend=False,
    )
    
    # 更新所有子图的坐标轴
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig


def create_organ_surface_plot(mask: np.ndarray, organ_name: str, 
                               color: str = 'red', opacity: float = 0.5) -> go.Figure:
    """
    创建器官的3D表面渲染
    
    Args:
        mask: 二值掩码 (0/1)
        organ_name: 器官名称
        color: 颜色
        opacity: 透明度
    """
    from skimage import measure
    
    # 降采样
    mask_ds = mask[::2, ::2, ::2]
    
    # 提取等值面
    try:
        verts, faces, _, _ = measure.marching_cubes(mask_ds.astype(float), level=0.5)
        
        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=opacity,
                name=organ_name,
            )
        ])
        
        fig.update_layout(
            title=dict(text=f'{organ_name} - 3D表面渲染', x=0.5, xanchor='center'),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
        )
        
        return fig
    except Exception as e:
        print(f"  警告: 无法为{organ_name}生成表面渲染: {e}")
        return None


def create_multi_window_comparison(windows_data: Dict[str, np.ndarray], 
                                   slice_idx: Optional[int] = 10) -> go.Figure:
    """
    创建多窗口对比图
    
    Args:
        windows_data: 字典 {窗口名称: 3D数组}
        slice_idx: Z轴切片索引
    """
    n_windows = len(windows_data)
    if n_windows == 0:
        return None
    
    # 获取中间切片
    first_volume = list(windows_data.values())[0]
    if slice_idx is None:
        slice_idx = first_volume.shape[2] // 2
    
    # 创建子图
    cols = min(4, n_windows)
    rows = (n_windows + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=list(windows_data.keys()),
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )
    
    for idx, (window_name, volume) in enumerate(windows_data.items()):
        row = idx // cols + 1
        col = idx % cols + 1
        
        fig.add_trace(
            go.Heatmap(
                z=volume[:, :, slice_idx].T, 
                colorscale='Gray', 
                showscale=(idx == n_windows - 1)
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=dict(text=f'多窗口对比 (Z={slice_idx})', x=0.5, xanchor='center'),
        height=300 * rows,
        showlegend=False,
    )
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig


def create_overview_dashboard(ct_volume: np.ndarray, 
                              windows_data: Dict[str, np.ndarray],
                              case_name: str) -> go.Figure:
    """
    创建总览仪表板
    """
    # 获取中间切片
    z_mid = ct_volume.shape[2] // 2
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'CT - 横断面', 'CT - 冠状面', 'CT - 矢状面',
            'CT分布直方图', '数据统计', '窗口对比'
        ),
        specs=[
            [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
            [{'type': 'histogram'}, {'type': 'table'}, {'type': 'bar'}]
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )
    
    # 三个正交切片
    fig.add_trace(
        go.Heatmap(z=ct_volume[:, :, z_mid].T, colorscale='Gray', showscale=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=ct_volume[:, ct_volume.shape[1]//2, :].T, colorscale='Gray', showscale=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=ct_volume[ct_volume.shape[0]//2, :, :].T, colorscale='Gray', showscale=False),
        row=1, col=3
    )
    
    # CT值分布直方图
    fig.add_trace(
        go.Histogram(x=ct_volume.flatten(), nbinsx=100, name='CT值分布'),
        row=2, col=1
    )
    
    # 数据统计表格
    stats_data = [
        ['形状', f'{ct_volume.shape}'],
        ['最小值', f'{ct_volume.min():.2f}'],
        ['最大值', f'{ct_volume.max():.2f}'],
        ['均值', f'{ct_volume.mean():.2f}'],
        ['标准差', f'{ct_volume.std():.2f}'],
        ['非零体素', f'{np.count_nonzero(ct_volume)}']
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['属性', '值'], fill_color='paleturquoise'),
            cells=dict(values=list(zip(*stats_data)), fill_color='lavender')
        ),
        row=2, col=2
    )
    
    # 窗口数据统计
    if windows_data:
        window_names = list(windows_data.keys())[:5]  # 最多5个
        window_means = [windows_data[w].mean() for w in window_names]
        
        fig.add_trace(
            go.Bar(x=window_names, y=window_means, name='窗口均值'),
            row=2, col=3
        )
    
    fig.update_layout(
        title=dict(text=f'数据集总览 - {case_name}', x=0.5, xanchor='center', font=dict(size=20)),
        height=900,
        showlegend=False,
    )
    
    fig.update_xaxes(showticklabels=False, row=1)
    fig.update_yaxes(showticklabels=False, row=1)
    
    return fig


def visualize_ct_dataset(dataset_path: str, output_dir: Optional[str] = None):
    """
    主可视化函数
    
    Args:
        dataset_path: 数据集路径，例如 /processed_dataset/processed/0000
        output_dir: 输出目录，默认为 dataset_path/visualization
    """
    print("=" * 80)
    print("CT数据集3D可视化")
    print("=" * 80)
    print(f"数据集路径: {dataset_path}")
    
    # 检查路径
    if not os.path.exists(dataset_path):
        print(f"错误: 路径不存在: {dataset_path}")
        return
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(dataset_path, 'visualization')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    
    case_name = os.path.basename(dataset_path)
    
    # 1. 加载主CT数据
    print("\n[1/5] 加载CT数据...")
    ct_files = [
        'ct_normalized_512.npy',
        'ct_normalized_1024.npy'
    ]
    ct_volume = None
    for ct_file in ct_files:
        ct_path = os.path.join(dataset_path, ct_file)
        ct_volume = load_npy_data(ct_path)
        if ct_volume is not None:
            break
    
    if ct_volume is None:
        print("错误: 未找到CT数据文件")
        return
    
    # 2. 加载窗口数据
    print("\n[2/5] 加载窗口数据...")
    windows_dir = os.path.join(dataset_path, 'windows')
    windows_data = {}
    
    if os.path.exists(windows_dir):
        for window_file in os.listdir(windows_dir):
            if window_file.endswith('.npy'):
                window_path = os.path.join(windows_dir, window_file)
                window_name = window_file.replace('.npy', '')
                window_volume = load_npy_data(window_path)
                if window_volume is not None:
                    windows_data[window_name] = window_volume
    else:
        print("  注意: 未找到windows目录")
    
    # 3. 加载器官数据
    print("\n[3/5] 加载器官数据...")
    organs_dir = os.path.join(dataset_path, 'organs')
    organs_data = {}
    
    if os.path.exists(organs_dir):
        for organ_name in os.listdir(organs_dir):
            organ_dir = os.path.join(organs_dir, organ_name)
            if os.path.isdir(organ_dir):
                # 查找器官的二值掩码
                for file in os.listdir(organ_dir):
                    if 'binary' in file and file.endswith('.npy'):
                        organ_path = os.path.join(organ_dir, file)
                        organ_mask = load_npy_data(organ_path)
                        if organ_mask is not None:
                            organs_data[organ_name] = organ_mask
                        break
    else:
        print("  注意: 未找到organs目录")
    
    # 4. 加载分割掩码
    print("\n[4/5] 加载分割掩码...")
    masks_dir = os.path.join(dataset_path, 'masks')
    segmentation_masks = None
    
    if os.path.exists(masks_dir):
        for mask_file in os.listdir(masks_dir):
            if mask_file.endswith('.npz'):
                mask_path = os.path.join(masks_dir, mask_file)
                segmentation_masks = load_npz_masks(mask_path)
                break
    else:
        print("  注意: 未找到masks目录")
    
    # 5. 生成可视化
    print("\n[5/5] 生成可视化...")
    print("-" * 80)
    
    # 5.1 总览仪表板
    print("  生成: 1. 总览仪表板...")
    overview_fig = create_overview_dashboard(ct_volume, windows_data, case_name)
    overview_path = os.path.join(output_dir, '01_overview_dashboard.html')
    overview_fig.write_html(overview_path)
    print(f"    ✓ 保存: {overview_path}")
    
    # 5.2 CT三切片视图
    print("  生成: 2. CT三切片视图...")
    slices_fig = create_slices_plot(ct_volume, f'CT数据 - {case_name}')
    slices_path = os.path.join(output_dir, '02_ct_slices.html')
    slices_fig.write_html(slices_path)
    print(f"    ✓ 保存: {slices_path}")
    
    # 5.3 CT 3D体渲染
    print("  生成: 3. CT 3D体渲染...")
    volume_fig = create_3d_volume_plot(ct_volume, f'CT 3D体渲染 - {case_name}', 
                                       opacity=0.1, colorscale='Gray')
    volume_path = os.path.join(output_dir, '03_ct_3d_volume.html')
    volume_fig.write_html(volume_path)
    print(f"    ✓ 保存: {volume_path}")
    
    # 5.4 窗口对比
    if windows_data:
        print(f"  生成: 4. 窗口对比 ({len(windows_data)}个窗口)...")
        windows_fig = create_multi_window_comparison(windows_data)
        if windows_fig:
            windows_path = os.path.join(output_dir, '04_windows_comparison.html')
            windows_fig.write_html(windows_path)
            print(f"    ✓ 保存: {windows_path}")
    
    # 5.5 器官3D表面渲染
    if organs_data:
        print(f"  生成: 5. 器官3D可视化 ({len(organs_data)}个器官)...")
        organ_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
        
        for idx, (organ_name, organ_mask) in enumerate(organs_data.items()):
            color = organ_colors[idx % len(organ_colors)]
            print(f"    处理器官: {organ_name}...")
            
            organ_fig = create_organ_surface_plot(organ_mask, organ_name, color=color)
            if organ_fig:
                organ_path = os.path.join(output_dir, f'05_organ_{organ_name}_3d.html')
                organ_fig.write_html(organ_path)
                print(f"      ✓ 保存: {organ_path}")
    
    # 5.6 生成索引页面
    print("  生成: 6. 索引页面...")
    create_index_html(output_dir, case_name, windows_data, organs_data)
    print(f"    ✓ 保存: {os.path.join(output_dir, 'index.html')}")
    
    print("-" * 80)
    print("\n" + "=" * 80)
    print("✓ 可视化完成！")
    print("=" * 80)
    print(f"\n打开以下文件查看可视化结果:")
    print(f"  {os.path.join(output_dir, 'index.html')}")
    print("\n或者在浏览器中打开:")
    index_path = os.path.abspath(os.path.join(output_dir, 'index.html'))
    print(f"  file://{index_path}")
    print("=" * 80)


def create_index_html(output_dir: str, case_name: str, 
                     windows_data: Dict, organs_data: Dict):
    """创建索引页面"""
    import datetime
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CT数据集可视化 - {case_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            font-size: 1.8em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .card {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            transition: transform 0.3s, box-shadow 0.3s;
            border: 2px solid #e9ecef;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
        }}
        .card h3 {{
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .card p {{
            color: #6c757d;
            line-height: 1.6;
            margin-bottom: 15px;
        }}
        .card a {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: bold;
            transition: transform 0.2s;
        }}
        .card a:hover {{
            transform: scale(1.05);
        }}
        .info-box {{
            background: #e7f3ff;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .info-box h3 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        .info-box ul {{
            list-style: none;
            padding-left: 0;
        }}
        .info-box li {{
            padding: 5px 0;
            color: #495057;
        }}
        .info-box li:before {{
            content: "✓ ";
            color: #667eea;
            font-weight: bold;
            margin-right: 8px;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 CT数据集3D可视化</h1>
            <p>病例: {case_name}</p>
        </div>
        
        <div class="content">
            <div class="info-box">
                <h3>📊 数据集信息</h3>
                <ul>
                    <li>病例编号: {case_name}</li>
                    <li>窗口数量: {len(windows_data)}</li>
                    <li>器官数量: {len(organs_data)}</li>
                    <li>可视化类型: 3D交互式HTML</li>
                </ul>
            </div>
            
            <!-- 主要可视化 -->
            <div class="section">
                <h2>🎯 主要可视化</h2>
                <div class="grid">
                    <div class="card">
                        <h3>📊 总览仪表板</h3>
                        <p>包含切片视图、统计信息和数据分布的综合仪表板。</p>
                        <a href="01_overview_dashboard.html" target="_blank">打开查看 →</a>
                    </div>
                    
                    <div class="card">
                        <h3>🔍 三切片视图</h3>
                        <p>矢状面、冠状面和横断面的CT切片可视化。</p>
                        <a href="02_ct_slices.html" target="_blank">打开查看 →</a>
                    </div>
                    
                    <div class="card">
                        <h3>🎮 3D体渲染</h3>
                        <p>CT数据的交互式3D体渲染，可旋转、缩放。</p>
                        <a href="03_ct_3d_volume.html" target="_blank">打开查看 →</a>
                    </div>
                </div>
            </div>
"""
    
    # 添加窗口可视化部分
    if windows_data:
        html_content += f"""
            <div class="section">
                <h2>🪟 窗口对比</h2>
                <div class="grid">
                    <div class="card">
                        <h3>多窗口对比</h3>
                        <p>肺窗、骨窗、软组织窗等{len(windows_data)}个窗口的对比视图。</p>
                        <a href="04_windows_comparison.html" target="_blank">打开查看 →</a>
                    </div>
                </div>
            </div>
"""
    
    # 添加器官可视化部分
    if organs_data:
        html_content += """
            <div class="section">
                <h2>🫀 器官3D渲染</h2>
                <div class="grid">
"""
        
        for organ_name in organs_data.keys():
            html_content += f"""
                    <div class="card">
                        <h3>{organ_name.capitalize()}</h3>
                        <p>{organ_name}的3D表面渲染，可交互查看。</p>
                        <a href="05_organ_{organ_name}_3d.html" target="_blank">打开查看 →</a>
                    </div>
"""
        
        html_content += """
                </div>
            </div>
"""
    
    html_content += f"""
            <div class="section">
                <h2>📖 使用说明</h2>
                <div class="info-box">
                    <h3>交互操作</h3>
                    <ul>
                        <li>鼠标拖动: 旋转3D视图</li>
                        <li>滚轮滚动: 缩放视图</li>
                        <li>双击: 重置视图</li>
                        <li>工具栏: 平移、缩放、保存图片等功能</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🔬 TRELLIS Med-3D-LLM CT数据预处理可视化工具</p>
            <p>生成时间: {current_time}</p>
        </div>
    </div>
</body>
</html>
"""
    
    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(
        description='CT数据集3D可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 可视化单个病例:
   python dataset_toolkits/visualize_ct_dataset.py \\
       --dataset_path /processed_dataset/processed/0000

2. 指定输出目录:
   python dataset_toolkits/visualize_ct_dataset.py \\
       --dataset_path /processed_dataset/processed/0000 \\
       --output_dir /custom/output/path

3. 使用bash脚本:
   bash scripts/visualize_ct.sh /processed_dataset/processed/0000

特性:
- 🎮 3D交互式可视化（可拖动、缩放）
- 📊 多种视图（切片、体渲染、表面渲染）
- 🪟 窗口对比（肺窗、骨窗等）
- 🫀 器官分割可视化
- 🌐 独立HTML文件，无需服务器
        """
    )
    
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='数据集路径，例如: /processed_dataset/processed/0000')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认: dataset_path/visualization）')
    
    args = parser.parse_args()
    
    # 检查必要的依赖
    try:
        import plotly
        import skimage
    except ImportError as e:
        print("错误: 缺少必要的依赖包")
        print("\n请安装以下依赖:")
        print("  pip install plotly scikit-image kaleido")
        sys.exit(1)
    
    # 执行可视化
    visualize_ct_dataset(args.dataset_path, args.output_dir)


if __name__ == '__main__':
    main()

