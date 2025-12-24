#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取Direct3D-S2模型的VAE权重
从HuggingFace预训练模型中提取VAE组件并单独保存
"""

import os
import sys
import argparse
from typing import Optional

# 延迟导入torch，提供更好的错误处理
def check_dependencies():
    """检查并导入必要的依赖"""
    try:
        import torch
        return True
    except OSError as e:
        if "DLL" in str(e) or "1114" in str(e):
            print("=" * 80)
            print("❌ PyTorch DLL加载失败")
            print("=" * 80)
            print("\n这是Windows系统上的常见问题，请尝试以下解决方案：")
            print("\n解决方案1: 安装Visual C++ Redistributable")
            print("  下载并安装: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("\n解决方案2: 重新安装PyTorch")
            print("  pip uninstall torch torchvision")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print("\n解决方案3: 使用CPU版本的PyTorch")
            print("  pip uninstall torch torchvision")
            print("  pip install torch torchvision")
            print("\n解决方案4: 检查系统环境")
            print("  - 确保已安装最新的Windows更新")
            print("  - 确保CUDA版本与PyTorch版本匹配（如果使用GPU）")
            print("\n" + "=" * 80)
            sys.exit(1)
        else:
            raise
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("\n请安装必要的依赖:")
        print("  pip install torch omegaconf huggingface_hub")
        sys.exit(1)

check_dependencies()

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf


def download_model_files(repo_id: str, subfolder: str, save_dir: str = "./downloaded_models"):
    """
    从HuggingFace下载模型文件
    
    Args:
        repo_id: HuggingFace仓库ID
        subfolder: 子文件夹名称
        save_dir: 保存目录
    
    Returns:
        包含所有文件路径的字典
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"📥 正在从 {repo_id}/{subfolder} 下载模型文件...")
    
    files = {
        'config': 'config.yaml',
        'dense': 'model_dense.ckpt',
        'sparse_512': 'model_sparse_512.ckpt',
        'sparse_1024': 'model_sparse_1024.ckpt',
    }
    
    file_paths = {}
    for key, filename in files.items():
        print(f"  ⬇️  下载 {filename}...")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                subfolder=subfolder,
                filename=filename,
                repo_type="model",
                cache_dir=save_dir
            )
            file_paths[key] = path
            print(f"  ✅ {filename} 下载完成")
        except Exception as e:
            print(f"  ⚠️  {filename} 下载失败: {e}")
            file_paths[key] = None
    
    return file_paths


def extract_vae_from_checkpoint(checkpoint_path: str, vae_name: str):
    """
    从检查点文件中提取VAE权重
    
    Args:
        checkpoint_path: 检查点文件路径
        vae_name: VAE名称（用于日志）
    
    Returns:
        VAE的state_dict
    """
    print(f"\n📦 加载 {vae_name} 从 {os.path.basename(checkpoint_path)}...")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # 提取VAE权重
    if 'vae' in checkpoint:
        vae_state_dict = checkpoint['vae']
        print(f"  ✅ 成功提取 {vae_name} 权重")
        print(f"  📊 权重参数数量: {len(vae_state_dict)} 个张量")
        
        # 显示权重结构信息
        total_params = sum(p.numel() for p in vae_state_dict.values())
        print(f"  📊 总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
        
        # 显示主要组件
        encoder_keys = [k for k in vae_state_dict.keys() if k.startswith('encoder.')]
        decoder_keys = [k for k in vae_state_dict.keys() if k.startswith('decoder.')]
        print(f"  🔧 Encoder 参数: {len(encoder_keys)} 个")
        print(f"  🔧 Decoder 参数: {len(decoder_keys)} 个")
        
        return vae_state_dict
    else:
        print(f"  ❌ 错误: 检查点中未找到 'vae' 键")
        print(f"  可用的键: {list(checkpoint.keys())}")
        return None


def save_vae_weights(vae_state_dict, output_path: str, vae_info: Optional[dict] = None):
    """
    保存VAE权重到文件
    
    Args:
        vae_state_dict: VAE的state_dict
        output_path: 输出文件路径
        vae_info: VAE的配置信息（可选）
    """
    if vae_state_dict is None:
        print(f"  ⚠️  跳过保存 {output_path}（权重为空）")
        return
    
    print(f"💾 保存VAE权重到 {output_path}...")
    
    # 准备保存的数据
    save_data = {
        'vae': vae_state_dict,
    }
    
    # 添加配置信息
    if vae_info:
        save_data['config'] = vae_info
    
    # 保存
    torch.save(save_data, output_path)
    
    # 检查文件大小
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"  ✅ 保存成功! 文件大小: {file_size:.2f} MB")


def load_and_extract_all_vaes(repo_id: str = "wushuang98/Direct3D-S2",
                               subfolder: str = "direct3d-s2-v-1-1",
                               output_dir: str = "./vae_weights",
                               local_path: Optional[str] = None):
    """
    从HuggingFace或本地路径加载并提取所有VAE权重
    
    Args:
        repo_id: HuggingFace仓库ID
        subfolder: 子文件夹名称
        output_dir: 输出目录
        local_path: 本地模型文件路径（如果提供则不从HF下载）
    """
    print("=" * 80)
    print("🚀 Direct3D-S2 VAE权重提取工具")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取模型文件路径
    if local_path and os.path.exists(local_path):
        print(f"📂 使用本地模型文件: {local_path}")
        file_paths = {
            'config': os.path.join(local_path, subfolder, 'config.yaml'),
            'dense': os.path.join(local_path, subfolder, 'model_dense.ckpt'),
            'sparse_512': os.path.join(local_path, subfolder, 'model_sparse_512.ckpt'),
            'sparse_1024': os.path.join(local_path, subfolder, 'model_sparse_1024.ckpt'),
        }
    else:
        # 从HuggingFace下载
        file_paths = download_model_files(repo_id, subfolder)
    
    # 加载配置文件
    config = None
    if file_paths.get('config') and os.path.exists(file_paths['config']):
        print(f"\n📄 加载配置文件...")
        config = OmegaConf.load(file_paths['config'])
        print("  ✅ 配置文件加载成功")
    
    # 提取各个VAE
    vae_configs = [
        ('dense', 'Dense VAE', 'dense_vae.pth', 'dense_vae'),
        ('sparse_512', 'Sparse VAE 512', 'sparse_vae_512.pth', 'sparse_vae_512'),
        ('sparse_1024', 'Sparse VAE 1024', 'sparse_vae_1024.pth', 'sparse_vae_1024'),
    ]
    
    print("\n" + "=" * 80)
    print("📦 开始提取VAE权重...")
    print("=" * 80)
    
    extracted_vaes = {}
    for file_key, vae_name, output_filename, config_key in vae_configs:
        if file_paths.get(file_key) and os.path.exists(file_paths[file_key]):
            # 提取VAE权重
            vae_state_dict = extract_vae_from_checkpoint(
                file_paths[file_key],
                vae_name
            )
            
            if vae_state_dict:
                # 获取配置信息
                vae_info = None
                if config and config_key in config:
                    vae_info = OmegaConf.to_container(config[config_key], resolve=True)
                
                # 保存权重
                output_path = os.path.join(output_dir, output_filename)
                save_vae_weights(vae_state_dict, output_path, vae_info)
                
                extracted_vaes[vae_name] = output_path
        else:
            print(f"\n⚠️  跳过 {vae_name}（文件不存在）")
    
    # 总结
    print("\n" + "=" * 80)
    print("✨ 提取完成!")
    print("=" * 80)
    print(f"\n📁 所有VAE权重已保存到: {os.path.abspath(output_dir)}")
    print("\n提取的VAE:")
    for vae_name, path in extracted_vaes.items():
        print(f"  ✅ {vae_name}: {os.path.basename(path)}")
    
    # 提供使用示例
    print("\n" + "=" * 80)
    print("💡 使用示例:")
    print("=" * 80)
    print("""
# 加载Dense VAE
import torch
from direct3d_s2.models.autoencoders.dense_vae import DenseShapeVAE
from omegaconf import OmegaConf

# 方式1: 从保存的文件加载完整数据（包含配置）
saved_data = torch.load('vae_weights/dense_vae.pth', map_location='cpu')
vae_state_dict = saved_data['vae']
vae_config = saved_data.get('config', {})

# 方式2: 如果有配置，创建模型并加载权重
if vae_config:
    vae = DenseShapeVAE(**vae_config['params'])
    vae.load_state_dict(vae_state_dict)
    vae.eval()
    print("✅ VAE加载成功!")

# 类似地加载Sparse VAE
from direct3d_s2.models.autoencoders.ss_vae import SparseSDFVAE
saved_data = torch.load('vae_weights/sparse_vae_1024.pth', map_location='cpu')
sparse_vae = SparseSDFVAE(**saved_data['config']['params'])
sparse_vae.load_state_dict(saved_data['vae'])
""")


def main():
    parser = argparse.ArgumentParser(
        description="从Direct3D-S2模型中提取VAE权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从HuggingFace下载并提取
  python extract_vae_weights.py
  
  # 从本地文件提取
  python extract_vae_weights.py --local-path /path/to/models
  
  # 指定输出目录
  python extract_vae_weights.py --output-dir ./my_vae_weights
  
  # 指定不同的模型版本
  python extract_vae_weights.py --subfolder direct3d-s2-v-1-0
        """
    )
    
    parser.add_argument(
        '--repo-id',
        type=str,
        default='wushuang98/Direct3D-S2',
        help='HuggingFace仓库ID (默认: wushuang98/Direct3D-S2)'
    )
    
    parser.add_argument(
        '--subfolder',
        type=str,
        default='direct3d-s2-v-1-1',
        help='模型子文件夹 (默认: direct3d-s2-v-1-1)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./vae_weights',
        help='输出目录 (默认: ./vae_weights)'
    )
    
    parser.add_argument(
        '--local-path',
        type=str,
        default=None,
        help='本地模型文件路径（如果提供则不从HuggingFace下载）'
    )
    
    args = parser.parse_args()
    
    load_and_extract_all_vaes(
        repo_id=args.repo_id,
        subfolder=args.subfolder,
        output_dir=args.output_dir,
        local_path=args.local_path
    )


if __name__ == '__main__':
    main()

