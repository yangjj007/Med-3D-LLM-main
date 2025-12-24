#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速开始示例 - 演示如何使用提取的VAE权重
这个脚本展示了最简单的使用方式
"""

import torch
import os


def quick_demo():
    """
    快速演示如何加载和使用VAE
    """
    print("=" * 80)
    print("🚀 Direct3D-S2 VAE快速开始示例")
    print("=" * 80)
    
    # 检查VAE权重文件是否存在
    vae_dir = "./vae_weights"
    dense_vae_path = os.path.join(vae_dir, "dense_vae.pth")
    
    if not os.path.exists(dense_vae_path):
        print("\n❌ 错误: 未找到VAE权重文件!")
        print("\n请先运行以下命令提取VAE权重:")
        print("  python extract_vae_weights.py")
        print("\n或者使用自定义路径:")
        print("  python extract_vae_weights.py --output-dir ./vae_weights")
        return
    
    print("\n✅ 找到VAE权重文件")
    
    # ==================== 示例1: 查看VAE信息 ====================
    print("\n" + "=" * 80)
    print("📋 示例1: 查看VAE基本信息")
    print("=" * 80)
    
    saved_data = torch.load(dense_vae_path, map_location='cpu')
    vae_state_dict = saved_data['vae']
    vae_config = saved_data.get('config', None)
    
    print(f"\n📦 Dense VAE信息:")
    print(f"  - 权重键数量: {len(vae_state_dict)}")
    print(f"  - 总参数量: {sum(p.numel() for p in vae_state_dict.values()):,}")
    
    if vae_config:
        print(f"\n⚙️  配置参数:")
        params = vae_config.get('params', vae_config)
        for key, value in list(params.items())[:5]:  # 只显示前5个
            print(f"    - {key}: {value}")
        print(f"    ... (共{len(params)}个参数)")
    
    # ==================== 示例2: 加载VAE模型 ====================
    print("\n" + "=" * 80)
    print("🔧 示例2: 加载VAE模型")
    print("=" * 80)
    
    try:
        from direct3d_s2.models.autoencoders.dense_vae import DenseShapeVAE
        
        # 创建模型
        if vae_config:
            params = vae_config.get('params', vae_config)
            vae = DenseShapeVAE(**params)
        else:
            # 使用默认配置
            vae = DenseShapeVAE(
                embed_dim=8,
                model_channels_encoder=[32, 128, 512],
                model_channels_decoder=[512, 128, 32],
            )
        
        # 加载权重
        vae.load_state_dict(vae_state_dict)
        vae.eval()
        
        print(f"\n✅ VAE模型加载成功!")
        print(f"  - 模型类型: {type(vae).__name__}")
        print(f"  - Embed维度: {vae.embed_dim}")
        print(f"  - 训练模式: {vae.training}")
        
        # ==================== 示例3: 测试前向传播 ====================
        print("\n" + "=" * 80)
        print("🧪 示例3: 测试前向传播")
        print("=" * 80)
        
        # 创建随机输入
        batch_size = 1
        resolution = 64
        batch = {
            'dense_index': torch.rand(batch_size, 1, resolution, resolution, resolution)
        }
        
        print(f"\n📥 输入:")
        print(f"  - 形状: {batch['dense_index'].shape}")
        print(f"  - 数据类型: {batch['dense_index'].dtype}")
        print(f"  - 值范围: [{batch['dense_index'].min():.3f}, {batch['dense_index'].max():.3f}]")
        
        with torch.no_grad():
            # 编码
            z, posterior = vae.encode(batch, sample_posterior=True)
            
            print(f"\n📤 编码结果:")
            print(f"  - 潜在表示形状: {z.shape}")
            print(f"  - 均值形状: {posterior.mean.shape}")
            print(f"  - 标准差形状: {posterior.std.shape}")
            
            # 解码
            reconst = vae.decoder(z)
            
            print(f"\n🔄 解码结果:")
            print(f"  - 重建形状: {reconst.shape}")
            print(f"  - 与输入形状一致: {reconst.shape == batch['dense_index'].shape}")
            
        print("\n✅ 前向传播测试成功!")
        
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print("\n请确保已安装direct3d_s2包:")
        print("  pip install -e .")
        return
    
    # ==================== 示例4: 简单使用建议 ====================
    print("\n" + "=" * 80)
    print("💡 使用建议")
    print("=" * 80)
    
    print("""
1. 用于特征提取:
   z, _ = vae.encode(batch, sample_posterior=False)  # 使用mode而不是sample
   
2. 用于生成:
   reconst = vae.decoder(z)
   
3. 用于微调训练:
   vae.train()
   optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
   
4. GPU加速:
   vae = vae.to('cuda')
   batch = {k: v.to('cuda') for k, v in batch.items()}

5. 查看更多示例:
   python use_extracted_vae.py --help
    """)
    
    # ==================== 检查其他VAE ====================
    print("\n" + "=" * 80)
    print("📁 检查其他VAE文件")
    print("=" * 80)
    
    other_vaes = [
        ("sparse_vae_512.pth", "Sparse VAE 512"),
        ("sparse_vae_1024.pth", "Sparse VAE 1024"),
    ]
    
    for filename, name in other_vaes:
        filepath = os.path.join(vae_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✅ {name}: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠️  {name}: {filename} (未找到)")
    
    print("\n" + "=" * 80)
    print("🎉 演示完成!")
    print("=" * 80)
    print("\n下一步:")
    print("  1. 查看详细文档: VAE_EXTRACTION_README.md")
    print("  2. 比较VAE架构: python use_extracted_vae.py --compare")
    print("  3. 测试其他VAE: python use_extracted_vae.py --vae-path vae_weights/sparse_vae_1024.pth --vae-type sparse_1024 --test")
    print()


if __name__ == '__main__':
    quick_demo()

