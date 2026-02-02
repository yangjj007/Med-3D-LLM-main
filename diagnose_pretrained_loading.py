#!/usr/bin/env python3
"""
诊断阶段二训练时预训练权重加载的问题
"""

import torch
import sys
import os

def diagnose_checkpoint(ckpt_path):
    """检查checkpoint文件的结构"""
    print("="*80)
    print(f"📦 检查 Checkpoint: {ckpt_path}")
    print("="*80)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ 文件不存在: {ckpt_path}")
        return None
    
    try:
        print(f"\n正在加载 checkpoint...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        print(f"✅ 加载成功")
        
        # 检查顶层键
        print(f"\n📋 顶层键:")
        for key in ckpt.keys():
            if isinstance(ckpt[key], dict):
                print(f"  - {key}: (字典, {len(ckpt[key])} 个键)")
            elif isinstance(ckpt[key], torch.Tensor):
                print(f"  - {key}: (Tensor, shape={ckpt[key].shape})")
            else:
                print(f"  - {key}: ({type(ckpt[key]).__name__})")
        
        # 提取 VAE state dict（模拟训练器的逻辑）
        print(f"\n🔍 提取 VAE state dict...")
        if 'vae' in ckpt:
            vae_state_dict = ckpt['vae']
            print(f"  使用键: 'vae'")
        elif 'state_dict' in ckpt:
            vae_state_dict = ckpt['state_dict']
            print(f"  使用键: 'state_dict'")
        else:
            vae_state_dict = ckpt
            print(f"  直接使用整个 checkpoint")
        
        # 分析 encoder、decoder、vq 的键
        print(f"\n📊 分析参数键:")
        
        encoder_keys = [k for k in vae_state_dict.keys() if k.startswith('encoder.')]
        decoder_keys = [k for k in vae_state_dict.keys() if k.startswith('decoder.')]
        vq_keys = [k for k in vae_state_dict.keys() if k.startswith('vq.')]
        other_keys = [k for k in vae_state_dict.keys() if not (k.startswith('encoder.') or k.startswith('decoder.') or k.startswith('vq.'))]
        
        print(f"\n  Encoder 参数: {len(encoder_keys)} 个")
        if encoder_keys:
            print(f"    前5个: {encoder_keys[:5]}")
        
        print(f"\n  Decoder 参数: {len(decoder_keys)} 个")
        if decoder_keys:
            print(f"    前5个: {decoder_keys[:5]}")
        
        print(f"\n  VQ 参数: {len(vq_keys)} 个")
        if vq_keys:
            print(f"    所有VQ键:")
            for key in vq_keys:
                value = vae_state_dict[key]
                if isinstance(value, torch.Tensor):
                    print(f"      - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"      - {key}: {type(value).__name__}")
            
            # 检查 embeddings.weight
            if 'vq.embeddings.weight' in vae_state_dict:
                embeddings = vae_state_dict['vq.embeddings.weight']
                print(f"\n  📌 VQ Embeddings (Codebook):")
                print(f"    - Shape: {embeddings.shape}")
                print(f"    - Dtype: {embeddings.dtype}")
                print(f"    - Min: {embeddings.min().item():.6f}")
                print(f"    - Max: {embeddings.max().item():.6f}")
                print(f"    - Mean: {embeddings.mean().item():.6f}")
                print(f"    - Std: {embeddings.std().item():.6f}")
                print(f"    - 前3个code的前5维:")
                for i in range(min(3, embeddings.shape[0])):
                    print(f"      Code {i}: {embeddings[i, :5].tolist()}")
        else:
            print(f"    ⚠️  没有找到任何以 'vq.' 开头的键！")
            print(f"    这意味着 VQ 参数不会被加载，会使用随机初始化！")
        
        if other_keys:
            print(f"\n  其他参数: {len(other_keys)} 个")
            print(f"    前10个: {other_keys[:10]}")
        
        # 模拟提取逻辑
        print(f"\n🔧 模拟训练器的提取逻辑:")
        
        encoder_state_dict = {
            k.replace('encoder.', ''): v
            for k, v in vae_state_dict.items()
            if k.startswith('encoder.')
        }
        print(f"  encoder_state_dict: {len(encoder_state_dict)} 个参数")
        
        decoder_state_dict = {
            k.replace('decoder.', ''): v
            for k, v in vae_state_dict.items()
            if k.startswith('decoder.')
        }
        print(f"  decoder_state_dict: {len(decoder_state_dict)} 个参数")
        
        vq_state_dict = {
            k.replace('vq.', ''): v
            for k, v in vae_state_dict.items()
            if k.startswith('vq.')
        }
        print(f"  vq_state_dict: {len(vq_state_dict)} 个参数")
        
        if vq_state_dict:
            print(f"    VQ 键（去除'vq.'前缀后）:")
            for key in vq_state_dict.keys():
                value = vq_state_dict[key]
                if isinstance(value, torch.Tensor):
                    print(f"      - {key}: shape={value.shape}")
        else:
            print(f"    ⚠️  vq_state_dict 是空的！")
        
        return vae_state_dict
        
    except Exception as e:
        print(f"❌ 加载失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python diagnose_pretrained_loading.py <checkpoint路径>")
        print("\n示例:")
        print("  python diagnose_pretrained_loading.py ./outputs/ct_vqvae_stage1_1024_kmeans/ckpts/vqvae_step0000300.pth")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    diagnose_checkpoint(ckpt_path)
    
    print("\n" + "="*80)
    print("✅ 诊断完成")
    print("="*80)
    print("\n💡 如果看到 '⚠️ 没有找到任何以 vq. 开头的键'，说明:")
    print("  1. checkpoint 文件中没有 VQ 参数")
    print("  2. 或者 VQ 参数的键名格式不对")
    print("  3. 这会导致阶段二训练时 VQ 使用随机初始化，而不是加载阶段一的权重")
    print("\n🔧 解决方法:")
    print("  - 确保阶段一训练保存了完整的模型权重（包括 vq.*）")
    print("  - 或者使用 --load_dir 参数从阶段一的输出目录加载完整 checkpoint")

if __name__ == '__main__':
    main()

