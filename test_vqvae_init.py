#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 SparseSDFVQVAE 模型是否可以正确初始化
"""
import torch
import json
from trellis.models import SparseSDFVQVAE

def test_vqvae_init_from_config():
    """测试从配置文件初始化模型"""
    # 读取配置文件
    config_path = "configs/vae/ct_vqvae_stage1.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_config = config['models']['vqvae']
    print(f"✅ 成功读取配置文件: {config_path}")
    print(f"   模型名: {model_config['name']}")
    print(f"   参数: {json.dumps(model_config['args'], indent=2)}")
    
    # 初始化模型
    try:
        model = SparseSDFVQVAE(**model_config['args'])
        print(f"\n✅ 成功初始化模型: {model_config['name']}")
        print(f"   Encoder: {type(model.encoder).__name__}")
        print(f"   Decoder: {type(model.decoder).__name__}")
        print(f"   VQ: {type(model.vq).__name__}")
        print(f"   Embed dim: {model.embed_dim}")
        print(f"   Num embeddings: {model.vq.num_embeddings}")
        return True
    except Exception as e:
        print(f"\n❌ 初始化模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vqvae_init_direct():
    """测试直接初始化模型（使用旧参数名）"""
    print("\n" + "="*80)
    print("测试使用旧参数名（兼容性测试）")
    print("="*80)
    
    try:
        model = SparseSDFVQVAE(
            resolution=64,
            model_channels=512,  # 旧参数名
            latent_channels=32,  # 旧参数名
            num_blocks=8,  # 旧参数名
            num_embeddings=8192,
            num_heads=None,
            num_head_channels=64,
            mlp_ratio=4.0,
            attn_mode="swin",
            window_size=8,
            pe_mode="ape",
            use_fp16=False,
            use_checkpoint=False,
            qk_rms_norm=False,
        )
        print(f"✅ 成功使用旧参数名初始化模型")
        print(f"   Embed dim: {model.embed_dim}")
        print(f"   Encoder model_channels: {model.encoder.model_channels}")
        print(f"   Decoder model_channels: {model.decoder.model_channels}")
        return True
    except Exception as e:
        print(f"❌ 使用旧参数名初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vqvae_init_new_params():
    """测试使用新参数名初始化模型"""
    print("\n" + "="*80)
    print("测试使用新参数名（VAE 风格）")
    print("="*80)
    
    try:
        model = SparseSDFVQVAE(
            resolution=64,
            model_channels_encoder=512,  # 新参数名
            model_channels_decoder=512,  # 新参数名
            embed_dim=32,  # 新参数名
            num_blocks_encoder=8,  # 新参数名
            num_blocks_decoder=8,  # 新参数名
            num_embeddings=8192,
            num_heads_encoder=8,
            num_heads_decoder=8,
            num_head_channels_encoder=64,
            num_head_channels_decoder=64,
            mlp_ratio=4.0,
            attn_mode="swin",
            window_size=8,
            pe_mode="ape",
            use_fp16=False,
            use_checkpoint=False,
            qk_rms_norm=False,
        )
        print(f"✅ 成功使用新参数名初始化模型")
        print(f"   Embed dim: {model.embed_dim}")
        print(f"   Encoder model_channels: {model.encoder.model_channels}")
        print(f"   Decoder model_channels: {model.decoder.model_channels}")
        return True
    except Exception as e:
        print(f"❌ 使用新参数名初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*80)
    print("测试 SparseSDFVQVAE 模型初始化")
    print("="*80)
    
    results = []
    
    # 测试1：从配置文件初始化
    results.append(("从配置文件初始化", test_vqvae_init_from_config()))
    
    # 测试2：使用旧参数名
    results.append(("使用旧参数名", test_vqvae_init_direct()))
    
    # 测试3：使用新参数名
    results.append(("使用新参数名", test_vqvae_init_new_params()))
    
    # 汇总结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} - {name}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 所有测试通过！模型可以正常使用。")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")

