# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

from ...modules import sparse as sp
from .encoder import SparseSDFEncoder
from .decoder import SparseSDFDecoder
from .base import SparseTransformerBase
from .distributions import DiagonalGaussianDistribution



class SparseSDFEncoder_VQ(SparseSDFEncoder):
    """
    改造后的VQ Encoder，继承自原始的SparseSDFEncoder
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__(
            resolution=resolution,
            in_channels=in_channels,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        
        # 重新初始化out_layer，不再输出mean和logvar，只输出latent
        self.out_layer = sp.SparseLinear(model_channels, latent_channels)
        
        # 重新初始化权重
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, factor: float = None, using_out_layer: bool = True):
        """
        Args:
            x: 输入的SparseTensor (稀疏坐标格式)
            factor: 缩放因子
            using_out_layer: 是否使用输出层
        Returns:
            如果using_out_layer=False，返回transformer的输出
            如果using_out_layer=True，返回最终的latent表示
        """
        x = self.input_layer1(x)
        for block in self.downsample:
            x = block(x)
        
        print("✅ Before transformer")
        h = SparseTransformerBase.forward(self, x, factor)
        print("✅ After transformer")
        h = h.type(x.dtype)
        
        if not using_out_layer:
            return h
        
        print("✅ Before layer_norm")
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        print("✅ After layer_norm")
        h = self.out_layer(h)
        
        return h


class SparseSDFDecoder_VQ(SparseSDFDecoder):
    """
    改造后的VQ Decoder，继承自原始的SparseSDFDecoder
    """
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        out_channels: int = 1,
        chunk_size: int = 1,
    ):
        super().__init__(
            resolution=resolution,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            representation_config=representation_config,
            out_channels=out_channels,
            chunk_size=chunk_size,
        )

    def forward(self, x: sp.SparseTensor, factor: float = None, using_input_layer: bool = True):
        """
        Args:
            x: 输入的SparseTensor (稀疏坐标格式)
            factor: 缩放因子
            using_input_layer: 是否使用输入层（从codebook解码时设为False）
        Returns:
            解码后的SparseTensor
        """
        if using_input_layer:
            h = SparseTransformerBase.forward(self, x, factor)
        else:
            # 直接使用输入，跳过transformer的input projection
            h = x
            
        h = h.type(self.dtype if hasattr(self, 'dtype') else x.dtype)
        
        for block in self.upsample:
            h = block(h)
        
        h = h.type(x.dtype)
        h = self.out_layer(h)
        h = self.out_active(h)
        
        return h


class SparseVectorQuantizer(nn.Module):
    """
    稀疏张量的Vector Quantizer
    """
    def __init__(self, num_embeddings: int = 81920, embedding_dim: int = 64, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z: sp.SparseTensor, only_return_indices: bool = False):
        """
        Args:
            z: SparseTensor with shape [N, embedding_dim] where N is number of active voxels
            only_return_indices: 是否只返回indices
        Returns:
            如果only_return_indices=True: 返回indices的SparseTensor
            否则: 返回(quantized, vq_loss, commitment_loss, encoding_indices)
        """
        # z.feats: [N, embedding_dim]
        z_flatten = z.feats  # [N, embedding_dim]
        
        # 计算距离并找到最近的codebook entry
        distances = torch.cdist(z_flatten, self.embeddings.weight)  # [N, num_embeddings]
        encoding_indices = torch.argmin(distances, dim=1)  # [N]
        
        if only_return_indices:
            # 返回indices作为SparseTensor，保持原始坐标
            return z.replace(encoding_indices.unsqueeze(-1).float())
        
        # 量化
        quantized_feats = self.embeddings(encoding_indices)  # [N, embedding_dim]
        
        # 计算损失
        commitment_loss = F.mse_loss(z_flatten, quantized_feats.detach())
        vq_loss = F.mse_loss(quantized_feats, z_flatten.detach())
        
        # Straight-through estimator
        quantized_feats = z_flatten + (quantized_feats - z_flatten).detach()
        
        # 创建新的SparseTensor
        quantized = z.replace(quantized_feats)
        encoding_indices_st = z.replace(encoding_indices.unsqueeze(-1).float())
        
        return quantized, vq_loss, commitment_loss, encoding_indices_st


class Direct3DS2_VQVAE(nn.Module):
    """
    Direct3D-S2的VQVAE版本
    """
    def __init__(
        self,
        resolution: int = 64,
        model_channels: int = 512,  # 必须是512的倍数，否则GroupNorm会报错
        latent_channels: int = 32,
        num_blocks: int = 8,
        num_embeddings: int = 8192,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
    ):
        super().__init__()
        
        self.resolution = resolution
        self.latent_channels = latent_channels
        
        # Encoder
        self.Encoder = SparseSDFEncoder_VQ(
            resolution=resolution,
            in_channels=1,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        
        # Decoder
        self.Decoder = SparseSDFDecoder_VQ(
            resolution=resolution,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            representation_config=representation_config,
            out_channels=1,
        )
        
        # Vector Quantizer
        self.vq = SparseVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_channels,
            beta=0.25
        )
    
    def Encode(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """
        编码输入的稀疏张量到离散indices
        Args:
            x: 输入的SparseTensor，坐标格式 [batch, x, y, z]，特征为SDF值
        Returns:
            encoding_indices: SparseTensor，包含量化后的indices
        """
        z = self.Encoder(x)
        encoding_indices = self.vq(z, only_return_indices=True)
        return encoding_indices
    
    def Decode(self, encoding_indices: sp.SparseTensor) -> sp.SparseTensor:
        """
        从离散indices解码到稀疏张量
        Args:
            encoding_indices: SparseTensor，包含量化indices
        Returns:
            recon: 重建的SparseTensor
        """
        # 从indices获取embedding
        indices = encoding_indices.feats.long().squeeze(-1)  # [N]
        quantized_feats = self.vq.embeddings(indices)  # [N, latent_channels]
        
        # 创建quantized SparseTensor
        quantized = encoding_indices.replace(quantized_feats)
        
        # 解码
        recon = self.Decoder(quantized, using_input_layer=True)
        return recon
    
    def forward(self, x: sp.SparseTensor):
        """
        完整的前向传播（训练时使用）
        Args:
            x: 输入的SparseTensor
        Returns:
            recon: 重建的SparseTensor
            vq_loss: VQ损失
            commitment_loss: commitment损失
        """
        # Encode
        z = self.Encoder(x)
        
        # Quantize
        quantized, vq_loss, commitment_loss, _ = self.vq(z)
        
        # Decode
        recon = self.Decoder(quantized, using_input_layer=True)
        
        return recon, vq_loss, commitment_loss
    
    @torch.no_grad()
    def load_pretrained_vae(self, pretrained_encoder, pretrained_decoder):
        """
        加载预训练的VAE参数
        Args:
            pretrained_encoder: 预训练的encoder state dict
            pretrained_decoder: 预训练的decoder state dict
        """
        # 加载encoder参数（除了out_layer）
        encoder_dict = self.Encoder.state_dict()
        pretrained_encoder_filtered = {
            k: v for k, v in pretrained_encoder.items() 
            if k in encoder_dict and 'out_layer' not in k
        }
        encoder_dict.update(pretrained_encoder_filtered)
        self.Encoder.load_state_dict(encoder_dict, strict=False)
        
        # 加载decoder参数（完整加载）
        self.Decoder.load_state_dict(pretrained_decoder, strict=False)
        
        print(f"Loaded pretrained VAE parameters")
        print(f"Encoder: {len(pretrained_encoder_filtered)}/{len(encoder_dict)} parameters loaded")
