# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import trimesh
from skimage import measure

from ...modules import sparse as sp
from .encoder import SparseSDFEncoder
from .decoder import SparseSDFDecoder
from .distributions import DiagonalGaussianDistribution


class SparseVectorQuantizer(nn.Module):
    """
    稀疏张量的 Vector Quantizer
    参考 vae_2_vqvae_example.py 中的 VectorQuantizer 实现
    """
    def __init__(self, num_embeddings: int = 8192, embedding_dim: int = 64, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z: sp.SparseTensor, only_return_indices: bool = False):
        """
        Args:
            z: SparseTensor，feats shape 为 [N, embedding_dim]，N 是激活体素数量
            only_return_indices: 是否只返回 indices
        Returns:
            如果 only_return_indices=True: 返回 indices 的 SparseTensor
            否则: 返回 (quantized, vq_loss, commitment_loss, encoding_indices)
        """
        # z.feats: [N, embedding_dim]
        z_flatten = z.feats  # [N, embedding_dim]
        
        # 计算距离并找到最近的 codebook entry
        distances = torch.cdist(z_flatten, self.embeddings.weight)  # [N, num_embeddings]
        encoding_indices = torch.argmin(distances, dim=1)  # [N]
        
        if only_return_indices:
            # 返回 indices 作为 SparseTensor，保持原始坐标
            return z.replace(encoding_indices.unsqueeze(-1).float())
        
        # 量化
        quantized_feats = self.embeddings(encoding_indices)  # [N, embedding_dim]
        
        # 计算损失
        commitment_loss = F.mse_loss(z_flatten, quantized_feats.detach())
        vq_loss = F.mse_loss(quantized_feats, z_flatten.detach())
        
        # Straight-through estimator
        quantized_feats = z_flatten + (quantized_feats - z_flatten).detach()
        
        # 创建新的 SparseTensor
        quantized = z.replace(quantized_feats)
        encoding_indices_st = z.replace(encoding_indices.unsqueeze(-1).float())
        
        return quantized, vq_loss, commitment_loss, encoding_indices_st


class SparseSDFVQVAE(nn.Module):
    """
    Direct3D-S2 的 VQVAE 版本
    严格遵循 SparseSDFVAE 的结构，只替换 VQ 部分
    """
    def __init__(self, *,
                 embed_dim: int = None,
                 latent_channels: int = None,  # 别名，兼容旧配置
                 resolution: int = 64,
                 model_channels_encoder: int = None,
                 model_channels_decoder: int = None,
                 model_channels: int = None,  # 别名，兼容旧配置
                 num_blocks_encoder: int = None,
                 num_blocks_decoder: int = None,
                 num_blocks: int = None,  # 别名，兼容旧配置
                 num_heads_encoder: int = None,
                 num_heads_decoder: int = None,
                 num_heads: int = None,  # 别名，兼容旧配置
                 num_head_channels_encoder: int = 64,
                 num_head_channels_decoder: int = 64,
                 num_head_channels: int = None,  # 别名，兼容旧配置
                 out_channels: int = 1,
                 use_fp16: bool = False,
                 use_checkpoint: bool = False,
                 chunk_size: int = 1,
                 latents_scale: float = 1.0,
                 latents_shift: float = 0.0,
                 num_embeddings: int = 8192,
                 mlp_ratio: float = 4,
                 attn_mode: str = "swin",
                 window_size: int = 8,
                 pe_mode: str = "ape",
                 qk_rms_norm: bool = False,
                 representation_config: dict = None):

        super().__init__()
        
        # 处理参数别名（兼容旧配置文件）
        if latent_channels is not None and embed_dim is None:
            embed_dim = latent_channels
        if embed_dim is None:
            embed_dim = 0
            
        if model_channels is not None:
            if model_channels_encoder is None:
                model_channels_encoder = model_channels
            if model_channels_decoder is None:
                model_channels_decoder = model_channels
        if model_channels_encoder is None:
            model_channels_encoder = 512
        if model_channels_decoder is None:
            model_channels_decoder = 512
            
        if num_blocks is not None:
            if num_blocks_encoder is None:
                num_blocks_encoder = num_blocks
            if num_blocks_decoder is None:
                num_blocks_decoder = num_blocks
        if num_blocks_encoder is None:
            num_blocks_encoder = 4
        if num_blocks_decoder is None:
            num_blocks_decoder = 4
            
        if num_heads is not None:
            if num_heads_encoder is None:
                num_heads_encoder = num_heads
            if num_heads_decoder is None:
                num_heads_decoder = num_heads
        if num_heads_encoder is None:
            num_heads_encoder = 8
        if num_heads_decoder is None:
            num_heads_decoder = 8
            
        if num_head_channels is not None:
            num_head_channels_encoder = num_head_channels
            num_head_channels_decoder = num_head_channels

        self.use_checkpoint = use_checkpoint
        self.resolution = resolution
        self.latents_scale = latents_scale
        self.latents_shift = latents_shift

        self.encoder = SparseSDFEncoder(
            resolution=resolution,
            in_channels=model_channels_encoder,
            model_channels=model_channels_encoder,
            latent_channels=embed_dim,
            num_blocks=num_blocks_encoder,
            num_heads=num_heads_encoder,
            num_head_channels=num_head_channels_encoder,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )

        self.decoder = SparseSDFDecoder(
            resolution=resolution,
            model_channels=model_channels_decoder,
            latent_channels=embed_dim,
            num_blocks=num_blocks_decoder,
            num_heads=num_heads_decoder,
            num_head_channels=num_head_channels_decoder,
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
        
        # Vector Quantizer（替代 VAE 的高斯分布）
        self.vq = SparseVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            beta=0.25
        )
        
        self.embed_dim = embed_dim

    def forward(self, batch):
        """
        训练时的完整前向传播
        """
        z, vq_loss, commitment_loss = self.encode(batch)

        reconst_x = self.decoder(z)
        outputs = {
            'reconst_x': reconst_x, 
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss
        }
        return outputs

    def encode(self, batch, only_return_indices: bool = False):
        """
        编码过程，替代 VAE 的采样过程
        Args:
            batch: 输入数据批次
            only_return_indices: 是否只返回量化索引（用于推理）
        Returns:
            如果 only_return_indices=True: 返回 encoding_indices
            否则: 返回 (z, vq_loss, commitment_loss)
        """
        feat, xyz, batch_idx = batch['sparse_sdf'], batch['sparse_index'], batch['batch_idx']
        if feat.ndim == 1:
            feat = feat.unsqueeze(-1)
        coords = torch.cat([batch_idx.unsqueeze(-1), xyz], dim=-1).int()
       
        x = sp.SparseTensor(feat, coords)
        h = self.encoder(x, batch.get('factor', None))
        
        # 获取 mean（替代 VAE 的 posterior.mode()）
        # encoder 输出的是 2*latent_channels，取前半部分作为 mean
        posterior = DiagonalGaussianDistribution(h.feats, feat_dim=1)
        mean_feats = posterior.mode()  # 使用 mode() 获取 mean，不采样
        h_mean = h.replace(mean_feats)
        
        if only_return_indices:
            # 只返回量化索引（用于 Encode 方法）
            encoding_indices = self.vq(h_mean, only_return_indices=True)
            return encoding_indices
        
        # 量化（替代 VAE 的采样）
        quantized, vq_loss, commitment_loss, _ = self.vq(h_mean)

        return quantized, vq_loss, commitment_loss
    
    def Encode(self, batch):
        """
        编码到离散索引（推理时使用）
        Args:
            batch: 输入数据批次
        Returns:
            encoding_indices: SparseTensor，包含量化后的 indices
        """
        return self.encode(batch, only_return_indices=True)
    
    def Decode(self, encoding_indices: sp.SparseTensor):
        """
        从离散索引解码（推理时使用）
        Args:
            encoding_indices: SparseTensor，包含量化 indices
        Returns:
            recon: 重建的 SparseTensor
        """
        # 从 indices 获取 embedding
        indices = encoding_indices.feats.long().squeeze(-1)  # [N]
        quantized_feats = self.vq.embeddings(indices)  # [N, latent_channels]
        
        # 创建 quantized SparseTensor
        quantized = encoding_indices.replace(quantized_feats)
        
        # 解码
        recon = self.decoder(quantized)
        return recon

    def decode_mesh(self,
                    latents,
                    voxel_resolution: int = 512,
                    mc_threshold: float = 0.2,
                    return_feat: bool = False,
                    factor: float = 1.0):
        voxel_resolution = int(voxel_resolution / factor)
        reconst_x = self.decoder(latents, factor=factor, return_feat=return_feat)
        if return_feat:
            return reconst_x
        outputs = self.sparse2mesh(reconst_x, voxel_resolution=voxel_resolution, mc_threshold=mc_threshold)
        
        return outputs

    def sparse2mesh(self,
                    reconst_x: torch.FloatTensor,
                    voxel_resolution: int = 512,
                    mc_threshold: float = 0.0):

        sparse_sdf, sparse_index = reconst_x.feats.float(), reconst_x.coords
        batch_size = int(sparse_index[..., 0].max().cpu().numpy() + 1)

        meshes = []
        for i in range(batch_size):
            idx = sparse_index[..., 0] == i
            sparse_sdf_i, sparse_index_i = sparse_sdf[idx].squeeze(-1).cpu(),  sparse_index[idx][..., 1:].detach().cpu()
            sdf = torch.ones((voxel_resolution, voxel_resolution, voxel_resolution))
            sdf[sparse_index_i[..., 0], sparse_index_i[..., 1], sparse_index_i[..., 2]] = sparse_sdf_i
            vertices, faces, _, _ = measure.marching_cubes(
                sdf.numpy(),
                mc_threshold,
                method="lewiner",
            )
            vertices = vertices / voxel_resolution * 2 - 1
            meshes.append(trimesh.Trimesh(vertices, faces))

        return meshes
    
    @torch.no_grad()
    def load_pretrained_vae(self, pretrained_vae_path: str):
        """
        加载预训练的 VAE 参数
        Args:
            pretrained_vae_path: 预训练 VAE 模型的路径
        """
        pretrained = torch.load(pretrained_vae_path, map_location='cpu')
        
        # 加载 encoder 参数
        encoder_dict = self.encoder.state_dict()
        pretrained_encoder = {k.replace('encoder.', ''): v for k, v in pretrained.items() if 'encoder.' in k}
        encoder_dict.update(pretrained_encoder)
        self.encoder.load_state_dict(encoder_dict, strict=False)
        
        # 加载 decoder 参数
        decoder_dict = self.decoder.state_dict()
        pretrained_decoder = {k.replace('decoder.', ''): v for k, v in pretrained.items() if 'decoder.' in k}
        decoder_dict.update(pretrained_decoder)
        self.decoder.load_state_dict(decoder_dict, strict=False)
        
        print(f"✅ Loaded pretrained VAE parameters from {pretrained_vae_path}")
        print(f"   Encoder: {len(pretrained_encoder)} parameters loaded")
        print(f"   Decoder: {len(pretrained_decoder)} parameters loaded")


# 向后兼容的别名
Direct3DS2_VQVAE = SparseSDFVQVAE
