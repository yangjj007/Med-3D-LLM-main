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
        print(f"[DEBUG VQ.forward] Input z.shape: {z.shape}, z.feats.shape: {z.feats.shape}")
        print(f"[DEBUG VQ.forward] z.feats min: {z.feats.min().item():.6f}, max: {z.feats.max().item():.6f}, mean: {z.feats.mean().item():.6f}")
        print(f"[DEBUG VQ.forward] z.feats std: {z.feats.std().item():.6f}")
        print(f"[DEBUG VQ.forward] Codebook weight min: {self.embeddings.weight.min().item():.6f}, max: {self.embeddings.weight.max().item():.6f}")
        
        # z.feats: [N, embedding_dim]
        z_flatten = z.feats  # [N, embedding_dim]
        
        # 计算距离并找到最近的 codebook entry
        print(f"[DEBUG VQ.forward] Computing distances...")
        distances = torch.cdist(z_flatten, self.embeddings.weight)  # [N, num_embeddings]
        print(f"[DEBUG VQ.forward] distances.shape: {distances.shape}")
        print(f"[DEBUG VQ.forward] distances min: {distances.min().item():.6f}, max: {distances.max().item():.6f}, mean: {distances.mean().item():.6f}")
        
        # 检查第一个样本的距离分布
        if len(distances) > 0:
            first_dists = distances[0]
            print(f"[DEBUG VQ.forward] First sample distances - min: {first_dists.min().item():.6f}, max: {first_dists.max().item():.6f}")
            print(f"[DEBUG VQ.forward] First sample top-5 closest indices: {torch.argsort(first_dists)[:5].tolist()}")
            print(f"[DEBUG VQ.forward] First sample top-5 closest distances: {first_dists[torch.argsort(first_dists)[:5]].tolist()}")
        
        encoding_indices = torch.argmin(distances, dim=1)  # [N]
        print(f"[DEBUG VQ.forward] encoding_indices.shape: {encoding_indices.shape}")
        print(f"[DEBUG VQ.forward] encoding_indices min: {encoding_indices.min().item()}, max: {encoding_indices.max().item()}")
        print(f"[DEBUG VQ.forward] encoding_indices unique values: {len(torch.unique(encoding_indices))}")
        print(f"[DEBUG VQ.forward] encoding_indices value counts (top 10): {torch.bincount(encoding_indices).sort(descending=True).values[:10].tolist()}")
        
        if only_return_indices:
            # 返回 indices 作为 SparseTensor，保持原始坐标
            result = z.replace(encoding_indices.unsqueeze(-1).float())
            print(f"[DEBUG VQ.forward] Returning indices only, result.shape: {result.shape}, result.feats.shape: {result.feats.shape}")
            return result
        
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
        # ===== DEBUG: 检查 forward 接收的 batch =====
        print(f"\n[DEBUG forward] batch type: {type(batch)}")
        print(f"[DEBUG forward] batch class: {batch.__class__.__name__}")
        if isinstance(batch, dict):
            print(f"[DEBUG forward] batch is dict, keys: {batch.keys()}")
        elif hasattr(batch, 'feats'):
            print(f"[DEBUG forward] batch is SparseTensor")
            print(f"[DEBUG forward] batch.shape: {batch.shape}")
            print(f"[DEBUG forward] batch.feats.shape: {batch.feats.shape}")
            print(f"[DEBUG forward] batch.coords.shape: {batch.coords.shape}")
        print("=" * 80)
        # ===== DEBUG END =====
        
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
            batch: 输入数据批次。可以是：
                  - SparseTensor：训练时使用
                  - dict：推理时使用，包含 'sparse_sdf', 'sparse_index', 'batch_idx' 键
            only_return_indices: 是否只返回量化索引（用于推理）
        Returns:
            如果 only_return_indices=True: 返回 encoding_indices
            否则: 返回 (z, vq_loss, commitment_loss)
        """
        # ===== DEBUG: 详细检查 batch 类型和内容 =====
        print(f"\n[DEBUG encode] batch type: {type(batch)}")
        print(f"[DEBUG encode] batch class name: {batch.__class__.__name__}")
        
        # 判断 batch 的类型并处理
        if hasattr(batch, 'feats') and hasattr(batch, 'coords'):
            # batch 是 SparseTensor（训练时的情况）
            print(f"[DEBUG encode] batch is SparseTensor (training mode)")
            print(f"[DEBUG encode] batch.shape: {batch.shape}")
            print(f"[DEBUG encode] batch.feats.shape: {batch.feats.shape}")
            print(f"[DEBUG encode] batch.coords.shape: {batch.coords.shape}")
            
            # 直接使用 SparseTensor
            x = batch
            
        elif isinstance(batch, dict):
            # batch 是字典（推理时的情况）
            print(f"[DEBUG encode] batch is dict (inference mode)")
            print(f"[DEBUG encode] batch keys: {batch.keys()}")
            
            # 从字典中提取数据并构建 SparseTensor
            feat, xyz, batch_idx = batch['sparse_sdf'], batch['sparse_index'], batch['batch_idx']
            print(f"[DEBUG encode] feat shape: {feat.shape}, dtype: {feat.dtype}")
            print(f"[DEBUG encode] xyz shape: {xyz.shape}, dtype: {xyz.dtype}")
            print(f"[DEBUG encode] batch_idx shape: {batch_idx.shape}, dtype: {batch_idx.dtype}")
            
            if feat.ndim == 1:
                feat = feat.unsqueeze(-1)
                print(f"[DEBUG encode] feat expanded to shape: {feat.shape}")
            
            coords = torch.cat([batch_idx.unsqueeze(-1), xyz], dim=-1).int()
            print(f"[DEBUG encode] coords shape: {coords.shape}, dtype: {coords.dtype}")
            
            x = sp.SparseTensor(feat, coords)
            print(f"[DEBUG encode] Created SparseTensor from dict")
            
        else:
            print(f"[DEBUG encode] ERROR: Unknown batch type!")
            print(f"[DEBUG encode] batch attributes: {dir(batch)}")
            
            # 打印调用栈
            import traceback
            print(f"[DEBUG encode] Call stack:")
            for line in traceback.format_stack()[:-1]:
                print(line.strip())
            print("=" * 80)
            
            raise TypeError(f"batch must be either SparseTensor or dict, got {type(batch)}")
        
        print("=" * 80)
        # ===== DEBUG END =====
        
        # 获取 factor 参数（如果有）
        factor = None
        if isinstance(batch, dict):
            factor = batch.get('factor', None)
        
        print(f"[DEBUG encode] About to call encoder...")
        print(f"[DEBUG encode] Input x.shape={x.shape}, x.feats.shape={x.feats.shape}, x.coords.shape={x.coords.shape}")
        print(f"[DEBUG encode] Input x.feats min: {x.feats.min().item():.6f}, max: {x.feats.max().item():.6f}, mean: {x.feats.mean().item():.6f}")
        print(f"[DEBUG encode] Input x.feats std: {x.feats.std().item():.6f}")
        print(f"[DEBUG encode] Input x.coords min: {x.coords.min(0).values}, max: {x.coords.max(0).values}")
        print(f"[DEBUG encode] Input x.dtype: {x.dtype}, x.feats.dtype: {x.feats.dtype}")
        print(f"[DEBUG encode] Encoder training mode: {self.encoder.training}")
        
        print(f"[DEBUG encode] Calling encoder with factor={factor}")
        h = self.encoder(x, factor)
        print(f"[DEBUG encode] Encoder output h.shape={h.shape}, h.feats.shape={h.feats.shape}")
        print(f"[DEBUG encode] Encoder output h.coords min: {h.coords.min(0).values}, max: {h.coords.max(0).values}")
        
        # 获取 mean（替代 VAE 的 posterior.mode()）
        # encoder 输出的是 2*latent_channels，取前半部分作为 mean
        print(f"[DEBUG encode] Creating DiagonalGaussianDistribution")
        print(f"[DEBUG encode] h.feats before DGD - shape: {h.feats.shape}, min: {h.feats.min().item():.6f}, max: {h.feats.max().item():.6f}, mean: {h.feats.mean().item():.6f}")
        posterior = DiagonalGaussianDistribution(h.feats, feat_dim=1)
        mean_feats = posterior.mode()  # 使用 mode() 获取 mean，不采样
        print(f"[DEBUG encode] mean_feats shape: {mean_feats.shape}")
        print(f"[DEBUG encode] mean_feats min: {mean_feats.min().item():.6f}, max: {mean_feats.max().item():.6f}, mean: {mean_feats.mean().item():.6f}, std: {mean_feats.std().item():.6f}")
        h_mean = h.replace(mean_feats)
        print(f"[DEBUG encode] h_mean shape: {h_mean.shape}")
        print(f"[DEBUG encode] h_mean.coords shape: {h_mean.coords.shape}, h_mean.feats shape: {h_mean.feats.shape}")
        
        if only_return_indices:
            # 只返回量化索引（用于 Encode 方法）
            print(f"[DEBUG encode] only_return_indices=True, calling vq...")
            encoding_indices = self.vq(h_mean, only_return_indices=True)
            print(f"[DEBUG encode] encoding_indices shape: {encoding_indices.shape}")
            return encoding_indices
        
        # 量化（替代 VAE 的采样）
        print(f"[DEBUG encode] Calling vq for quantization...")
        quantized, vq_loss, commitment_loss, _ = self.vq(h_mean)
        print(f"[DEBUG encode] Quantization done")
        print(f"[DEBUG encode] quantized shape: {quantized.shape}")
        print(f"[DEBUG encode] vq_loss: {vq_loss.item()}, commitment_loss: {commitment_loss.item()}")

        return quantized, vq_loss, commitment_loss
    
    def Encode(self, batch):
        """
        编码到离散索引（推理时使用）
        Args:
            batch: 输入数据批次
        Returns:
            encoding_indices: SparseTensor，包含量化后的 indices
        """
        # ===== DEBUG: 在 Encode 方法入口检查 batch =====
        print(f"\n[DEBUG Encode] Entering Encode method")
        print(f"[DEBUG Encode] batch type: {type(batch)}")
        print(f"[DEBUG Encode] batch class: {batch.__class__.__name__}")
        if isinstance(batch, dict):
            print(f"[DEBUG Encode] batch is dict, keys: {batch.keys()}")
        else:
            print(f"[DEBUG Encode] batch is NOT dict!")
        print("=" * 80)
        # ===== DEBUG END =====
        
        encoding_indices = self.encode(batch, only_return_indices=True)
        print(f"DEBUG Encode: Indices max: {encoding_indices.feats.max()}, min: {encoding_indices.feats.min()}")
        # 确保 max 值没有超过你的 codebook_size

        return encoding_indices
    
    def Decode(self, encoding_indices: sp.SparseTensor):
        """
        从离散索引解码（推理时使用）
        Args:
            encoding_indices: SparseTensor，包含量化 indices
        Returns:
            recon: 重建的 SparseTensor
        """
        print(f"[DEBUG Decode] encoding_indices type: {type(encoding_indices)}")
        print(f"[DEBUG Decode] encoding_indices.shape: {encoding_indices.shape}")
        print(f"[DEBUG Decode] encoding_indices.feats.shape: {encoding_indices.feats.shape}")
        print(f"[DEBUG Decode] encoding_indices.coords.shape: {encoding_indices.coords.shape}")
        print(f"[DEBUG Decode] encoding_indices.feats dtype: {encoding_indices.feats.dtype}")
        print(f"[DEBUG Decode] encoding_indices.coords dtype: {encoding_indices.coords.dtype}")
        
        # 从 indices 获取 embedding
        indices = encoding_indices.feats.long().squeeze(-1)  # [N]
        print(f"[DEBUG Decode] After squeeze, indices.shape: {indices.shape}")
        print(f"[DEBUG Decode] indices min: {indices.min().item()}, max: {indices.max().item()}, unique count: {len(torch.unique(indices))}")
        print(f"[DEBUG Decode] First 10 indices: {indices[:10].tolist() if len(indices) >= 10 else indices.tolist()}")
        print(f"[DEBUG Decode] VQ codebook size: {self.vq.embeddings.num_embeddings}")
        print(f"[DEBUG Decode] VQ embedding dim: {self.vq.embeddings.embedding_dim}")
        
        # 检查索引是否在有效范围内
        if indices.max() >= self.vq.embeddings.num_embeddings:
            print(f"[ERROR Decode] Index out of range! max index: {indices.max().item()}, codebook size: {self.vq.embeddings.num_embeddings}")
        
        quantized_feats = self.vq.embeddings(indices)  # [N, latent_channels]
        print(f"[DEBUG Decode] quantized_feats.shape: {quantized_feats.shape}")
        print(f"[DEBUG Decode] quantized_feats min: {quantized_feats.min().item():.4f}, max: {quantized_feats.max().item():.4f}, mean: {quantized_feats.mean().item():.4f}")
        
        # 创建 quantized SparseTensor
        quantized = encoding_indices.replace(quantized_feats)
        print(f"[DEBUG Decode] quantized SparseTensor created")
        print(f"[DEBUG Decode] quantized.shape: {quantized.shape}")
        print(f"[DEBUG Decode] quantized.feats.shape: {quantized.feats.shape}")
        print(f"[DEBUG Decode] quantized.coords.shape: {quantized.coords.shape}")
        print(f"[DEBUG Decode] quantized.coords min: {quantized.coords.min(0).values}, max: {quantized.coords.max(0).values}")
        
        # 解码
        print(f"[DEBUG Decode] Calling decoder...")
        recon = self.decoder(quantized)
        print(f"[DEBUG Decode] Decoder returned successfully")
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
    def load_pretrained_vae(self, encoder_state_dict: dict, decoder_state_dict: dict):
        """
        加载预训练的 VAE 参数
        Args:
            encoder_state_dict: 预训练的 encoder 权重字典
            decoder_state_dict: 预训练的 decoder 权重字典
        """
        # 加载 encoder 参数
        encoder_dict = self.encoder.state_dict()
        encoder_dict.update(encoder_state_dict)
        self.encoder.load_state_dict(encoder_dict, strict=False)
        
        # 加载 decoder 参数
        decoder_dict = self.decoder.state_dict()
        decoder_dict.update(decoder_state_dict)
        self.decoder.load_state_dict(decoder_dict, strict=False)
        
        print(f"✅ Loaded pretrained VAE parameters")
        print(f"   Encoder: {len(encoder_state_dict)} parameters loaded")
        print(f"   Decoder: {len(decoder_state_dict)} parameters loaded")


# 向后兼容的别名
Direct3DS2_VQVAE = SparseSDFVQVAE
