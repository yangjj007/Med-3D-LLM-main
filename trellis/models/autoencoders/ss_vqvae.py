# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import trimesh
from skimage import measure

from ...modules import sparse as sp
from .encoder import SparseSDFEncoder
from .decoder import SparseSDFDecoder
# VQVAE不需要DiagonalGaussianDistribution（移除VAE的高斯采样机制）


class SparseVectorQuantizer(nn.Module):
    """
    稀疏张量的 Vector Quantizer
    支持两种码本更新模式：
    1. 梯度更新模式 (use_ema_update=False): 通过反向传播更新码本
    2. EMA更新模式 (use_ema_update=True): 通过指数移动平均统计更新码本
    """
    def __init__(self, num_embeddings: int = 8192, embedding_dim: int = 64, beta: float = 0.25,
                 use_ema_update: bool = False, decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.use_ema_update = use_ema_update
        self.decay = decay
        self.epsilon = epsilon
        
        # 码本嵌入
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.normal_(mean=0.0, std=1.0)
        
        # 根据更新模式设置requires_grad和初始化buffer
        if use_ema_update:
            # EMA模式：禁用梯度，注册统计buffer
            self.embeddings.weight.requires_grad = False
            # 🔧 使用拉普拉斯先验（伪计数）初始化，避免未使用码本坍塌为0
            # ema_cluster_size = 1: 每个码本有1次先验计数
            # ema_w = embedding: 当c=1时，e=w/c=embedding，保持初始分布
            # 这样未使用的码本会保持原值，而不是衰减到0向量
            self.register_buffer('ema_cluster_size', torch.ones(num_embeddings))
            self.register_buffer('ema_w', self.embeddings.weight.data.clone())
        # else: 梯度模式保持默认requires_grad=True
    

    def forward(self, z: sp.SparseTensor, only_return_indices: bool = False):
        """
        Args:
            z: SparseTensor，feats shape 为 [N, embedding_dim]，N 是激活体素数量
            only_return_indices: 是否只返回 indices
        Returns:
            如果 only_return_indices=True: 返回 indices 的 SparseTensor
            否则: 返回 (quantized, vq_loss, commitment_loss, encoding_indices)
            注意：当use_ema_update=True时，vq_loss为None
        """
        print(f"\n[DEBUG VQ] Input z.feats: shape={z.feats.shape}, min={z.feats.min().item():.6f}, max={z.feats.max().item():.6f}, mean={z.feats.mean().item():.6f}, std={z.feats.std().item():.6f}")
        print(f"[DEBUG VQ] Codebook: min={self.embeddings.weight.min().item():.6f}, max={self.embeddings.weight.max().item():.6f}, mean={self.embeddings.weight.mean().item():.6f}, std={self.embeddings.weight.std().item():.6f}")
        print(f"[DEBUG VQ] Codebook requires_grad: {self.embeddings.weight.requires_grad}, use_ema_update: {self.use_ema_update}")
        
        # 检查码本是否有异常（太多零向量）
        codebook_norms = torch.norm(self.embeddings.weight, dim=1)  # [num_embeddings]
        zero_codes = (codebook_norms < 0.01).sum().item()
        print(f"[DEBUG VQ] Codebook norms: min={codebook_norms.min().item():.6f}, max={codebook_norms.max().item():.6f}, mean={codebook_norms.mean().item():.6f}")
        print(f"[DEBUG VQ] Near-zero codes (norm<0.01): {zero_codes}/{self.num_embeddings}")
        
        # z.feats: [N, embedding_dim]
        z_flatten = z.feats  # [N, embedding_dim]
        
        # 计算距离并找到最近的 codebook entry
        distances = torch.cdist(z_flatten, self.embeddings.weight)  # [N, num_embeddings]
        print(f"[DEBUG VQ] Distances: min={distances.min().item():.6f}, max={distances.max().item():.6f}, mean={distances.mean().item():.6f}")
        
        # 统计最小距离的分布
        min_distances = distances.min(dim=1)[0]  # [N]
        print(f"[DEBUG VQ] Min distances: mean={min_distances.mean().item():.6f}, std={min_distances.std().item():.6f}, median={min_distances.median().item():.6f}")
        
        encoding_indices = torch.argmin(distances, dim=1)  # [N]
        unique_codes = torch.unique(encoding_indices)
        print(f"[DEBUG VQ] Encoding indices: unique codes used={len(unique_codes)}/{self.num_embeddings}")
        
        # 统计每个码本被使用的次数
        if len(unique_codes) < 100:  # 只在激活少时打印
            counts = torch.bincount(encoding_indices, minlength=self.num_embeddings)
            used_counts = counts[counts > 0]
            print(f"[DEBUG VQ] Usage distribution: min={used_counts.min().item()}, max={used_counts.max().item()}, mean={used_counts.float().mean().item():.1f}")
        
        if only_return_indices:
            # 返回 indices 作为 SparseTensor，保持原始坐标
            result = z.replace(encoding_indices.unsqueeze(-1).float())
            return result
        
        # 量化
        quantized_feats = self.embeddings(encoding_indices)  # [N, embedding_dim]
        print(f"[DEBUG VQ] Quantized feats: min={quantized_feats.min().item():.6f}, max={quantized_feats.max().item():.6f}, mean={quantized_feats.mean().item():.6f}")
        
        # 计算commitment loss（两种模式都需要）
        commitment_loss = F.mse_loss(z_flatten, quantized_feats.detach())
        
        # 根据更新模式选择不同的处理方式
        if self.use_ema_update:
            # EMA模式：在训练时调用EMA更新
            if self.training:
                self._update_ema(encoding_indices, z_flatten)
            vq_loss = None  # EMA模式不需要vq_loss
            print(f"[DEBUG VQ] EMA mode - Commitment Loss: {commitment_loss.item():.6f}, VQ Loss: None")
        else:
            # 梯度模式：计算vq_loss用于反向传播
            vq_loss = F.mse_loss(quantized_feats, z_flatten.detach())
            print(f"[DEBUG VQ] Gradient mode - VQ Loss: {vq_loss.item():.6f}, Commitment Loss: {commitment_loss.item():.6f}")
            print(f"[DEBUG VQ] VQ Loss requires_grad: {vq_loss.requires_grad}, Commitment Loss requires_grad: {commitment_loss.requires_grad}")
        
        # Straight-through estimator
        quantized_feats = z_flatten + (quantized_feats - z_flatten).detach()
        
        # 创建新的 SparseTensor
        quantized = z.replace(quantized_feats)
        encoding_indices_st = z.replace(encoding_indices.unsqueeze(-1).float())
        
        print(f"[DEBUG VQ] Output quantized feats: min={quantized.feats.min().item():.6f}, max={quantized.feats.max().item():.6f}, requires_grad={quantized.feats.requires_grad}\n")
        
        return quantized, vq_loss, commitment_loss, encoding_indices_st
    
    @torch.no_grad()
    def _update_ema(self, encoding_indices, z_flatten):
        """
        使用EMA更新码本（仅在use_ema_update=True时调用）
        
        Args:
            encoding_indices: 分配的码本索引 [N]
            z_flatten: encoder输出的特征向量 [N, embedding_dim]
        """
        print(f"[DEBUG EMA] === Starting EMA Update ===")
        print(f"[DEBUG EMA] Input z_flatten: shape={z_flatten.shape}, min={z_flatten.min().item():.6f}, max={z_flatten.max().item():.6f}, mean={z_flatten.mean().item():.6f}, std={z_flatten.std().item():.6f}")
        print(f"[DEBUG EMA] Encoding indices: shape={encoding_indices.shape}, unique codes={len(torch.unique(encoding_indices))}/{self.num_embeddings}")
        
        # 计算one-hot编码
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # [N, num_embeddings]
        print(f"[DEBUG EMA] One-hot encodings: shape={encodings.shape}, sum={encodings.sum().item():.1f}")
        
        # 检查当前EMA状态
        print(f"[DEBUG EMA] OLD ema_cluster_size: sum={self.ema_cluster_size.sum().item():.1f}, min={self.ema_cluster_size.min().item():.6f}, max={self.ema_cluster_size.max().item():.6f}")
        print(f"[DEBUG EMA] OLD ema_w: min={self.ema_w.min().item():.6f}, max={self.ema_w.max().item():.6f}, mean={self.ema_w.mean().item():.6f}")
        print(f"[DEBUG EMA] OLD embeddings: min={self.embeddings.weight.data.min().item():.6f}, max={self.embeddings.weight.data.max().item():.6f}, mean={self.embeddings.weight.data.mean().item():.6f}")
        
        # EMA更新统计量
        batch_cluster_size = encodings.sum(0)  # [num_embeddings]
        print(f"[DEBUG EMA] Batch cluster size: sum={batch_cluster_size.sum().item():.1f}, nonzero={(batch_cluster_size > 0).sum().item()}/{self.num_embeddings}")
        
        new_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * batch_cluster_size
        print(f"[DEBUG EMA] NEW cluster_size: sum={new_cluster_size.sum().item():.1f}, min={new_cluster_size.min().item():.6f}, max={new_cluster_size.max().item():.6f}")
        
        # 计算batch的加权特征和
        batch_w = encodings.t() @ z_flatten  # [num_embeddings, embedding_dim]
        print(f"[DEBUG EMA] Batch_w (encodings.t() @ z_flatten): shape={batch_w.shape}, min={batch_w.min().item():.6f}, max={batch_w.max().item():.6f}, mean={batch_w.mean().item():.6f}")
        
        new_w = self.decay * self.ema_w + (1 - self.decay) * batch_w
        print(f"[DEBUG EMA] NEW ema_w: min={new_w.min().item():.6f}, max={new_w.max().item():.6f}, mean={new_w.mean().item():.6f}")
        
        # 拉普拉斯平滑（避免某些码本从未被使用）
        n = new_cluster_size.sum()
        print(f"[DEBUG EMA] Total cluster size n: {n.item():.1f}")
        
        smoothed_cluster_size = (
            (new_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
        )
        print(f"[DEBUG EMA] Smoothed cluster size: min={smoothed_cluster_size.min().item():.6f}, max={smoothed_cluster_size.max().item():.6f}, mean={smoothed_cluster_size.mean().item():.6f}")
        
        # 更新码本向量（所有码本，包括未使用的）
        new_embeddings = new_w / (smoothed_cluster_size.unsqueeze(1) + 1e-7)
        print(f"[DEBUG EMA] NEW embeddings (all codes): min={new_embeddings.min().item():.6f}, max={new_embeddings.max().item():.6f}, mean={new_embeddings.mean().item():.6f}, std={new_embeddings.std().item():.6f}")
        
        # 统计实际使用的码本数量（去除先验计数影响）
        used_codes = (new_cluster_size > 1.5).sum().item()  # > 1.5表示除了先验1次外，实际被使用过
        print(f"[DEBUG EMA] Actually used codes: {used_codes}/{self.num_embeddings} (cluster_size > 1.5)")
        
        self.embeddings.weight.data.copy_(new_embeddings)
        
        # 更新buffer
        self.ema_cluster_size.copy_(new_cluster_size)
        self.ema_w.copy_(new_w)
        
        print(f"[DEBUG EMA] === EMA Update Complete ===\n")


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
                 use_ema_update: bool = False,  # 新增：是否使用EMA更新码本
                 vq_decay: float = 0.99,        # 新增：EMA衰减率
                 vq_epsilon: float = 1e-5,      # 新增：拉普拉斯平滑系数
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
            beta=0.25,
            use_ema_update=use_ema_update,
            decay=vq_decay,
            epsilon=vq_epsilon
        )
        
        self.embed_dim = embed_dim
        self.use_ema_update = use_ema_update

    def forward(self, batch):
        """
        训练时的完整前向传播
        """
        z, vq_loss, commitment_loss = self.encode(batch)

        print(f"[DEBUG forward] Calling decoder...")
        reconst_x = self.decoder(z)
        print(f"[DEBUG forward] Decoder output: shape={reconst_x.shape}, feats.shape={reconst_x.feats.shape}")
        print(f"[DEBUG forward] Decoder output feats: min={reconst_x.feats.min().item():.6f}, max={reconst_x.feats.max().item():.6f}, mean={reconst_x.feats.mean().item():.6f}")
        print(f"[DEBUG forward] Decoder output requires_grad: {reconst_x.feats.requires_grad}")
        
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
        # 判断 batch 的类型并处理
        if hasattr(batch, 'feats') and hasattr(batch, 'coords'):
            # batch 是 SparseTensor（训练时的情况）
            x = batch
            factor = None
        elif isinstance(batch, dict):
            # batch 是字典（推理时的情况）
            feat, xyz, batch_idx = batch['sparse_sdf'], batch['sparse_index'], batch['batch_idx']
            
            if feat.ndim == 1:
                feat = feat.unsqueeze(-1)
            
            coords = torch.cat([batch_idx.unsqueeze(-1), xyz], dim=-1).int()
            x = sp.SparseTensor(feat, coords)
            factor = batch.get('factor', None)
        else:
            raise TypeError(f"batch must be either SparseTensor or dict, got {type(batch)}")
        
        print(f"[DEBUG encode] Input x.feats: shape={x.feats.shape}, min={x.feats.min().item():.6f}, max={x.feats.max().item():.6f}, mean={x.feats.mean().item():.6f}, std={x.feats.std().item():.6f}")
        print(f"[DEBUG encode] Encoder training: {self.encoder.training}")
        
        h = self.encoder(x, factor)
        print(f"[DEBUG encode] Encoder output h.feats: shape={h.feats.shape}, min={h.feats.min().item():.6f}, max={h.feats.max().item():.6f}, mean={h.feats.mean().item():.6f}, std={h.feats.std().item():.6f}")
        print(f"[DEBUG encode] h.feats requires_grad: {h.feats.requires_grad}")
        
        if only_return_indices:
            # 只返回量化索引（用于 Encode 方法）
            encoding_indices = self.vq(h, only_return_indices=True)
            return encoding_indices
        
        # 量化（替代 VAE 的采样）
        quantized, vq_loss, commitment_loss, _ = self.vq(h)
        if vq_loss is not None:
            print(f"[DEBUG encode] Quantization results: vq_loss={vq_loss.item():.6f}, commitment_loss={commitment_loss.item():.6f}")
        else:
            print(f"[DEBUG encode] Quantization results: vq_loss=None (EMA mode), commitment_loss={commitment_loss.item():.6f}")

        return quantized, vq_loss, commitment_loss
    
    def Encode(self, batch):
        """
        编码到离散索引（推理时使用）
        Args:
            batch: 输入数据批次
        Returns:
            encoding_indices: SparseTensor，包含量化后的 indices
        """
        encoding_indices = self.encode(batch, only_return_indices=True)
        return encoding_indices
    
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
        
        # 检查索引是否在有效范围内
        if indices.max() >= self.vq.embeddings.num_embeddings:
            print(f"[ERROR Decode] Index out of range! max index: {indices.max().item()}, codebook size: {self.vq.embeddings.num_embeddings}")
        
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
    def load_pretrained_vae(self, encoder_state_dict: dict, decoder_state_dict: dict, vq_state_dict: dict = None):
        """
        加载预训练的 VAE 参数
        Args:
            encoder_state_dict: 预训练的 encoder 权重字典
            decoder_state_dict: 预训练的 decoder 权重字典
            vq_state_dict: 预训练的 VQ 权重字典（可选）
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
        
        # 加载 VQ 参数（如果提供）
        if vq_state_dict is not None and len(vq_state_dict) > 0:
            vq_dict = self.vq.state_dict()
            
            # 筛选可用的参数（避免形状不匹配）
            loaded_keys = []
            skipped_keys = []
            for key, value in vq_state_dict.items():
                if key in vq_dict:
                    if vq_dict[key].shape == value.shape:
                        vq_dict[key] = value
                        loaded_keys.append(key)
                    else:
                        skipped_keys.append(f"{key} (shape mismatch: {vq_dict[key].shape} vs {value.shape})")
                else:
                    skipped_keys.append(f"{key} (not found in current model)")
            
            # 加载更新后的参数
            self.vq.load_state_dict(vq_dict, strict=False)
            
            print(f"   VQ: {len(loaded_keys)} parameters loaded")
            if loaded_keys:
                print(f"      Loaded: {', '.join(loaded_keys)}")
            if skipped_keys:
                print(f"      Skipped: {', '.join(skipped_keys)}")
            
            # 特别说明EMA buffer的处理
            if self.use_ema_update:
                if 'ema_cluster_size' in loaded_keys and 'ema_w' in loaded_keys:
                    print(f"      ℹ️  EMA buffers loaded from pretrained model")
                else:
                    print(f"      ⚠️  EMA buffers not found in pretrained model, will be initialized from scratch")
        else:
            print(f"   VQ: No pretrained VQ parameters provided, using random initialization")
    
    # def convert_to_fp16(self) -> None:
    #     """
    #     Convert encoder, decoder, and VQ codebook to float16.
    #     This method is called by the trainer when loading checkpoints with fp16_mode='inflat_all'.
    #     """
    #     if hasattr(self.encoder, 'convert_to_fp16'):
    #         self.encoder.convert_to_fp16()
    #     if hasattr(self.decoder, 'convert_to_fp16'):
    #         self.decoder.convert_to_fp16()
    #     # Convert VQ codebook embeddings to fp16
    #     if hasattr(self.vq, 'embeddings'):
    #         self.vq.embeddings.weight.data = self.vq.embeddings.weight.data.half()
    
    # def convert_to_fp32(self) -> None:
    #     """
    #     Convert encoder, decoder, and VQ codebook to float32.
    #     """
    #     if hasattr(self.encoder, 'convert_to_fp32'):
    #         self.encoder.convert_to_fp32()
    #     if hasattr(self.decoder, 'convert_to_fp32'):
    #         self.decoder.convert_to_fp32()
    #     # Convert VQ codebook embeddings to fp32
    #     if hasattr(self.vq, 'embeddings'):
    #         self.vq.embeddings.weight.data = self.vq.embeddings.weight.data.float()


# 向后兼容的别名
Direct3DS2_VQVAE = SparseSDFVQVAE
