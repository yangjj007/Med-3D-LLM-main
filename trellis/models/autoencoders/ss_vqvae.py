# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple
import re
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import trimesh
import numpy as np

from ...modules import sparse as sp
from ...utils.sparse_sdf_marching_cubes import sparse_feats_coords_to_trimeshes
from .encoder import SparseSDFEncoder
from .decoder import SparseSDFDecoder
# VQVAE不需要DiagonalGaussianDistribution（移除VAE的高斯采样机制）


# =============================================================================
# ShapeLLM VQVAE3D 风格的 2×2×2（或通用 block_side^3）通道拼接
# -----------------------------------------------------------------------------
# 参考 ShapeLLM-Omni `trellis/models/sparse_structure_vqvae.py::VQVAE3D.Encode`：
#     z = z.permute(0, 2, 3, 4, 1).contiguous()
#     z = z.view(bs, 8, 8, 16, 32)
#     encoding_indices = self.vq(z, only_return_indices=True)
# 它在 **稠密** 16³×C 的 latent 上做 reshape，把空间上相邻的若干个格点沿通道维拼起来，
# 再送进 VQ。我们的 SparseSDFVQVAE 输出是 **稀疏** SparseTensor，因此等价实现为：
#   1) 按 `x // bs, y // bs, z // bs` 将每个体素分配到块 (block)。
#   2) 在每个块内，按 `offset = lx*bs^2 + ly*bs + lz`（x 最慢、z 最快，对齐 ShapeLLM view）
#      沿通道维拼接；块内缺失的体素以 0 填充。
# 这既处理了「维度除不尽」也处理了稀疏块不满的情况：缺失位置统一补 0（常用做法）。
# 丢弃不满的块会在物体表面附近丢失大量信息，所以默认采用 **zero-padding**。
# =============================================================================


def _cluster_centers_for_codebook_reinit(
    encodings_np: np.ndarray,
    num_embeddings: int,
    *,
    max_samples: int = 65536,
    max_iter: int = 25,
    faiss_gpu_device: Optional[int] = None,
) -> np.ndarray:
    """
    使用 FAISS GPU KMeans 生成码本周期性重估计用的聚类中心。

    ``sklearn`` 的 CPU KMeans/MiniBatchKMeans 在 ``n_clusters`` 为几千时可能让
    rank0 长时间停在聚类阶段，进而使其它 rank 等待 ``dist.broadcast`` 到 NCCL
    watchdog 超时。这里显式使用当前 rank0 所在 GPU，避免 FAISS 默认占用所有可见 GPU。
    """
    if encodings_np.dtype != np.float32:
        encodings_np = encodings_np.astype(np.float32, copy=False)

    n = int(encodings_np.shape[0])
    if n < num_embeddings:
        raise ValueError(
            f'FAISS KMeans 需要样本数 >= 聚类数，但收到 {n} < {num_embeddings}'
        )

    sample_cap = max(max_samples, num_embeddings)
    if n > sample_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=sample_cap, replace=False)
        encodings_np = encodings_np[idx]
        n = sample_cap

    if not encodings_np.flags.c_contiguous:
        encodings_np = np.ascontiguousarray(encodings_np)

    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            'use_kmeans_reinit=True 现在需要 FAISS GPU。请在训练环境安装 faiss-gpu，'
            '例如：conda install -c pytorch -c nvidia faiss-gpu'
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError('FAISS GPU KMeans 需要 CUDA 可用，但当前 torch.cuda.is_available() 为 False')

    gpu_device = 0 if faiss_gpu_device is None else int(faiss_gpu_device)
    kmeans = faiss.Kmeans(
        d=int(encodings_np.shape[1]),
        k=int(num_embeddings),
        niter=int(max_iter),
        nredo=1,
        verbose=True,
        gpu=gpu_device,
        seed=0,
    )
    try:
        kmeans.train(encodings_np)
    except Exception as exc:
        raise RuntimeError(
            f'FAISS GPU KMeans 失败：gpu_device={gpu_device}, '
            f'n_samples={n}, n_clusters={num_embeddings}, dim={encodings_np.shape[1]}。'
            '请确认安装的是 faiss-gpu 而不是 faiss-cpu，并且 CUDA/FAISS 版本匹配。'
        ) from exc

    return np.asarray(kmeans.centroids, dtype=np.float32)


def _pack_block_channel(
    z: sp.SparseTensor,
    block_side: int,
) -> Tuple[sp.SparseTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 `block_side^3` 个空间邻居的特征沿通道维拼接。

    Args:
        z: SparseTensor；feats ``[N, C]``；coords ``[N, 4]`` = (batch, x, y, z)。
        block_side: 每维合并的体素数（如 2 表示 2×2×2 = 8）。

    Returns:
        z_merged: SparseTensor；feats ``[M, C * G]``；coords 为 block 级（``x,y,z = orig // block_side``）。
        inverse:  ``[N]`` 每个原始体素所属 block 在 ``z_merged`` 中的行索引。
        offset:   ``[N]`` 每个原始体素在块内的位移（``0 .. G-1``）。
        mask:     ``[M, G]`` bool，记录每个 block 的哪些 offset 位置有原始体素（其余为 0 填充）。
    """
    if block_side == 1:
        M = z.feats.shape[0]
        device = z.feats.device
        inverse = torch.arange(M, device=device)
        offset = torch.zeros(M, dtype=torch.long, device=device)
        mask = torch.ones(M, 1, dtype=torch.bool, device=device)
        return z, inverse, offset, mask

    N, C = z.feats.shape
    G = block_side ** 3
    device = z.feats.device

    coords = z.coords  # [N, 4] int
    batch_ids = coords[:, 0]
    xyz = coords[:, 1:4]
    block_xyz = xyz // block_side
    local = xyz - block_xyz * block_side  # [N, 3] in [0, block_side)
    # 块内线性偏移：x 最慢、z 最快（与 ShapeLLM view 的 reshape 顺序一致）
    offset = (
        local[:, 0].to(torch.long) * (block_side * block_side)
        + local[:, 1].to(torch.long) * block_side
        + local[:, 2].to(torch.long)
    )

    block_coords = torch.stack(
        [batch_ids, block_xyz[:, 0], block_xyz[:, 1], block_xyz[:, 2]], dim=1
    )  # [N, 4]，与 coords 同 dtype（int32）
    unique_blocks, inverse = torch.unique(block_coords, dim=0, return_inverse=True)
    M = unique_blocks.shape[0]

    block_feats = z.feats.new_zeros((M, G, C))
    # 同一 (block, offset) 在同一个 SparseTensor 里不应出现重复坐标；直接赋值即可
    block_feats[inverse, offset] = z.feats
    merged_feats = block_feats.reshape(M, G * C)

    mask = torch.zeros(M, G, dtype=torch.bool, device=device)
    mask[inverse, offset] = True

    z_merged = sp.SparseTensor(merged_feats, unique_blocks.to(coords.dtype))
    return z_merged, inverse, offset, mask


def _unpack_block_channel(
    z_merged: sp.SparseTensor,
    inverse: torch.Tensor,
    offset: torch.Tensor,
    original_coords: torch.Tensor,
    block_side: int,
    channels: int,
) -> sp.SparseTensor:
    """
    `_pack_block_channel` 的逆操作：按原始稀疏布局 (inverse, offset) 把合并后的
    ``[M, C*G]`` 特征散回 ``[N, C]``。训练/encode 路径复用之。
    """
    if block_side == 1:
        return z_merged
    G = block_side ** 3
    M = z_merged.feats.shape[0]
    merged = z_merged.feats.view(M, G, channels)
    per_voxel = merged[inverse, offset]  # [N, C]
    return sp.SparseTensor(per_voxel, original_coords)


def _expand_blocks_full(
    z_merged: sp.SparseTensor,
    block_side: int,
    channels: int,
) -> sp.SparseTensor:
    """
    `_pack_block_channel` 的「满块」反操作：将每个 block 扩展为全部 ``block_side^3``
    个子体素（不使用原始稀疏掩码），用于仅凭索引的推理解码（``Decode`` 路径）。
    """
    if block_side == 1:
        return z_merged
    G = block_side ** 3
    M = z_merged.feats.shape[0]
    device = z_merged.feats.device
    merged = z_merged.feats.view(M, G, channels)

    # 生成 G 个块内偏移坐标，顺序与 `offset = lx*bs^2 + ly*bs + lz` 一致
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(block_side, device=device),
            torch.arange(block_side, device=device),
            torch.arange(block_side, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(G, 3)  # [G, 3]

    block_coords = z_merged.coords  # [M, 4]
    batch_ids = block_coords[:, 0:1]  # [M, 1]
    bxyz = block_coords[:, 1:4] * block_side  # [M, 3]
    new_xyz = (bxyz.unsqueeze(1) + grid.unsqueeze(0)).reshape(M * G, 3)  # [M*G, 3]
    new_batch = batch_ids.expand(M, G).reshape(M * G, 1)  # [M*G, 1]
    new_coords = torch.cat([new_batch, new_xyz], dim=1).to(block_coords.dtype)
    new_feats = merged.reshape(M * G, channels)
    return sp.SparseTensor(new_feats, new_coords)


def _infer_block_side(vq_group_size: int) -> int:
    """vq_group_size 必须是某个整数的立方（1, 8, 27, 64 ...）。"""
    if vq_group_size <= 0:
        raise ValueError(f"vq_group_size 必须 > 0，但传入了 {vq_group_size}")
    side = int(round(vq_group_size ** (1.0 / 3.0)))
    # round 可能不精确（浮点），再上下试探一次
    for s in (side - 1, side, side + 1):
        if s >= 1 and s ** 3 == vq_group_size:
            return s
    raise ValueError(
        f"vq_group_size={vq_group_size} 不是整数立方，不能表示为 block_side^3。"
        f" 允许值：1, 8, 27, 64, 125, ..."
    )


class ReservoirSampler(nn.Module):
    """
    水塘采样器，用于持续收集训练特征
    使用经典水塘采样算法，维护固定大小的特征池
    """
    def __init__(self, num_samples: int = 16384):
        super(ReservoirSampler, self).__init__()
        self.n = num_samples  # 容量
        self.register_buffer('buffer', None, persistent=False)
        self.register_buffer('i', torch.tensor(0), persistent=False)
        
    def add(self, samples: torch.Tensor):
        """
        添加样本到水塘（每个 GPU 独立维护）
        Args:
            samples: [N, D] 特征张量
        """
        if samples.numel() == 0:
            return
            
        samples = samples.detach()
        
        # 初始化 buffer
        if self.buffer is None:
            self.buffer = torch.empty(self.n, samples.size(-1), 
                                     device=samples.device, dtype=samples.dtype)
            self.i = torch.tensor(0, device=samples.device)
        
        # 确保 buffer 和 samples 在同一设备上
        if self.buffer.device != samples.device:
            self.buffer = self.buffer.to(samples.device)
            self.i = self.i.to(samples.device)
        
        # 每个 GPU 独立进行水塘采样（不进行跨 GPU 同步）
        for sample in samples:
            if self.i < self.n:
                # 缓冲区未满，直接添加
                self.buffer[self.i] = sample
                self.i += 1
            else:
                # 缓冲区已满，随机替换（水塘采样算法）
                j = torch.randint(0, self.i + 1, (1,), device=sample.device).item()
                if j < self.n:
                    self.buffer[j] = sample
                self.i += 1
    
    def contents(self) -> torch.Tensor:
        """
        获取采样结果
        Returns:
            [min(i, n), D] 已收集的特征
        """
        if self.buffer is None:
            return torch.empty(0)
        return self.buffer[:min(self.i.item(), self.n)]
    
    def reset(self):
        """清空缓冲区"""
        if self.buffer is not None:
            self.i.fill_(0)


class SparseVectorQuantizer(nn.Module):
    """
    稀疏张量的 Vector Quantizer
    支持两种码本更新模式：
    1. 梯度更新模式 (use_ema_update=False): 通过反向传播更新码本
    2. EMA更新模式 (use_ema_update=True): 通过指数移动平均统计更新码本
    支持可选的 K-means 周期性重新初始化：
    3. K-means 重估计模式 (use_kmeans_reinit=True): 周期性使用 K-means 聚类重置码本
    """
    def __init__(self, num_embeddings: int = 8192, embedding_dim: int = 64, beta: float = 0.25,
                 use_ema_update: bool = False, decay: float = 0.99, epsilon: float = 1e-5,
                 use_kmeans_reinit: bool = False, kmeans_interval: int = 2000, 
                 reservoir_size: int = 16384):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.use_ema_update = use_ema_update
        self.decay = decay
        self.epsilon = epsilon
        
        # K-means 重估计参数
        self.use_kmeans_reinit = use_kmeans_reinit
        self.kmeans_interval = kmeans_interval
        
        # 码本嵌入
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.normal_(mean=0.0, std=1.0)
        
        # 根据更新模式设置requires_grad和初始化buffer
        if use_ema_update:
            # EMA模式：禁用梯度，注册统计buffer
            self.embeddings.weight.requires_grad = False
            # 🔧 修复：使用0初始化，在第一个batch后才开始EMA更新
            # 这避免了初始化带来的假设，让码本完全由数据驱动
            # 对于未使用的码本，通过拉普拉斯平滑保持为初始值
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_w', torch.zeros(num_embeddings, embedding_dim))
            # 标记是否是第一次EMA更新
            self.register_buffer('_ema_initialized', torch.tensor(False))
        # else: 梯度模式保持默认requires_grad=True
        
        # K-means 重估计器
        if use_kmeans_reinit:
            self.reestimation_reservoir = ReservoirSampler(reservoir_size)
            print(f"[K-means 重估计] 已启用，间隔={kmeans_interval}步，水塘容量={reservoir_size}")
    

    def forward(
        self,
        z: sp.SparseTensor,
        only_return_indices: bool = False,
        current_step: int = -1,
        update_codebook: bool = True,
    ):
        """
        Args:
            z: SparseTensor，feats shape 为 [N, embedding_dim]，N 是激活体素数量
            only_return_indices: 是否只返回 indices
            current_step: 当前训练步数，用于 K-means 重估计触发（-1 表示不使用）
            update_codebook: 是否允许 EMA/K-means 更新码本统计。
        Returns:
            如果 only_return_indices=True: 返回 indices 的 SparseTensor
            否则: 返回 (quantized, vq_loss, commitment_loss, encoding_indices, codebook_stats)
            注意：当use_ema_update=True时，vq_loss为None
            codebook_stats: 包含 perplexity, entropy, unique_count, utilization_ratio 的字典
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
        z_flatten_orig = z.feats  # [N, embedding_dim]，保留原始 dtype 用于 loss 计算
        input_dtype = z_flatten_orig.dtype
        emb_dtype = self.embeddings.weight.dtype
        
        # 将 z_flatten 转换为与 embeddings 相同的 dtype，避免 cdist dtype 不匹配
        z_flatten = z_flatten_orig.to(emb_dtype)
        
        # 计算距离并找到最近的 codebook entry
        distances = torch.cdist(z_flatten, self.embeddings.weight)  # [N, num_embeddings]
        print(f"[DEBUG VQ] Distances: min={distances.min().item():.6f}, max={distances.max().item():.6f}, mean={distances.mean().item():.6f}")
        
        # 统计最小距离的分布
        min_distances = distances.min(dim=1)[0]  # [N]
        print(f"[DEBUG VQ] Min distances: mean={min_distances.mean().item():.6f}, std={min_distances.std().item():.6f}, median={min_distances.median().item():.6f}")
        
        encoding_indices = torch.argmin(distances, dim=1)  # [N]
        unique_codes_batch = torch.unique(encoding_indices)
        print(f"[DEBUG VQ] Encoding indices: batch unique codes={len(unique_codes_batch)}/{self.num_embeddings}")
        
        # 统计每个码本被使用的次数
        if len(unique_codes_batch) < 100:
            counts = torch.bincount(encoding_indices, minlength=self.num_embeddings)
            used_counts = counts[counts > 0]
            print(f"[DEBUG VQ] Usage distribution: min={used_counts.min().item()}, max={used_counts.max().item()}, mean={used_counts.float().mean().item():.1f}")
        
        # ============ 码本利用率统计（按每个样本单独计算后取均值）============
        batch_ids = z.coords[:, 0]  # [N] 每个体素所属的样本索引
        unique_batch_ids = torch.unique(batch_ids)
        num_samples = len(unique_batch_ids)
        
        sample_perplexities = []
        sample_entropies = []
        sample_unique_counts = []
        sample_seq_lens: list[int] = []
        sample_unique_ratio_pct: list[float] = []
        epsilon = 1e-10
        
        for bid in unique_batch_ids:
            mask = batch_ids == bid
            sample_indices = encoding_indices[mask]  # 当前样本的码本索引
            
            sample_unique = torch.unique(sample_indices)
            n_u = len(sample_unique)
            sample_unique_counts.append(n_u)
            seq_len = int(mask.sum().item())
            sample_seq_lens.append(seq_len)
            if seq_len > 0:
                sample_unique_ratio_pct.append(100.0 * n_u / float(seq_len))
            
            sample_onehot = F.one_hot(sample_indices, self.num_embeddings).float()
            sample_probs = torch.mean(sample_onehot, dim=0)
            sample_entropy = -torch.sum(sample_probs * torch.log(sample_probs + epsilon))
            sample_entropies.append(sample_entropy.item())
            sample_perplexities.append(torch.exp(sample_entropy).item())
        
        avg_perplexity = sum(sample_perplexities) / num_samples
        avg_entropy = sum(sample_entropies) / num_samples
        avg_unique_count = sum(sample_unique_counts) / num_samples
        avg_utilization_ratio = (avg_unique_count / self.num_embeddings) * 100.0
        avg_seq_len = sum(sample_seq_lens) / max(num_samples, 1)
        avg_unique_ratio_pct = (
            sum(sample_unique_ratio_pct) / len(sample_unique_ratio_pct)
            if sample_unique_ratio_pct
            else 0.0
        )
        min_u_pct = min(sample_unique_ratio_pct) if sample_unique_ratio_pct else 0.0
        max_u_pct = max(sample_unique_ratio_pct) if sample_unique_ratio_pct else 0.0
        total_vq_positions = int(encoding_indices.shape[0])
        batch_unique_div_len_pct = (
            100.0 * len(unique_codes_batch) / float(total_vq_positions)
            if total_vq_positions > 0
            else 0.0
        )
        
        print(
            f"[DEBUG VQ] Per-sample stats (mean of {num_samples} samples): "
            f"unique={avg_unique_count:.1f}, seq_len={avg_seq_len:.1f}, "
            f"unique/len%={avg_unique_ratio_pct:.2f} (min={min_u_pct:.2f}, max={max_u_pct:.2f}), "
            f"perplexity={avg_perplexity:.2f}, entropy={avg_entropy:.4f}"
        )
        print(
            f"[DEBUG VQ] Batch pooled: distinct_codes={len(unique_codes_batch)}/{self.num_embeddings}, "
            f"total_VQ_positions={total_vq_positions}, "
            f"distinct_codes/total_positions%={batch_unique_div_len_pct:.2f}% "
            f"(仅 batch_size=1 时常与 per-sample unique/len% 一致；多样本时后者更有代表性)"
        )
        
        codebook_stats = {
            'perplexity': avg_perplexity,
            'entropy': avg_entropy,
            'unique_count': avg_unique_count,
            'utilization_ratio': avg_utilization_ratio,
            'batch_unique_count': len(unique_codes_batch),
            'batch_unique_codes': unique_codes_batch.detach().cpu().tolist(),
            'avg_seq_len': avg_seq_len,
            'avg_unique_ratio_pct': avg_unique_ratio_pct,
            'min_unique_ratio_pct': min_u_pct,
            'max_unique_ratio_pct': max_u_pct,
            'batch_unique_div_total_positions_pct': batch_unique_div_len_pct,
            'total_vq_positions': float(total_vq_positions),
        }
        
        if only_return_indices:
            # 返回 indices 作为 SparseTensor，保持原始坐标
            result = z.replace(encoding_indices.unsqueeze(-1).float())
            return result
        
        # 量化（embeddings 返回 emb_dtype）
        quantized_feats = self.embeddings(encoding_indices)  # [N, embedding_dim]，dtype=emb_dtype
        # 转回原始 dtype，保证后续 loss 和 straight-through 与 z_flatten_orig 一致
        quantized_feats = quantized_feats.to(input_dtype)
        print(f"[DEBUG VQ] Quantized feats: min={quantized_feats.min().item():.6f}, max={quantized_feats.max().item():.6f}, mean={quantized_feats.mean().item():.6f}")
        
        # 计算commitment loss（两种模式都需要，均使用原始 dtype）
        commitment_loss = F.mse_loss(z_flatten_orig, quantized_feats.detach())
        
        # 根据更新模式选择不同的处理方式
        if self.use_ema_update:
            # EMA模式：在训练时调用EMA更新（使用 emb_dtype 的 z_flatten）
            if self.training and update_codebook:
                self._update_ema(encoding_indices, z_flatten)
            vq_loss = None  # EMA模式不需要vq_loss
            print(f"[DEBUG VQ] EMA mode - Commitment Loss: {commitment_loss.item():.6f}, VQ Loss: None")
        else:
            # 梯度模式：计算vq_loss用于反向传播
            vq_loss = F.mse_loss(quantized_feats, z_flatten_orig.detach())
            print(f"[DEBUG VQ] Gradient mode - VQ Loss: {vq_loss.item():.6f}, Commitment Loss: {commitment_loss.item():.6f}")
            print(f"[DEBUG VQ] VQ Loss requires_grad: {vq_loss.requires_grad}, Commitment Loss requires_grad: {commitment_loss.requires_grad}")
        
        # Straight-through estimator（均在原始 dtype 下执行）
        quantized_feats = z_flatten_orig + (quantized_feats - z_flatten_orig).detach()
        
        # 创建新的 SparseTensor
        quantized = z.replace(quantized_feats)
        encoding_indices_st = z.replace(encoding_indices.unsqueeze(-1).float())
        
        print(f"[DEBUG VQ] Output quantized feats: min={quantized.feats.min().item():.6f}, max={quantized.feats.max().item():.6f}, requires_grad={quantized.feats.requires_grad}\n")
        
        # ============ K-means 特征收集和周期性重估计 ============
        if self.use_kmeans_reinit and self.training and update_codebook and current_step >= 0:
            # 收集特征到水塘采样器
            self.reestimation_reservoir.add(z_flatten)
            
            # 周期性触发 K-means 重估计
            if current_step > 0 and current_step % self.kmeans_interval == 0:
                if dist.is_initialized():
                    print(
                        f'[K-means DEBUG] forward -> reestimate | step={current_step} '
                        f'interval={self.kmeans_interval} rank={dist.get_rank()}/{dist.get_world_size()}',
                        flush=True,
                    )
                self.reestimate()
        
        return quantized, vq_loss, commitment_loss, encoding_indices_st, codebook_stats
    
    @torch.no_grad()
    def reestimate(self):
        """
        使用 K-means 重新初始化码本
        参考 VQFR 实现，用 K-means 聚类中心整体替换码本权重
        在分布式训练中，收集所有 GPU 的样本后进行聚类
        """
        if not self.use_kmeans_reinit:
            return
        
        # 获取当前 GPU 收集的特征
        encodings = self.reestimation_reservoir.contents()
        
        # 如果是分布式训练，收集所有 GPU 的样本
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            t0 = time.perf_counter()

            def _dbg(msg: str) -> None:
                print(
                    f'[K-means DEBUG | rank={rank}/{world_size} | +{time.perf_counter() - t0:.2f}s] {msg}',
                    flush=True,
                )

            _dbg(
                f'enter reestimate | local_encodings={tuple(encodings.shape)} '
                f'device={encodings.device} dtype={encodings.dtype}'
            )
            
            # 获取所有 GPU 的样本数量
            local_size = torch.tensor([encodings.shape[0]], device=encodings.device)
            size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
            t_ag0 = time.perf_counter()
            dist.all_gather(size_list, local_size)
            _dbg(f'all_gather sizes done in {time.perf_counter() - t_ag0:.3f}s | sizes={[s.item() for s in size_list]}')
            
            # 计算总样本数
            total_samples = sum(s.item() for s in size_list)
            
            if rank == 0:
                print(
                    f'[K-means 重估计] 收集到各 GPU 样本数: {[s.item() for s in size_list]}，总计: {total_samples}',
                    flush=True,
                )
            
            # 检查总样本数是否足够
            if total_samples < self.num_embeddings:
                if rank == 0:
                    print(
                        f'[K-means 重估计] 跳过：总样本数不足 ({total_samples} < {self.num_embeddings})',
                        flush=True,
                    )
                # 所有 GPU 同步跳过
                return
            
            # 收集所有 GPU 的样本（只在 rank 0 执行聚类）
            if encodings.shape[0] > 0:
                # 确保张量连续
                encodings = encodings.contiguous()
            
            # 使用 gather 而非 all_gather（只在 rank 0 收集）
            t_p2p0 = time.perf_counter()
            if rank == 0:
                gathered_encodings = []
                _dbg('rank0: start recv chain from other ranks')
                
                for i in range(world_size):
                    if i == 0:
                        # rank 0 的样本
                        if encodings.shape[0] > 0:
                            gathered_encodings.append(encodings)
                    else:
                        # 从其他 rank 接收
                        recv_size = size_list[i].item()
                        if recv_size > 0:
                            recv_tensor = torch.empty(recv_size, encodings.shape[1], 
                                                     device=encodings.device, dtype=encodings.dtype)
                            dist.recv(recv_tensor, src=i)
                            gathered_encodings.append(recv_tensor)
                            _dbg(f'rank0: recv from rank {i} ok | tensor {tuple(recv_tensor.shape)}')
                        else:
                            _dbg(f'rank0: skip recv from rank {i} (empty)')
                
                # 合并所有样本
                all_encodings = torch.cat(gathered_encodings, dim=0) if gathered_encodings else encodings
                _dbg(
                    f'rank0: p2p gather done in {time.perf_counter() - t_p2p0:.3f}s | '
                    f'all_encodings={tuple(all_encodings.shape)}'
                )
            else:
                # 其他 rank 发送样本到 rank 0
                if encodings.shape[0] > 0:
                    _dbg(f'send encodings to rank0 | {tuple(encodings.shape)}')
                    dist.send(encodings, dst=0)
                    _dbg(f'send to rank0 finished in {time.perf_counter() - t_p2p0:.3f}s')
                else:
                    _dbg('no local samples; skip send')
                all_encodings = None

            # 确认所有 rank 都完成点对点，再让 rank0 独占运行 K-means（其它 rank 会早到 broadcast 等待）
            t_bar0 = time.perf_counter()
            dist.barrier()
            _dbg(f'post-p2p barrier ok in {time.perf_counter() - t_bar0:.3f}s')
            
            # 只在 rank 0 执行 K-means 聚类
            if rank == 0:
                print(
                    f'[K-means 重估计] 开始，使用 {all_encodings.shape[0]} 个样本重建 {self.num_embeddings} 个码本向量...',
                    flush=True,
                )
                
                try:
                    t_cpu0 = time.perf_counter()
                    encodings_np = all_encodings.cpu().numpy()
                    _dbg(
                        f'rank0: cpu().numpy() done in {time.perf_counter() - t_cpu0:.3f}s | '
                        f'np_shape={encodings_np.shape} dtype={encodings_np.dtype}'
                    )

                    t_km0 = time.perf_counter()
                    faiss_gpu_device = all_encodings.device.index
                    print(
                        f'[K-means DEBUG | rank=0] FAISS GPU KMeans '
                        f'(gpu={faiss_gpu_device}, n_samples={encodings_np.shape[0]}, '
                        f'n_clusters={self.num_embeddings}) ...',
                        flush=True,
                    )
                    clustered = _cluster_centers_for_codebook_reinit(
                        encodings_np,
                        self.num_embeddings,
                        faiss_gpu_device=faiss_gpu_device,
                    )
                    print(
                        f'[K-means DEBUG | rank=0] FAISS GPU KMeans finished in {time.perf_counter() - t_km0:.2f}s',
                        flush=True,
                    )
                    
                    # 用 K-means 的聚类中心整体替换码本
                    new_embeddings = torch.tensor(clustered, 
                                                  dtype=self.embeddings.weight.dtype,
                                                  device=self.embeddings.weight.device)
                    self.embeddings.weight.data[...] = new_embeddings
                    
                    print(f'[K-means 重估计] 完成！码本已更新', flush=True)
                    
                except Exception as e:
                    print(f'[K-means 重估计] 失败：{e}', flush=True)
                    traceback.print_exc()
            
            if rank != 0:
                print(
                    '[K-means DEBUG] 本 rank 在 dist.broadcast 上等待 rank0 完成 FAISS GPU 聚类 '
                    '（rank0 完成后会广播码本）',
                    flush=True,
                )
            _dbg('about to enter dist.broadcast(embeddings.weight)')
            t_bc0 = time.perf_counter()
            dist.broadcast(self.embeddings.weight.data, src=0)
            _dbg(f'dist.broadcast done in {time.perf_counter() - t_bc0:.3f}s')
            
            # 所有 GPU 同步清空水塘采样器
            self.reestimation_reservoir.reset()
            
            # 如果是 EMA 模式，也重置 EMA 统计量
            if self.use_ema_update:
                self.ema_cluster_size.zero_()
                self.ema_w.zero_()
                self._ema_initialized.fill_(False)
                if rank == 0:
                    print(f'[K-means 重估计] 同时重置了 EMA 统计量')
        
        else:
            # 单 GPU 模式
            t0 = time.perf_counter()
            if encodings.shape[0] < self.num_embeddings:
                print(f'[K-means 重估计] 跳过：样本数不足 ({encodings.shape[0]} < {self.num_embeddings})')
                return
            
            print(
                f'[K-means 重估计] 开始，使用 {encodings.shape[0]} 个样本重建 {self.num_embeddings} 个码本向量...',
                flush=True,
            )
            
            encodings_np = encodings.cpu().numpy()
            print(
                f'[K-means DEBUG | single-GPU] numpy shape={encodings_np.shape} '
                f'cpu_numpy_s={time.perf_counter() - t0:.3f}',
                flush=True,
            )
            
            try:
                t_km = time.perf_counter()
                faiss_gpu_device = encodings.device.index
                clustered = _cluster_centers_for_codebook_reinit(
                    encodings_np,
                    self.num_embeddings,
                    faiss_gpu_device=faiss_gpu_device,
                )
                print(
                    f'[K-means DEBUG | single-GPU] FAISS GPU KMeans '
                    f'(gpu={faiss_gpu_device}) done in {time.perf_counter() - t_km:.2f}s',
                    flush=True,
                )
                
                # 用 K-means 的聚类中心整体替换码本
                self.embeddings.weight.data[...] = torch.tensor(clustered, 
                                                                dtype=self.embeddings.weight.dtype,
                                                                device=self.embeddings.weight.device)
                
                # 清空水塘采样器，重新收集特征
                self.reestimation_reservoir.reset()
                
                # 如果是 EMA 模式，也重置 EMA 统计量
                if self.use_ema_update:
                    self.ema_cluster_size.zero_()
                    self.ema_w.zero_()
                    self._ema_initialized.fill_(False)
                    print(f'[K-means 重估计] 同时重置了 EMA 统计量')
                
                print(f'[K-means 重估计] 完成！码本已更新')
                
            except Exception as e:
                print(f'[K-means 重估计] 失败：{e}')
    
    @torch.no_grad()
    def _update_ema(self, encoding_indices, z_flatten):
        """
        使用EMA更新码本（仅在use_ema_update=True时调用）
        
        Args:
            encoding_indices: 分配的码本索引 [N]
            z_flatten: encoder输出的特征向量 [N, embedding_dim]
        """
        print(f"[DEBUG EMA] === Starting EMA Update ===")
        # EMA 统计全程在 float32 下进行，保证数值稳定；只在写回 embeddings.weight 时转换回原 dtype
        z_flatten = z_flatten.float()
        print(f"[DEBUG EMA] Input z_flatten: shape={z_flatten.shape}, min={z_flatten.min().item():.6f}, max={z_flatten.max().item():.6f}, mean={z_flatten.mean().item():.6f}, std={z_flatten.std().item():.6f}")
        print(f"[DEBUG EMA] Encoding indices: shape={encoding_indices.shape}, unique codes={len(torch.unique(encoding_indices))}/{self.num_embeddings}")
        
        # 计算one-hot编码
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # [N, num_embeddings]
        print(f"[DEBUG EMA] One-hot encodings: shape={encodings.shape}, sum={encodings.sum().item():.1f}")
        
        # 检查当前EMA状态
        print(f"[DEBUG EMA] OLD ema_cluster_size: sum={self.ema_cluster_size.sum().item():.1f}, min={self.ema_cluster_size.min().item():.6f}, max={self.ema_cluster_size.max().item():.6f}")
        print(f"[DEBUG EMA] OLD ema_w: min={self.ema_w.min().item():.6f}, max={self.ema_w.max().item():.6f}, mean={self.ema_w.mean().item():.6f}")
        print(f"[DEBUG EMA] OLD embeddings: min={self.embeddings.weight.data.min().item():.6f}, max={self.embeddings.weight.data.max().item():.6f}, mean={self.embeddings.weight.data.mean().item():.6f}")
        
        # 第一次初始化：直接用batch统计初始化EMA
        if not self._ema_initialized:
            print(f"[DEBUG EMA] ⚠️  First EMA update - Initializing from batch statistics")
            batch_cluster_size = encodings.sum(0)  # [num_embeddings]
            batch_w = encodings.t() @ z_flatten  # [num_embeddings, embedding_dim]
            
            # 对于未使用的码本，保持原始初始化值
            # 对于使用过的码本，用batch统计初始化
            self.ema_cluster_size.copy_(batch_cluster_size)
            self.ema_w.copy_(batch_w)
            self._ema_initialized.fill_(True)
            
            print(f"[DEBUG EMA] Initialized ema_cluster_size: sum={self.ema_cluster_size.sum().item():.1f}, nonzero={(self.ema_cluster_size > 0).sum().item()}/{self.num_embeddings}")
            print(f"[DEBUG EMA] Initialized ema_w: min={self.ema_w.min().item():.6f}, max={self.ema_w.max().item():.6f}")
        
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
        print(f"[DEBUG EMA] Smoothed cluster size: min={smoothed_cluster_size.min().item():.6f}, max={smoothed_cluster_size.max().item():.6f}, mean={smoothed_cluster_size.mean():.6f}")
        
        # 更新码本向量（在 float32 下计算，最后转回 embeddings 原有 dtype 再写回）
        emb_dtype = self.embeddings.weight.data.dtype
        new_embeddings = torch.zeros(self.num_embeddings, self.embedding_dim,
                                     dtype=torch.float32, device=self.embeddings.weight.device)
        used_mask = new_cluster_size > 0
        new_embeddings[used_mask] = new_w[used_mask] / (smoothed_cluster_size[used_mask].unsqueeze(1) + 1e-7)
        new_embeddings[~used_mask] = self.embeddings.weight.data[~used_mask].float()  # 保持未使用码本不变
        
        print(f"[DEBUG EMA] NEW embeddings (all codes): min={new_embeddings.min().item():.6f}, max={new_embeddings.max().item():.6f}, mean={new_embeddings.mean().item():.6f}, std={new_embeddings.std().item():.6f}")
        print(f"[DEBUG EMA] Used codes: {used_mask.sum().item()}/{self.num_embeddings}, Unused codes: {(~used_mask).sum().item()}/{self.num_embeddings}")
        
        # 检查更新后的码本中是否有near-zero向量
        updated_norms = torch.norm(new_embeddings[used_mask], dim=1)
        if len(updated_norms) > 0:
            print(f"[DEBUG EMA] Updated codes norms: min={updated_norms.min().item():.6f}, max={updated_norms.max().item():.6f}, mean={updated_norms.mean().item():.6f}")
        
        # 写回 embeddings.weight，转换回原始 dtype（fp16 or fp32）
        self.embeddings.weight.data.copy_(new_embeddings.to(emb_dtype))
        
        # 更新buffer（buffer 本身是 float32，直接 copy）
        self.ema_cluster_size.copy_(new_cluster_size)
        self.ema_w.copy_(new_w)
        
        print(f"[DEBUG EMA] === EMA Update Complete ===\n")


# 仅第 0 维为码本条数、可与预训练 checkpoint 对齐合并的 VQ 张量
_VQ_CODEBOOK_DIM0_KEYS = frozenset({'embeddings.weight', 'ema_cluster_size', 'ema_w'})


def _merge_vq_tensor_codebook_dim0(
    current: torch.Tensor,
    pretrained: torch.Tensor,
    key: str,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """
    当预训练与当前模型仅在码本条数（dim0）上不同时，将预训练权重写入前缀；
    多出的码本位置保留 current 中已有值（embeddings 为构造时的 N(0,1)，EMA buffer 为 0）。

    Returns:
        (merged_tensor, log_msg)；若无法按码本条数合并则返回 (None, None)。
    """
    if key not in _VQ_CODEBOOK_DIM0_KEYS:
        return None, None
    if current.ndim != pretrained.ndim or current.shape[1:] != pretrained.shape[1:]:
        return None, None

    n_cur = int(current.shape[0])
    n_pt = int(pretrained.shape[0])
    out = current.clone()
    n_copy = min(n_cur, n_pt)
    out[:n_copy] = pretrained[:n_copy].to(device=out.device, dtype=out.dtype)

    if n_cur == n_pt:
        return out, None

    if n_cur > n_pt:
        if key == 'embeddings.weight':
            detail = (
                f'索引 [0:{n_pt}) 自预训练加载，'
                f'索引 [{n_pt}:{n_cur}) 保留当前模型中的随机初始化 (N(0,1)，与 nn.Embedding 一致)'
            )
        elif key == 'ema_cluster_size':
            detail = f'索引 [0:{n_pt}) 自预训练加载，索引 [{n_pt}:{n_cur}) 置 0（与新建 VQ 一致）'
        elif key == 'ema_w':
            detail = f'索引 [0:{n_pt}) 自预训练加载，索引 [{n_pt}:{n_cur}) 置 0（与新建 VQ 一致）'
        else:
            detail = ''
        msg = (
            f'码本条数扩展: checkpoint={n_pt} -> 当前模型={n_cur}；'
            f'键 `{key}`：{detail}'
        )
    else:
        msg = (
            f'码本条数收缩: checkpoint={n_pt} -> 当前模型={n_cur}；'
            f'键 `{key}`：仅加载前 {n_copy} 条，其余权重已丢弃'
        )
    return out, msg


def _decoder_upsample_block_count(state_dict: dict) -> int:
    """从 state_dict 键名推断 decoder 中 upsample 的块数（upsample.0 .. upsample.(n-1)）。"""
    mx = -1
    for k in state_dict:
        m = re.match(r"^upsample\.(\d+)\.", k)
        if m:
            mx = max(mx, int(m.group(1)))
    return (mx + 1) if mx >= 0 else 0


def _encoder_downsample_block_count(state_dict: dict) -> int:
    """从 state_dict 键名推断 encoder 中 downsample 的块数（downsample.0 .. downsample.(n-1)）。"""
    mx = -1
    for k in state_dict:
        m = re.match(r"^downsample\.(\d+)\.", k)
        if m:
            mx = max(mx, int(m.group(1)))
    return (mx + 1) if mx >= 0 else 0


def _remap_decoder_upsample_state_dict(state_dict: dict, shift: int) -> dict:
    """checkpoint 比当前网少 shift 个 upsample 块时，把 ckpt 的 upsample.i 映射为 upsample.(i+shift)（前置块保持随机初始化）。"""
    if shift <= 0 or not state_dict:
        return state_dict
    remapped = {}
    for k, v in state_dict.items():
        m = re.match(r"^upsample\.(\d+)\.(.+)$", k)
        if m:
            idx = int(m.group(1))
            remapped[f"upsample.{idx + shift}.{m.group(2)}"] = v
        else:
            remapped[k] = v
    return remapped


def _remap_decoder_upsample_state_dict_drop_leading(state_dict: dict, drop: int) -> dict:
    """checkpoint 多 drop 个前置 upsample 时，丢弃 upsample.0..drop-1，并把 upsample.(drop+k) 重命名为 upsample.k。"""
    if drop <= 0 or not state_dict:
        return state_dict
    remapped: Dict[str, Any] = {}
    for k, v in state_dict.items():
        m = re.match(r"^upsample\.(\d+)\.(.+)$", k)
        if m:
            idx = int(m.group(1))
            new_idx = idx - drop
            if new_idx < 0:
                continue
            remapped[f"upsample.{new_idx}.{m.group(2)}"] = v
        else:
            remapped[k] = v
    return remapped


def _remap_decoder_ckpt_old3_to_new4(state_dict: dict) -> dict:
    """
    旧版 3 个 upsample（512→128→64→32）→ 新版 4 个渐进块（512→256→128→64→32）。
    丢弃旧 upsample.0；旧 upsample.1/2 重映射到新 upsample.2/3（形状一致可加载）。
    """
    if not state_dict:
        return state_dict
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith("upsample.0."):
            continue
        if k.startswith("upsample.1."):
            out["upsample.2." + k[len("upsample.1.") :]] = v
        elif k.startswith("upsample.2."):
            out["upsample.3." + k[len("upsample.2.") :]] = v
        else:
            out[k] = v
    return out


def _prepare_encoder_ckpt_for_progressive_load(enc_ckpt: dict, model_n: int) -> dict:
    """
    旧 encoder 为「3 级主干 + 末尾 1×512→512」共 4 块时，最后一层在 downsample.3；
    新 encoder 为「4 级渐进 + 末尾 1×512→512」共 5 块时，最后一层在 downsample.4。
    将 ckpt 的 downsample.3.* 重命名为 downsample.4.*，以便与当前模型对齐加载。
    """
    if not enc_ckpt:
        return enc_ckpt
    ckpt_n = _encoder_downsample_block_count(enc_ckpt)
    if ckpt_n == 4 and model_n == 5:
        remapped: Dict[str, Any] = {}
        for k, v in enc_ckpt.items():
            if k.startswith("downsample.3."):
                remapped[k.replace("downsample.3.", "downsample.4.", 1)] = v
            else:
                remapped[k] = v
        return remapped
    return enc_ckpt


def _merge_state_dict_shape_strict(
    model_sd: dict, src_sd: dict
) -> Tuple[dict, int, int]:
    """
    以 model_sd 的键为全集：若 src 中同名张量形状一致则拷贝，否则保留 model 中的张量（随机初始化权重）。
    返回 (merged_dict, n_loaded, n_shape_mismatch_or_missing_in_src)。
    """
    out: Dict[str, Any] = {}
    n_ok = 0
    n_skip = 0
    for k, cur in model_sd.items():
        if k not in src_sd:
            out[k] = cur.detach().clone() if isinstance(cur, torch.Tensor) else cur
            n_skip += 1
            continue
        v = src_sd[k]
        if isinstance(v, torch.Tensor) and isinstance(cur, torch.Tensor) and v.shape == cur.shape:
            out[k] = v.clone().to(dtype=cur.dtype)
            n_ok += 1
        else:
            out[k] = cur.detach().clone() if isinstance(cur, torch.Tensor) else cur
            n_skip += 1
    return out, n_ok, n_skip


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
                 use_kmeans_reinit: bool = False,  # 新增：是否使用K-means重新初始化
                 kmeans_interval: int = 2000,   # 新增：K-means重估计间隔
                 reservoir_size: int = 16384,   # 新增：水塘采样器容量
                 # 新增：ShapeLLM VQVAE3D 风格的「块内通道拼接」
                 # vq_group_size=G 表示把空间上 (G^(1/3))^3 个相邻格点沿通道维拼成 C*G 维再送 VQ，
                 # VQ 码本 embedding_dim = embed_dim * vq_group_size。默认 8 对应 2×2×2 立方块；
                 # 1 表示不拼接（旧行为）。块内缺失/不满的 offset 统一 0 填充（与 ShapeLLM view 等价）。
                 vq_group_size: int = 8,
                 # 训练时是否与 Decode 推理一致：把每个 block 展开为完整 block_side^3 子格后送 decoder。
                 # False 为旧训练路径：只按原始 sparse mask 反拼接。
                 train_full_block_decode: bool = True,
                 mlp_ratio: float = 4,
                 attn_mode: str = "swin",
                 window_size: int = 8,
                 pe_mode: str = "ape",
                 qk_rms_norm: bool = False,
                 representation_config: dict = None,
                 # 在渐进 4 级上下采样（共 ×16 空间）之外，再追加 L 对：encoder 末尾同通道下采样、decoder 开头同通道 subdivide；0 表示无额外级。
                 extra_down_up_levels: int = 0):

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
        self.extra_down_up_levels = int(extra_down_up_levels)
        if self.extra_down_up_levels < 0:
            raise ValueError("extra_down_up_levels must be >= 0")

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
            extra_down_up_levels=self.extra_down_up_levels,
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
            extra_down_up_levels=self.extra_down_up_levels,
        )
        
        # 块内通道拼接设置
        self.vq_group_size = int(vq_group_size)
        self.vq_block_side = _infer_block_side(self.vq_group_size)  # 立方根，1/2/3/...
        self.train_full_block_decode = bool(train_full_block_decode)
        self.freeze_codebook_updates = False
        vq_embed_dim = embed_dim * self.vq_group_size

        # Vector Quantizer（替代 VAE 的高斯分布）
        # 注意：启用 vq_group_size>1 时，每个 VQ 位置的向量维度为 embed_dim * vq_group_size
        self.vq = SparseVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=vq_embed_dim,
            beta=0.25,
            use_ema_update=use_ema_update,
            decay=vq_decay,
            epsilon=vq_epsilon,
            use_kmeans_reinit=use_kmeans_reinit,
            kmeans_interval=kmeans_interval,
            reservoir_size=reservoir_size
        )
        
        self.embed_dim = embed_dim
        self.use_ema_update = use_ema_update
        self.use_kmeans_reinit = use_kmeans_reinit

    def forward(
        self,
        batch,
        current_step: int = -1,
        full_block_decode: Optional[bool] = None,
        band_prune: Optional[dict] = None,
        gt_prune: Optional[dict] = None,
    ):
        """
        训练时的完整前向传播
        Args:
            batch: 输入数据批次
            current_step: 当前训练步数，用于 K-means 重估计（-1 表示不使用）
            band_prune: 可选，传给 ``SparseSDFDecoder``；见 ``decoder.build_band_prune_aabb``。
                若 ``batch`` 为 dict 且含 ``band_prune`` 且此处为 None，则使用 ``batch['band_prune']``。
                无 GT / 无种子坐标时传 ``{"enabled": False}`` 或不传本参数以关闭裁剪。
            gt_prune: 可选，传给 ``SparseSDFDecoder``；GT 键集合对齐精确裁剪，
                见 ``SparseSDFDecoder.forward`` 文档。训练时由 trainer 构造并传入；
                推理时可传 ``{'mode': 'geometry', 'extra_band_factor': 4.0, 'resolution': 512}``。
        """
        if band_prune is None and isinstance(batch, dict):
            band_prune = batch.get("band_prune")
        if gt_prune is None and isinstance(batch, dict):
            gt_prune = batch.get("gt_prune")

        z, vq_loss, commitment_loss, codebook_stats = self.encode(
            batch,
            current_step=current_step,
            full_block_decode=full_block_decode,
        )

        print(f"[DEBUG forward] Calling decoder...")
        reconst_x = self.decoder(z, band_prune=band_prune, gt_prune=gt_prune)
        print(f"[DEBUG forward] Decoder output: shape={reconst_x.shape}, feats.shape={reconst_x.feats.shape}")
        print(f"[DEBUG forward] Decoder output feats: min={reconst_x.feats.min().item():.6f}, max={reconst_x.feats.max().item():.6f}, mean={reconst_x.feats.mean().item():.6f}")
        print(f"[DEBUG forward] Decoder output requires_grad: {reconst_x.feats.requires_grad}")

        outputs = {
            'reconst_x': reconst_x,
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'codebook_stats': codebook_stats
        }
        return outputs

    def encode(
        self,
        batch,
        only_return_indices: bool = False,
        current_step: int = -1,
        full_block_decode: Optional[bool] = None,
    ):
        """
        编码过程，替代 VAE 的采样过程
        Args:
            batch: 输入数据批次。可以是：
                  - SparseTensor：训练时使用
                  - dict：推理时使用，包含 'sparse_sdf', 'sparse_index', 'batch_idx' 键
            only_return_indices: 是否只返回量化索引（用于推理）
            current_step: 当前训练步数，用于 K-means 重估计（-1 表示不使用）
            full_block_decode: 非 only_return_indices 时是否使用满块展开后送 decoder；
                None 时使用 ``self.train_full_block_decode``。
        Returns:
            如果 only_return_indices=True: 返回 encoding_indices
            否则: 返回 (z, vq_loss, commitment_loss, codebook_stats)
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

        # —— ShapeLLM 风格：在送入 VQ 之前，把 block_side^3 个空间邻居沿通道维拼接 ——
        block_side = self.vq_block_side
        original_coords = h.coords
        channels = h.feats.shape[1]
        h_merged, inv_blk, off_blk, _mask_blk = _pack_block_channel(h, block_side)
        if block_side != 1:
            print(
                f"[DEBUG encode] block_channel pack: block_side={block_side}, "
                f"N={original_coords.shape[0]}→M={h_merged.feats.shape[0]}, "
                f"C={channels}→{h_merged.feats.shape[1]} (=C*{block_side**3})"
            )

        if only_return_indices:
            # 仅返回 block 级量化索引（Encode 路径）
            encoding_indices = self.vq(h_merged, only_return_indices=True, current_step=current_step)
            return encoding_indices

        # 量化（替代 VAE 的采样），在 block 级别上进行
        quantized_merged, vq_loss, commitment_loss, _, codebook_stats = self.vq(
            h_merged,
            current_step=current_step,
            update_codebook=not bool(getattr(self, "freeze_codebook_updates", False)),
        )
        if vq_loss is not None:
            print(f"[DEBUG encode] Quantization results: vq_loss={vq_loss.item():.6f}, commitment_loss={commitment_loss.item():.6f}")
        else:
            print(f"[DEBUG encode] Quantization results: vq_loss=None (EMA mode), commitment_loss={commitment_loss.item():.6f}")

        if full_block_decode is None:
            full_block_decode = bool(self.train_full_block_decode)

        if full_block_decode:
            # 与推理 Decode 一致：每个 block 展开为完整 block_side^3 子格后交给 decoder。
            quantized = _expand_blocks_full(quantized_merged, block_side, channels)
        else:
            # 旧训练路径：只在原始 sparse mask 上反拼接。
            quantized = _unpack_block_channel(
                quantized_merged, inv_blk, off_blk, original_coords, block_side, channels
            )

        return quantized, vq_loss, commitment_loss, codebook_stats
    
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
    
    def Decode(
        self,
        encoding_indices: sp.SparseTensor,
        band_prune: Optional[dict] = None,
        gt_prune: Optional[dict] = None,
    ):
        """
        从离散索引解码（推理时使用）。

        Args:
            encoding_indices: SparseTensor；coords 为 **block 级别**（与 Encode 返回一致），
                feats 为每个 block 的码本索引（``[M, 1]``）。启用 ``vq_group_size>1`` 时，
                每个 block 的 ``embedding`` 被拆分为 ``block_side^3`` 份按通道复原到
                原始格点（``x = bx*bs + lx`` 等），缺失掩码信息不可知，故默认 **满块展开**。
            band_prune: 可选；无 GT 时传 ``seed_coords`` + ``seed_resolution`` + ``output_resolution``
                + ``extra_band_factor``（与预处理 ``preprocessing_extra_band_factor`` 一致），
                见 ``decoder.build_band_prune_aabb``。
            gt_prune: 可选；推理时传入 ``mode='geometry'`` 字典，以量化后的 latent 坐标为锚点，
                按 ``extra_band_factor`` 在每个上采样步后裁剪解码体素，使 decoder 输出严格限于
                ``preprocessing_extra_band_factor`` 定义的宽频壳层内。字典格式::

                    {
                        'mode':              'geometry',
                        'extra_band_factor': float,  # 如 4.0，与预处理保持一致
                        'resolution':        int,    # 输出分辨率，如 512
                    }

                见 ``SparseSDFDecoder.forward`` 的 gt_prune 文档。
        Returns:
            recon: 重建的 SparseTensor，体素范围被 gt_prune 约束在宽频壳层内。
        """
        indices = encoding_indices.feats.long().squeeze(-1)  # [M]

        if indices.max() >= self.vq.embeddings.num_embeddings:
            print(f"[ERROR Decode] Index out of range! max index: {indices.max().item()}, codebook size: {self.vq.embeddings.num_embeddings}")

        # 查表得到 block 级合并特征 [M, C*G]
        quantized_merged_feats = self.vq.embeddings(indices)
        quantized_merged = encoding_indices.replace(quantized_merged_feats)

        # 满块展开到 [M*G, C] 对应的原始分辨率稀疏张量
        quantized = _expand_blocks_full(
            quantized_merged, self.vq_block_side, self.embed_dim
        )

        recon = self.decoder(quantized, band_prune=band_prune, gt_prune=gt_prune)
        return recon

    def decode_mesh(self,
                    latents,
                    voxel_resolution: int = 512,
                    mc_threshold: float = 0.0,
                    return_feat: bool = False,
                    factor: float = 1.0,
                    band_prune: Optional[dict] = None):
        voxel_resolution = int(voxel_resolution / factor)
        reconst_x = self.decoder(
            latents, factor=factor, return_feat=return_feat, band_prune=band_prune
        )
        if return_feat:
            return reconst_x
        outputs = self.sparse2mesh(reconst_x, voxel_resolution=voxel_resolution, mc_threshold=mc_threshold)
        
        return outputs

    def sparse2mesh(self,
                    reconst_x: torch.FloatTensor,
                    voxel_resolution: int = 512,
                    mc_threshold: float = 0.0):

        raw = sparse_feats_coords_to_trimeshes(
            reconst_x.feats.float(),
            reconst_x.coords,
            int(voxel_resolution),
            float(mc_threshold),
        )
        meshes = []
        for m in raw:
            if m is None:
                meshes.append(trimesh.Trimesh(vertices=[], faces=[], process=False))
            else:
                meshes.append(m)
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
        print(f"\n{'='*80}")
        print(f"🔧 [DEBUG] SparseSDFVQVAE.load_pretrained_vae 被调用")
        print(f"{'='*80}")
        print(f"📊 输入参数:")
        print(f"   encoder_state_dict: {len(encoder_state_dict) if encoder_state_dict else 0} 个参数")
        print(f"   decoder_state_dict: {len(decoder_state_dict) if decoder_state_dict else 0} 个参数")
        print(f"   vq_state_dict: {len(vq_state_dict) if vq_state_dict else 0} 个参数")
        print(f"   vq_state_dict is None: {vq_state_dict is None}")
        print(f"   vq_state_dict is not None and len(vq_state_dict) > 0: {vq_state_dict is not None and len(vq_state_dict) > 0}")
        
        if vq_state_dict:
            print(f"\n📋 VQ state_dict 详情:")
            for key, value in vq_state_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
        
        # 加载 encoder 参数（仅拷贝形状完全一致的键；渐进 4 级与旧 3 级块索引对齐由 _prepare_encoder_ckpt_for_progressive_load 处理）
        print(f"\n📥 加载 Encoder 参数...")
        enc_ckpt = dict(encoder_state_dict) if encoder_state_dict else {}
        enc_ckpt = _prepare_encoder_ckpt_for_progressive_load(enc_ckpt, len(self.encoder.downsample))
        enc_merged, n_e_ok, n_e_skip = _merge_state_dict_shape_strict(self.encoder.state_dict(), enc_ckpt)
        _menc, _uenc = self.encoder.load_state_dict(enc_merged, strict=False)
        print(
            f"   ℹ️  encoder：形状匹配并成功加载 {n_e_ok} 个张量；"
            f"形状不符或 ckpt 缺键处保留当前初始化（计 {n_e_skip} 项）"
        )
        if _menc:
            print(f"   [encoder] missing_keys ({len(_menc)}): {_menc[:12]}{'...' if len(_menc) > 12 else ''}")
        if _uenc:
            print(f"   [encoder] unexpected_keys ({len(_uenc)}): {_uenc[:12]}{'...' if len(_uenc) > 12 else ''}")
        print(f"   ✅ Encoder 加载完成")
        
        # 加载 decoder 参数（旧 3 upsample → 新 4 渐进时做键重映射；块数不等时 shift / drop_leading）
        print(f"\n📥 加载 Decoder 参数...")
        dec_ckpt = dict(decoder_state_dict) if decoder_state_dict else {}
        if dec_ckpt:
            model_n = len(self.decoder.upsample)
            ckpt_n = _decoder_upsample_block_count(dec_ckpt)
            if ckpt_n == 3 and model_n == 4:
                dec_ckpt = _remap_decoder_ckpt_old3_to_new4(dec_ckpt)
                print(
                    "   ℹ️  decoder：检测到旧版 3 个 upsample 权重，已重映射到新版 4 块 "
                    "（丢弃旧 upsample.0；旧 1→新 2、旧 2→新 3）；新 upsample.0/1 保持随机初始化"
                )
            else:
                shift = model_n - ckpt_n
                if shift > 0:
                    dec_ckpt = _remap_decoder_upsample_state_dict(dec_ckpt, shift)
                    print(
                        f"   ℹ️  decoder upsample：checkpoint 含 {ckpt_n} 块，当前模型 {model_n} 块，"
                        f"已对 upsample.* 做 +{shift} 键重映射（前置 {shift} 块保持初始化）"
                    )
                elif ckpt_n > model_n:
                    drop = ckpt_n - model_n
                    dec_ckpt = _remap_decoder_upsample_state_dict_drop_leading(dec_ckpt, drop)
                    print(
                        f"   ℹ️  decoder upsample：checkpoint {ckpt_n} 块 > 当前 {model_n} 块，"
                        f"已丢弃 ckpt 前 {drop} 个前置 upsample 并重排键以对齐"
                    )
        dec_merged, n_d_ok, n_d_skip = _merge_state_dict_shape_strict(self.decoder.state_dict(), dec_ckpt)
        _mdec, _udec = self.decoder.load_state_dict(dec_merged, strict=False)
        print(
            f"   ℹ️  decoder：形状匹配并成功加载 {n_d_ok} 个张量；"
            f"形状不符或 ckpt 缺键处保留当前初始化（计 {n_d_skip} 项）"
        )
        if _mdec:
            print(f"   [decoder] missing_keys ({len(_mdec)}): {_mdec[:12]}{'...' if len(_mdec) > 12 else ''}")
        if _udec:
            print(f"   [decoder] unexpected_keys ({len(_udec)}): {_udec[:12]}{'...' if len(_udec) > 12 else ''}")
        print(f"   ✅ Decoder 加载完成")
        
        # 强制将encoder和decoder转换为正确的dtype
        # 这样可以确保即使checkpoint中的权重是float16，也能正确转换
        print(f"\n🔧 检查并修正 dtype...")
        if not self.encoder.use_fp16:
            print(f"   Encoder use_fp16=False，转换为 float32")
            self.encoder.convert_to_fp32()
        else:
            print(f"   Encoder use_fp16=True，转换为 float16")
            self.encoder.convert_to_fp16()
            
        if not self.decoder.use_fp16:
            print(f"   Decoder use_fp16=False，转换为 float32")
            self.decoder.convert_to_fp32()
        else:
            print(f"   Decoder use_fp16=True，转换为 float16")
            self.decoder.convert_to_fp16()
        
        print(f"\n✅ Loaded pretrained VAE parameters")
        print(f"   Encoder: {len(encoder_state_dict)} parameters loaded")
        print(f"   Decoder: {len(decoder_state_dict)} parameters loaded")
        
        # 加载 VQ 参数（如果提供）
        if vq_state_dict is not None and len(vq_state_dict) > 0:
            print(f"\n📥 加载 VQ 参数...")
            print(f"   VQ state_dict 包含 {len(vq_state_dict)} 个键")
            
            # 记录加载前的codebook
            print(f"\n📊 加载前的 VQ Codebook:")
            before_embeddings = self.vq.embeddings.weight.data.clone()
            print(f"   Shape: {before_embeddings.shape}")
            print(f"   Min: {before_embeddings.min().item():.6f}, Max: {before_embeddings.max().item():.6f}")
            print(f"   Mean: {before_embeddings.mean().item():.6f}, Std: {before_embeddings.std().item():.6f}")
            print(f"   前3个code的前5维:")
            for i in range(min(3, before_embeddings.shape[0])):
                print(f"     Code {i}: {before_embeddings[i, :5].tolist()}")
            
            vq_dict = self.vq.state_dict()
            print(f"\n🔍 当前 VQ 模型的 state_dict 包含 {len(vq_dict)} 个键:")
            for key in vq_dict.keys():
                val = vq_dict[key]
                print(f"   - {key}: shape={val.shape if isinstance(val, torch.Tensor) else type(val)}")

            # 预检：码本条数（num_embeddings）是否与 checkpoint 一致
            cur_n = int(self.vq.num_embeddings)
            ckpt_n = None
            if 'embeddings.weight' in vq_state_dict and isinstance(vq_state_dict['embeddings.weight'], torch.Tensor):
                ckpt_n = int(vq_state_dict['embeddings.weight'].shape[0])
            if ckpt_n is not None and ckpt_n != cur_n:
                print(f"\n{'!'*80}")
                print(f"📣 【码本大小变化】预训练 checkpoint 的码本条数为 {ckpt_n}，当前模型为 {cur_n}")
                if cur_n > ckpt_n:
                    print(
                        f"   → 将加载前 {ckpt_n} 个 code 的权重；"
                        f"多出的 {cur_n - ckpt_n} 个 code 使用当前模型中的初始化 "
                        f"(embeddings: 随机 N(0,1)；EMA 统计 buffer: 0)"
                    )
                else:
                    print(
                        f"   → 仅加载前 {cur_n} 个 code 的权重，checkpoint 中多出的 {ckpt_n - cur_n} 个 code 将被忽略"
                    )
                print(f"{'!'*80}\n")
            
            # 筛选可用的参数（避免形状不匹配）；支持仅第 0 维码本条数变化的合并加载
            loaded_keys = []
            skipped_keys = []
            codebook_merge_msgs: list[str] = []
            print(f"\n🔄 开始匹配和加载参数...")
            for key, value in vq_state_dict.items():
                print(f"\n   检查键: {key}")
                if key in vq_dict:
                    print(f"     ✓ 键存在于模型中")
                    print(f"     预训练 shape: {value.shape}")
                    print(f"     当前模型 shape: {vq_dict[key].shape}")
                    if vq_dict[key].shape == value.shape:
                        print(f"     ✓ Shape 匹配！正在更新...")
                        vq_dict[key] = value.to(device=vq_dict[key].device, dtype=vq_dict[key].dtype)
                        loaded_keys.append(key)
                        print(f"     ✅ 已更新到 vq_dict")
                        
                        # 如果是 embeddings.weight，打印详细信息
                        if key == 'embeddings.weight':
                            print(f"     📊 预训练 embeddings 统计:")
                            print(f"        Min: {value.min().item():.6f}, Max: {value.max().item():.6f}")
                            print(f"        Mean: {value.mean().item():.6f}, Std: {value.std().item():.6f}")
                            print(f"        前3个code的前5维:")
                            for i in range(min(3, value.shape[0])):
                                print(f"          Code {i}: {value[i, :5].tolist()}")
                    else:
                        merged, merge_msg = _merge_vq_tensor_codebook_dim0(vq_dict[key], value, key)
                        if merged is not None:
                            print(f"     🔀 码本条数不一致，已按前缀合并加载（见下方说明）")
                            vq_dict[key] = merged
                            loaded_keys.append(key)
                            if merge_msg:
                                codebook_merge_msgs.append(merge_msg)
                                print(f"        {merge_msg}")
                        else:
                            print(f"     ✗ Shape 不匹配且无法按码本条数合并，跳过")
                            skipped_keys.append(f"{key} (shape mismatch: {vq_dict[key].shape} vs {value.shape})")
                else:
                    print(f"     ✗ 键不存在于当前模型")
                    skipped_keys.append(f"{key} (not found in current model)")

            if codebook_merge_msgs:
                print(f"\n{'='*80}")
                print("📋 【码本尺寸变化】合并加载摘要（各张量）")
                for line in codebook_merge_msgs:
                    print(f"   • {line}")
                print(f"{'='*80}")
            
            # 加载更新后的参数
            print(f"\n📥 调用 self.vq.load_state_dict()...")
            self.vq.load_state_dict(vq_dict, strict=False)
            print(f"   ✅ load_state_dict 完成")
            
            # 验证加载后的codebook
            print(f"\n📊 加载后的 VQ Codebook:")
            after_embeddings = self.vq.embeddings.weight.data
            print(f"   Shape: {after_embeddings.shape}")
            print(f"   Min: {after_embeddings.min().item():.6f}, Max: {after_embeddings.max().item():.6f}")
            print(f"   Mean: {after_embeddings.mean().item():.6f}, Std: {after_embeddings.std().item():.6f}")
            print(f"   前3个code的前5维:")
            for i in range(min(3, after_embeddings.shape[0])):
                print(f"     Code {i}: {after_embeddings[i, :5].tolist()}")
            
            # 计算变化
            diff = (after_embeddings - before_embeddings).abs().max().item()
            print(f"\n   🔍 加载前后的最大差异: {diff:.6e}")
            if diff < 1e-6:
                print(f"   ⚠️  警告: Codebook 几乎没有变化！可能加载失败！")
            else:
                print(f"   ✅ Codebook 已更新！")
            
            print(f"\n   VQ: {len(loaded_keys)} parameters loaded")
            if loaded_keys:
                print(f"      ✅ Loaded: {', '.join(loaded_keys)}")
            if skipped_keys:
                print(f"      ⚠️  Skipped: {', '.join(skipped_keys)}")
            
            # 特别说明EMA buffer的处理
            if self.use_ema_update:
                print(f"\n   📋 EMA 模式检查 (use_ema_update=True):")
                if 'ema_cluster_size' in loaded_keys and 'ema_w' in loaded_keys:
                    print(f"      ✅ EMA buffers loaded from pretrained model")
                else:
                    print(f"      ⚠️  EMA buffers not found in pretrained model, will be initialized from scratch")
                    print(f"      可用的VQ键: {list(vq_state_dict.keys())}")
        else:
            print(f"\n   ⚠️  VQ: No pretrained VQ parameters provided or empty dict, using random initialization")
            if vq_state_dict is None:
                print(f"      原因: vq_state_dict is None")
            elif len(vq_state_dict) == 0:
                print(f"      原因: vq_state_dict is empty")
        
        print(f"{'='*80}\n")
    
    def convert_to_fp16(self) -> None:
        """
        Convert encoder, decoder, and VQ codebook to float16.
        This method is called by the trainer when loading checkpoints with fp16_mode='inflat_all'.
        """
        if hasattr(self.encoder, 'convert_to_fp16'):
            self.encoder.convert_to_fp16()
        if hasattr(self.decoder, 'convert_to_fp16'):
            self.decoder.convert_to_fp16()
        # Convert VQ codebook embeddings to fp16
        if hasattr(self.vq, 'embeddings'):
            self.vq.embeddings.weight.data = self.vq.embeddings.weight.data.half()
    
    def convert_to_fp32(self) -> None:
        """
        Convert encoder, decoder, and VQ codebook to float32.
        """
        if hasattr(self.encoder, 'convert_to_fp32'):
            self.encoder.convert_to_fp32()
        if hasattr(self.decoder, 'convert_to_fp32'):
            self.decoder.convert_to_fp32()
        # Convert VQ codebook embeddings to fp32
        if hasattr(self.vq, 'embeddings'):
            self.vq.embeddings.weight.data = self.vq.embeddings.weight.data.float()


# 向后兼容的别名
Direct3DS2_VQVAE = SparseSDFVQVAE
