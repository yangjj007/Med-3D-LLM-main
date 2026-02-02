"""
Trainer for Sparse SDF VQVAE model.
"""

from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...modules import sparse as sp


class SparseSDF_VQVAETrainer(BasicTrainer):
    """
    Trainer for Sparse SDF VQVAE.
    
    Args:
        models (dict[str, nn.Module]): Models to train. Should contain 'vqvae' key.
        dataset (torch.utils.data.Dataset): Dataset returning sparse SDF data.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
        
        lambda_vq (float): VQ loss weight.
        lambda_commitment (float): Commitment loss weight.
        loss_type (str): Reconstruction loss type. 'mse' for MSE loss, 'l1' for L1 loss, 'l1_l2' for combined L1+L2 loss.
        pretrained_vae_path (str): Path to pretrained VAE checkpoint to initialize from.
        training_stage (int): Training stage. 1 = freeze VAE, train codebook only; 2 = joint training (default: 1).
    """
    
    def __init__(
        self,
        models,
        dataset,
        *,
        lambda_vq: float = 1.0,
        lambda_commitment: float = 0.25,
        loss_type: str = 'mse',
        pretrained_vae_path: str = None,
        training_stage: int = 1,
        load_dir=None,
        step=None,
        **kwargs
    ):
        # 保存配置，稍后使用
        self.lambda_vq = lambda_vq
        self.lambda_commitment = lambda_commitment
        self.loss_type = loss_type
        self.training_stage = training_stage
        self.pretrained_vae_path = pretrained_vae_path
        
        # 检查是否有实际的checkpoint要加载
        # load_dir 可能被设置为 output_dir，但如果没有checkpoint文件，就不算"恢复训练"
        has_checkpoint_to_load = False
        if load_dir is not None and step is not None:
            # 显式提供了step，肯定要加载checkpoint
            has_checkpoint_to_load = True
        elif load_dir is not None:
            # 只提供了load_dir，检查是否真的有checkpoint文件
            import os
            import glob
            ckpt_files = glob.glob(os.path.join(load_dir, 'ckpts', 'misc_*.pt'))
            has_checkpoint_to_load = len(ckpt_files) > 0
        
        # 只有当：1) 提供了pretrained_vae_path，且 2) 没有checkpoint要恢复时，才加载预训练权重
        self._should_load_pretrained = (pretrained_vae_path is not None and 
                                       not has_checkpoint_to_load)
        
        print(f"\n{'='*80}")
        print(f"🔍 [DEBUG] SparseSDF_VQVAETrainer.__init__ 参数检查")
        print(f"{'='*80}")
        print(f"  pretrained_vae_path: {pretrained_vae_path}")
        print(f"  load_dir: {load_dir}")
        print(f"  step: {step}")
        print(f"  has_checkpoint_to_load: {has_checkpoint_to_load}")
        print(f"  _should_load_pretrained: {self._should_load_pretrained}")
        print(f"{'='*80}\n")
        
        # 调用父类初始化
        super().__init__(models, dataset, load_dir=load_dir, step=step, **kwargs)
    
    def init_models_and_more(self, **kwargs):
        """
        重写父类方法，在初始化DDP和收集参数之前先配置训练阶段（冻结参数）
        
        关键时序：
        1. 先加载预训练权重（如果需要且不是从checkpoint恢复）
        2. 配置训练阶段（冻结不需要训练的参数）
        3. 调用父类方法（收集参数、初始化DDP、optimizer等）
        
        这样可以确保DDP和optimizer只包含真正需要训练的参数，避免
        "parameters that were not used in producing loss" 错误
        """
        # ===== 阶段1：加载预训练权重（仅当不从checkpoint恢复时）=====
        if self._should_load_pretrained:
            if self.is_master:
                print(f"\n{'='*80}")
                print(f"[INFO] Loading pretrained VAE weights...")
                print(f"{'='*80}")
            self._load_pretrained_vae(self.pretrained_vae_path)
        
        # ===== 阶段2：配置训练阶段（冻结参数）=====
        # ⚠️ 关键：必须在父类的init_models_and_more之前调用
        # 这样父类收集model_params时就不会包含冻结的参数
        self._configure_training_stage()
        
        # ===== 阶段3：调用父类方法 =====
        # 此时会：
        # - 收集model_params（只包含requires_grad=True的参数）✅
        # - 初始化DDP（只包含可训练参数，避免unused parameter错误）✅
        # - 初始化optimizer（只优化可训练参数）✅
        if self.is_master:
            print(f"\n{'='*80}")
            print(f"[INFO] Initializing trainer components with correct parameter set...")
            print(f"{'='*80}")
        super().init_models_and_more(**kwargs)
        
        if self.is_master:
            print(f"\n{'='*80}")
            print(f"[SUCCESS] Trainer initialization complete!")
            print(f"  - Model parameters: {len(self.model_params)}")
            print(f"  - Optimizer parameters: {sum(len(g['params']) for g in self.optimizer.param_groups)}")
            print(f"{'='*80}\n")
    
    def _load_pretrained_vae(self, pretrained_vae_path: str):
        """
        Load pretrained VAE weights.
        
        Args:
            pretrained_vae_path: Path to pretrained VAE checkpoint (.pth file)
        """
        if self.is_master:
            print(f'\n{"="*80}')
            print(f'🔧 [DEBUG] _load_pretrained_vae 被调用')
            print(f'{"="*80}')
            print(f'📁 预训练权重路径: {pretrained_vae_path}')
            print(f'📋 self._should_load_pretrained: {self._should_load_pretrained}')
        
        # Load checkpoint
        if self.is_master:
            print(f'\n📦 正在加载 checkpoint...')
        checkpoint = torch.load(pretrained_vae_path, map_location='cpu', weights_only=True)
        if self.is_master:
            print(f'✅ Checkpoint 加载成功')
            print(f'   顶层键: {list(checkpoint.keys())}')
        
        # Extract VAE state dict
        if 'vae' in checkpoint:
            vae_state_dict = checkpoint['vae']
            if self.is_master:
                print(f'   使用键: "vae"')
        elif 'state_dict' in checkpoint:
            vae_state_dict = checkpoint['state_dict']
            if self.is_master:
                print(f'   使用键: "state_dict"')
        else:
            vae_state_dict = checkpoint
            if self.is_master:
                print(f'   直接使用整个 checkpoint')
        
        # Get encoder and decoder state dicts
        encoder_state_dict = {
            k.replace('encoder.', ''): v
            for k, v in vae_state_dict.items()
            if k.startswith('encoder.')
        }
        decoder_state_dict = {
            k.replace('decoder.', ''): v
            for k, v in vae_state_dict.items()
            if k.startswith('decoder.')
        }
        
        # Get VQ state dict (if exists)
        vq_state_dict = {
            k.replace('vq.', ''): v
            for k, v in vae_state_dict.items()
            if k.startswith('vq.')
        }
        
        if self.is_master:
            print(f'\n📊 提取的参数统计:')
            print(f'   Encoder: {len(encoder_state_dict)} 个参数')
            print(f'   Decoder: {len(decoder_state_dict)} 个参数')
            print(f'   VQ: {len(vq_state_dict)} 个参数')
            
            if vq_state_dict:
                print(f'\n   VQ 参数详情:')
                for key, value in vq_state_dict.items():
                    if isinstance(value, torch.Tensor):
                        print(f'     - {key}: shape={value.shape}, dtype={value.dtype}')
                        if key == 'embeddings.weight':
                            print(f'       统计: min={value.min().item():.6f}, max={value.max().item():.6f}, '
                                  f'mean={value.mean().item():.6f}, std={value.std().item():.6f}')
                            print(f'       前3个code的前5维:')
                            for i in range(min(3, value.shape[0])):
                                print(f'         Code {i}: {value[i, :5].tolist()}')
            else:
                print(f'   ⚠️  VQ state dict 是空的！')
        
        # Load into VQVAE model
        vqvae = self.models['vqvae']
        if hasattr(vqvae, 'module'):
            vqvae = vqvae.module
            if self.is_master:
                print(f'\n🔧 模型被 DDP 包装，使用 module 属性')
        
        if self.is_master:
            print(f'\n🔍 检查模型是否有 load_pretrained_vae 方法...')
            print(f'   hasattr(vqvae, "load_pretrained_vae"): {hasattr(vqvae, "load_pretrained_vae")}')
        
        if hasattr(vqvae, 'load_pretrained_vae'):
            if self.is_master:
                print(f'\n📥 调用 vqvae.load_pretrained_vae()...')
                # 在加载前记录当前codebook
                print(f'\n📊 加载前的 Codebook 统计:')
                current_embeddings = vqvae.vq.embeddings.weight.data
                print(f'   Shape: {current_embeddings.shape}')
                print(f'   Min: {current_embeddings.min().item():.6f}, Max: {current_embeddings.max().item():.6f}')
                print(f'   Mean: {current_embeddings.mean().item():.6f}, Std: {current_embeddings.std().item():.6f}')
                print(f'   前3个code的前5维:')
                for i in range(min(3, current_embeddings.shape[0])):
                    print(f'     Code {i}: {current_embeddings[i, :5].tolist()}')
            
            vqvae.load_pretrained_vae(encoder_state_dict, decoder_state_dict, vq_state_dict)
            
            if self.is_master:
                print(f'\n📊 加载后的 Codebook 统计:')
                new_embeddings = vqvae.vq.embeddings.weight.data
                print(f'   Shape: {new_embeddings.shape}')
                print(f'   Min: {new_embeddings.min().item():.6f}, Max: {new_embeddings.max().item():.6f}')
                print(f'   Mean: {new_embeddings.mean().item():.6f}, Std: {new_embeddings.std().item():.6f}')
                print(f'   前3个code的前5维:')
                for i in range(min(3, new_embeddings.shape[0])):
                    print(f'     Code {i}: {new_embeddings[i, :5].tolist()}')
                
                # 检查是否真的改变了
                if vq_state_dict and 'embeddings.weight' in vq_state_dict:
                    original_embeddings = vq_state_dict['embeddings.weight']
                    diff = (new_embeddings.cpu() - original_embeddings).abs().max().item()
                    print(f'\n   ✅ 与预训练权重的最大差异: {diff:.6e}')
                    if diff < 1e-6:
                        print(f'   ✅ Codebook 已成功加载！')
                    else:
                        print(f'   ⚠️  Codebook 与预训练权重有差异！')
                
                print(f'\n✅ 预训练 VAE 权重加载完成')
                print(f'{"="*80}\n')
        else:
            if self.is_master:
                print(f'❌ 警告: VQVAE 模型没有 load_pretrained_vae 方法')
                print(f'{"="*80}\n')
    
    def _configure_training_stage(self):
        """
        Configure model parameters based on training stage.
        
        Stage 1: Freeze encoder and decoder, train only codebook
        Stage 2: Unfreeze all parameters for joint training
        """
        vqvae = self.models['vqvae']
        if hasattr(vqvae, 'module'):
            vqvae = vqvae.module
        
        if self.training_stage == 1:
            # Stage 1: Freeze encoder and decoder
            if hasattr(vqvae, 'encoder'):
                for param in vqvae.encoder.parameters():
                    param.requires_grad = False
            
            if hasattr(vqvae, 'decoder'):
                for param in vqvae.decoder.parameters():
                    param.requires_grad = False
            
            # Ensure VQ parameters are trainable (except for EMA mode where embeddings have no grad)
            if hasattr(vqvae, 'vq'):
                for param in vqvae.vq.parameters():
                    param.requires_grad = True
                # 但如果是EMA模式，码本权重不需要梯度
                if hasattr(vqvae, 'use_ema_update') and vqvae.use_ema_update:
                    vqvae.vq.embeddings.weight.requires_grad = False
            
            if self.is_master:
                print("\n" + "=" * 80)
                print("[Stage 1] Encoder and Decoder frozen, training Codebook only")
                
                # 打印码本更新模式
                if hasattr(vqvae, 'use_ema_update'):
                    if vqvae.use_ema_update:
                        print(f"[Codebook Update Mode] EMA (decay={vqvae.vq.decay}, epsilon={vqvae.vq.epsilon})")
                    else:
                        print(f"[Codebook Update Mode] Gradient (lambda_vq={self.lambda_vq})")
                
                print("=" * 80)
                
                # Count trainable parameters
                total_params = sum(p.numel() for p in vqvae.parameters())
                trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print(f"Frozen parameters: {total_params - trainable_params:,}")
                print("=" * 80 + "\n")
        
        elif self.training_stage == 2:
            # Stage 2: Unfreeze all parameters
            for param in vqvae.parameters():
                param.requires_grad = True
            
            # 但如果是EMA模式，码本权重不需要梯度
            if hasattr(vqvae, 'use_ema_update') and vqvae.use_ema_update:
                vqvae.vq.embeddings.weight.requires_grad = False
            
            if self.is_master:
                print("\n" + "=" * 80)
                print("[Stage 2] Joint training: Encoder + Decoder + Codebook")
                
                # 打印码本更新模式
                if hasattr(vqvae, 'use_ema_update'):
                    if vqvae.use_ema_update:
                        print(f"[Codebook Update Mode] EMA (decay={vqvae.vq.decay}, epsilon={vqvae.vq.epsilon})")
                    else:
                        print(f"[Codebook Update Mode] Gradient (lambda_vq={self.lambda_vq})")
                
                print("=" * 80)
                
                # Count trainable parameters
                total_params = sum(p.numel() for p in vqvae.parameters())
                trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print("=" * 80 + "\n")
        
        else:
            raise ValueError(f"Invalid training_stage: {self.training_stage}. Must be 1 or 2.")
    
    def training_losses(
        self,
        sparse_sdf: torch.Tensor,
        sparse_index: torch.Tensor,
        batch_idx: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.
        
        Args:
            sparse_sdf: SDF values [total_N, 1]
            sparse_index: 3D coordinates [total_N, 3]
            batch_idx: Batch indices [total_N]
        
        Returns:
            Tuple of (loss_dict, status_dict)
        """
        print(f"\n{'='*100}")
        print(f"[DEBUG training_losses] 开始损失计算 - Step {self.step}")
        print(f"{'='*100}")
        
        # 输入数据检查
        print(f"[DEBUG training_losses] 输入数据形状:")
        print(f"  sparse_sdf.shape: {sparse_sdf.shape}, dtype: {sparse_sdf.dtype}")
        print(f"  sparse_index.shape: {sparse_index.shape}, dtype: {sparse_index.dtype}")
        print(f"  batch_idx.shape: {batch_idx.shape}, dtype: {batch_idx.dtype}")
        print(f"[DEBUG training_losses] 输入数据统计:")
        print(f"  sparse_sdf - min: {sparse_sdf.min().item():.6f}, max: {sparse_sdf.max().item():.6f}, mean: {sparse_sdf.mean().item():.6f}")
        print(f"  sparse_index - min: {sparse_index.min().item()}, max: {sparse_index.max().item()}")
        print(f"  batch_idx - unique: {torch.unique(batch_idx).tolist()}")
        
        # Construct sparse tensor input
        coords = torch.cat([batch_idx.unsqueeze(-1), sparse_index], dim=-1).int()
        x = sp.SparseTensor(sparse_sdf, coords)
        print(f"[DEBUG training_losses] 构建SparseTensor: shape={x.shape}, feats.shape={x.feats.shape}")
        
        # Forward pass through VQVAE
        vqvae = self.training_models['vqvae']
        print(f"[DEBUG training_losses] 调用VQVAE forward...")
        print(f"[DEBUG training_losses] VQVAE训练模式: {vqvae.training}")
        print(f"[DEBUG training_losses] 当前训练步数: {self.step}")
        
        # 检查模型参数是否正常
        if hasattr(vqvae, 'module'):
            vqvae_module = vqvae.module
        else:
            vqvae_module = vqvae
        
        # 检查encoder参数
        encoder_requires_grad = any(p.requires_grad for p in vqvae_module.encoder.parameters())
        decoder_requires_grad = any(p.requires_grad for p in vqvae_module.decoder.parameters())
        vq_requires_grad = any(p.requires_grad for p in vqvae_module.vq.parameters())
        
        print(f"[DEBUG training_losses] 参数训练状态:")
        print(f"  encoder requires_grad: {encoder_requires_grad}")
        print(f"  decoder requires_grad: {decoder_requires_grad}")
        print(f"  vq requires_grad: {vq_requires_grad}")
        
        # 检查VQ codebook权重
        vq_embeddings = vqvae_module.vq.embeddings.weight
        print(f"[DEBUG training_losses] VQ Codebook统计:")
        print(f"  shape: {vq_embeddings.shape}")
        print(f"  min: {vq_embeddings.min().item():.6f}, max: {vq_embeddings.max().item():.6f}")
        print(f"  mean: {vq_embeddings.mean().item():.6f}, std: {vq_embeddings.std().item():.6f}")
        print(f"  requires_grad: {vq_embeddings.requires_grad}")
        
        # 传递当前步数到模型，用于 K-means 重估计
        outputs = vqvae(x, current_step=self.step)
        print(f"[DEBUG training_losses] VQVAE forward完成")
        
        # Extract outputs from dictionary
        recon = outputs['reconst_x']
        vq_loss = outputs.get('vq_loss')  # 可能为None（EMA模式）
        commitment_loss = outputs['commitment_loss']
        codebook_stats = outputs.get('codebook_stats', {})
        
        print(f"[DEBUG training_losses] VQVAE输出:")
        print(f"  recon.shape: {recon.shape}, recon.feats.shape: {recon.feats.shape}")
        print(f"  recon.feats - min: {recon.feats.min().item():.6f}, max: {recon.feats.max().item():.6f}, mean: {recon.feats.mean().item():.6f}")
        if vq_loss is not None:
            print(f"  vq_loss: {vq_loss.item():.6f} (type: {type(vq_loss)})")
            print(f"  vq_loss requires_grad: {vq_loss.requires_grad}")
        else:
            print(f"  vq_loss: None (EMA update mode)")
        print(f"  commitment_loss: {commitment_loss.item():.6f} (type: {type(commitment_loss)})")
        print(f"  commitment_loss requires_grad: {commitment_loss.requires_grad}")
        
        # Align reconstruction with input coordinates (ShapeLLM方法)
        # Decoder可能生成扩展体素，需要只对输入位置计算损失
        input_coords = x.coords  # [N_input, 4] (batch, x, y, z)
        output_coords = recon.coords  # [N_output, 4]
        
        print(f"[DEBUG training_losses] 坐标对齐:")
        print(f"  input_coords.shape: {input_coords.shape}")
        print(f"  output_coords.shape: {output_coords.shape}")
        
        # 构建坐标到索引的映射字典（更高效的方法）
        # 将输入坐标转换为字符串键
        input_coord_dict = {}
        for i, coord in enumerate(input_coords):
            key = tuple(coord.cpu().tolist())
            input_coord_dict[key] = i
        
        # 找到输出中匹配输入的体素
        aligned_indices_output = []
        aligned_indices_input = []
        for i, coord in enumerate(output_coords):
            key = tuple(coord.cpu().tolist())
            if key in input_coord_dict:
                aligned_indices_output.append(i)
                aligned_indices_input.append(input_coord_dict[key])
        
        if len(aligned_indices_output) == 0:
            raise RuntimeError("❌ 没有找到输入输出之间匹配的体素！这不应该发生。")
        
        print(f"  对齐的体素数: {len(aligned_indices_output)} / {len(input_coords)} ({len(aligned_indices_output)/len(input_coords)*100:.2f}%)")
        
        # 提取对齐的特征
        aligned_indices_output = torch.tensor(aligned_indices_output, device=recon.feats.device)
        aligned_indices_input = torch.tensor(aligned_indices_input, device=sparse_sdf.device)
        
        recon_aligned = recon.feats[aligned_indices_output]
        target_aligned = sparse_sdf[aligned_indices_input]
        
        print(f"[DEBUG training_losses] 对齐后的特征:")
        print(f"  recon_aligned.shape: {recon_aligned.shape}")
        print(f"  target_aligned.shape: {target_aligned.shape}")
        print(f"  recon_aligned - min: {recon_aligned.min().item():.6f}, max: {recon_aligned.max().item():.6f}, mean: {recon_aligned.mean().item():.6f}")
        print(f"  target_aligned - min: {target_aligned.min().item():.6f}, max: {target_aligned.max().item():.6f}, mean: {target_aligned.mean().item():.6f}")
        print(f"  recon_aligned requires_grad: {recon_aligned.requires_grad}")
        
        # Compute reconstruction loss on aligned voxels
        if self.loss_type == 'mse':
            recon_loss = F.mse_loss(recon_aligned, target_aligned, reduction='mean')
        elif self.loss_type == 'l1':
            recon_loss = F.l1_loss(recon_aligned, target_aligned, reduction='mean')
        elif self.loss_type == 'l1_l2':
            # Combined L1 + L2 loss for stage 2
            recon_loss = 0.5 * F.l1_loss(recon_aligned, target_aligned, reduction='mean') + \
                         0.5 * F.mse_loss(recon_aligned, target_aligned, reduction='mean')
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')
        
        print(f"[DEBUG training_losses] 损失计算 (loss_type={self.loss_type}):")
        print(f"  recon_loss: {recon_loss.item():.6f}")
        print(f"  recon_loss requires_grad: {recon_loss.requires_grad}")
        print(f"  lambda_vq: {self.lambda_vq}")
        print(f"  lambda_commitment: {self.lambda_commitment}")
        
        # Total loss - 根据vq_loss是否为None选择不同的计算方式
        if vq_loss is not None:
            # 梯度模式：使用vq_loss
            total_loss = recon_loss + self.lambda_vq * vq_loss + self.lambda_commitment * commitment_loss
            print(f"[DEBUG training_losses] 总损失 (梯度模式):")
            print(f"  total_loss: {total_loss.item():.6f}")
            print(f"  total_loss requires_grad: {total_loss.requires_grad}")
            print(f"  计算: {recon_loss.item():.6f} + {self.lambda_vq}*{vq_loss.item():.6f} + {self.lambda_commitment}*{commitment_loss.item():.6f}")
        else:
            # EMA模式：不使用vq_loss
            total_loss = recon_loss + self.lambda_commitment * commitment_loss
            print(f"[DEBUG training_losses] 总损失 (EMA模式):")
            print(f"  total_loss: {total_loss.item():.6f}")
            print(f"  total_loss requires_grad: {total_loss.requires_grad}")
            print(f"  计算: {recon_loss.item():.6f} + {self.lambda_commitment}*{commitment_loss.item():.6f}")
        print(f"{'='*100}\n")
        
        # Loss dictionary
        terms = edict(
            loss=total_loss,
            recon=recon_loss,
            commitment=commitment_loss,
        )
        if vq_loss is not None:
            terms['vq'] = vq_loss
        
        # Status dictionary with additional metrics
        status = edict(
            num_input_voxels=len(input_coords),
            num_output_voxels=len(output_coords),
            num_aligned_voxels=len(aligned_indices_output),
            alignment_ratio=len(aligned_indices_output) / len(input_coords),
            codebook_perplexity=codebook_stats.get('perplexity', 0.0),
            codebook_entropy=codebook_stats.get('entropy', 0.0),
            codebook_utilization_ratio=codebook_stats.get('utilization_ratio', 0.0),
            codebook_unique_count=codebook_stats.get('unique_count', 0),
        )
        
        return terms, status
    
    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=16, batch_size=1, verbose=False):  # 默认batch_size=1避免内存问题
        """Take a snapshot of the model's performance."""
        # Use training batch_size if not specified, default to 2 for safety
        if batch_size is None:
            batch_size = getattr(self, 'batch_size_per_gpu', 2)
        super().snapshot(suffix=suffix, num_samples=num_samples, batch_size=batch_size, verbose=verbose)
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Run snapshot inference.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for inference
            verbose: Whether to print verbose output
        
        Returns:
            Dictionary of samples for visualization
        """
        try:
            # Create a dataset copy with potentially reduced max_points for stability
            dataset_copy = copy.deepcopy(self.dataset)
            original_max_points = getattr(dataset_copy, 'max_points', None)
            
            dataloader = DataLoader(
                dataset_copy,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=dataset_copy.collate_fn if hasattr(dataset_copy, 'collate_fn') else None,
            )
            
            # Get VQVAE model
            vqvae = self.models['vqvae']
            if hasattr(vqvae, 'module'):
                vqvae = vqvae.module
            
            # Get model dtype for fp16 compatibility
            model_dtype = vqvae.encoder.dtype if hasattr(vqvae.encoder, 'dtype') else torch.float32
            
            # 检查encoder的out_layer权重（仅在首次快照时检查）
            if self.step == 0 or self.step % (self.i_sample * 10) == 0:
                if hasattr(vqvae.encoder, 'out_layer'):
                    out_layer = vqvae.encoder.out_layer
                    weight_sum = out_layer.weight.abs().sum().item()
                    if weight_sum < 1e-6:
                        print(f"\n⚠️  警告: Encoder out_layer权重几乎为零！")
                        print(f"  这可能表示模型未被训练或checkpoint未正确加载")
                        print(f"  权重绝对值总和: {weight_sum:.6e}\n")
            
            # Inference
            gts = []
            recons = []
            
            for i in range(0, num_samples, batch_size):
                batch = min(batch_size, num_samples - i)
                
                data = next(iter(dataloader))
                
                # Move to device (data already collated, no need to slice)
                sparse_sdf = data['sparse_sdf'].cuda() if isinstance(data['sparse_sdf'], torch.Tensor) else data['sparse_sdf']
                sparse_index = data['sparse_index'].cuda() if isinstance(data['sparse_index'], torch.Tensor) else data['sparse_index']
                batch_idx = data['batch_idx'].cuda() if isinstance(data['batch_idx'], torch.Tensor) else data['batch_idx']
                
                # Only keep points belonging to current batch
                mask = batch_idx < batch
                sparse_sdf = sparse_sdf[mask]
                sparse_index = sparse_index[mask]
                batch_idx = batch_idx[mask]
                
                # Convert to model dtype (for fp16 compatibility)
                sparse_sdf = sparse_sdf.to(dtype=model_dtype)
                
                # Construct sparse tensor
                coords = torch.cat([batch_idx.unsqueeze(-1), sparse_index], dim=-1).int()
                x = sp.SparseTensor(sparse_sdf, coords)
                
                # Check for any NaN or Inf
                if torch.isnan(sparse_sdf).any() or torch.isinf(sparse_sdf).any():
                    print(f"⚠️  警告: 检测到NaN或Inf值在sparse_sdf中！")
                
                # Encode and decode
                try:
                    # 构建字典格式的 batch，符合 Encode 方法的输入要求
                    batch_dict = {
                        'sparse_sdf': sparse_sdf,
                        'sparse_index': sparse_index,
                        'batch_idx': batch_idx,
                    }
                    
                    encoding_indices = vqvae.Encode(batch_dict)
                except Exception as e:
                    print(f"❌ 编码错误: {type(e).__name__}: {e}")
                    raise
                
                recon = vqvae.Decode(encoding_indices)
                
                # 清理CUDA缓存，避免内存累积
                torch.cuda.empty_cache()
                
                # Store results
                gts.append({
                    'sparse_sdf': sparse_sdf,
                    'sparse_index': sparse_index,
                    'batch_idx': batch_idx,
                })
                recons.append({
                    'sparse_sdf': recon.feats,
                    'sparse_index': recon.coords[:, 1:],
                    'batch_idx': recon.coords[:, 0],
                })
            
            # Combine results
            gt_combined = {
                'sparse_sdf': torch.cat([g['sparse_sdf'] for g in gts], dim=0),
                'sparse_index': torch.cat([g['sparse_index'] for g in gts], dim=0),
                'batch_idx': torch.cat([g['batch_idx'] for g in gts], dim=0),
            }
            
            recon_combined = {
                'sparse_sdf': torch.cat([r['sparse_sdf'] for r in recons], dim=0),
                'sparse_index': torch.cat([r['sparse_index'] for r in recons], dim=0),
                'batch_idx': torch.cat([r['batch_idx'] for r in recons], dim=0),
            }
            
            sample_dict = {
                'gt': {'value': gt_combined, 'type': 'sample'},
                'recon': {'value': recon_combined, 'type': 'sample'},
            }
            
            return sample_dict
            
        except Exception as e:
            print(f"\n❌ run_snapshot错误:")
            print(f"  异常类型: {type(e).__name__}")
            print(f"  异常信息: {str(e)}")
            import traceback
            print(f"\n完整堆栈跟踪:")
            traceback.print_exc()
            raise

